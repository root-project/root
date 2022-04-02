/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

/**
\file RooCompositeDataStore.cxx
\class RooCompositeDataStore
\ingroup Roofitcore

RooCompositeDataStore combines several disjunct datasets into one. This is useful for simultaneous PDFs
that do not depend on the same observable such as a PDF depending on `x` combined with another one depending
on `y`.
The composite storage will store two different datasets, `{x}` and `{y}`, but they can be passed as a single
dataset to RooFit operations. A category tag will define which dataset has to be passed to which likelihood.

When iterated from start to finish, datasets will be traversed in the order of the category index.
**/

#include "RooCompositeDataStore.h"

#include "RooMsgService.h"
#include "RooFormulaVar.h"
#include "RooRealVar.h"
#include "RooTrace.h"
#include "RooCategory.h"

#include "TTree.h"
#include "TChain.h"

#include <iomanip>
#include <iostream>

using namespace std;

ClassImp(RooCompositeDataStore);


////////////////////////////////////////////////////////////////////////////////

RooCompositeDataStore::RooCompositeDataStore()
{
  TRACE_CREATE
}



////////////////////////////////////////////////////////////////////////////////
/// Convert map by label to map by index for more efficient internal use

RooCompositeDataStore::RooCompositeDataStore(
        RooStringView name, RooStringView title,
        const RooArgSet& vars, RooCategory& indexCat,map<std::string,RooAbsDataStore*> const& inputData) :
  RooAbsDataStore(name,title,RooArgSet(vars,indexCat)), _indexCat(&indexCat)
{
  for (const auto& iter : inputData) {
    const RooAbsCategory::value_type idx = indexCat.lookupIndex(iter.first);
    _dataMap[idx] = iter.second;
  }
  TRACE_CREATE
}


////////////////////////////////////////////////////////////////////////////////
/// Convert map by label to map by index for more efficient internal use

RooCompositeDataStore::RooCompositeDataStore(const RooCompositeDataStore& other, const char* newname) :
  RooAbsDataStore(other,newname), _indexCat(other._indexCat), _curStore(other._curStore), _curIndex(other._curIndex), _ownComps(kTRUE)
{
  for (const auto& item : other._dataMap) {
    RooAbsDataStore* clonedata = item.second->clone() ;
    _dataMap[item.first] = clonedata ;
  }
  TRACE_CREATE
}


////////////////////////////////////////////////////////////////////////////////
/// Update index category pointer, if it is contained in input argument vars

RooCompositeDataStore::RooCompositeDataStore(const RooCompositeDataStore& other, const RooArgSet& vars, const char* newname) :
  RooAbsDataStore(other,vars,newname), _indexCat(other._indexCat), _curStore(other._curStore), _curIndex(other._curIndex), _ownComps(kTRUE)
{
  RooCategory* newIdx = (RooCategory*) vars.find(other._indexCat->GetName()) ;
  if (newIdx) {
    _indexCat = newIdx ;
  }

  // Convert map by label to map by index for more efficient internal use
  for (const auto& item : other._dataMap) {
    RooAbsDataStore* clonedata = item.second->clone(vars) ;
    _dataMap[item.first] = clonedata ;
  }
  TRACE_CREATE
}




////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooCompositeDataStore::~RooCompositeDataStore()
{
  if (_ownComps) {
    for (auto& item : _dataMap) {
      delete item.second;
    }
  }
  TRACE_DESTROY
}


RooAbsDataStore* RooCompositeDataStore::reduce(
        RooStringView name, RooStringView title, const RooArgSet& vars, const RooFormulaVar* cutVar,
        const char* cutRange, std::size_t nStart, std::size_t nStop)
{
  // for the components, we need the set of variables without the index category
  RooArgSet varsNoIndex{vars};
  if (RooAbsArg * indexCat = varsNoIndex.find(*_indexCat)) {
    varsNoIndex.remove(*indexCat,true) ;
  }

  // create an empty RooCompositeDataStore
  auto * out = new RooCompositeDataStore{name, title, varsNoIndex, *_indexCat, std::map<std::string,RooAbsDataStore*>{}};

  // fill it with reduced versions of components
  for (const auto& item : _dataMap) {
    out->_dataMap[item.first] = item.second->reduce(name, title, varsNoIndex, cutVar, cutRange, nStart, nStop);
  }

  // indiceate component ownership and return
  out->_ownComps = true;
  return out;
}


////////////////////////////////////////////////////////////////////////////////
/// Forward recalculate request to all subsets

void RooCompositeDataStore::recalculateCache(const RooArgSet* proj, Int_t firstEvent, Int_t lastEvent, Int_t stepSize, Bool_t skipZeroWeights)
{
  for (auto const& item : _dataMap) {
    item.second->recalculateCache(proj,firstEvent,lastEvent,stepSize,skipZeroWeights) ;
  }
}


////////////////////////////////////////////////////////////////////////////////

Bool_t RooCompositeDataStore::hasFilledCache() const
{
  Bool_t ret(kFALSE) ;
  for (auto const& item : _dataMap) {
    ret |= item.second->hasFilledCache() ;
  }
  return ret ;
}


////////////////////////////////////////////////////////////////////////////////

void RooCompositeDataStore::forceCacheUpdate()
{
  for (auto const& item : _dataMap) {
    item.second->forceCacheUpdate() ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Forward fill request to appropriate subset

Int_t RooCompositeDataStore::fill()
{
  RooAbsDataStore* subset = _dataMap[_indexCat->getCurrentIndex()] ;
  const_cast<RooArgSet*>((subset->get()))->assignValueOnly(_vars) ;
  return subset->fill() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Forward fill request to appropriate subset

Double_t RooCompositeDataStore::sumEntries() const
{
  Double_t sum(0) ;

  for (auto const& item : _dataMap) {
    sum+= item.second->sumEntries() ;
  }
  return sum ;
}



////////////////////////////////////////////////////////////////////////////////
/// Load the n-th data point (n='idx') in memory
/// and return a pointer to the internal RooArgSet
/// holding its coordinates.

const RooArgSet* RooCompositeDataStore::get(Int_t idx) const
{
  Int_t offset(0) ;
  for (auto const& item : _dataMap) {
    if (idx>=(offset+item.second->numEntries())) {
      offset += item.second->numEntries() ;
      continue ;
    }
    _vars.assign(*item.second->get(idx-offset)) ;

    _indexCat->setIndex(item.first) ;
    _curStore = item.second ;
    _curIndex = idx-offset ;

    return &_vars ;
  }
  return 0 ;
}



////////////////////////////////////////////////////////////////////////////////

Double_t RooCompositeDataStore::weight() const
{
  if (!_curStore) get(0) ;
  // coverity[FORWARD_NULL]
  return _curStore->weight(_curIndex) ;
}


////////////////////////////////////////////////////////////////////////////////

Double_t RooCompositeDataStore::weightError(RooAbsData::ErrorType etype) const
{
  if (!_curStore) get(0) ;
  // coverity[FORWARD_NULL]
  return _curStore->weightError(etype) ;
}




////////////////////////////////////////////////////////////////////////////////

void RooCompositeDataStore::weightError(Double_t& lo, Double_t& hi, RooAbsData::ErrorType etype) const
{
  if (!_curStore) get(0) ;
  // coverity[FORWARD_NULL]
  return _curStore->weightError(lo,hi,etype) ;
}




////////////////////////////////////////////////////////////////////////////////

Bool_t RooCompositeDataStore::isWeighted() const
{
  for (auto const& item : _dataMap) {
    if (item.second->isWeighted()) return kTRUE ;
  }
  return kFALSE ; ;
}


////////////////////////////////////////////////////////////////////////////////

void RooCompositeDataStore::loadValues(const RooAbsDataStore*, const RooFormulaVar*, const char*, std::size_t, std::size_t)
{
  throw(std::runtime_error("RooCompositeDataSore::loadValues() NOT IMPLEMENTED")) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Change name of internal observable named 'from' into 'to'

Bool_t RooCompositeDataStore::changeObservableName(const char* from, const char* to)
{

  // Find observable to be changed
  RooAbsArg* var = _vars.find(from) ;

  // Check that we found it
  if (!var) {
    coutE(InputArguments) << "RooCompositeDataStore::changeObservableName(" << GetName() << " no observable " << from << " in this dataset" << endl ;
    return kTRUE ;
  }

  // Process name change
  var->SetName(to) ;

  // Forward name change request to component datasets
  Bool_t ret(kFALSE) ;
  for (auto const& item : _dataMap) {
    ret |= item.second->changeObservableName(from,to) ;
  }

  return ret ;
}



////////////////////////////////////////////////////////////////////////////////
/// WVE ownership issue here!! Caller (a RooAbsData) should take ownership of all
/// arguments, but only does for the first one here...

RooAbsArg* RooCompositeDataStore::addColumn(RooAbsArg& newVar, Bool_t adjustRange)
{
  RooAbsArg* ret(0) ;
  for (auto const& item : _dataMap) {
    ret = item.second->addColumn(newVar,adjustRange) ;
  }
  if (ret) {
    _vars.add(*ret) ;
  }
  return ret ;
}



////////////////////////////////////////////////////////////////////////////////
/// WVE ownership issue here!! Caller (a RooAbsData) should take ownership of all
/// arguments, but only does for the first one here...

RooArgSet* RooCompositeDataStore::addColumns(const RooArgList& varList)
{
  RooArgSet* ret(0) ;
  for (auto const& item : _dataMap) {
    ret = item.second->addColumns(varList) ;
  }
  if (ret) {
    _vars.add(*ret) ;
  }
  return ret ;
}




////////////////////////////////////////////////////////////////////////////////

RooAbsDataStore* RooCompositeDataStore::merge(const RooArgSet& /*allVars*/, list<RooAbsDataStore*> /*dstoreList*/)
{
  throw string("RooCompositeDataStore::merge() is not implemented yet") ;
}





////////////////////////////////////////////////////////////////////////////////

void RooCompositeDataStore::append(RooAbsDataStore& other)
{
  Int_t nevt = other.numEntries() ;
  for (int i=0 ; i<nevt ; i++) {
    _vars.assign(*other.get(i)) ;
    fill() ;
  }
}



////////////////////////////////////////////////////////////////////////////////

Int_t RooCompositeDataStore::numEntries() const
{
  Int_t n(0) ;
  for (auto const& item : _dataMap) {
    n += item.second->numEntries() ;
  }
  return n ;
}




////////////////////////////////////////////////////////////////////////////////

void RooCompositeDataStore::reset()
{
  for (auto const& item : _dataMap) {
    item.second->reset() ;
  }
}



////////////////////////////////////////////////////////////////////////////////

void RooCompositeDataStore::cacheArgs(const RooAbsArg* owner, RooArgSet& newVarSet, const RooArgSet* nset, Bool_t skipZeroWeights)
{
  for (auto const& item : _dataMap) {
    item.second->cacheArgs(owner,newVarSet,nset,skipZeroWeights) ;
  }
}



////////////////////////////////////////////////////////////////////////////////

void RooCompositeDataStore::setArgStatus(const RooArgSet& set, Bool_t active)
{
  for (auto const& item : _dataMap) {
    RooArgSet* subset = (RooArgSet*) set.selectCommon(*item.second->get()) ;
    item.second->setArgStatus(*subset,active) ;
    delete subset ;
  }
  return ;
}



////////////////////////////////////////////////////////////////////////////////
/// Initialize cache of dataset: attach variables of cache ArgSet
/// to the corresponding TTree branches

void RooCompositeDataStore::attachCache(const RooAbsArg* newOwner, const RooArgSet& inCachedVars)
{
  for (auto const& item : _dataMap) {
    item.second->attachCache(newOwner,inCachedVars) ;
  }
  return ;
}



////////////////////////////////////////////////////////////////////////////////

void RooCompositeDataStore::resetCache()
{
  for (auto const& item : _dataMap) {
    item.second->resetCache() ;
  }
  return ;
}



////////////////////////////////////////////////////////////////////////////////

void RooCompositeDataStore::attachBuffers(const RooArgSet& extObs)
{
  for (auto const& item : _dataMap) {
    item.second->attachBuffers(extObs);
  }
  return ;
}



////////////////////////////////////////////////////////////////////////////////

void RooCompositeDataStore::resetBuffers()
{
  for (auto const& item : _dataMap) {
    item.second->resetBuffers();
  }
  return ;
}


////////////////////////////////////////////////////////////////////////////////

void RooCompositeDataStore::dump()
{
  cout << "RooCompositeDataStore::dump()" << endl ;
  for (auto const& item : _dataMap) {
    cout << "state number " << item.first << " has store " << item.second->IsA()->GetName() << " with variables " << *item.second->get() ;
    if (item.second->isWeighted()) cout << " and is weighted " ;
    cout << endl ;
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Get the weights of the events in the range [first, first+len).
/// This implementation will fill a vector with every event retrieved one by one
/// (even if the weight is constant). Then, it returns a span.
RooSpan<const double> RooCompositeDataStore::getWeightBatch(std::size_t first, std::size_t len) const {
  if (!_weightBuffer) {
    _weightBuffer.reset(new std::vector<double>());
    _weightBuffer->reserve(len);

    for (std::size_t i = 0; i < static_cast<std::size_t>(numEntries()); ++i) {
      _weightBuffer->push_back(weight(i));
    }
  }

  return {_weightBuffer->data() + first, len};
}
