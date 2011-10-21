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

//////////////////////////////////////////////////////////////////////////////
//
// BEGIN_HTML
// RooCompositeDataStore is the abstract base class for data collection that
// use a TTree as internal storage mechanism
// END_HTML
//

#include "RooFit.h"
#include "RooMsgService.h"
#include "RooCompositeDataStore.h"

#include "Riostream.h"
#include "TTree.h"
#include "TChain.h"
#include "TDirectory.h"
#include "TROOT.h"
#include "RooFormulaVar.h"
#include "RooRealVar.h"
#include "RooCategory.h"
#include <iomanip>
using namespace std ;

ClassImp(RooCompositeDataStore)
;


//_____________________________________________________________________________
RooCompositeDataStore::RooCompositeDataStore() : _curStore(0), _curIndex(0)
{
}



//_____________________________________________________________________________
RooCompositeDataStore::RooCompositeDataStore(const char* name, const char* title, const RooArgSet& vars, RooCategory& indexCat,map<string,RooAbsDataStore*> inputData) :
  RooAbsDataStore(name,title,RooArgSet(vars,indexCat)), _dataMap(inputData), _indexCat(&indexCat), _curStore(0), _curIndex(0)
{
}




//_____________________________________________________________________________
RooCompositeDataStore::RooCompositeDataStore(const RooCompositeDataStore& other, const char* newname) :
  RooAbsDataStore(other,newname), _dataMap(other._dataMap), _indexCat(other._indexCat), _curStore(other._curStore), _curIndex(other._curIndex)
{
}


//_____________________________________________________________________________
RooCompositeDataStore::RooCompositeDataStore(const RooCompositeDataStore& other, const RooArgSet& vars, const char* newname) :
  RooAbsDataStore(other,vars,newname), _dataMap(other._dataMap), _indexCat(other._indexCat), _curStore(other._curStore), _curIndex(other._curIndex)
{
}




//_____________________________________________________________________________
RooCompositeDataStore::~RooCompositeDataStore()
{
  // Destructor
}


//_____________________________________________________________________________
Bool_t RooCompositeDataStore::valid() const 
{
  // Return true if currently loaded coordinate is considered valid within
  // the current range definitions of all observables
  return kTRUE ;
}




//_____________________________________________________________________________
Int_t RooCompositeDataStore::fill()
{
  // Forward fill request to appropriate subset
  RooAbsDataStore* subset = _dataMap[_indexCat->getLabel()] ;
  *const_cast<RooArgSet*>((subset->get())) = _vars ;
  return subset->fill() ;
}
 


//_____________________________________________________________________________
const RooArgSet* RooCompositeDataStore::get(Int_t index) const 
{
  // Load the n-th data point (n='index') in memory
  // and return a pointer to the internal RooArgSet
  // holding its coordinates.

  Int_t offset(0) ;
  map<string,RooAbsDataStore*>::const_iterator iter ;
  for (iter = _dataMap.begin() ; iter!=_dataMap.end() ; ++iter) {    
    if (index>=(offset+iter->second->numEntries())) {
      offset += iter->second->numEntries() ;
      continue ;
    }    
    const_cast<RooCompositeDataStore*>(this)->_vars = (*iter->second->get(index-offset)) ;
    _indexCat->setLabel(iter->first.c_str()) ;
    _curStore = iter->second ;
    _curIndex = index-offset ;
    
    return &_vars ;
  }
  return 0 ;
}



//_____________________________________________________________________________
Double_t RooCompositeDataStore::weight() const 
{  
  if (!_curStore) get(0) ;
  // coverity[FORWARD_NULL]
  return _curStore->weight(_curIndex) ;
}





//_____________________________________________________________________________
Double_t RooCompositeDataStore::weight(Int_t index) const 
{
  get(index) ;
  return weight() ;
}




//_____________________________________________________________________________
Double_t RooCompositeDataStore::weightError(RooAbsData::ErrorType etype) const 
{  
  if (!_curStore) get(0) ;
  // coverity[FORWARD_NULL]
  return _curStore->weightError(etype) ;
}




//_____________________________________________________________________________
void RooCompositeDataStore::weightError(Double_t& lo, Double_t& hi, RooAbsData::ErrorType etype) const 
{
  if (!_curStore) get(0) ;
  // coverity[FORWARD_NULL]
  return _curStore->weightError(lo,hi,etype) ;
}




//_____________________________________________________________________________
Bool_t RooCompositeDataStore::isWeighted() const 
{
  map<string,RooAbsDataStore*>::const_iterator iter ;
  for (iter = _dataMap.begin() ; iter!=_dataMap.end() ; ++iter) {    
    if (iter->second->isWeighted()) return kTRUE ;
  }
  return kFALSE ; ;
}


//_____________________________________________________________________________
void RooCompositeDataStore::loadValues(const RooAbsDataStore*, const RooFormulaVar*, const char*, Int_t, Int_t) 
{
  throw(std::string("RooCompositeDataSore::loadValues() NOT IMPLEMENTED")) ;
}



//_____________________________________________________________________________
Bool_t RooCompositeDataStore::changeObservableName(const char* from, const char* to) 
{
  // Change name of internal observable named 'from' into 'to'


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
  map<string,RooAbsDataStore*>::const_iterator iter ;
  for (iter = _dataMap.begin() ; iter!=_dataMap.end() ; ++iter) {    
    ret |= iter->second->changeObservableName(from,to) ;
  }
    
  return ret ;
}

  

//_____________________________________________________________________________
RooAbsArg* RooCompositeDataStore::addColumn(RooAbsArg& newVar, Bool_t adjustRange)
{
  // WVE ownership issue here!! Caller (a RooAbsData) should take ownership of all
  // arguments, but only does for the first one here...
  RooAbsArg* ret(0) ;
  map<string,RooAbsDataStore*>::const_iterator iter ;
  for (iter = _dataMap.begin() ; iter!=_dataMap.end() ; ++iter) {    
    ret = iter->second->addColumn(newVar,adjustRange) ;
  }
  return ret ;
}



//_____________________________________________________________________________
RooArgSet* RooCompositeDataStore::addColumns(const RooArgList& varList)
{
  // WVE ownership issue here!! Caller (a RooAbsData) should take ownership of all
  // arguments, but only does for the first one here...
  RooArgSet* ret(0) ;
  map<string,RooAbsDataStore*>::const_iterator iter ;
  for (iter = _dataMap.begin() ; iter!=_dataMap.end() ; ++iter) {    
    ret = iter->second->addColumns(varList) ;
  }
  return ret ;
}




//_____________________________________________________________________________
RooAbsDataStore* RooCompositeDataStore::merge(const RooArgSet& /*allVars*/, list<RooAbsDataStore*> /*dstoreList*/)
{
  throw string("RooCompositeDataStore::merge() is not implemented yet") ;
}





//_____________________________________________________________________________
void RooCompositeDataStore::append(RooAbsDataStore& other) 
{
  Int_t nevt = other.numEntries() ;
  for (int i=0 ; i<nevt ; i++) {  
    _vars = *other.get(i) ;
    fill() ;
  }
}



//_____________________________________________________________________________
Int_t RooCompositeDataStore::numEntries() const 
{
  Int_t n(0) ;
  map<string,RooAbsDataStore*>::const_iterator iter ;
  for (iter = _dataMap.begin() ; iter!=_dataMap.end() ; ++iter) {    
    n += iter->second->numEntries() ;
  }
  return n ;
}




//_____________________________________________________________________________
void RooCompositeDataStore::reset() 
{
  map<string,RooAbsDataStore*>::const_iterator iter ;
  for (iter = _dataMap.begin() ; iter!=_dataMap.end() ; ++iter) {    
    iter->second->reset() ;
  }
}



//_____________________________________________________________________________
void RooCompositeDataStore::cacheArgs(const RooAbsArg* owner, RooArgSet& newVarSet, const RooArgSet* nset) 
{
  map<string,RooAbsDataStore*>::const_iterator iter ;
  for (iter = _dataMap.begin() ; iter!=_dataMap.end() ; ++iter) {    
    iter->second->cacheArgs(owner,newVarSet,nset) ;
  }
}



//_____________________________________________________________________________
void RooCompositeDataStore::setArgStatus(const RooArgSet& set, Bool_t active) 
{
  map<string,RooAbsDataStore*>::const_iterator iter ;
  for (iter = _dataMap.begin() ; iter!=_dataMap.end() ; ++iter) {    
    RooArgSet* subset = (RooArgSet*) set.selectCommon(*iter->second->get()) ;
    iter->second->setArgStatus(*subset,active) ;
    delete subset ;
  }
  return ;
}



//_____________________________________________________________________________
void RooCompositeDataStore::attachCache(const RooAbsArg* newOwner, const RooArgSet& inCachedVars) 
{
  // Initialize cache of dataset: attach variables of cache ArgSet
  // to the corresponding TTree branches
  map<string,RooAbsDataStore*>::const_iterator iter ;
  for (iter = _dataMap.begin() ; iter!=_dataMap.end() ; ++iter) {    
    iter->second->attachCache(newOwner,inCachedVars) ;
  }
  return ;
}



//_____________________________________________________________________________
void RooCompositeDataStore::resetCache() 
{
  map<string,RooAbsDataStore*>::const_iterator iter ;
  for (iter = _dataMap.begin() ; iter!=_dataMap.end() ; ++iter) {    
    iter->second->resetCache() ;
  }
  return ;
}




