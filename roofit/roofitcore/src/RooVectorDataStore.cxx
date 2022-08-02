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
\file RooVectorDataStore.cxx
\class RooVectorDataStore
\ingroup Roofitcore

RooVectorDataStore uses std::vectors to store data columns. Each of these vectors
is associated to an instance of a RooAbsReal, whose values it represents. Those
RooAbsReal are the observables of the dataset.
In addition to the observables, a data column can be bound to a different instance
of a RooAbsReal (e.g., the column "x" can be bound to the observable "x" of a computation
graph using attachBuffers()). In this case, a get() operation writes the value of
the requested column into the bound real.

As a faster alternative to loading values one-by-one, one can use the function getBatches(),
which returns spans pointing directly to the data.
**/

#include "RooVectorDataStore.h"

#include "RooMsgService.h"
#include "RooTreeDataStore.h"
#include "RooFormulaVar.h"
#include "RooRealVar.h"
#include "RooCategory.h"
#include "RooHistError.h"
#include "RooTrace.h"
#include "RooHelpers.h"

#include "Math/Util.h"
#include "ROOT/StringUtils.hxx"
#include "TBuffer.h"

#include <iomanip>
using namespace std;

ClassImp(RooVectorDataStore);
ClassImp(RooVectorDataStore::RealVector);


////////////////////////////////////////////////////////////////////////////////

RooVectorDataStore::RooVectorDataStore()
{
  TRACE_CREATE
}



////////////////////////////////////////////////////////////////////////////////

RooVectorDataStore::RooVectorDataStore(RooStringView name, RooStringView title, const RooArgSet& vars, const char* wgtVarName) :
  RooAbsDataStore(name,title,varsNoWeight(vars,wgtVarName)),
  _varsww(vars),
  _wgtVar(weightVar(vars,wgtVarName))
{
  for (auto arg : _varsww) {
    arg->attachToVStore(*this) ;
  }

  setAllBuffersNative() ;
  TRACE_CREATE
}



////////////////////////////////////////////////////////////////////////////////

void RooVectorDataStore::setAllBuffersNative()
{
  for (auto realVec : _realStoreList) {
    realVec->setNativeBuffer();
  }

  for (auto fullVec : _realfStoreList) {
    fullVec->setNativeBuffer();
  }

  for (auto catVec : _catStoreList) {
    catVec->setNativeBuffer();
  }
}




////////////////////////////////////////////////////////////////////////////////
/// Utility function for constructors
/// Return RooArgSet that is copy of allVars minus variable matching wgtName if specified

RooArgSet RooVectorDataStore::varsNoWeight(const RooArgSet& allVars, const char* wgtName)
{
  RooArgSet ret(allVars) ;
  if(wgtName) {
    RooAbsArg* wgt = allVars.find(wgtName) ;
    if (wgt) {
      ret.remove(*wgt,true,true) ;
    }
  }
  return ret ;
}



////////////////////////////////////////////////////////////////////////////////
/// Utility function for constructors
/// Return pointer to weight variable if it is defined

RooRealVar* RooVectorDataStore::weightVar(const RooArgSet& allVars, const char* wgtName)
{
  if(wgtName) {
    RooRealVar* wgt = dynamic_cast<RooRealVar*>(allVars.find(wgtName)) ;
    return wgt ;
  }
  return 0 ;
}




////////////////////////////////////////////////////////////////////////////////
/// Regular copy ctor

RooVectorDataStore::RooVectorDataStore(const RooVectorDataStore& other, const char* newname) :
  RooAbsDataStore(other,newname),
  _varsww(other._varsww),
  _wgtVar(other._wgtVar),
  _sumWeight(other._sumWeight),
  _sumWeightCarry(other._sumWeightCarry),
  _extWgtArray(other._extWgtArray),
  _extWgtErrLoArray(other._extWgtErrLoArray),
  _extWgtErrHiArray(other._extWgtErrHiArray),
  _extSumW2Array(other._extSumW2Array),
  _currentWeightIndex(other._currentWeightIndex)
{
  for (const auto realVec : other._realStoreList) {
    _realStoreList.push_back(new RealVector(*realVec, (RooAbsReal*)_varsww.find(realVec->_nativeReal->GetName()))) ;
  }

  for (const auto realFullVec : other._realfStoreList) {
    _realfStoreList.push_back(new RealFullVector(*realFullVec, (RooAbsReal*)_varsww.find(realFullVec->_nativeReal->GetName()))) ;
  }

  for (const auto catVec : other._catStoreList) {
    _catStoreList.push_back(new CatVector(*catVec, (RooAbsCategory*)_varsww.find(catVec->_cat->GetName()))) ;
 }

  setAllBuffersNative() ;

  TRACE_CREATE
}


////////////////////////////////////////////////////////////////////////////////

RooVectorDataStore::RooVectorDataStore(const RooTreeDataStore& other, const RooArgSet& vars, const char* newname) :
  RooAbsDataStore(other,varsNoWeight(vars,other._wgtVar?other._wgtVar->GetName():0),newname),
  _varsww(vars),
  _wgtVar(weightVar(vars,other._wgtVar?other._wgtVar->GetName():0))
{
  for (const auto arg : _varsww) {
    arg->attachToVStore(*this) ;
  }

  setAllBuffersNative() ;

  // now copy contents of tree storage here
  reserve(other.numEntries());
  for (Int_t i=0 ; i<other.numEntries() ; i++) {
    other.get(i) ;
    _varsww.assign(other._varsww) ;
    fill() ;
  }
  TRACE_CREATE

}


////////////////////////////////////////////////////////////////////////////////
/// Clone ctor, must connect internal storage to given new external set of vars

RooVectorDataStore::RooVectorDataStore(const RooVectorDataStore& other, const RooArgSet& vars, const char* newname) :
  RooAbsDataStore(other,varsNoWeight(vars,other._wgtVar?other._wgtVar->GetName():0),newname),
  _varsww(vars),
  _wgtVar(other._wgtVar?weightVar(vars,other._wgtVar->GetName()):0),
  _sumWeight(other._sumWeight),
  _sumWeightCarry(other._sumWeightCarry),
  _extWgtArray(other._extWgtArray),
  _extWgtErrLoArray(other._extWgtErrLoArray),
  _extWgtErrHiArray(other._extWgtErrHiArray),
  _extSumW2Array(other._extSumW2Array),
  _currentWeightIndex(other._currentWeightIndex)
{
  for (const auto realVec : other._realStoreList) {
    auto real = static_cast<RooAbsReal*>(vars.find(realVec->bufArg()->GetName()));
    if (real) {
      // Clone vector
      _realStoreList.push_back(new RealVector(*realVec, real)) ;
      // Adjust buffer pointer
      real->attachToVStore(*this) ;
    }
  }

  vector<RealFullVector*>::const_iterator fiter = other._realfStoreList.begin() ;
  for (; fiter!=other._realfStoreList.end() ; ++fiter) {
    RooAbsReal* real = (RooAbsReal*) vars.find((*fiter)->bufArg()->GetName()) ;
    if (real) {
      // Clone vector
      _realfStoreList.push_back(new RealFullVector(**fiter,real)) ;
      // Adjust buffer pointer
      real->attachToVStore(*this) ;
    }
  }

  vector<CatVector*>::const_iterator citer = other._catStoreList.begin() ;
  for (; citer!=other._catStoreList.end() ; ++citer) {
    RooAbsCategory* cat = (RooAbsCategory*) vars.find((*citer)->bufArg()->GetName()) ;
    if (cat) {
      // Clone vector
      _catStoreList.push_back(new CatVector(**citer,cat)) ;
      // Adjust buffer pointer
      cat->attachToVStore(*this) ;
    }
  }

  setAllBuffersNative() ;

  TRACE_CREATE

}


RooAbsDataStore* RooVectorDataStore::reduce(RooStringView name, RooStringView title,
                        const RooArgSet& vars, const RooFormulaVar* cutVar, const char* cutRange,
                        std::size_t nStart, std::size_t nStop) {
  RooArgSet tmp(vars) ;
  if(_wgtVar && !tmp.contains(*_wgtVar)) {
    tmp.add(*_wgtVar) ;
  }
  const char* wgtVarName = _wgtVar ? _wgtVar->GetName() : nullptr;
  return new RooVectorDataStore(name, title, *this, tmp, cutVar, cutRange, nStart, nStop, wgtVarName);
}



////////////////////////////////////////////////////////////////////////////////

RooVectorDataStore::RooVectorDataStore(RooStringView name, RooStringView title, RooAbsDataStore& tds,
          const RooArgSet& vars, const RooFormulaVar* cutVar, const char* cutRange,
          std::size_t nStart, std::size_t nStop, const char* wgtVarName) :

  RooAbsDataStore(name,title,varsNoWeight(vars,wgtVarName)),
  _varsww(vars),
  _wgtVar(weightVar(vars,wgtVarName))
{
  for (const auto arg : _varsww) {
    arg->attachToVStore(*this) ;
  }

  setAllBuffersNative() ;

  // Deep clone cutVar and attach clone to this dataset
  RooFormulaVar* cloneVar = 0;
  if (cutVar) {
    cloneVar = (RooFormulaVar*) cutVar->cloneTree() ;
    cloneVar->attachDataStore(tds) ;
  }

  RooVectorDataStore* vds = dynamic_cast<RooVectorDataStore*>(&tds) ;
  if (vds && vds->_cache) {
    _cache = new RooVectorDataStore(*vds->_cache) ;
  }

  loadValues(&tds,cloneVar,cutRange,nStart,nStop);

  delete cloneVar ;
  TRACE_CREATE
}






////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooVectorDataStore::~RooVectorDataStore()
{
  for (auto elm : _realStoreList) {
    delete elm;
  }

  for (auto elm : _realfStoreList) {
    delete elm;
  }

  for (auto elm : _catStoreList) {
    delete elm;
  }

  delete _cache ;
  TRACE_DESTROY
}


////////////////////////////////////////////////////////////////////////////////
/// Interface function to TTree::Fill

Int_t RooVectorDataStore::fill()
{
  for (auto realVec : _realStoreList) {
    realVec->fill() ;
  }

  for (auto fullVec : _realfStoreList) {
    fullVec->fill() ;
  }

  for (auto catVec : _catStoreList) {
    catVec->fill() ;
  }
  // use Kahan's algorithm to sum up weights to avoid loss of precision
  double y = (_wgtVar ? _wgtVar->getVal() : 1.) - _sumWeightCarry;
  double t = _sumWeight + y;
  _sumWeightCarry = (t - _sumWeight) - y;
  _sumWeight = t;

  return 0 ;
}



////////////////////////////////////////////////////////////////////////////////
/// Load the n-th data point (n='index') into the variables of this dataset,
/// and return a pointer to the RooArgSet that holds them.
const RooArgSet* RooVectorDataStore::get(Int_t index) const
{
  if (index < 0 || static_cast<std::size_t>(index) >= size()) return 0;

  for (const auto realV : _realStoreList) {
    realV->load(index);
  }

  for (const auto fullRealP : _realfStoreList) {
    fullRealP->get(index);
  }

  for (const auto catP : _catStoreList) {
    catP->load(index);
  }

  if (_doDirtyProp) {
    // Raise all dirty flags
    for (auto var : _vars) {
      var->setValueDirty(); // This triggers recalculation of all clients
    }
  }

  // Update current weight cache
  _currentWeightIndex = index;

  if (_cache) {
    _cache->get(index) ;
  }

  return &_vars;
}


////////////////////////////////////////////////////////////////////////////////
/// Load the n-th data point (n='index') into the variables of this dataset,
/// and return a pointer to the RooArgSet that holds them.
const RooArgSet* RooVectorDataStore::getNative(Int_t index) const
{
  if (index < 0 || static_cast<std::size_t>(index) >= size()) return 0;

  for (const auto realV : _realStoreList) {
    realV->loadToNative(index) ;
  }

  for (const auto fullRealP : _realfStoreList) {
    fullRealP->loadToNative(index);
  }

  for (const auto catP : _catStoreList) {
    catP->loadToNative(index);
  }

  if (_doDirtyProp) {
    // Raise all dirty flags
    for (auto var : _vars) {
      var->setValueDirty() ; // This triggers recalculation of all clients
    }
  }

  // Update current weight cache
  _currentWeightIndex = index;

  if (_cache) {
    _cache->getNative(index) ;
  }

  return &_vars;
}


////////////////////////////////////////////////////////////////////////////////
/// Return the error of the current weight.
/// @param[in] etype Switch between simple Poisson or sum-of-weights statistics

double RooVectorDataStore::weightError(RooAbsData::ErrorType etype) const
{
  if (_extWgtArray) {

    // We have a weight array, use that info

    // Return symmetric error on current bin calculated either from Poisson statistics or from SumOfWeights
    double lo = 0, hi = 0 ;
    weightError(lo,hi,etype) ;
    return (lo+hi)/2 ;

   } else if (_wgtVar) {

    // We have a a weight variable, use that info
    if (_wgtVar->hasAsymError()) {
      return ( _wgtVar->getAsymErrorHi() - _wgtVar->getAsymErrorLo() ) / 2 ;
    } else if (_wgtVar->hasError(false)) {
      return _wgtVar->getError();
    } else {
      return 0 ;
    }

  } else {

    // We have no weights
    return 0 ;

  }
}



////////////////////////////////////////////////////////////////////////////////

void RooVectorDataStore::weightError(double& lo, double& hi, RooAbsData::ErrorType etype) const
{
  if (_extWgtArray) {
    double wgt;

    // We have a weight array, use that info
    switch (etype) {

    case RooAbsData::Auto:
      throw string(Form("RooDataHist::weightError(%s) error type Auto not allowed here",GetName())) ;
      break ;

    case RooAbsData::Expected:
      throw string(Form("RooDataHist::weightError(%s) error type Expected not allowed here",GetName())) ;
      break ;

    case RooAbsData::Poisson:
      // Weight may be preset or precalculated
      if (_extWgtErrLoArray && _extWgtErrLoArray[_currentWeightIndex] >= 0) {
        lo = _extWgtErrLoArray[_currentWeightIndex];
        hi = _extWgtErrHiArray[_currentWeightIndex];
        return ;
      }

      // Otherwise Calculate poisson errors
      wgt = weight();
      double ym,yp ;
      RooHistError::instance().getPoissonInterval(Int_t(wgt+0.5),ym,yp,1);
      lo = wgt-ym;
      hi = yp-wgt;
      return ;

    case RooAbsData::SumW2:
      lo = sqrt( _extSumW2Array ? _extSumW2Array[_currentWeightIndex] : _extWgtArray[_currentWeightIndex] );
      hi = lo;
      return ;

    case RooAbsData::None:
      lo = 0 ;
      hi = 0 ;
      return ;
    }

  } else if (_wgtVar) {

    // We have a a weight variable, use that info
    if (_wgtVar->hasAsymError()) {
      hi = _wgtVar->getAsymErrorHi() ;
      lo = _wgtVar->getAsymErrorLo() ;
    } else {
      hi = _wgtVar->getError() ;
      lo = _wgtVar->getError() ;
    }

  } else {

    // We are unweighted
    lo=0 ;
    hi=0 ;

  }
}



////////////////////////////////////////////////////////////////////////////////
///

void RooVectorDataStore::loadValues(const RooAbsDataStore *ads, const RooFormulaVar* select, const char* rangeName, std::size_t nStart, std::size_t nStop)
{
  // Load values from dataset 't' into this data collection, optionally
  // selecting events using 'select' RooFormulaVar
  //

  // Redirect formula servers to source data row
  std::unique_ptr<RooFormulaVar> selectClone;
  if (select) {
    selectClone.reset( static_cast<RooFormulaVar*>(select->cloneTree()) );
    selectClone->recursiveRedirectServers(*ads->get()) ;
    selectClone->setOperMode(RooAbsArg::ADirty,true) ;
  }

  // Force DS internal initialization
  ads->get(0) ;

  // Loop over events in source tree
  const auto numEntr = static_cast<std::size_t>(ads->numEntries());
  const std::size_t nevent = nStop < numEntr ? nStop : numEntr;

  auto TDS = dynamic_cast<const RooTreeDataStore*>(ads);
  auto VDS = dynamic_cast<const RooVectorDataStore*>(ads);

  // Check if weight is being renamed - if so set flag to enable special handling in copy loop
  bool weightRename(false) ;
  bool newWeightVar = _wgtVar ? _wgtVar->getAttribute("NewWeight") : false ;

  if (_wgtVar && VDS && ((RooVectorDataStore*)(ads))->_wgtVar) {
    if (string(_wgtVar->GetName())!=((RooVectorDataStore*)(ads))->_wgtVar->GetName() && !newWeightVar) {
      weightRename=true ;
    }
  }
  if (_wgtVar && TDS && ((RooTreeDataStore*)(ads))->_wgtVar) {
    if (string(_wgtVar->GetName())!=((RooTreeDataStore*)(ads))->_wgtVar->GetName() && !newWeightVar) {
      weightRename=true ;
    }
  }

  std::vector<std::string> ranges;
  if (rangeName) {
   ranges = ROOT::Split(rangeName, ",");
  }

  reserve(numEntries() + (nevent - nStart));
  for(auto i=nStart; i < nevent ; ++i) {
    ads->get(i);

    // Does this event pass the cuts?
    if (selectClone && selectClone->getVal()==0) {
      continue ;
    }

    if (TDS) {
      _varsww.assignValueOnly(TDS->_varsww) ;
      if (weightRename) {
        _wgtVar->setVal(TDS->_wgtVar->getVal()) ;
      }
    } else if (VDS) {
      _varsww.assignValueOnly(VDS->_varsww) ;
      if (weightRename) {
        _wgtVar->setVal(VDS->_wgtVar->getVal()) ;
      }
    } else {
      _varsww.assignValueOnly(*ads->get()) ;
    }

    // Check that all copied values are valid and in range
    bool allValid = true;
    for (const auto arg : _varsww) {
      allValid &= arg->isValid();
      if (allValid && !ranges.empty()) {
        // If we have one or multiple ranges to be selected, the value
        // must be in one of them to be valid
        allValid &= std::any_of(ranges.begin(), ranges.end(), [arg](const std::string& range){
          return arg->inRange(range.c_str());});
      }
      if (!allValid)
        break ;
    }

    if (!allValid) {
      continue ;
    }

    fill() ;
  }

  SetTitle(ads->GetTitle());
}





////////////////////////////////////////////////////////////////////////////////

bool RooVectorDataStore::changeObservableName(const char* /*from*/, const char* /*to*/)
{
  return false ;
}



////////////////////////////////////////////////////////////////////////////////
/// Add a new column to the data set which holds the pre-calculated values
/// of 'newVar'. This operation is only meaningful if 'newVar' is a derived
/// value.
///
/// The return value points to the added element holding 'newVar's value
/// in the data collection. The element is always the corresponding fundamental
/// type of 'newVar' (e.g. a RooRealVar if 'newVar' is a RooFormulaVar)
///
/// Note: This function is explicitly NOT intended as a speed optimization
///       opportunity for the user. Components of complex PDFs that can be
///       precalculated with the dataset are automatically identified as such
///       and will be precalculated when fitting to a dataset
///
///       By forcibly precalculating functions with non-trivial Jacobians,
///       or functions of multiple variables occurring in the data set,
///       using addColumn(), you may alter the outcome of the fit.
///
///       Only in cases where such a modification of fit behaviour is intentional,
///       this function should be used.

RooAbsArg* RooVectorDataStore::addColumn(RooAbsArg& newVar, bool /*adjustRange*/)
{
  // Create a fundamental object of the right type to hold newVar values
  RooAbsArg* valHolder= newVar.createFundamental();
  // Sanity check that the holder really is fundamental
  if(!valHolder->isFundamental()) {
    coutE(InputArguments) << GetName() << "::addColumn: holder argument is not fundamental: \""
    << valHolder->GetName() << "\"" << endl;
    return 0;
  }

  // Attention: need to do this now, as adding an empty column might give 0 as size
  const std::size_t numEvt = size();

  // Clone variable and attach to cloned tree
  RooAbsArg* newVarClone = newVar.cloneTree() ;
  newVarClone->recursiveRedirectServers(_vars,false) ;

  // Attach value place holder to this tree
  valHolder->attachToVStore(*this) ;
  _vars.add(*valHolder) ;
  _varsww.add(*valHolder) ;

  // Fill values of placeholder
  RealVector* rv(0) ;
  CatVector* cv(0) ;
  assert(numEvt != 0);
  if (dynamic_cast<RooAbsReal*>(valHolder)) {
    rv = addReal((RooAbsReal*)valHolder);
    rv->resize(numEvt) ;
  } else if (dynamic_cast<RooAbsCategory*>((RooAbsCategory*)valHolder)) {
    cv = addCategory((RooAbsCategory*)valHolder) ;
    cv->resize(numEvt) ;
  }

  for (std::size_t i=0; i < numEvt; i++) {
    getNative(i) ;

    newVarClone->syncCache(&_vars) ;
    valHolder->copyCache(newVarClone) ;

    if (rv) rv->write(i) ;
    if (cv) cv->write(i) ;
  }

  delete newVarClone ;
  return valHolder ;

}



////////////////////////////////////////////////////////////////////////////////
/// Merge columns of supplied data set(s) with this data set.  All
/// data sets must have equal number of entries.  In case of
/// duplicate columns the column of the last dataset in the list
/// prevails

RooAbsDataStore* RooVectorDataStore::merge(const RooArgSet& allVars, list<RooAbsDataStore*> dstoreList)
{
  RooVectorDataStore* mergedStore = new RooVectorDataStore("merged","merged",allVars) ;

  const auto nevt = dstoreList.front()->numEntries();
  mergedStore->reserve(nevt);
  for (int i=0 ; i<nevt ; i++) {

    // Copy data from self
    mergedStore->_vars.assign(*get(i)) ;

    // Copy variables from merge sets
    for (list<RooAbsDataStore*>::iterator iter = dstoreList.begin() ; iter!=dstoreList.end() ; ++iter) {
      const RooArgSet* partSet = (*iter)->get(i) ;
      mergedStore->_vars.assign(*partSet) ;
    }

    mergedStore->fill() ;
  }
  return mergedStore ;
}



void RooVectorDataStore::reserve(Int_t nEvts)
{
  for (auto elm : _realStoreList) {
    elm->reserve(nEvts);
  }

  for (auto elm : _realfStoreList) {
    elm->reserve(nEvts);
  }

  for (auto elm : _catStoreList) {
    elm->reserve(nEvts);
  }
}

////////////////////////////////////////////////////////////////////////////////

void RooVectorDataStore::append(RooAbsDataStore& other)
{
  Int_t nevt = other.numEntries() ;
  reserve(nevt + numEntries());
  for (int i=0 ; i<nevt ; i++) {
    _vars.assign(*other.get(i)) ;
    if (_wgtVar) {
      _wgtVar->setVal(other.weight()) ;
    }

    fill() ;
  }
}



////////////////////////////////////////////////////////////////////////////////

void RooVectorDataStore::reset()
{
  _sumWeight=_sumWeightCarry=0 ;

  for (auto elm : _realStoreList) {
    elm->reset() ;
  }

  for (auto elm : _realfStoreList) {
    elm->reset() ;
  }

  for (auto elm : _catStoreList) {
    elm->reset() ;
  }

}

////////////////////////////////////////////////////////////////////////////////
/// Cache given RooAbsArgs: The tree is
/// given direct write access of the args internal cache
/// the args values is pre-calculated for all data points
/// in this data collection. Upon a get() call, the
/// internal cache of 'newVar' will be loaded with the
/// precalculated value and it's dirty flag will be cleared.

void RooVectorDataStore::cacheArgs(const RooAbsArg* owner, RooArgSet& newVarSet, const RooArgSet* nset, bool skipZeroWeights)
{
  // Delete previous cache, if any
  delete _cache ;
  _cache = 0 ;

  // Reorder cached elements. First constant nodes, then tracked nodes in order of dependence

  // Step 1 - split in constant and tracked
  RooArgSet newVarSetCopy(newVarSet);
  RooArgSet orderedArgs;
  vector<RooAbsArg*> trackArgs;
  for (const auto arg : newVarSetCopy) {
    if (arg->getAttribute("ConstantExpression") && !arg->getAttribute("NOCacheAndTrack")) {
      orderedArgs.add(*arg) ;
    } else {

      // Explicitly check that arg depends on any of the observables, if this
      // is not the case, skip it, as inclusion would result in repeated
      // calculation of a function that has the same value for every event
      // in the likelihood
      if (arg->dependsOn(_vars) && !arg->getAttribute("NOCacheAndTrack")) {
        trackArgs.push_back(arg) ;
      } else {
        newVarSet.remove(*arg) ;
      }
    }
  }

  // Step 2 - reorder tracked nodes
  std::sort(trackArgs.begin(), trackArgs.end(), [](RooAbsArg* left, RooAbsArg* right){
    //LM: exclude same comparison. This avoids an issue when using sort in MacOS  versions
    if (left == right) return false;
    return right->dependsOn(*left);
  });

  // Step 3 - put back together
  for (const auto trackedArg : trackArgs) {
    orderedArgs.add(*trackedArg);
  }

  // WVE need to prune tracking entries _below_ constant nodes as the're not needed
//   cout << "Number of Cache-and-Tracked args are " << trackArgs.size() << endl ;
//   cout << "Compound ordered cache parameters = " << endl ;
//   orderedArgs.Print("v") ;

  checkInit() ;

  std::vector<RooArgSet*> vlist;
  RooArgList cloneSet;

  for (const auto var : orderedArgs) {

    // Clone variable and attach to cloned tree
    RooArgSet* newVarCloneList = (RooArgSet*) RooArgSet(*var).snapshot() ;
    RooAbsArg* newVarClone = newVarCloneList->find(var->GetName()) ;
    newVarClone->recursiveRedirectServers(_vars,false) ;

    vlist.push_back(newVarCloneList) ;
    cloneSet.add(*newVarClone) ;
  }

  _cacheOwner = (RooAbsArg*) owner ;
  RooVectorDataStore* newCache = new RooVectorDataStore("cache","cache",orderedArgs) ;


  RooAbsArg::setDirtyInhibit(true) ;

  std::vector<RooArgSet*> nsetList ;
  std::vector<RooArgSet*> argObsList ;

  // Now need to attach branch buffers of clones
  for (const auto arg : cloneSet) {
    arg->attachToVStore(*newCache) ;

    RooArgSet* argObs = nset ? arg->getObservables(*nset) : arg->getVariables() ;
    argObsList.push_back(argObs) ;

    RooArgSet* normSet(0) ;
    const char* catNset = arg->getStringAttribute("CATNormSet") ;
    if (catNset) {
//       cout << "RooVectorDataStore::cacheArgs() cached node " << arg->GetName() << " has a normalization set specification CATNormSet = " << catNset << endl ;

      RooArgSet anset = RooHelpers::selectFromArgSet(nset ? *nset : RooArgSet{}, catNset);
      normSet = (RooArgSet*) anset.selectCommon(*argObs) ;

    }
    const char* catCset = arg->getStringAttribute("CATCondSet") ;
    if (catCset) {
//       cout << "RooVectorDataStore::cacheArgs() cached node " << arg->GetName() << " has a conditional observable set specification CATCondSet = " << catCset << endl ;

      RooArgSet acset = RooHelpers::selectFromArgSet(nset ? *nset : RooArgSet{}, catCset);
      argObs->remove(acset,true,true) ;
      normSet = argObs ;
    }

    // now construct normalization set for component from cset/nset spec
//     if (normSet) {
//       cout << "RooVectorDaraStore::cacheArgs() component " << arg->GetName() << " has custom normalization set " << *normSet << endl ;
//     }
    nsetList.push_back(normSet) ;
  }


  // Fill values of of placeholder
  const std::size_t numEvt = size();
  newCache->reserve(numEvt);
  for (std::size_t i=0; i < numEvt; i++) {
    getNative(i) ;
    if (weight()!=0 || !skipZeroWeights) {
      for (unsigned int j = 0; j < cloneSet.size(); ++j) {
        auto& cloneArg = cloneSet[j];
        auto argNSet = nsetList[j];
        // WVE need to intervene here for condobs from ProdPdf
        cloneArg.syncCache(argNSet ? argNSet : nset) ;
      }
    }
    newCache->fill() ;
  }

  RooAbsArg::setDirtyInhibit(false) ;


  // Now need to attach branch buffers of original function objects
  for (const auto arg : orderedArgs) {
    arg->attachToVStore(*newCache) ;

    // Activate change tracking mode, if requested
    if (!arg->getAttribute("ConstantExpression") && dynamic_cast<RooAbsReal*>(arg)) {
      RealVector* rv = newCache->addReal((RooAbsReal*)arg) ;
      RooArgSet* deps = arg->getParameters(_vars) ;
      rv->setDependents(*deps) ;

      // WV lookup normalization set and associate with RealVector
      // find ordinal number of arg in original list
      Int_t idx = cloneSet.index(arg->GetName()) ;

      coutI(Optimization) << "RooVectorDataStore::cacheArg() element " << arg->GetName() << " has change tracking enabled on parameters " << *deps << endl ;
      rv->setNset(nsetList[idx]) ;
      delete deps ;
    }

  }


  for (auto set : vlist) {
    delete set;
  }
  for (auto set : argObsList) {
    delete set;
  }

  _cache = newCache ;
  _cache->setDirtyProp(_doDirtyProp) ;
}


void RooVectorDataStore::forceCacheUpdate()
{
  if (_cache) _forcedUpdate = true ;
}



////////////////////////////////////////////////////////////////////////////////

void RooVectorDataStore::recalculateCache( const RooArgSet *projectedArgs, Int_t firstEvent, Int_t lastEvent, Int_t stepSize, bool skipZeroWeights)
{
  if (!_cache) return ;

  std::vector<RooVectorDataStore::RealVector *> tv;
  tv.reserve(static_cast<std::size_t>(_cache->_realStoreList.size() * 0.7)); // Typically, 30..60% need to be recalculated

  // Check which items need recalculation
  for (const auto realVec : _cache->_realStoreList) {
    if (_forcedUpdate || realVec->needRecalc()) {
       tv.push_back(realVec);
       realVec->_nativeReal->setOperMode(RooAbsArg::ADirty);
       realVec->_nativeReal->_operMode = RooAbsArg::Auto;
    }
  }
  _forcedUpdate = false ;

  // If no recalculations are needed stop here
  if (tv.empty()) {
     return;
  }


  // Refill caches of elements that require recalculation
  RooArgSet* ownedNset = 0 ;
  RooArgSet* usedNset = 0 ;
  if (projectedArgs && projectedArgs->getSize()>0) {
    ownedNset = (RooArgSet*) _vars.snapshot(false) ;
    ownedNset->remove(*projectedArgs,false,true);
    usedNset = ownedNset ;
  } else {
    usedNset = &_vars ;
  }


  for (int i=firstEvent ; i<lastEvent ; i+=stepSize) {
    get(i) ;
    bool zeroWeight = (weight()==0) ;
    if (!zeroWeight || !skipZeroWeights) {
       for (auto realVector : tv) {
          realVector->_nativeReal->_valueDirty = true;
          realVector->_nativeReal->getValV(realVector->_nset ? realVector->_nset : usedNset);
          realVector->write(i);
      }
    }
  }

  for (auto realVector : tv) {
     realVector->_nativeReal->setOperMode(RooAbsArg::AClean);
  }

  delete ownedNset ;

}


////////////////////////////////////////////////////////////////////////////////
/// Initialize cache of dataset: attach variables of cache ArgSet
/// to the corresponding TTree branches

void RooVectorDataStore::attachCache(const RooAbsArg* newOwner, const RooArgSet& cachedVarsIn)
{
  // Only applicable if a cache exists
  if (!_cache) return ;

  // Clone ctor, must connect internal storage to given new external set of vars
  std::vector<RealVector*> cacheElements(_cache->realStoreList());
  cacheElements.insert(cacheElements.end(), _cache->_realfStoreList.begin(), _cache->_realfStoreList.end());

  for (const auto elm : cacheElements) {
    auto real = static_cast<RooAbsReal*>(cachedVarsIn.find(elm->bufArg()->GetName()));
    if (real) {
      // Adjust buffer pointer
      real->attachToVStore(*_cache) ;
    }
  }

  for (const auto catVec : _cache->_catStoreList) {
    auto cat = static_cast<RooAbsCategory*>(cachedVarsIn.find(catVec->bufArg()->GetName()));
    if (cat) {
      // Adjust buffer pointer
      cat->attachToVStore(*_cache) ;
    }
  }

  _cacheOwner = (RooAbsArg*) newOwner ;
}




////////////////////////////////////////////////////////////////////////////////

void RooVectorDataStore::resetCache()
{
  delete _cache;
  _cache = nullptr;
  _cacheOwner = nullptr;
  return ;
}





////////////////////////////////////////////////////////////////////////////////
/// Disabling of branches is (intentionally) not implemented in vector
/// data stores (as the doesn't result in a net saving of time)

void RooVectorDataStore::setArgStatus(const RooArgSet& /*set*/, bool /*active*/)
{
  return ;
}




////////////////////////////////////////////////////////////////////////////////

void RooVectorDataStore::attachBuffers(const RooArgSet& extObs)
{
  for (auto arg : _varsww) {
    RooAbsArg* extArg = extObs.find(arg->GetName()) ;
    if (extArg) {
      extArg->attachToVStore(*this) ;
    }
  }
}



////////////////////////////////////////////////////////////////////////////////

void RooVectorDataStore::resetBuffers()
{
  for (auto arg : _varsww) {
    arg->attachToVStore(*this);
  }
}



////////////////////////////////////////////////////////////////////////////////

void RooVectorDataStore::dump()
{
  cout << "RooVectorDataStor::dump()" << endl ;

  cout << "_varsww = " << endl ; _varsww.Print("v") ;
  cout << "realVector list is" << endl ;

  for (const auto elm : _realStoreList) {
    cout << "RealVector " << elm << " _nativeReal = " << elm->_nativeReal << " = " << elm->_nativeReal->GetName() << " bufptr = " << elm->_buf  << endl ;
    cout << " values : " ;
    Int_t imax = elm->_vec.size()>10 ? 10 : elm->_vec.size() ;
    for (Int_t i=0 ; i<imax ; i++) {
      cout << elm->_vec[i] << " " ;
    }
    cout << endl ;
  }

  for (const auto elm : _realfStoreList) {
    cout << "RealFullVector " << elm << " _nativeReal = " << elm->_nativeReal << " = " << elm->_nativeReal->GetName()
    << " bufptr = " << elm->_buf  << " errbufptr = " << elm->_bufE << endl ;

    cout << " values : " ;
    Int_t imax = elm->_vec.size()>10 ? 10 : elm->_vec.size() ;
    for (Int_t i=0 ; i<imax ; i++) {
      cout << elm->_vec[i] << " " ;
    }
    cout << endl ;
    if (elm->_vecE) {
      cout << " errors : " ;
      for (Int_t i=0 ; i<imax ; i++) {
   cout << (*elm->_vecE)[i] << " " ;
      }
      cout << endl ;

    }
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Stream an object of class RooVectorDataStore.

void RooVectorDataStore::Streamer(TBuffer &R__b)
{
  if (R__b.IsReading()) {
    R__b.ReadClassBuffer(RooVectorDataStore::Class(),this);

    for (auto elm : _realStoreList) {
      RooAbsArg* arg = _varsww.find(elm->_nativeReal->GetName()) ;
      arg->attachToVStore(*this) ;
    }
    for (auto elm : _realfStoreList) {
      RooAbsArg* arg = _varsww.find(elm->_nativeReal->GetName()) ;
      arg->attachToVStore(*this) ;
    }
    for (auto elm : _catStoreList) {
      RooAbsArg* arg = _varsww.find(elm->_cat->GetName()) ;
      arg->attachToVStore(*this) ;
    }

  } else {
    R__b.WriteClassBuffer(RooVectorDataStore::Class(),this);
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Stream an object of class RooVectorDataStore::RealFullVector.

void RooVectorDataStore::RealFullVector::Streamer(TBuffer &R__b)
{
   if (R__b.IsReading()) {
     R__b.ReadClassBuffer(RooVectorDataStore::RealFullVector::Class(),this);

     // WVE - It seems that ROOT persistence turns null pointers to vectors into pointers to null-sized vectors
     //       Intervene here to remove those null-sized vectors and replace with null pointers to not break
     //       assumptions made elsewhere in this class
     if (_vecE  && _vecE->empty()) { delete _vecE   ; _vecE = 0 ; }
     if (_vecEL && _vecEL->empty()) { delete _vecEL ; _vecEL = 0 ; }
     if (_vecEH && _vecEH->empty()) { delete _vecEH ; _vecEH = 0 ; }
   } else {
     R__b.WriteClassBuffer(RooVectorDataStore::RealFullVector::Class(),this);
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Return batches of the data columns for the requested events.
/// \param[in] first First event in the batches.
/// \param[in] len   Number of events in batches.
/// \return Spans with the associated data.
RooAbsData::RealSpans RooVectorDataStore::getBatches(std::size_t first, std::size_t len) const {
  RooAbsData::RealSpans evalData;

  auto emplace = [this,&evalData,first,len](const RealVector* realVec) {
    auto span = realVec->getRange(first, first + len);
    auto result = evalData.emplace(realVec->_nativeReal, span);
    if (result.second == false || result.first->second.size() != len) {
      const auto size = result.second ? result.first->second.size() : 0;
      coutE(DataHandling) << "A batch of data for '" << realVec->_nativeReal->GetName()
          << "' was requested from " << first << " to " << first+len
          << ", but only the events [" << first << ", " << first + size << ") are available." << std::endl;
    }
    if (realVec->_real) {
      // If a buffer is attached, i.e. we are ready to load into a RooAbsReal outside of our dataset,
      // we can directly map our spans to this real.
      evalData.emplace(realVec->_real, std::move(span));
    }
  };

  for (const auto realVec : _realStoreList) {
    emplace(realVec);
  }
  for (const auto realVec : _realfStoreList) {
    emplace(realVec);
  }

  if (_cache) {
    for (const auto realVec : _cache->_realStoreList) {
      emplace(realVec);
    }
    for (const auto realVec : _cache->_realfStoreList) {
      emplace(realVec);
    }
  }

  return evalData;
}


RooAbsData::CategorySpans RooVectorDataStore::getCategoryBatches(std::size_t first, std::size_t len) const {
  RooAbsData::CategorySpans evalData;

  auto emplace = [this,&evalData,first,len](const CatVector* catVec) {
    auto span = catVec->getRange(first, first + len);
    auto result = evalData.emplace(catVec->_cat, span);
    if (result.second == false || result.first->second.size() != len) {
      const auto size = result.second ? result.first->second.size() : 0;
      coutE(DataHandling) << "A batch of data for '" << catVec->_cat->GetName()
          << "' was requested from " << first << " to " << first+len
          << ", but only the events [" << first << ", " << first + size << ") are available." << std::endl;
    }
  };

  for (const auto& catVec : _catStoreList) {
    emplace(catVec);
  }

  return evalData;
}


////////////////////////////////////////////////////////////////////////////////
/// Return the weights of all events in the range [first, first+len).
/// If an array with weights is stored, a batch with these weights will be returned. If
/// no weights are stored, an empty batch is returned. Use weight() to check if there's
/// a constant weight.
RooSpan<const double> RooVectorDataStore::getWeightBatch(std::size_t first, std::size_t len) const
{
  if (_extWgtArray) {
    return RooSpan<const double>(_extWgtArray + first, _extWgtArray + first + len);
  }

  if (_wgtVar) {
    auto findWeightVar = [this](const RealVector* realVec) {
      return realVec->_nativeReal == _wgtVar || realVec->_nativeReal->GetName() == _wgtVar->GetName();
    };

    auto storageIter = std::find_if(_realStoreList.begin(), _realStoreList.end(), findWeightVar);
    if (storageIter != _realStoreList.end())
      return (*storageIter)->getRange(first, first + len);

    auto fstorageIter = std::find_if(_realfStoreList.begin(), _realfStoreList.end(), findWeightVar);
    if (fstorageIter != _realfStoreList.end())
      return (*fstorageIter)->getRange(first, first + len);

    throw std::logic_error("RooVectorDataStore::getWeightBatch(): Could not retrieve data for _wgtVar.");
  }
  return {};
}


RooVectorDataStore::CatVector* RooVectorDataStore::addCategory(RooAbsCategory* cat) {

  // First try a match by name
  for (auto catVec : _catStoreList) {
    if (std::string(catVec->bufArg()->GetName())==cat->GetName()) {
      return catVec;
    }
  }

  // If nothing found this will make an entry
  _catStoreList.push_back(new CatVector(cat)) ;

  return _catStoreList.back() ;
}


RooVectorDataStore::RealVector* RooVectorDataStore::addReal(RooAbsReal* real) {

  // First try a match by name
  for (auto realVec : _realStoreList) {
    if (realVec->bufArg()->namePtr()==real->namePtr()) {
      return realVec;
    }
  }

  // Then check if an entry already exists for a full real
  for (auto fullVec : _realfStoreList) {
    if (fullVec->bufArg()->namePtr()==real->namePtr()) {
      // Return full vector as RealVector base class here
      return fullVec;
    }
  }

  // If nothing found this will make an entry
  _realStoreList.push_back(new RealVector(real)) ;

  return _realStoreList.back() ;
}


bool RooVectorDataStore::isFullReal(RooAbsReal* real) {

  // First try a match by name
  for (auto fullVec : _realfStoreList) {
    if (std::string(fullVec->bufArg()->GetName())==real->GetName()) {
      return true ;
    }
  }
  return false ;
}


bool RooVectorDataStore::hasError(RooAbsReal* real) {

  // First try a match by name
  for (auto fullVec : _realfStoreList) {
    if (std::string(fullVec->bufArg()->GetName())==real->GetName()) {
      return fullVec->_vecE ? true : false ;
    }
  }
  return false ;
}


bool RooVectorDataStore::hasAsymError(RooAbsReal* real) {

  // First try a match by name
  for (auto fullVec : _realfStoreList) {
    if (std::string(fullVec->bufArg()->GetName())==real->GetName()) {
      return fullVec->_vecEL ? true : false ;
    }
  }
  return false ;
}


RooVectorDataStore::RealFullVector* RooVectorDataStore::addRealFull(RooAbsReal* real) {

  // First try a match by name
  for (auto fullVec : _realfStoreList) {
    if (std::string(fullVec->bufArg()->GetName())==real->GetName()) {
    return fullVec;
    }
  }

  // Then check if an entry already exists for a bare real
  for (auto realVec : _realStoreList) {
    if (std::string(realVec->bufArg()->GetName())==real->GetName()) {

      // Convert element to full and add to full list
      _realfStoreList.push_back(new RealFullVector(*realVec,real)) ;

      // Delete bare element
      _realStoreList.erase(std::find(_realStoreList.begin(), _realStoreList.end(), realVec));
      delete realVec;

      return _realfStoreList.back() ;
    }
  }

  // If nothing found this will make an entry
  _realfStoreList.push_back(new RealFullVector(real)) ;

  return _realfStoreList.back() ;
}


/// Trigger a recomputation of the cached weight sums. Meant for use by RooFit
/// dataset converter functions such as the NumPy converter functions
/// implemented as pythonizations.
void RooVectorDataStore::recomputeSumWeight() {
  double const* arr = nullptr;
  if (_extWgtArray) {
    arr = _extWgtArray;
  }
  if (_wgtVar) {
    const std::string wgtName = _wgtVar->GetName();
    for(auto const* real : _realStoreList) {
      if(wgtName == real->_nativeReal->GetName())
        arr = real->_vec.data();
    }
    for(auto const* real : _realfStoreList) {
      if(wgtName == real->_nativeReal->GetName())
        arr = real->_vec.data();
    }
  }
  if(arr == nullptr) {
    _sumWeight = size();
    return;
  }
  auto result = ROOT::Math::KahanSum<double, 4>::Accumulate(arr, arr + size(), 0.0);
  _sumWeight = result.Sum();
  _sumWeightCarry = result.Carry();
}


/// Exports all arrays in this RooVectorDataStore into a simple datastructure
/// to be used by RooFit internal export functions.
RooVectorDataStore::ArraysStruct  RooVectorDataStore::getArrays() const {
  ArraysStruct out;
  out.size = size();

  for(auto const* real : _realStoreList) {
    out.reals.emplace_back(real->_nativeReal->GetName(), real->_vec.data());
  }
  for(auto const* realf : _realfStoreList) {
    std::string name = realf->_nativeReal->GetName();
    out.reals.emplace_back(name, realf->_vec.data());
    if(realf->_vecE) out.reals.emplace_back(name + "Err", realf->_vecE->data());
    if(realf->_vecEL) out.reals.emplace_back(name + "ErrLo", realf->_vecEL->data());
    if(realf->_vecEH) out.reals.emplace_back(name + "ErrHi", realf->_vecEH->data());
  }
  for(auto const* cat : _catStoreList) {
    out.cats.emplace_back(cat->_cat->GetName(), cat->_vec.data());
  }

  if(_extWgtArray) out.reals.emplace_back("weight", _extWgtArray);
  if(_extWgtErrLoArray) out.reals.emplace_back("wgtErrLo", _extWgtErrLoArray);
  if(_extWgtErrHiArray) out.reals.emplace_back("wgtErrHi", _extWgtErrHiArray);
  if(_extSumW2Array) out.reals.emplace_back("sumW2",_extSumW2Array);

  return out;
}
