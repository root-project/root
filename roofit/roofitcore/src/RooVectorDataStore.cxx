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

Uses `std::vector` to store data columns. Each of these vectors
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
#include "RooFitImplHelpers.h"

#include "Math/Util.h"
#include "ROOT/StringUtils.hxx"
#include "TBuffer.h"

#include <iomanip>
using std::string, std::vector, std::list;



////////////////////////////////////////////////////////////////////////////////

RooVectorDataStore::RooVectorDataStore()
{
  TRACE_CREATE;
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
  TRACE_CREATE;
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
  return nullptr ;
}




////////////////////////////////////////////////////////////////////////////////
/// Regular copy constructor.

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
    _realStoreList.push_back(new RealVector(*realVec, static_cast<RooAbsReal*>(_varsww.find(realVec->_nativeReal->GetName())))) ;
  }

  for (const auto realFullVec : other._realfStoreList) {
    _realfStoreList.push_back(new RealFullVector(*realFullVec, static_cast<RooAbsReal*>(_varsww.find(realFullVec->_nativeReal->GetName())))) ;
  }

  for (const auto catVec : other._catStoreList) {
    _catStoreList.push_back(new CatVector(*catVec, static_cast<RooAbsCategory*>(_varsww.find(catVec->_cat->GetName())))) ;
 }

  setAllBuffersNative() ;

  TRACE_CREATE;
}


////////////////////////////////////////////////////////////////////////////////

RooVectorDataStore::RooVectorDataStore(const RooTreeDataStore& other, const RooArgSet& vars, const char* newname) :
  RooAbsDataStore(other,varsNoWeight(vars,other._wgtVar?other._wgtVar->GetName():nullptr),newname),
  _varsww(vars),
  _wgtVar(weightVar(vars,other._wgtVar?other._wgtVar->GetName():nullptr))
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
  TRACE_CREATE;

}


////////////////////////////////////////////////////////////////////////////////
/// Clone constructor, must connect internal storage to given new external set of variables.

RooVectorDataStore::RooVectorDataStore(const RooVectorDataStore& other, const RooArgSet& vars, const char* newname) :
  RooAbsDataStore(other,varsNoWeight(vars,other._wgtVar?other._wgtVar->GetName():nullptr),newname),
  _varsww(vars),
  _wgtVar(other._wgtVar?weightVar(vars,other._wgtVar->GetName()):nullptr),
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

  auto forwardIter = other._realfStoreList.begin() ;
  for (; forwardIter!=other._realfStoreList.end() ; ++forwardIter) {
    RooAbsReal* real = static_cast<RooAbsReal*>(vars.find((*forwardIter)->bufArg()->GetName())) ;
    if (real) {
      // Clone vector
      _realfStoreList.push_back(new RealFullVector(**forwardIter,real)) ;
      // Adjust buffer pointer
      real->attachToVStore(*this) ;
    }
  }

  vector<CatVector*>::const_iterator citer = other._catStoreList.begin() ;
  for (; citer!=other._catStoreList.end() ; ++citer) {
    RooAbsCategory* cat = static_cast<RooAbsCategory*>(vars.find((*citer)->bufArg()->GetName())) ;
    if (cat) {
      // Clone vector
      _catStoreList.push_back(new CatVector(**citer,cat)) ;
      // Adjust buffer pointer
      cat->attachToVStore(*this) ;
    }
  }

  setAllBuffersNative() ;

  TRACE_CREATE;

}


std::unique_ptr<RooAbsDataStore> RooVectorDataStore::reduce(RooStringView name, RooStringView title,
                        const RooArgSet& vars, const RooFormulaVar* cutVar, const char* cutRange,
                        std::size_t nStart, std::size_t nStop) {
  RooArgSet tmp(vars) ;
  if(_wgtVar && !tmp.contains(*_wgtVar)) {
    tmp.add(*_wgtVar) ;
  }
  const char* wgtVarName = _wgtVar ? _wgtVar->GetName() : nullptr;
  return std::make_unique<RooVectorDataStore>(name, title, *this, tmp, cutVar, cutRange, nStart, nStop, wgtVarName);
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
  std::unique_ptr<RooFormulaVar> cloneVar;
  if (cutVar) {
    cloneVar.reset(static_cast<RooFormulaVar*>(cutVar->cloneTree()));
    cloneVar->attachDataStore(tds) ;
  }

  loadValues(&tds,cloneVar.get(),cutRange,nStart,nStop);

  TRACE_CREATE;
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

  TRACE_DESTROY;
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
  if (index < 0 || static_cast<std::size_t>(index) >= size()) return nullptr;

  for (const auto realV : _realStoreList) {
    realV->load(index);
  }

  for (const auto fullRealP : _realfStoreList) {
    fullRealP->load(index);
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
    double lo = 0;
    double hi = 0;
    weightError(lo,hi,etype) ;
    return (lo+hi)/2 ;

   } else if (_wgtVar) {

    // We have a weight variable, use that info
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
      double ym;
      double yp;
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

    // We have a weight variable, use that info
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

  auto treeDS = dynamic_cast<const RooTreeDataStore*>(ads);
  auto vectorDS = dynamic_cast<const RooVectorDataStore*>(ads);

  // Check if weight is being renamed - if so set flag to enable special handling in copy loop
  bool weightRename(false) ;
  const bool newWeightVar = _wgtVar ? _wgtVar->getAttribute("NewWeight") : false ;

  if (_wgtVar && vectorDS && vectorDS->_wgtVar) {
    if (std::string(_wgtVar->GetName()) != vectorDS->_wgtVar->GetName() && !newWeightVar) {
      weightRename=true ;
    }
  }
  if (_wgtVar && treeDS && treeDS->_wgtVar) {
    if (std::string(_wgtVar->GetName()) != treeDS->_wgtVar->GetName() && !newWeightVar) {
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

    RooArgSet const* otherVarsww = nullptr;

    if (treeDS) {
      otherVarsww = &treeDS->_varsww;
      if (weightRename) {
        _wgtVar->setVal(treeDS->_wgtVar->getVal()) ;
      }
    } else if (vectorDS) {
      otherVarsww = &vectorDS->_varsww;
      if (weightRename) {
        _wgtVar->setVal(vectorDS->_wgtVar->getVal()) ;
      }
    } else {
      otherVarsww = ads->get();
    }

    // Check that all copied values are valid and in range
    bool allValid = true;
    for (const auto arg : *otherVarsww) {
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

    _varsww.assign(*otherVarsww) ;

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
  auto valHolder = std::unique_ptr<RooAbsArg>{newVar.createFundamental()}.release();
  // Sanity check that the holder really is fundamental
  if(!valHolder->isFundamental()) {
    coutE(InputArguments) << GetName() << "::addColumn: holder argument is not fundamental: \""
    << valHolder->GetName() << "\"" << std::endl;
    return nullptr;
  }

  // Attention: need to do this now, as adding an empty column might give 0 as size
  const std::size_t numEvt = size();

  // Clone variable and attach to cloned tree
  std::unique_ptr<RooAbsArg> newVarClone{newVar.cloneTree()};
  newVarClone->recursiveRedirectServers(_vars,false) ;

  // Attach value place holder to this tree
  valHolder->attachToVStore(*this) ;
  _vars.add(*valHolder) ;
  _varsww.add(*valHolder) ;

  // Fill values of placeholder
  RealVector* rv(nullptr) ;
  CatVector* cv(nullptr) ;
  assert(numEvt != 0);
  if (dynamic_cast<RooAbsReal*>(valHolder)) {
    rv = addReal(static_cast<RooAbsReal*>(valHolder));
    rv->resize(numEvt) ;
  } else if (dynamic_cast<RooAbsCategory*>(static_cast<RooAbsCategory*>(valHolder))) {
    cv = addCategory(static_cast<RooAbsCategory*>(valHolder)) ;
    cv->resize(numEvt) ;
  }

  for (std::size_t i=0; i < numEvt; i++) {
    get(i) ;

    newVarClone->syncCache(&_vars) ;
    valHolder->copyCache(newVarClone.get()) ;

    if (rv) rv->write(i) ;
    if (cv) cv->write(i) ;
  }

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
  std::cout << "RooVectorDataStor::dump()" << std::endl ;

  std::cout << "_varsww = " << std::endl ; _varsww.Print("v") ;
  std::cout << "realVector list is" << std::endl ;

  for (const auto elm : _realStoreList) {
    std::cout << "RealVector " << elm << " _nativeReal = " << elm->_nativeReal << " = " << elm->_nativeReal->GetName() << " bufptr = " << elm->_buf  << std::endl ;
    std::cout << " values : " ;
    Int_t imax = elm->_vec.size()>10 ? 10 : elm->_vec.size() ;
    for (Int_t i=0 ; i<imax ; i++) {
      std::cout << elm->_vec[i] << " " ;
    }
    std::cout << std::endl ;
  }

  for (const auto elm : _realfStoreList) {
    std::cout << "RealFullVector " << elm << " _nativeReal = " << elm->_nativeReal << " = " << elm->_nativeReal->GetName()
    << " bufptr = " << elm->_buf  << " errbufptr = " << elm->bufE() << std::endl ;

    std::cout << " values : " ;
    Int_t imax = elm->_vec.size()>10 ? 10 : elm->_vec.size() ;
    for (Int_t i=0 ; i<imax ; i++) {
      std::cout << elm->_vec[i] << " " ;
    }
    std::cout << std::endl ;
    if (elm->bufE()) {
      std::cout << " errors : " ;
      for (Int_t i=0 ; i<imax ; i++) {
   std::cout << elm->dataE()[i] << " " ;
      }
      std::cout << std::endl ;

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
      evalData.emplace(realVec->_real, span);
    }
  };

  for (const auto realVec : _realStoreList) {
    emplace(realVec);
  }
  for (const auto realVec : _realfStoreList) {
    emplace(realVec);
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
std::span<const double> RooVectorDataStore::getWeightBatch(std::size_t first, std::size_t len) const
{
  if (_extWgtArray) {
    return std::span<const double>(_extWgtArray + first, _extWgtArray + first + len);
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
      return fullVec->bufE();
    }
  }
  return false ;
}


bool RooVectorDataStore::hasAsymError(RooAbsReal* real) {

  // First try a match by name
  for (auto fullVec : _realfStoreList) {
    if (std::string(fullVec->bufArg()->GetName())==real->GetName()) {
      return fullVec->bufEL();
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
    if(realf->bufE()) out.reals.emplace_back(name + "Err", realf->dataE().data());
    if(realf->bufEL()) out.reals.emplace_back(name + "ErrLo", realf->dataEL().data());
    if(realf->bufEH()) out.reals.emplace_back(name + "ErrHi", realf->dataEH().data());
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
