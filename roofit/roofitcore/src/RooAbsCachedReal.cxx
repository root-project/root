 /*****************************************************************************
  * Project: RooFit                                                           *
  *                                                                           *
  * Copyright (c) 2000-2005, Regents of the University of California          *
  *                          and Stanford University. All rights reserved.    *
  *                                                                           *
  * Redistribution and use in source and binary forms,                        *
  * with or without modification, are permitted according to the terms        *
  * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
  *****************************************************************************/

/**
\file RooAbsCachedReal.cxx
\class RooAbsCachedReal
\ingroup Roofitcore

RooAbsCachedReal is the abstract base class for functions that need or
want to cache their evaluate() output in a RooHistFunc defined in
terms of the used observables. This base class manages the creation
and storage of all RooHistFunc cache p.d.fs and the RooDataHists
that define their shape. Implementations of RooAbsCachedReal must
define member function fillCacheObject() which serves to fill an
already created RooDataHist with the functions function values. In
addition the member functions actualObservables() and
actualParameters() must be define which report what the actual
observables to be cached are for a given set of observables passed
by the user to getVal() and on which parameters need to be tracked
for changes to trigger a refilling of the cache histogram.
**/

#include "Riostream.h"
using namespace std ;

#include "TString.h"
#include "RooAbsCachedReal.h"
#include "RooAbsReal.h"
#include "RooMsgService.h"
#include "RooDataHist.h"
#include "RooHistFunc.h"
#include "RooChangeTracker.h"
#include "RooExpensiveObjectCache.h"

ClassImp(RooAbsCachedReal);



////////////////////////////////////////////////////////////////////////////////
/// Constructor

RooAbsCachedReal::RooAbsCachedReal(const char *name, const char *title, Int_t ipOrder) :
  RooAbsReal(name,title),
  _cacheMgr(this,10),
  _ipOrder(ipOrder),
  _disableCache(false)
 {
 }



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooAbsCachedReal::RooAbsCachedReal(const RooAbsCachedReal& other, const char* name) :
   RooAbsReal(other,name),
   _cacheMgr(other._cacheMgr,this),
   _ipOrder(other._ipOrder),
   _disableCache(other._disableCache)
 {
 }



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooAbsCachedReal::~RooAbsCachedReal()
{
}



////////////////////////////////////////////////////////////////////////////////
/// Implementation of getVal() overriding default implementation
/// of RooAbsReal. Return value stored in cache p.d.f
/// rather than return value of evaluate() which is undefined
/// for RooAbsCachedReal

double RooAbsCachedReal::getValV(const RooArgSet* nset) const
{
  if (_disableCache) {
    return RooAbsReal::getValV(nset) ;
  }

  // Cannot call cached p.d.f w.o nset
  // if (!nset) return evaluate() ;

  // Calculate current unnormalized value of object
  // coverity[NULL_RETURNS]
  FuncCacheElem* cache = getCache(nset) ;

  _value = cache->func()->getVal() ;

  return _value ;
}


////////////////////////////////////////////////////////////////////////////////
/// Interface function to create an internal cache object that represent
/// each cached function configuration. This interface allows to create and
/// return a class derived from RooAbsCachedReal::FuncCacheElem so that
/// a derived class fillCacheObject implementation can utilize extra functionality
/// defined in such a derived cache class

RooAbsCachedReal::FuncCacheElem* RooAbsCachedReal::createCache(const RooArgSet* nset) const
{
  return new FuncCacheElem(const_cast<RooAbsCachedReal&>(*this),nset) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Retrieve cache corresponding to observables in nset

RooAbsCachedReal::FuncCacheElem* RooAbsCachedReal::getCache(const RooArgSet* nset) const
{
  // Check if this configuration was created becfore
  Int_t sterileIdx(-1) ;
  FuncCacheElem* cache = (FuncCacheElem*) _cacheMgr.getObj(nset,0,&sterileIdx) ;
  if (cache) {
    if (cache->paramTracker()->hasChanged(true)) {
      ccoutD(Eval) << "RooAbsCachedReal::getCache(" << GetName() << ") cached function "
        << cache->func()->GetName() << " requires recalculation as parameters changed" << endl ;
      fillCacheObject(*cache) ;
      cache->func()->setValueDirty() ;
    }
    return cache ;
  }

  cache = createCache(nset) ;

  // Set cache function data to ADirty since function will need update every time in cache update process
  for (auto* arg : *(cache->hist()->get()) ) {
    arg->setOperMode(ADirty);
  }

  // Check if we have contents registered already in global expensive object cache
  RooDataHist* htmp = (RooDataHist*) expensiveObjectCache().retrieveObject(cache->hist()->GetName(),RooDataHist::Class(),cache->paramTracker()->parameters()) ;

  if (htmp) {

    cache->hist()->reset() ;
    cache->hist()->add(*htmp) ;

  } else {

    fillCacheObject(*cache) ;

    RooDataHist* eoclone = new RooDataHist(*cache->hist()) ;
    eoclone->removeSelfFromDir() ;
    expensiveObjectCache().registerObject(GetName(),cache->hist()->GetName(),*eoclone,cache->paramTracker()->parameters()) ;
  }

  // Store this cache configuration
  Int_t code = _cacheMgr.setObj(nset,0,((RooAbsCacheElement*)cache),0) ;
  ccoutD(Caching) << "RooAbsCachedReal("<<this<<")::getCache(" << GetName() << ") creating new cache " << cache->func()->GetName() << " for nset " << (nset?*nset:RooArgSet()) << " with code " << code << endl ;

  return cache ;
}



////////////////////////////////////////////////////////////////////////////////
/// Constructor of cache storage unit class
///
/// Create RooDataHist that will cache function values and create
/// RooHistFunc that represent s RooDataHist shape as function, create
/// meta object that tracks changes in declared parameters of p.d.f
/// through actualParameters()

RooAbsCachedReal::FuncCacheElem::FuncCacheElem(const RooAbsCachedReal& self, const RooArgSet* nset)
{
  // Disable source caching by default
  _cacheSource = false ;
  _sourceClone = 0 ;

  RooArgSet* nset2 = self.actualObservables(nset?*nset:RooArgSet()) ;

  RooArgSet orderedObs ;
  self.preferredObservableScanOrder(*nset2,orderedObs) ;

  // Create RooDataHist
  auto hname = std::string(self.inputBaseName()) + "_CACHEHIST" + self.cacheNameSuffix(*nset2).Data();

  _hist = new RooDataHist(hname,hname,*nset2,self.binningName()) ;
  _hist->removeSelfFromDir() ;

  RooArgSet* observables= self.actualObservables(*nset2) ;

  // Create RooHistFunc
  TString funcname = self.inputBaseName() ;
  funcname.Append("_CACHE") ;
  funcname.Append(self.cacheNameSuffix(*nset2)) ;
  _func = new RooHistFunc(funcname,funcname,*observables,*_hist,self.getInterpolationOrder()) ;
  if (self.operMode()==ADirty) _func->setOperMode(ADirty) ;

  // Set initial state of cache to dirty
  _func->setValueDirty() ;

  // Create pseudo-object that tracks changes in parameter values
  std::unique_ptr<RooArgSet> params{self.actualParameters(orderedObs)};
  string name= Form("%s_CACHEPARAMS",_func->GetName()) ;
  _paramTracker = new RooChangeTracker(name.c_str(),name.c_str(),*params,true) ;
  _paramTracker->hasChanged(true) ; // clear dirty flag as cache is up-to-date upon creation

  // Introduce formal dependency of RooHistFunc on parameters so that const optimization code
  // makes the correct decisions
  _func->addServerList(*params) ;

  delete observables ;
  delete nset2 ;
}


////////////////////////////////////////////////////////////////////////////////

RooAbsCachedReal::FuncCacheElem::~FuncCacheElem()
{
  if (_sourceClone) { delete _sourceClone ; }
  delete _paramTracker ;
  delete _func ;
  delete _hist ;
}




////////////////////////////////////////////////////////////////////////////////
/// Construct unique suffix name for cache p.d.f object

TString RooAbsCachedReal::cacheNameSuffix(const RooArgSet& nset) const
{
  TString name ;
  name.Append("_Obs[") ;
  if (nset.getSize()>0) {
    bool first(true) ;
    for (RooAbsArg * arg : nset) {
      if (first) {
   first=false ;
      } else {
   name.Append(",") ;
      }
      name.Append(arg->GetName()) ;
    }
  }

  name.Append("]") ;
  const char* payloadUS = payloadUniqueSuffix() ;
  if (payloadUS) {
    name.Append(payloadUS) ;
  }
  return name ;
}



////////////////////////////////////////////////////////////////////////////////
/// Set interpolation order of RooHistFunct representing cache histogram

void RooAbsCachedReal::setInterpolationOrder(Int_t order)
{
  _ipOrder = order ;

  for (Int_t i=0 ; i<_cacheMgr.cacheSize() ; i++) {
    FuncCacheElem* cache = (FuncCacheElem*) _cacheMgr.getObjByIndex(i) ;
    if (cache) {
      cache->func()->setInterpolationOrder(order) ;
    }
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Return list of contained RooAbsArg objects

RooArgList RooAbsCachedReal::FuncCacheElem::containedArgs(Action)
{
  RooArgList ret(*func()) ;

  ret.add(*_paramTracker) ;
  if (_sourceClone) {
    ret.add(*_sourceClone) ;
  }
  return ret ;
}


////////////////////////////////////////////////////////////////////////////////
/// Print contents of cache when printing self as part of object tree

void RooAbsCachedReal::FuncCacheElem::printCompactTreeHook(ostream& os, const char* indent, Int_t curElem, Int_t maxElem)
{
  if (curElem==0) {
    os << indent << "--- RooAbsCachedReal begin cache ---" << endl ;
  }

  TString indent2(indent) ;
  indent2 += Form("[%d] ",curElem) ;
  func()->printCompactTree(os,indent2) ;

  if (curElem==maxElem) {
    os << indent << "--- RooAbsCachedReal end cache --- " << endl ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Return analytical integration capabilities of the RooHistFunc that corresponds to the set of observables in allVars

Int_t RooAbsCachedReal::getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet, const char* rangeName) const
{
  FuncCacheElem* cache = getCache(normSet?normSet:&allVars) ;
  Int_t code = cache->func()->getAnalyticalIntegralWN(allVars,analVars,normSet,rangeName) ;
  _anaIntMap[code].first = &allVars ;
  _anaIntMap[code].second = normSet ;
  return code ;
}



////////////////////////////////////////////////////////////////////////////////
/// Forward call to implementation in relevant RooHistFunc instance

double RooAbsCachedReal::analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName) const
{
  if (code==0) {
    return getVal(normSet) ;
  }

  const RooArgSet* anaVars = _anaIntMap[code].first ;
  const RooArgSet* normSet2 = _anaIntMap[code].second ;

  FuncCacheElem* cache = getCache(normSet2?normSet2:anaVars) ;
  return cache->func()->analyticalIntegralWN(code,normSet,rangeName) ;

}





