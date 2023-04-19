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
\file RooCachedReal.cxx
\class RooCachedReal
\ingroup Roofitcore

RooCachedReal is an implementation of RooAbsCachedReal that can cache
any external RooAbsReal input function provided in the constructor.
**/

#include "Riostream.h"

#include "RooAbsPdf.h"
#include "RooCachedReal.h"
#include "RooAbsReal.h"
#include "RooAbsCategory.h"
#include "RooMsgService.h"
#include "RooDataHist.h"
#include "RooHistPdf.h"
#include "RooChangeTracker.h"

using namespace std;

ClassImp(RooCachedReal);
  ;


////////////////////////////////////////////////////////////////////////////////
/// Constructor taking name, title and function to be cached. To control
/// granularity of the binning of the cache histogram set the desired properties
/// in the binning named "cache" in the observables of the function

RooCachedReal::RooCachedReal(const char *name, const char *title, RooAbsReal& _func) :
   RooAbsCachedReal(name,title),
   func("func","func",this,_func),
   _useCdfBoundaries(false),
   _cacheSource(false)
 {
   // Choose same expensive object cache as input function
   setExpensiveObjectCache(_func.expensiveObjectCache()) ;
 }




////////////////////////////////////////////////////////////////////////////////
/// Constructor taking name, title and function to be cached and
/// fixed choice of variable to cache. To control granularity of the
/// binning of the cache histogram set the desired properties in the
/// binning named "cache" in the observables of the function.
/// If the fixed set of cache observables does not match the observables
/// defined in the use context of the p.d.f the cache is still filled
/// completely. Ee.g. when it is specified to cache x and p and only x
/// is a observable in the given use context the cache histogram will
/// store sampled values for all values of observable x and parameter p.
/// In such a mode of operation the cache will also not be recalculated
/// if the observable p changes

RooCachedReal::RooCachedReal(const char *name, const char *title, RooAbsReal& _func, const RooArgSet& cacheObs) :
   RooAbsCachedReal(name,title),
   func("func","func",this,_func),
   _cacheObs("cacheObs","cacheObs",this,false,false),
   _useCdfBoundaries(false),
   _cacheSource(false)
 {
   _cacheObs.add(cacheObs) ;

   // Choose same expensive object cache as input function
   setExpensiveObjectCache(_func.expensiveObjectCache()) ;
 }




////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooCachedReal::RooCachedReal(const RooCachedReal& other, const char* name) :
   RooAbsCachedReal(other,name),
   func("func",this,other.func),
   _cacheObs("cacheObs",this,other._cacheObs),
   _useCdfBoundaries(other._useCdfBoundaries),
   _cacheSource(other._cacheSource)
 {
 }



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooCachedReal::~RooCachedReal()
{
}


////////////////////////////////////////////////////////////////////////////////
/// Interface function to create an internal cache object that represent
/// each cached function configuration. This interface allows to create and
/// return a class derived from RooAbsCachedReal::FuncCacheElem so that
/// a derived class fillCacheObject implementation can utilize extra functionality
/// defined in such a derived cache class

RooAbsCachedReal::FuncCacheElem* RooCachedReal::createCache(const RooArgSet* nset) const
{
  FuncCacheElem* ret = RooAbsCachedReal::createCache(nset) ;
  if (_cacheSource) {
    ret->setCacheSource(true) ;
  }
  return ret ;
}



////////////////////////////////////////////////////////////////////////////////
/// Update contents of cache histogram by resampling the input function

void RooCachedReal::fillCacheObject(RooAbsCachedReal::FuncCacheElem& cache) const
{
  unsigned nDim = cache.hist()->get()->getSize();
  if (nDim>1) {
    unsigned nCat(0);
    for(RooAbsArg * arg : *cache.hist()->get()) {
      if (dynamic_cast<RooAbsCategory*>(arg)) ++nCat;
    }
    if (nDim>nCat+1) {
        coutP(Eval) << "RooCachedReal::fillCacheObject(" << GetName() << ") filling "
                    << nCat << " + " << nDim-nCat <<" dimensional cache (" << cache.hist()->numEntries() << " points)" <<endl;
    }
  }

  // Make deep clone of self and attach to dataset observables
  if (!cache.sourceClone()) {
    RooAbsArg* sourceClone = func.arg().cloneTree() ;
    cache.setSourceClone((RooAbsReal*)sourceClone) ;
    cache.sourceClone()->recursiveRedirectServers(*cache.hist()->get()) ;
    cache.sourceClone()->recursiveRedirectServers(cache.paramTracker()->parameters());
  }

  // Iterator over all bins of RooDataHist and fill weights
  for (Int_t i=0 ; i<cache.hist()->numEntries() ; i++) {
    const RooArgSet* obs = cache.hist()->get(i) ;
    double binVal = cache.sourceClone()->getVal(obs) ;
    cache.hist()->set(i, binVal, 0.);
  }

  // Delete source clone if we don't cache it
  if (!cache.cacheSource()) {
    cache.setSourceClone(0) ;
  }

  cache.func()->setCdfBoundaries(_useCdfBoundaries) ;

}



////////////////////////////////////////////////////////////////////////////////
/// If this pdf is operated with a fixed set of observables, return
/// the subset of the fixed observables that are actual dependents
/// of the external input p.d.f. If this p.d.f is operated without
/// a fixed set of cache observables, return the actual observables
/// of the external input p.d.f given the choice of observables defined
/// in nset

RooArgSet* RooCachedReal::actualObservables(const RooArgSet& nset) const
{
  if (_cacheObs.getSize()>0) {
    return func.arg().getObservables(_cacheObs) ;
  }

  return func.arg().getObservables(nset) ;
}



////////////////////////////////////////////////////////////////////////////////
/// If this p.d.f is operated with a fixed set of observables, return
/// all variables of the external input p.d.f that are not one of
/// the cache observables. If this p.d.f is operated in automatic mode,
/// return the parameters of the external input p.d.f

RooFit::OwningPtr<RooArgSet> RooCachedReal::actualParameters(const RooArgSet& nset) const
{
   return func->getParameters(_cacheObs.empty() ? nset : _cacheObs);
}


void RooCachedReal::operModeHook()
{
  if (operMode()==ADirty) {
    ((RooAbsArg*)func.absArg())->setOperMode(ADirty) ;
  }
}




