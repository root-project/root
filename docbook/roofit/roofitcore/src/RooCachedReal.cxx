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

//////////////////////////////////////////////////////////////////////////////
// 
// BEGIN_HTML
// RooCachedReal is an implementation of RooAbsCachedReal that can cache
// any external RooAbsReal input function provided in the constructor. 
// END_HTML
//

#include "Riostream.h" 

#include "RooAbsPdf.h"
#include "RooCachedReal.h" 
#include "RooAbsReal.h" 
#include "RooMsgService.h"
#include "RooDataHist.h"
#include "RooHistPdf.h"

ClassImp(RooCachedReal) 
  ;


//_____________________________________________________________________________
RooCachedReal::RooCachedReal(const char *name, const char *title, RooAbsReal& _func) :
   RooAbsCachedReal(name,title), 
   func("func","func",this,_func),
   _useCdfBoundaries(kFALSE)
 { 
   // Constructor taking name, title and function to be cached. To control
   // granularity of the binning of the cache histogram set the desired properties
   // in the binning named "cache" in the observables of the function
   
   // Choose same expensive object cache as input function
   setExpensiveObjectCache(_func.expensiveObjectCache()) ;
 } 




//_____________________________________________________________________________
RooCachedReal::RooCachedReal(const char *name, const char *title, RooAbsReal& _func, const RooArgSet& cacheObs) :
   RooAbsCachedReal(name,title), 
   func("func","func",this,_func),
   _cacheObs("cacheObs","cacheObs",this,kFALSE,kFALSE)  
 { 
   // Constructor taking name, title and function to be cached and
   // fixed choice of variable to cache. To control granularity of the
   // binning of the cache histogram set the desired properties in the
   // binning named "cache" in the observables of the function.
   // If the fixed set of cache observables does not match the observables
   // defined in the use context of the p.d.f the cache is still filled
   // completely. Ee.g. when it is specified to cache x and p and only x 
   // is a observable in the given use context the cache histogram will
   // store sampled values for all values of observable x and parameter p.
   // In such a mode of operation the cache will also not be recalculated
   // if the observable p changes

   _cacheObs.add(cacheObs) ;

   // Choose same expensive object cache as input function
   setExpensiveObjectCache(_func.expensiveObjectCache()) ;
 } 




//_____________________________________________________________________________
RooCachedReal::RooCachedReal(const RooCachedReal& other, const char* name) :  
   RooAbsCachedReal(other,name), 
   func("func",this,other.func),
   _cacheObs("cacheObs",this,other._cacheObs),
   _useCdfBoundaries(other._useCdfBoundaries)
 { 
   // Copy constructor
 } 



//_____________________________________________________________________________
RooCachedReal::~RooCachedReal() 
{
  // Destructor
}



//_____________________________________________________________________________
void RooCachedReal::fillCacheObject(RooAbsCachedReal::FuncCacheElem& cache) const 
{
  // Update contents of cache histogram by resampling the input function

  if (cache.hist()->get()->getSize()>1) {
    coutP(Eval) << "RooCachedReal::fillCacheObject(" << GetName() << ") filling multi-dimensional cache (" << cache.hist()->numEntries() << " points)" ;
  }

  func.arg().fillDataHist(cache.hist(),0,1.0,kFALSE,kTRUE) ;
  cache.func()->setCdfBoundaries(_useCdfBoundaries) ;

  if (cache.hist()->get()->getSize()>1) {
    ccoutP(Eval) << endl ;
  }

}



//_____________________________________________________________________________
RooArgSet* RooCachedReal::actualObservables(const RooArgSet& nset) const 
{ 
  // If this pdf is operated with a fixed set of observables, return
  // the subset of the fixed observables that are actual dependents
  // of the external input p.d.f. If this p.d.f is operated without
  // a fixed set of cache observables, return the actual observables
  // of the external input p.d.f given the choice of observables defined
  // in nset

  if (_cacheObs.getSize()>0) {
    return func.arg().getObservables(_cacheObs) ;
  } 

  return func.arg().getObservables(nset) ; 
}



//_____________________________________________________________________________
RooArgSet* RooCachedReal::actualParameters(const RooArgSet& nset) const 
{ 
  // If this p.d.f is operated with a fixed set of observables, return
  // all variables of the external input p.d.f that are not one of
  // the cache observables. If this p.d.f is operated in automatic mode,
  // return the parameters of the external input p.d.f

  if (_cacheObs.getSize()>0) {
    return func.arg().getParameters(_cacheObs) ;
  } 
  return func.arg().getParameters(nset) ; 
}





