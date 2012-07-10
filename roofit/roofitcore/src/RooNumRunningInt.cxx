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
// Class RooNumRunningInt is an implementation of RooAbsCachedReal that represents a running integral
// <pre>
// RI(f(x)) = Int[x_lo,x] f(x') dx'
// </pre>
// that is calculated internally with a numeric technique: The input function
// is first sampled into a histogram, which is then numerically integrated.
// The output function is an interpolated version of the integrated histogram.
// The sampling density is controlled by the binning named "cache" in the observable x.
// The shape of the p.d.f is always calculated for the entire domain in x and
// cached in a histogram. The cache histogram is automatically recalculated
// when any of the parameters of the input p.d.f. has changed.
// END_HTML
//

#include "Riostream.h" 

#include "RooAbsPdf.h"
#include "RooNumRunningInt.h" 
#include "RooAbsReal.h" 
#include "RooMsgService.h"
#include "RooDataHist.h"
#include "RooHistPdf.h"
#include "RooRealVar.h"

using namespace std;

ClassImp(RooNumRunningInt) 
  ;



//_____________________________________________________________________________
RooNumRunningInt::RooNumRunningInt(const char *name, const char *title, RooAbsReal& _func, RooRealVar& _x, const char* bname) :
   RooAbsCachedReal(name,title), 
   func("func","func",this,_func),
   x("x","x",this,_x),
   _binningName(bname?bname:"cache")
 { 
   // Construct running integral of function '_func' over x_print from
   // the lower bound on _x to the present value of _x using a numeric
   // sampling technique. The sampling frequency is controlled by the
   // binning named 'bname' and a default second order interpolation
   // is applied to smooth the histogram-based c.d.f.

   setInterpolationOrder(2) ;
 } 




//_____________________________________________________________________________
RooNumRunningInt::RooNumRunningInt(const RooNumRunningInt& other, const char* name) :  
   RooAbsCachedReal(other,name), 
   func("func",this,other.func),
   x("x",this,other.x),
   _binningName(other._binningName)
 { 
   // Copy constructor
 } 



//_____________________________________________________________________________
RooNumRunningInt::~RooNumRunningInt() 
{
  // Destructor
}


//_____________________________________________________________________________
const char* RooNumRunningInt::inputBaseName() const 
{
  // Return unique name for RooAbsCachedPdf cache components
  // constructed from input function name

  static string ret ;
  ret = func.arg().GetName() ;
  ret += "_NUMRUNINT" ;
  return ret.c_str() ;
} ;



//_____________________________________________________________________________
RooNumRunningInt::RICacheElem::RICacheElem(const RooNumRunningInt& self, const RooArgSet* nset) : 
  FuncCacheElem(self,nset), _self(&const_cast<RooNumRunningInt&>(self))
{
  // Construct RunningIntegral CacheElement
  
  // Instantiate temp arrays
  _ax = new Double_t[hist()->numEntries()] ;
  _ay = new Double_t[hist()->numEntries()] ;

  // Copy X values from histo  
  _xx = (RooRealVar*) hist()->get()->find(self.x.arg().GetName()) ;
  for (int i=0 ; i<hist()->numEntries() ; i++) {
    hist()->get(i) ;
    _ax[i] = _xx->getVal() ;
    _ay[i] = -1 ;
  }

}


//_____________________________________________________________________________
RooNumRunningInt::RICacheElem::~RICacheElem() 
{
  // Destructor

  // Delete temp arrays 
  delete[] _ax ;
  delete[] _ay ;    
}


//_____________________________________________________________________________
RooArgList RooNumRunningInt::RICacheElem::containedArgs(Action action) 
{
  // Return all RooAbsArg components contained in cache element

  RooArgList ret ;
  ret.add(FuncCacheElem::containedArgs(action)) ;
  ret.add(*_self) ;
  ret.add(*_xx) ;
  return ret ;  
}



//_____________________________________________________________________________
void RooNumRunningInt::RICacheElem::calculate(Bool_t cdfmode) 
{
  // Calculate the numeric running integral and store
  // the result in the cache histogram provided
  // by RooAbsCachedPdf

  // Update contents of histogram
  Int_t nbins = hist()->numEntries() ;
  
  Double_t xsave = _self->x ;

  Int_t lastHi=0 ;
  Int_t nInitRange=32 ;
  for (int i=1 ; i<=nInitRange ; i++) {
    Int_t hi = (i*nbins)/nInitRange -1 ;
    Int_t lo = lastHi ;
    addRange(lo,hi,nbins) ;
    lastHi=hi ;
  }

  // Perform numeric integration
  for (int i=1 ; i<nbins ; i++) {
    _ay[i] += _ay[i-1] ;
  }
 
  // Normalize and transfer to cache histogram
  Double_t binv = (_self->x.max()-_self->x.min())/nbins ;
  for (int i=0 ; i<nbins ; i++) {
    hist()->get(i) ;
    if (cdfmode) {
      hist()->set(_ay[i]/_ay[nbins-1]) ;
    } else {
      hist()->set(_ay[i]*binv) ;
    }
  }

  if (cdfmode) {
    func()->setCdfBoundaries(kTRUE) ;
  }
  _self->x = xsave ;
}



//_____________________________________________________________________________
void RooNumRunningInt::RICacheElem::addRange(Int_t ixlo, Int_t ixhi, Int_t nbins) 
{
  // Fill all empty histogram bins in the range [ixlo,ixhi] where nbins is the
  // total number of histogram bins. This method samples the mid-point of the
  // range and if the mid-point value is within small tolerance of the interpolated
  // mid-point value fills all remaining elements through linear interpolation.
  // If the tolerance is exceeded, the algorithm is recursed on the two subranges
  // [xlo,xmid] and [xmid,xhi]

  // Add first and last point, if not there already
  if (_ay[ixlo]<0) {
    addPoint(ixlo) ;
  }
  if (_ay[ixhi]<0) {
    addPoint(ixhi) ;
  }

  // Terminate here if there is no gap
  if (ixhi-ixlo==1) {
    return ;
  }

  // If gap size is one, simply fill gap and return
  if (ixhi-ixlo==2) {
    addPoint(ixlo+1) ;
    return ;
  }

  // Add mid-point
  Int_t ixmid = (ixlo+ixhi)/2 ;
  addPoint(ixmid) ;
  
  // Calculate difference of mid-point w.r.t interpolated value
  Double_t yInt = _ay[ixlo] + (_ay[ixhi]-_ay[ixlo])*(ixmid-ixlo)/(ixhi-ixlo) ;
  
  // If relative deviation is greater than tolerance divide and iterate
  if (fabs(yInt-_ay[ixmid])*(_ax[nbins-1]-_ax[0])>1e-6) {    
    addRange(ixlo,ixmid,nbins) ;
    addRange(ixmid,ixhi,nbins) ;
  } else {
    for (Int_t j=ixlo+1 ; j<ixmid ; j++) { 
      _ay[j] = _ay[ixlo] + (_ay[ixmid]-_ay[ixlo])*(j-ixlo)/(ixmid-ixlo) ; 
    }
    for (Int_t j=ixmid+1 ; j<ixhi ; j++) { 
      _ay[j] = _ay[ixmid] + (_ay[ixhi]-_ay[ixmid])*(j-ixmid)/(ixhi-ixmid) ; 
    }
  }

}


//_____________________________________________________________________________
void RooNumRunningInt::RICacheElem::addPoint(Int_t ix)
{
  // Sample function at bin ix

  hist()->get(ix) ;
  _self->x = _xx->getVal() ;
  _ay[ix] = _self->func.arg().getVal(*_xx) ;

}



//_____________________________________________________________________________
void RooNumRunningInt::fillCacheObject(RooAbsCachedReal::FuncCacheElem& cache) const 
{
  // Fill the cache object by calling its calculate() method
  RICacheElem& riCache = static_cast<RICacheElem&>(cache) ;
  riCache.calculate(kFALSE) ;
}



//_____________________________________________________________________________
RooArgSet* RooNumRunningInt::actualObservables(const RooArgSet& /*nset*/) const 
{
  // Return observable in nset to be cached by RooAbsCachedPdf
  // this is always the x observable that is integrated

  RooArgSet* ret = new RooArgSet ;
  ret->add(x.arg()) ;
  return ret ;
}



//_____________________________________________________________________________
RooArgSet* RooNumRunningInt::actualParameters(const RooArgSet& /*nset*/) const 
{
  // Return the parameters of the cache created by RooAbsCachedPdf.
  // These are always the input functions parameter, but never the
  // integrated variable x.

  RooArgSet* ret = func.arg().getParameters(RooArgSet()) ;
  ret->remove(x.arg(),kTRUE,kTRUE) ;
  return ret ;
}


//_____________________________________________________________________________
RooAbsCachedReal::FuncCacheElem* RooNumRunningInt::createCache(const RooArgSet* nset) const 
{
  // Create custom cache element for running integral calculations

  return new RICacheElem(*const_cast<RooNumRunningInt*>(this),nset) ; 
}


//_____________________________________________________________________________
Double_t RooNumRunningInt::evaluate() const 
{
  // Dummy function that is never called

  cout << "RooNumRunningInt::evaluate(" << GetName() << ")" << endl ;
  return 0 ;
}


