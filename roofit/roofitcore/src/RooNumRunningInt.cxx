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
\file RooNumRunningInt.cxx
\class RooNumRunningInt
\ingroup Roofitcore

Class RooNumRunningInt is an implementation of RooAbsCachedReal that represents a running integral
\f[ RI(f(x)) = \int_{xlow}^{x} f(x') dx'                 \f]
that is calculated internally with a numeric technique: The input function
is first sampled into a histogram, which is then numerically integrated.
The output function is an interpolated version of the integrated histogram.
The sampling density is controlled by the binning named "cache" in the observable x.
The shape of the p.d.f is always calculated for the entire domain in x and
cached in a histogram. The cache histogram is automatically recalculated
when any of the parameters of the input p.d.f. has changed.
**/

#include "Riostream.h"

#include "RooAbsPdf.h"
#include "RooNumRunningInt.h"
#include "RooAbsReal.h"
#include "RooMsgService.h"
#include "RooDataHist.h"
#include "RooHistPdf.h"
#include "RooRealVar.h"

using namespace std;

ClassImp(RooNumRunningInt);
  ;



////////////////////////////////////////////////////////////////////////////////
/// Construct running integral of function '_func' over x_print from
/// the lower bound on _x to the present value of _x using a numeric
/// sampling technique. The sampling frequency is controlled by the
/// binning named 'bname' and a default second order interpolation
/// is applied to smooth the histogram-based c.d.f.

RooNumRunningInt::RooNumRunningInt(const char *name, const char *title, RooAbsReal& _func, RooRealVar& _x, const char* bname) :
   RooAbsCachedReal(name,title),
   func("func","func",this,_func),
   x("x","x",this,_x),
   _binningName(bname?bname:"cache")
 {
   setInterpolationOrder(2) ;
 }




////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooNumRunningInt::RooNumRunningInt(const RooNumRunningInt& other, const char* name) :
   RooAbsCachedReal(other,name),
   func("func",this,other.func),
   x("x",this,other.x),
   _binningName(other._binningName)
 {
 }



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooNumRunningInt::~RooNumRunningInt()
{
}


////////////////////////////////////////////////////////////////////////////////
/// Return unique name for RooAbsCachedPdf cache components
/// constructed from input function name

const char* RooNumRunningInt::inputBaseName() const
{
  static string ret ;
  ret = func.arg().GetName() ;
  ret += "_NUMRUNINT" ;
  return ret.c_str() ;
} ;



////////////////////////////////////////////////////////////////////////////////
/// Construct RunningIntegral CacheElement

RooNumRunningInt::RICacheElem::RICacheElem(const RooNumRunningInt& self, const RooArgSet* nset) :
  FuncCacheElem(self,nset), _self(&const_cast<RooNumRunningInt&>(self))
{
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


////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooNumRunningInt::RICacheElem::~RICacheElem()
{
  // Delete temp arrays
  delete[] _ax ;
  delete[] _ay ;
}


////////////////////////////////////////////////////////////////////////////////
/// Return all RooAbsArg components contained in cache element

RooArgList RooNumRunningInt::RICacheElem::containedArgs(Action action)
{
  RooArgList ret ;
  ret.add(FuncCacheElem::containedArgs(action)) ;
  ret.add(*_self) ;
  ret.add(*_xx) ;
  return ret ;
}



////////////////////////////////////////////////////////////////////////////////
/// Calculate the numeric running integral and store
/// the result in the cache histogram provided
/// by RooAbsCachedPdf

void RooNumRunningInt::RICacheElem::calculate(bool cdfmode)
{
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
      hist()->set(i, _ay[i]/_ay[nbins-1], 0.);
    } else {
      hist()->set(i, _ay[i]*binv, 0.);
    }
  }

  if (cdfmode) {
    func()->setCdfBoundaries(true) ;
  }
  _self->x = xsave ;
}



////////////////////////////////////////////////////////////////////////////////
/// Fill all empty histogram bins in the range [ixlo,ixhi] where nbins is the
/// total number of histogram bins. This method samples the mid-point of the
/// range and if the mid-point value is within small tolerance of the interpolated
/// mid-point value fills all remaining elements through linear interpolation.
/// If the tolerance is exceeded, the algorithm is recursed on the two subranges
/// [xlo,xmid] and [xmid,xhi]

void RooNumRunningInt::RICacheElem::addRange(Int_t ixlo, Int_t ixhi, Int_t nbins)
{
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


////////////////////////////////////////////////////////////////////////////////
/// Sample function at bin ix

void RooNumRunningInt::RICacheElem::addPoint(Int_t ix)
{
  hist()->get(ix) ;
  _self->x = _xx->getVal() ;
  _ay[ix] = _self->func.arg().getVal(*_xx) ;

}



////////////////////////////////////////////////////////////////////////////////
/// Fill the cache object by calling its calculate() method

void RooNumRunningInt::fillCacheObject(RooAbsCachedReal::FuncCacheElem& cache) const
{
  RICacheElem& riCache = static_cast<RICacheElem&>(cache) ;
  riCache.calculate(false) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return observable in nset to be cached by RooAbsCachedPdf
/// this is always the x observable that is integrated

RooArgSet* RooNumRunningInt::actualObservables(const RooArgSet& /*nset*/) const
{
  RooArgSet* ret = new RooArgSet ;
  ret->add(x.arg()) ;
  return ret ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return the parameters of the cache created by RooAbsCachedPdf.
/// These are always the input functions parameter, but never the
/// integrated variable x.

RooArgSet* RooNumRunningInt::actualParameters(const RooArgSet& /*nset*/) const
{
  RooArgSet* ret = func.arg().getParameters(RooArgSet()) ;
  ret->remove(x.arg(),true,true) ;
  return ret ;
}


////////////////////////////////////////////////////////////////////////////////
/// Create custom cache element for running integral calculations

RooAbsCachedReal::FuncCacheElem* RooNumRunningInt::createCache(const RooArgSet* nset) const
{
  return new RICacheElem(*const_cast<RooNumRunningInt*>(this),nset) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Dummy function that is never called

Double_t RooNumRunningInt::evaluate() const
{
  cout << "RooNumRunningInt::evaluate(" << GetName() << ")" << endl ;
  return 0 ;
}


