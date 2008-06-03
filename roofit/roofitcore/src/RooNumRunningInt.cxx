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

 // -- CLASS DESCRIPTION [PDF] -- 
 // Your description goes here... 

#include "Riostream.h" 

#include "RooAbsPdf.h"
#include "RooNumRunningInt.h" 
#include "RooAbsReal.h" 
#include "RooMsgService.h"
#include "RooDataHist.h"
#include "RooHistPdf.h"
#include "RooRealVar.h"

ClassImp(RooNumRunningInt) 
  ;


RooNumRunningInt::RooNumRunningInt(const char *name, const char *title, RooAbsReal& _func, RooRealVar& _x, const char* bname) :
   RooAbsCachedReal(name,title), 
   func("func","func",this,_func),
   x("x","x",this,_x),
   _binningName(bname?bname:"cache")
 { 
   setInterpolationOrder(2) ;
 } 




RooNumRunningInt::RooNumRunningInt(const RooNumRunningInt& other, const char* name) :  
   RooAbsCachedReal(other,name), 
   func("func",this,other.func),
   x("x",this,other.x),
   _binningName(other._binningName)
 { 
 } 


RooNumRunningInt::~RooNumRunningInt() 
{
}

const char* RooNumRunningInt::inputBaseName() const 
{
  static string ret ;
  ret = func.arg().GetName() ;
  ret += "_NUMRUNINT" ;
  return ret.c_str() ;
} ;


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

RooNumRunningInt::RICacheElem::~RICacheElem() 
{
  // Delete temp arrays 
  delete[] _ax ;
  delete[] _ay ;    
}

RooArgList RooNumRunningInt::RICacheElem::containedArgs(Action action) 
{
  RooArgList ret ;
  ret.add(FuncCacheElem::containedArgs(action)) ;
  ret.add(*_self) ;
  ret.add(*_xx) ;
  return ret ;  
}


void RooNumRunningInt::RICacheElem::calculate(Bool_t cdfmode) 
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
      hist()->set(_ay[i]/_ay[nbins-1]*binv) ;
    } else {
      hist()->set(_ay[i]*binv) ;
    }
  }

  if (cdfmode) {
    func()->setCdfBoundaries(kTRUE) ;
  }
  _self->x = xsave ;
}


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
  if (fabs(yInt-_ay[ixmid])*(_ax[nbins]-_ax[0])>1e-6) {    
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

void RooNumRunningInt::RICacheElem::addPoint(Int_t ix)
{
  hist()->get(ix) ;
  _self->x = _xx->getVal() ;
  _ay[ix] = _self->func.arg().getVal(*_xx) ;

}


void RooNumRunningInt::fillCacheObject(RooAbsCachedReal::FuncCacheElem& cache) const 
{
  RICacheElem& riCache = static_cast<RICacheElem&>(cache) ;
  riCache.calculate(kFALSE) ;
}



RooArgSet* RooNumRunningInt::actualObservables(const RooArgSet& /*nset*/) const 
{
  RooArgSet* ret = new RooArgSet ;
  ret->add(x.arg()) ;
  return ret ;
}



RooArgSet* RooNumRunningInt::actualParameters(const RooArgSet& /*nset*/) const 
{
  RooArgSet* ret = func.arg().getParameters(RooArgSet()) ;
  ret->remove(x.arg(),kTRUE,kTRUE) ;
  return ret ;
}

RooAbsCachedReal::FuncCacheElem* RooNumRunningInt::createCache(const RooArgSet* nset) const 
{
  return new RICacheElem(*const_cast<RooNumRunningInt*>(this),nset) ; 
}

Double_t RooNumRunningInt::evaluate() const 
{
  cout << "RooNumRunningInt::evaluate(" << GetName() << ")" << endl ;
  return 0 ;
}


