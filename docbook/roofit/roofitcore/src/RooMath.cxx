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

// -- CLASS DESCRIPTION [MISC] --
// RooMath is a singleton class implementing various mathematical
// functions not found in TMath, mostly involving complex algebra
//
//

#include "RooFit.h"

#include "RooMath.h"
#include "RooMath.h"
#include "TMath.h"
#include <math.h>
#include "Riostream.h"
#include "RooMsgService.h"
#include "RooSentinel.h"
#include "TSystem.h"

ClassImp(RooMath)
;


RooComplex RooMath::FastComplexErrFunc(const RooComplex& z)
{
  return ITPComplexErrFunc(z,z.im()>0?3:4) ;
}
  
Double_t RooMath::FastComplexErrFuncRe(const RooComplex& z) 
{
  return ITPComplexErrFuncRe(z,z.im()>0?3:4) ;
}

Double_t RooMath::FastComplexErrFuncIm(const RooComplex& z) 
{
  return ITPComplexErrFuncIm(z,z.im()>0?3:4) ;
}

void RooMath::cacheCERF(Bool_t flag) 
{ 
  _cacheTable = flag ; 
}


RooComplex RooMath::ComplexErrFunc(Double_t re, Double_t im) {
  // Return CERNlib complex error function for Z(re,im)
  return ComplexErrFunc(RooComplex(re,im));
}

RooComplex RooMath::ComplexErrFunc(const RooComplex& Z) {
  // Return CERNlib complex error function
  //
  // This code is translated from the fortran version in the CERN mathlib.
  // (see ftp://asisftp.cern.ch/cernlib/share/pro/src/mathlib/gen/c/cwerf64.F)

  RooComplex ZH,S,T,V;
  static RooComplex R[38];

  static const Double_t Z1= 1, HF= Z1/2, Z10= 10;
  static const Double_t C1= 74/Z10, C2= 83/Z10, C3= Z10/32, C4 = 16/Z10;
  static const Double_t C = 1.12837916709551257, P = pow(2*C4,33);
  static const RooComplex zero(0);

  Double_t X(Z.re()),Y(Z.im()), XA(fabs(X)), YA(fabs(Y));
  int N;
  if((YA < C1) && (XA < C2)) {
    ZH= RooComplex(YA+C4,XA);
    R[37]=zero;
    N= 36;
    while(N > 0) {
      T=ZH+R[N+1].conj()*N;
      R[N--]=(T*HF)/T.abs2();
    }
    Double_t XL=P;
    S=zero;
    N= 33;
    while(N > 0) {
      XL=C3*XL;
      S=R[N--]*(S+XL);
    }
    V=S*C;
  }
  else {
    ZH=RooComplex(YA,XA);
    R[1]=zero;
    N= 9;
    while(N > 0) {
      T=ZH+R[1].conj()*N;
      R[1]=(T*HF)/T.abs2();
      N--;
    }
    V=R[1]*C;
  }
  if(YA==0) V=RooComplex(exp(-(XA*XA)),V.im());

  if(Y < 0) {
    RooComplex tmp(XA,YA);
    tmp= -tmp*tmp;
    V=tmp.exp()*2-V;
    if(X > 0) V= V.conj();
  }
  else {
    if(X < 0) V= V.conj();
  }
  return V;
}



void RooMath::initFastCERF(Int_t reBins, Double_t reMin, Double_t reMax, Int_t imBins, Double_t imMin, Double_t imMax) 
{
  // Allocate and initialize lookup table for interpolated complex error function
  // for given grid parameters

  // Store grid dimensions and related parameters
  _reBins = reBins ;
  _imBins = imBins ;
  _reMin = reMin ;
  _reMax = reMax ;
  _reRange = _reMax-_reMin ;
  _reStep  = _reRange/_reBins ;

  _imMin = imMin ;
  _imMax = imMax ;
  _imRange = _imMax-_imMin ;
  _imStep = _imRange/_imBins ;

  oocxcoutD((TObject*)0,Eval) << endl
			      << "RooMath::initFastCERF: Allocating Complex Error Function lookup table" << endl
			      << "                       Re: " << _reBins << " bins in range (" << _reMin << "," << _reMax << ")" << endl
			      << "                       Im: " << _imBins << " bins in range (" << _imMin << "," << _imMax << ")" << endl
			      << "                       Allocation size : " << _reBins*_imBins * 2 * sizeof(Double_t) / 1024 << " kB" << endl ;


  // Allocate storage matrix for Im(cerf) and Re(cerf) and fill it using ComplexErrFunc()
  RooSentinel::activate() ;
  Int_t imIdx,reIdx ;
  _reCerfArray = new pDouble_t[_imBins] ;
  _imCerfArray = new pDouble_t[_imBins] ;
  for (imIdx=0 ; imIdx<_imBins ; imIdx++) {
    _reCerfArray[imIdx] = new Double_t[_reBins] ;
    _imCerfArray[imIdx] = new Double_t[_reBins] ;
  }
  
  Bool_t cacheLoaded(kFALSE) ;
  if (!_cacheTable || !(cacheLoaded=loadCache())) {
    
    oocxcoutD((TObject*)0,Eval) << endl
				<< "                       Filling table: |..................................................|\r" 
				<< "                       Filling table: |" ;
    
    // Allocate storage matrix for Im(cerf) and Re(cerf) and fill it using ComplexErrFunc()
    for (imIdx=0 ; imIdx<_imBins ; imIdx++) {
      if (imIdx % (_imBins/50) ==0) {
	ooccxcoutD((TObject*)0,Eval) << ">" ; cout.flush() ;
      }
      for (reIdx=0 ; reIdx<_reBins ; reIdx++) {
	RooComplex val=ComplexErrFunc(_reMin+reIdx*_reStep,_imMin+imIdx*_imStep) ;
	_reCerfArray[imIdx][reIdx] = val.re();
	_imCerfArray[imIdx][reIdx] = val.im() ;
      }
    }
    ooccoutI((TObject*)0,Eval) << endl ;
  } 

  if (_cacheTable && !cacheLoaded) storeCache() ;
}


void RooMath::cleanup()
{
  if (_reCerfArray) {
    for (Int_t imIdx=0 ; imIdx<_imBins ; imIdx++) {
      delete[] _reCerfArray[imIdx] ;
      delete[] _imCerfArray[imIdx] ;
    }
    delete[] _reCerfArray ;
    delete[] _imCerfArray ;
    _reCerfArray = 0 ;
    _imCerfArray = 0 ;
  }
  
}


RooComplex RooMath::ITPComplexErrFunc(const RooComplex& z, Int_t nOrder)
{
  // Return complex error function interpolated from lookup tabel created
  // by initFastCERF(). Interpolation is performed in Im and Re plane
  // to specified order.

  // Initialize grid
  if (!_reCerfArray) initFastCERF() ;

  // Located nearest grid point
  Double_t imPrime = (z.im()-_imMin) / _imStep ;
  Double_t rePrime = (z.re()-_reMin) / _reStep ;

  // Calculate corners of nOrder X nOrder grid box
  Int_t imIdxLo = Int_t(imPrime - 1.0*nOrder/2 + 0.5) ;
  Int_t imIdxHi = imIdxLo+nOrder-1 ;
  Int_t reIdxLo = Int_t(rePrime - 1.0*nOrder/2 + 0.5) ;
  Int_t reIdxHi = reIdxLo+nOrder-1 ;

  // Check if the box is fully contained in the grid
  if (imIdxLo<0 || imIdxHi>_imBins-1 || reIdxLo<0 || reIdxHi>_reBins-1) {
    return ComplexErrFunc(z) ;
  }

  // Allocate temporary array space for interpolation
  Int_t imIdx /*, reIdx*/ ;
  Double_t imYR[10] ;
  Double_t imYI[10] ;

  // Loop over imaginary grid points
  for (imIdx=imIdxLo ; imIdx<=imIdxHi ; imIdx++) {
    // Interpolate real array and store as array point for imaginary interpolation
    imYR[imIdx-imIdxLo] = interpolate(&_reCerfArray[imIdx][reIdxLo],nOrder,rePrime-reIdxLo) ;
    imYI[imIdx-imIdxLo] = interpolate(&_imCerfArray[imIdx][reIdxLo],nOrder,rePrime-reIdxLo) ;
  }
  // Interpolate imaginary arrays and construct complex return value
  Double_t re = interpolate(imYR,nOrder,imPrime-imIdxLo) ;
  Double_t im = interpolate(imYI,nOrder,imPrime-imIdxLo) ;
  return RooComplex(re,im) ;
}





Double_t RooMath::ITPComplexErrFuncRe(const RooComplex& z, Int_t nOrder)
{
  // Return real component of complex error function interpolated from 
  // lookup table created by initFastCERF(). Interpolation is performed in 
  // Im and Re plane to specified order. This functions is noticably faster
  // than ITPComplexErrrFunc().re() because only the real lookup table
  // is interpolated

  // Initialize grid
  if (!_reCerfArray) initFastCERF() ;

  // Located nearest grid point
  Double_t imPrime = (z.im()-_imMin) / _imStep ;
  Double_t rePrime = (z.re()-_reMin) / _reStep ;

  // Calculate corners of nOrder X nOrder grid box
  Int_t imIdxLo = Int_t(imPrime - 1.0*nOrder/2 + 0.5) ;
  Int_t imIdxHi = imIdxLo+nOrder-1 ;
  Int_t reIdxLo = Int_t(rePrime - 1.0*nOrder/2 + 0.5) ;
  Int_t reIdxHi = reIdxLo+nOrder-1 ;
  
  // Check if the box is fully contained in the grid
  if (imIdxLo<0 || imIdxHi>_imBins-1 || reIdxLo<0 || reIdxHi>_reBins-1) {
    //cout << "RooMath::ITPComplexErrFuncRe: (" << z.re() << "," << z.im() << ") outside interpolation grid" << endl ;
    return ComplexErrFunc(z).re() ;
  }

  if (nOrder==1) return _reCerfArray[imIdxLo][reIdxLo] ;

  Int_t imIdx ;
  Double_t imYR[10] ;

  // Allocate temporary array space for interpolation
  for (imIdx=imIdxLo ; imIdx<=imIdxHi ; imIdx++) {
    // Interpolate real array and store as array point for imaginary interpolation
    imYR[imIdx-imIdxLo] = interpolate(&_reCerfArray[imIdx][reIdxLo],nOrder,rePrime-reIdxLo) ;
  }
  // Interpolate imaginary arrays and construct complex return value
  return interpolate(imYR,nOrder,imPrime-imIdxLo) ;
}



Double_t RooMath::ITPComplexErrFuncIm(const RooComplex& z, Int_t nOrder)
{
  // Return real component of complex error function interpolated from 
  // lookup table created by initFastCERF(). Interpolation is performed in 
  // Im and Re plane to specified order. This functions is noticably faster
  // than ITPComplexErrrFunc().im() because only the imaginary lookup table
  // is interpolated

  // Initialize grid
  if (!_reCerfArray) initFastCERF() ;

  // Located nearest grid point
  Double_t imPrime = (z.im()-_imMin) / _imStep ;
  Double_t rePrime = (z.re()-_reMin) / _reStep ;

  // Calculate corners of nOrder X nOrder grid box
  Int_t imIdxLo = Int_t(imPrime - 1.0*nOrder/2 + 0.5) ;
  Int_t imIdxHi = imIdxLo+nOrder-1 ;
  Int_t reIdxLo = Int_t(rePrime - 1.0*nOrder/2 + 0.5) ;
  Int_t reIdxHi = reIdxLo+nOrder-1 ;

  // Check if the box is fully contained in the grid
  if (imIdxLo<0 || imIdxHi>_imBins-1 || reIdxLo<0 || reIdxHi>_reBins-1) {
    return ComplexErrFunc(z).im() ;
  }

  // Allocate temporary array space for interpolation
  Int_t imIdx ;
  Double_t imYI[10] ;

  // Loop over imaginary grid points
  for (imIdx=imIdxLo ; imIdx<=imIdxHi ; imIdx++) {
    // Interpolate real array and store as array point for imaginary interpolation
    imYI[imIdx-imIdxLo] = interpolate(&_imCerfArray[imIdx][reIdxLo],nOrder,rePrime-reIdxLo) ;
  }
  // Interpolate imaginary arrays and construct complex return value
  return interpolate(imYI,nOrder,imPrime-imIdxLo) ;
}






Double_t RooMath::interpolate(Double_t ya[], Int_t n, Double_t x) 
{
  // Interpolate array 'ya' with 'n' elements for 'x' (between 0 and 'n'-1)

  // Int to Double conversion is faster via array lookup than type conversion!
  static Double_t itod[20] = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
			      10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0} ;
  int i,m,ns=1 ;
  Double_t den,dif,dift/*,ho,hp,w*/,y,dy ;
  Double_t c[20], d[20] ;

  dif = fabs(x) ;
  for(i=1 ; i<=n ; i++) {
    if ((dift=fabs(x-itod[i-1]))<dif) {
      ns=i ;
      dif=dift ;
    }
    c[i] = ya[i-1] ;
    d[i] = ya[i-1] ;
  }
  
  y=ya[--ns] ;
  for(m=1 ; m<n; m++) {       
    for(i=1 ; i<=n-m ; i++) { 
      den=(c[i+1] - d[i])/itod[m] ;
      d[i]=(x-itod[i+m-1])*den ;
      c[i]=(x-itod[i-1])*den ;
    }
    dy = (2*ns)<(n-m) ? c[ns+1] : d[ns--] ;
    y += dy ;
  }
  return y ;
}



Double_t RooMath::interpolate(Double_t xa[], Double_t ya[], Int_t n, Double_t x) 
{
  // Interpolate array 'ya' with 'n' elements for 'xa' 

  // Int to Double conversion is faster via array lookup than type conversion!
//   static Double_t itod[20] = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
// 			      10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0} ;
  int i,m,ns=1 ;
  Double_t den,dif,dift,ho,hp,w,y,dy ;
  Double_t c[20], d[20] ;

  dif = fabs(x-xa[0]) ;
  for(i=1 ; i<=n ; i++) {
    if ((dift=fabs(x-xa[i-1]))<dif) {
      ns=i ;
      dif=dift ;
    }
    c[i] = ya[i-1] ;
    d[i] = ya[i-1] ;
  }
  
  y=ya[--ns] ;
  for(m=1 ; m<n; m++) {       
    for(i=1 ; i<=n-m ; i++) { 
      ho=xa[i-1]-x ;
      hp=xa[i-1+m]-x ;
      w=c[i+1]-d[i] ;
      den=ho-hp ;
      if (den==0.) {
	oocoutE((TObject*)0,Eval) << "RooMath::interpolate ERROR: zero distance between points not allowed" << endl ;
	return 0 ;
      }
      den = w/den ;
      d[i]=hp*den ;
      c[i]=ho*den;
    }
    dy = (2*ns)<(n-m) ? c[ns+1] : d[ns--] ;
    y += dy ;
  }
  return y ;
}



Bool_t RooMath::loadCache() 
{
  // Load the complex error function lookup table from the cache file

  const char* fName = cacheFileName() ;
  // Open cache file
  ifstream ifs(fName) ;

  // Return immediately if file doesn't exist
  if (ifs.fail()) return kFALSE ;

  oocxcoutD((TObject*)0,Eval) << endl << "                       Filling table from cache file " << fName << endl ;

  // Load data in memory arrays
  Bool_t ok(kTRUE) ;
  Int_t i ;
  for (i=0 ; i<_imBins ; i++) {
    ifs.read((char*)_imCerfArray[i],_reBins*sizeof(Double_t)) ;
    if (ifs.fail()) ok=kFALSE ;
    ifs.read((char*)_reCerfArray[i],_reBins*sizeof(Double_t)) ;
    if (ifs.fail()) ok=kFALSE ;
  }  

  // Issue error message on partial read failure
  if (!ok) {
    oocoutE((TObject*)0,Eval) << "RooMath::loadCERFCache: error reading file " << cacheFileName() << endl ;
  }
  return ok ;
}


void RooMath::storeCache()
{
  // Store the complex error function lookup table in the cache file

  ofstream ofs(cacheFileName()) ;
  
  oocxcoutI((TObject*)0,Eval) << endl << "                       Writing table to cache file " << cacheFileName() << endl ;
  Int_t i ;
  for (i=0 ; i<_imBins ; i++) {
    ofs.write((char*)_imCerfArray[i],_reBins*sizeof(Double_t)) ;
    ofs.write((char*)_reCerfArray[i],_reBins*sizeof(Double_t)) ;
  }
}



const char* RooMath::cacheFileName() 
{
  // Construct and return the name of the complex error function cache file
  static char fileName[1024] ;  
  snprintf(fileName,1024,"/tmp/RooMath_CERFcache_R%04d_I%04d_%d.dat",_reBins,_imBins,gSystem->GetUid()) ;
  return fileName ;
}


Double_t RooMath::erf(Double_t x) 
{
  return TMath::Erf(x) ;
}

Double_t RooMath::erfc(Double_t x)
{
  return TMath::Erfc(x) ;
}



// Instantiation of static members
Int_t RooMath::_reBins(0) ;
Int_t RooMath::_imBins(0) ;
Double_t RooMath::_reMin(0) ;
Double_t RooMath::_reMax(0) ;
Double_t RooMath::_reRange(0) ;
Double_t RooMath::_reStep(0) ;
Double_t RooMath::_imMin(0) ;
Double_t RooMath::_imMax(0) ;
Double_t RooMath::_imRange(0) ;
Double_t RooMath::_imStep(0) ;
pDouble_t* RooMath::_reCerfArray = 0;
pDouble_t* RooMath::_imCerfArray = 0;
Bool_t RooMath::_cacheTable(kTRUE) ;

