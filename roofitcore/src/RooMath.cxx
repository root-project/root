/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooMath.cc,v 1.3 2001/08/03 02:04:32 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   20-Jun-2000 DK Created initial version from RooProbDens.rdl
 *   18-Jun-2001 WV Imported from RooFitTools
 *
 * Copyright (C) 2000 Stanford University
 *****************************************************************************/

#include "RooFitCore/RooMath.hh"
#include <math.h>
#include <iostream.h>

ClassImp(RooMath)
;

RooComplex RooMath::ComplexErrFunc(Double_t re, Double_t im) {
  return ComplexErrFunc(RooComplex(re,im));
}

RooComplex RooMath::ComplexErrFunc(const RooComplex& Z) {
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

  cout << "RooMath::initFastCERF: Allocating Complex Error Function lookup table" << endl
       << "                       Re: " << _reBins << " bins in range (" << _reMin << "," << _reMax << ")" << endl
       << "                       Im: " << _imBins << " bins in range (" << _imMin << "," << _imMax << ")" << endl
       << "                       Allocation size : " << _reBins*_imBins * 2 * sizeof(Double_t) / 1024 << " kB" << endl
       << "                       Filling table: |..................................................|\r" 
       << "                       Filling table: |" ;

  // Allocate storage matrix for Im(cerf) and Re(cerf) and fill it using ComplexErrFunc()
  Int_t imIdx,reIdx ;
  _reCerfArray = new pDouble_t[_imBins] ;
  _imCerfArray = new pDouble_t[_imBins] ;
  for (imIdx=0 ; imIdx<_imBins ; imIdx++) {
    _reCerfArray[imIdx] = new Double_t[_reBins] ;
    _imCerfArray[imIdx] = new Double_t[_reBins] ;
    if (imIdx % (_imBins/50) ==0) {
      cout << ">" ; cout.flush() ;
    }
    for (reIdx=0 ; reIdx<_reBins ; reIdx++) {
      RooComplex val=ComplexErrFunc(_reMin+reIdx*_reStep,_imMin+imIdx*_imStep) ;
      _reCerfArray[imIdx][reIdx] = val.re();
      _imCerfArray[imIdx][reIdx] = val.im() ;
    }
  }
  cout << endl ;
}




RooComplex RooMath::ITPComplexErrFunc(const RooComplex& z, Int_t nOrder)
{
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
  Int_t imIdx, reIdx ;
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






// Adapted from 'Numerical Recipes, C edition' p90-91, modified for fixed grid
Double_t RooMath::interpolate(Double_t ya[], Int_t n, Double_t x) 
{
  // Int to Double conversion is faster via array lookup than type conversion!
  static Double_t itod[20] = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
			      10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0} ;
  int i,m,ns=1 ;
  Double_t den,dif,dift,ho,hp,w,y,dy ;
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
pDouble_t* RooMath::_reCerfArray(0) ;
pDouble_t* RooMath::_imCerfArray(0) ;


