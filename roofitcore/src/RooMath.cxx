/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
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

