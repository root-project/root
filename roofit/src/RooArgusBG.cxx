/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id$
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   05-Jan-2000 DK Created initial version from RooArgusBGProb
 *   02-May-2001 WV Port to RooFitModels/RooFitCore
 *
 * Copyright (C) 1999 Stanford University
 *****************************************************************************/
#include "BaBar/BaBar.hh"

#include <iostream.h>
#include <math.h>

#include "RooFitModels/RooArgusBG.hh"
#include "RooFitCore/RooRealVar.hh"

ClassImp(RooArgusBG)

RooArgusBG::RooArgusBG(const char *name, const char *title,
		       RooAbsReal& _m, RooAbsReal& _m0, RooAbsReal& _c) :
  RooAbsPdf(name, title), 
  m("m","Mass",this,_m),
  m0("m0","Resonance mass",this,_m0),
  c("c","Slope parameter",this,_c)
{
}

RooArgusBG::RooArgusBG(const RooArgusBG& other, const char* name) :
  RooAbsPdf(other,name), m("m",this,other.m), m0("m0",this,other.m0), c("c",this,other.c)
{
}


Double_t RooArgusBG::evaluate() const {
  Double_t t= m/m0;
  if(t >= 1) return 0;

  Double_t u= 1 - t*t;
  return m*sqrt(u)*exp(c*u) ;
}



// void RooArgusBG::initGenerator() {
//   // calculate our value at m(min)
//   Double_t tmin= _mmin/m0;
//   Double_t umin= 1 - tmin*tmin;
//   _maxProb= _mmin*sqrt(umin)*exp(c*umin)/_norm;
//   // check the roots of this PDF to see if there are any local maxima
//   // within our range of m values
//   if(c != 0) {
//     Double_t tmp1= 1+c;
//     Double_t tmp2= sqrt(1+c*c);
//     Double_t t1= (tmp1+tmp2)/(2*c);
//     Double_t t2= (tmp1-tmp2)/(2*c);
//     if(t1 > 0) {
//       Double_t u1= 1 - t1;
//       t1= m0*sqrt(t1);
//       if(t1 > _mmin && t1 < _mmax) {
// 	Double_t prob1= t1*sqrt(u1)*exp(c*u1)/_norm;
// 	if(prob1 > _maxProb) _maxProb= prob1;
//       }
//     }
//     if(t2 > 0) {
//       Double_t u2= 1 - t2;
//       t2= m0*sqrt(t2);
//       if(t2 > _mmin && t2 < _mmax) {
// 	Double_t prob2= t2*sqrt(u2)*exp(c*u2)/_norm;
// 	if(prob2 > _maxProb) _maxProb= prob2;
//       }
//     }
//   }
//   else {
//     Double_t u1= 0.5;
//     Double_t t1= m0/sqrt(2);
//     if(t1 > _mmin && t1 < _mmax) {
//       Double_t prob1= t1*sqrt(u1)*exp(c*u1)/_norm;
//       if(prob1 > _maxProb) _maxProb= prob1;
//     }
//   }
// }
