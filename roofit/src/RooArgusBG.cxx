/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooArgusBG.cc,v 1.8 2002/09/10 02:01:30 verkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2002, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

// -- CLASS DESCRIPTION [PDF] --

//#include "BaBar/BaBar.hh"
#include <iostream.h>
#include <math.h>

#include "RooFitModels/RooArgusBG.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooRealConstant.hh"

ClassImp(RooArgusBG)

RooArgusBG::RooArgusBG(const char *name, const char *title,
		       RooAbsReal& _m, RooAbsReal& _m0, RooAbsReal& _c) :
  RooAbsPdf(name, title), 
  m("m","Mass",this,_m),
  m0("m0","Resonance mass",this,_m0),
  c("c","Slope parameter",this,_c),
  p("p","Power",this,(RooRealVar&)RooRealConstant::value(0.5))
{
}

RooArgusBG::RooArgusBG(const char *name, const char *title,
		       RooAbsReal& _m, RooAbsReal& _m0, RooAbsReal& _c, RooAbsReal& _p) :
  RooAbsPdf(name, title), 
  m("m","Mass",this,_m),
  m0("m0","Resonance mass",this,_m0),
  c("c","Slope parameter",this,_c),
  p("p","Power",this,_p)
{
}

RooArgusBG::RooArgusBG(const RooArgusBG& other, const char* name) :
  RooAbsPdf(other,name), 
  m("m",this,other.m), 
  m0("m0",this,other.m0), 
  c("c",this,other.c),
  p("p",this,other.p)
{
}


Double_t RooArgusBG::evaluate() const {
  Double_t t= m/m0;
  if(t >= 1) return 0;

  Double_t u= 1 - t*t;
  return m*pow(u,p)*exp(c*u) ;
}


Int_t RooArgusBG::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const
{
  if (p.arg().isConstant()) {
    // We can integrate over m if power = 0.5
    if (matchArgs(allVars,analVars,m) && p == 0.5) return 1;
  }
  return 0;

}


Double_t RooArgusBG::analyticalIntegral(Int_t code) const
{
  assert(code==1);
  // Formula for integration over m when p=0.5
  static const Double_t pi = atan2(0.0,-1.0);
  Double_t min = (m.min() < m0) ? m.min() : m0;
  Double_t max = (m.max() < m0) ? m.max() : m0;
  Double_t f1 = (1.-pow(min/m0,2));
  Double_t f2 = (1.-pow(max/m0,2));
  Double_t aLow  = -0.5*m0*m0*(exp(c*f1)*sqrt(f1)/c + 0.5/pow(-c,1.5)*sqrt(pi)*erf(sqrt(-c*f1)));
  Double_t aHigh = -0.5*m0*m0*(exp(c*f2)*sqrt(f2)/c + 0.5/pow(-c,1.5)*sqrt(pi)*erf(sqrt(-c*f2)));
  Double_t area = aHigh - aLow;
  return area;

}


