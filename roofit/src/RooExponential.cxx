/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id$
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

#include "RooFitModels/RooExponential.hh"
#include "RooFitCore/RooRealVar.hh"

ClassImp(RooExponential)

RooExponential::RooExponential(const char *name, const char *title,
			       RooAbsReal& _x, RooAbsReal& _c) :
  RooAbsPdf(name, title), 
  x("x","Dependent",this,_x),
  c("c","Exponent",this,_c)
{
}

RooExponential::RooExponential(const RooExponential& other, const char* name) :
  RooAbsPdf(other, name), x("x",this,other.x), c("c",this,other.c)
{
}

Double_t RooExponential::evaluate() const{
  return exp(c*x);
}

Int_t RooExponential::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const 
{
  if (matchArgs(allVars,analVars,x)) return 1 ;
  return 0 ;
}

Double_t RooExponential::analyticalIntegral(Int_t code) const 
{
  switch(code) {
  case 1: 
    {
      if(c == 0.0) return 0;
      return ( exp( c*x.max() ) - exp( c*x.min() ) )/c;
    }
  }
  
  assert(0) ;
  return 0 ;
}

