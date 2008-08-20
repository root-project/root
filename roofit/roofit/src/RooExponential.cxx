/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 * @(#)root/roofit:$Id$
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

//////////////////////////////////////////////////////////////////////////////
//
// BEGIN_HTML
// Exponential p.d.f
// END_HTML
//

#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include <math.h>

#include "RooExponential.h"
#include "RooRealVar.h"

ClassImp(RooExponential)


//_____________________________________________________________________________
RooExponential::RooExponential(const char *name, const char *title,
			       RooAbsReal& _x, RooAbsReal& _c) :
  RooAbsPdf(name, title), 
  x("x","Dependent",this,_x),
  c("c","Exponent",this,_c)
{
}


//_____________________________________________________________________________
RooExponential::RooExponential(const RooExponential& other, const char* name) :
  RooAbsPdf(other, name), x("x",this,other.x), c("c",this,other.c)
{
}


//_____________________________________________________________________________
Double_t RooExponential::evaluate() const{
  //cout << "exp(x=" << x << ",c=" << c << ")=" << exp(c*x) << endl ;
  return exp(c*x);
}


//_____________________________________________________________________________
Int_t RooExponential::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /*rangeName*/) const 
{
  if (matchArgs(allVars,analVars,x)) return 1 ;
  return 0 ;
}


//_____________________________________________________________________________
Double_t RooExponential::analyticalIntegral(Int_t code, const char* rangeName) const 
{
  switch(code) {
  case 1: 
    {
      Double_t ret(0) ;
      if(c == 0.0) {
	ret = (x.max(rangeName) - x.min(rangeName));
      } else {
	ret =  ( exp( c*x.max(rangeName) ) - exp( c*x.min(rangeName) ) )/c;
      }

      //cout << "Int_exp_dx(c=" << c << ", xmin=" << x.min(rangeName) << ", xmax=" << x.max(rangeName) << ")=" << ret << endl ;
      return ret ;
    }
  }
  
  assert(0) ;
  return 0 ;
}

