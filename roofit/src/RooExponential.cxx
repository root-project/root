/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooExponential.cc,v 1.3 2001/09/20 01:41:48 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   AB, Adrian Bevan, Liverpool University, bevan@slac.stanford.edu
 * History:
 *   14-Feb-2000 DK Created initial version from RooLifetime
 *   21-Aug-2001 AB Ported to RooFitModels/RooFitCore
 *
 * Copyright (C) 1999 Stanford University
 *****************************************************************************/
#include "BaBar/BaBar.hh"

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

/*
void  RooExponential::useParametersImpl() {
  RooPdf::useParametersImpl();
  if(c != 0) {
    _exp1= exp(_xptr->GetMin()*c);
    _exp2= exp(_xptr->GetMax()*c);
    _norm= (_exp2 - _exp1)/c;
  }
  else {
    _exp1= _xptr->GetMin();
    _exp2= _xptr->GetMax();
    _norm= _exp2 - _exp1;
  }
}

Double_t RooExponential::evaluate() {
  return exp(c*x)/_norm;
}

void RooExponential::initGenerator() {
}

Int_t RooExponential::generateDependents() {
  Double_t r= RooRandom::instance().Rndm();
  x= r*_exp1 + (1-r)*_exp2;
  if(c != 0) x= log(x)/c;
  *_xptr= x;
  return 1;
  }*/
