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

/** \class RooUniform
    \ingroup Roofit

Flat p.d.f. in N dimensions
**/

#include "RooArgSet.h"
#include "RooRealVar.h"
#include "RooUniform.h"


ClassImp(RooUniform);

////////////////////////////////////////////////////////////////////////////////

RooUniform::RooUniform(const char *name, const char *title, const RooArgSet& _x) :
  RooAbsPdf(name,title),
  x("x","Observables",this,true,false)
{
  x.add(_x) ;
}

////////////////////////////////////////////////////////////////////////////////

RooUniform::RooUniform(const RooUniform& other, const char* name) :
  RooAbsPdf(other,name), x("x",this,other.x)
{
}

////////////////////////////////////////////////////////////////////////////////

double RooUniform::evaluate() const
{
  return 1 ;
}

////////////////////////////////////////////////////////////////////////////////
/// Advertise analytical integral

Int_t RooUniform::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /*rangeName*/) const
{
  Int_t nx = x.size() ;
  if (nx>31) {
    // Warn that analytical integration is only provided for the first 31 observables
    coutW(Integration) << "RooUniform::getAnalyticalIntegral(" << GetName() << ") WARNING: p.d.f. has " << x.size()
             << " observables, analytical integration is only implemented for the first 31 observables" << std::endl ;
    nx=31 ;
  }

  Int_t code(0) ;
  for (std::size_t i=0 ; i<x.size() ; i++) {
    if (allVars.find(x.at(i)->GetName())) {
      code |= (1<<i) ;
      analVars.add(*allVars.find(x.at(i)->GetName())) ;
    }
  }
  return code ;
}

////////////////////////////////////////////////////////////////////////////////
/// Implement analytical integral

double RooUniform::analyticalIntegral(Int_t code, const char* rangeName) const
{
  double ret(1) ;
  for (int i=0 ; i<32 ; i++) {
    if (code&(1<<i)) {
      RooAbsRealLValue* var = static_cast<RooAbsRealLValue*>(x.at(i)) ;
      ret *= (var->getMax(rangeName) - var->getMin(rangeName)) ;
    }
  }
  return ret ;
}

////////////////////////////////////////////////////////////////////////////////
/// Advertise internal generator

Int_t RooUniform::getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, bool /*staticInitOK*/) const
{
  Int_t nx = x.size() ;
  if (nx>31) {
    // Warn that analytical integration is only provided for the first 31 observables
    coutW(Integration) << "RooUniform::getGenerator(" << GetName() << ") WARNING: p.d.f. has " << x.size()
             << " observables, internal integrator is only implemented for the first 31 observables" << std::endl ;
    nx=31 ;
  }

  Int_t code(0) ;
  for (std::size_t i=0 ; i<x.size() ; i++) {
    if (directVars.find(x.at(i)->GetName())) {
      code |= (1<<i) ;
      generateVars.add(*directVars.find(x.at(i)->GetName())) ;
    }
  }
  return code ;
}

////////////////////////////////////////////////////////////////////////////////
/// Implement internal generator

void RooUniform::generateEvent(Int_t code)
{
  // Fast-track handling of one-observable case
  if (code==1) {
    (static_cast<RooAbsRealLValue*>(x.at(0)))->randomize() ;
    return ;
  }

  for (int i=0 ; i<32 ; i++) {
    if (code&(1<<i)) {
      RooAbsRealLValue* var = static_cast<RooAbsRealLValue*>(x.at(i)) ;
      var->randomize() ;
    }
  }
}
