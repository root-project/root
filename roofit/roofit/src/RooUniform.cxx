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
// Flat p.d.f. in N dimensions
// END_HTML
//

#include "RooFit.h"

#include "Riostream.h"
#include <math.h>

#include "RooUniform.h"
#include "RooAbsReal.h"
#include "RooRealVar.h"
#include "RooRandom.h"
#include "RooMath.h"
#include "RooArgSet.h"

using namespace std;

ClassImp(RooUniform)


//_____________________________________________________________________________
RooUniform::RooUniform(const char *name, const char *title, const RooArgSet& _x) :
  RooAbsPdf(name,title),
  x("x","Observables",this,kTRUE,kFALSE)
{
  x.add(_x) ;
}



//_____________________________________________________________________________
RooUniform::RooUniform(const RooUniform& other, const char* name) : 
  RooAbsPdf(other,name), x("x",this,other.x)
{
}



//_____________________________________________________________________________
Double_t RooUniform::evaluate() const
{
  return 1 ;
}



//_____________________________________________________________________________
Int_t RooUniform::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /*rangeName*/) const 
{
  // Advertise analytical integral

  Int_t nx = x.getSize() ;
  if (nx>31) {
    // Warn that analytical integration is only provided for the first 31 observables
    coutW(Integration) << "RooUniform::getAnalyticalIntegral(" << GetName() << ") WARNING: p.d.f. has " << x.getSize() 
		       << " observables, analytical integration is only implemented for the first 31 observables" << endl ;
    nx=31 ;
  }

  Int_t code(0) ;
  for (int i=0 ; i<x.getSize() ; i++) {
    if (allVars.find(x.at(i)->GetName())) {
      code |= (1<<i) ;
      analVars.add(*allVars.find(x.at(i)->GetName())) ;
    }
  }    
  return code ;
}



//_____________________________________________________________________________
Double_t RooUniform::analyticalIntegral(Int_t code, const char* rangeName) const 
{
  // Implement analytical integral
  Double_t ret(1) ;
  for (int i=0 ; i<32 ; i++) {
    if (code&(1<<i)) {
      RooAbsRealLValue* var = (RooAbsRealLValue*)x.at(i) ;
      ret *= (var->getMax(rangeName) - var->getMin(rangeName)) ;
    }    
  }
  return ret ;
}




//_____________________________________________________________________________
Int_t RooUniform::getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, Bool_t /*staticInitOK*/) const
{
  // Advertise internal generator 

  Int_t nx = x.getSize() ;
  if (nx>31) {
    // Warn that analytical integration is only provided for the first 31 observables
    coutW(Integration) << "RooUniform::getGenerator(" << GetName() << ") WARNING: p.d.f. has " << x.getSize() 
		       << " observables, internal integrator is only implemented for the first 31 observables" << endl ;
    nx=31 ;
  }
  
  Int_t code(0) ;
  for (int i=0 ; i<x.getSize() ; i++) {
    if (directVars.find(x.at(i)->GetName())) {
      code |= (1<<i) ;
      generateVars.add(*directVars.find(x.at(i)->GetName())) ;
    }
  }    
  return code ;
  return 0 ;
}



//_____________________________________________________________________________
void RooUniform::generateEvent(Int_t code)
{
  // Implement internal generator

  // Fast-track handling of one-observable case
  if (code==1) {
    ((RooAbsRealLValue*)x.at(0))->randomize() ;
    return ;
  }

  for (int i=0 ; i<32 ; i++) {
    if (code&(1<<i)) {
      RooAbsRealLValue* var = (RooAbsRealLValue*)x.at(i) ;
      var->randomize() ;
    }    
  }
}


