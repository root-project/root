/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Name:  $:$Id: RooAbsIntegrator.cxx,v 1.20 2007/05/11 09:11:58 verkerke Exp $
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

// -- CLASS DESCRIPTION [AUX] --
// RooAbsIntegrator is the abstract interface for integrating real-valued
// functions that implement the RooAbsFunc interface.

#include "RooFit.h"

#include "RooAbsIntegrator.h"
#include "RooAbsIntegrator.h"
#include "TClass.h"

ClassImp(RooAbsIntegrator)
;

RooAbsIntegrator::RooAbsIntegrator() : _function(0), _valid(kFALSE), _printEvalCounter(kFALSE) 
{
}

RooAbsIntegrator::RooAbsIntegrator(const RooAbsFunc& function, Bool_t printEvalCounter) :
  _function(&function), _valid(function.isValid()), _printEvalCounter(printEvalCounter)
{
}

Double_t RooAbsIntegrator::calculate(const Double_t *yvec) 
{
  if (_printEvalCounter) integrand()->resetNumCall() ;
  Double_t ret = integral(yvec) ; 
  if (_printEvalCounter) {
    cout << IsA()->GetName() << "::calculate() number of function calls = " << integrand()->numCall() << endl ;
  }
  return ret ;
}

Bool_t RooAbsIntegrator::setLimits(Double_t, Double_t) 
{ 
  return kFALSE ; 
}
 
Bool_t RooAbsIntegrator::setUseIntegrandLimits(Bool_t) { 
  return kFALSE ; 
} 
