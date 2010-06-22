/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
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
// RooAbsIntegrator is the abstract interface for integrators of real-valued
// functions that implement the RooAbsFunc interface.
// END_HTML
//
//

#include "RooFit.h"

#include "Riostream.h"


#include "RooAbsIntegrator.h"
#include "RooAbsIntegrator.h"
#include "RooMsgService.h"
#include "TClass.h"

ClassImp(RooAbsIntegrator)
;


//_____________________________________________________________________________
RooAbsIntegrator::RooAbsIntegrator() : _function(0), _valid(kFALSE), _printEvalCounter(kFALSE) 
{
  // Default constructor
}



//_____________________________________________________________________________
RooAbsIntegrator::RooAbsIntegrator(const RooAbsFunc& function, Bool_t doPrintEvalCounter) :
  _function(&function), _valid(function.isValid()), _printEvalCounter(doPrintEvalCounter)
{
  // Copy constructor
}



//_____________________________________________________________________________
Double_t RooAbsIntegrator::calculate(const Double_t *yvec) 
{
  // Calculate integral value with given array of parameter values

  integrand()->resetNumCall() ;

  integrand()->saveXVec() ;
  Double_t ret = integral(yvec) ; 
  integrand()->restoreXVec() ;
  
  cxcoutD(NumIntegration) << IsA()->GetName() << "::calculate(" << _function->getName() << ") number of function calls = " << integrand()->numCall()<<", result  = "<<ret << endl ;
  return ret ;
}



//_____________________________________________________________________________
Bool_t RooAbsIntegrator::setLimits(Double_t xmin, Double_t xmax) 
{ 
  // Interface to set limits on integration
  return setLimits(&xmin,&xmax) ; 
}
 


//_____________________________________________________________________________
Bool_t RooAbsIntegrator::setUseIntegrandLimits(Bool_t) 
{ 
  // Interface function that allows to defer limit definition to integrand definition
  return kFALSE ; 
} 
