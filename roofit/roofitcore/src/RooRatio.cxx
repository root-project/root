/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   Rahul Balasubramanian, Nikhef, rahulb@nikhef.nl                         *
 *                                                                           *
 * Copyright (c) 2000-2007, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

/**
\file RooRatio.cxx
\class RooRatio
\ingroup Roofitcore

A RooRatio represents the ratio of two given RooAbsReal objects.

**/


#include <memory>

#include "Riostream.h" 
#include "RooRatio.h" 
#include <math.h> 
#include "TMath.h" 
#include "RooMsgService.h"
#include "RooTrace.h"

ClassImp(RooRatio);

RooRatio::RooRatio() 
{
   TRACE_CREATE
}
 
RooRatio::RooRatio(const char *name, const char *title,
                   RooAbsReal& nr,
                   RooAbsReal& dr) :
  RooAbsReal(name,title),
  _numerator("numerator", "numerator", this, nr),
  _denominator("denominator", "denominator", this, dr)
{
    TRACE_CREATE
}

RooRatio::~RooRatio()
{
  TRACE_DESTROY
}

RooRatio::RooRatio(const RooRatio& other, const char* name) :
  RooAbsReal(other, name),
  _numerator("numerator",this, other._numerator),
  _denominator("denominator", this, other._denominator)
{
  TRACE_CREATE
}


Double_t RooRatio::evaluate() const 
{ 

  if(_denominator == 0.0) {
    if(_numerator == 0.0) return std::numeric_limits<double>::quiet_NaN();
    else return (_numerator > 0.0) ? RooNumber::infinity() : -1.0*RooNumber::infinity();
  }
  else
    return _numerator/_denominator;  
} 




