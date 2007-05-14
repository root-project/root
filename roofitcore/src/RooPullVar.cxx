/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Name:  $:$Id: RooPullVar.cxx,v 1.5 2007/05/11 09:11:58 verkerke Exp $
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

// -- CLASS DESCRIPTION [REAL] --
//
// RooPullVar calculates the pull of measurement w.r.t to true value
// using the measurement value and its error. If an asymmetric error
// is defined on a given measurement the proper side of that asymmetric
// error will be used

#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include <math.h>

#include "RooPullVar.h"
#include "RooAbsReal.h"
#include "RooRealVar.h"

ClassImp(RooPullVar)
;

RooPullVar::RooPullVar()
{
}


RooPullVar::RooPullVar(const char* name, const char* title, RooRealVar& meas, RooAbsReal& truth) :
  RooAbsReal(name, title),
  _meas("meas","Measurement",this,meas),
  _true("true","Truth",this,truth)
{
}





RooPullVar::RooPullVar(const RooPullVar& other, const char* name) :
  RooAbsReal(other, name), 
  _meas("meas",this,other._meas),
  _true("true",this,other._true)
{
}


RooPullVar::~RooPullVar() 
{
}



Double_t RooPullVar::evaluate() const 
{
  const RooRealVar& meas = static_cast<const RooRealVar&>(_meas.arg()) ;  
  if (meas.hasAsymError()) {
    Double_t delta = _meas-_true ;
    if (delta<0) {
      return delta/meas.getAsymErrorHi() ;
    } else {
      return -delta/meas.getAsymErrorLo() ;
    }
  } else if (meas.hasError()) {
    return (_meas-_true)/meas.getError() ;    
  } else {
    return 0 ;
  }
}


