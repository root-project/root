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
// Implementation of BlindTools' precision blinding method
// A RooUnblindPrecision object is a real valued function
// object, constructed from a blind value holder and a 
// set of unblinding parameters. When supplied to a PDF
// in lieu of a regular parameter, the blind value holder
// supplied to the unblinder objects will in a fit be minimized 
// to blind value corresponding to the actual minimum of the
// parameter. The transformation is chosen such that the
// the error on the blind parameters is indentical to that
// of the unblind parameter
// END_HTML
//

#include "RooFit.h"

#include "RooArgSet.h"
#include "RooArgSet.h"
#include "RooUnblindPrecision.h"


ClassImp(RooUnblindPrecision)
;



//_____________________________________________________________________________
RooUnblindPrecision::RooUnblindPrecision() : _blindEngine("") 
{
  // Default constructor
}



//_____________________________________________________________________________
RooUnblindPrecision::RooUnblindPrecision(const char *name, const char *title,
					 const char *blindString, Double_t centralValue, 
					 Double_t scale, RooAbsReal& value,
					 Bool_t sin2betaMode)
  : RooAbsHiddenReal(name,title), 
  _value("value","Precision blinded value",this,value),
  _blindEngine(blindString,RooBlindTools::full,centralValue,scale,sin2betaMode)
{  
  // Constructor from a given RooAbsReal (to hold the blind value) and a set of blinding parameters
}



//_____________________________________________________________________________
RooUnblindPrecision::RooUnblindPrecision(const char *name, const char *title,
					 const char *blindString, Double_t centralValue, 
					 Double_t scale, RooAbsReal& value, RooAbsCategory& blindState,
					 Bool_t sin2betaMode)
  : RooAbsHiddenReal(name,title,blindState), 
  _value("value","Precision blinded value",this,value),
  _blindEngine(blindString,RooBlindTools::full,centralValue,scale,sin2betaMode) 
{  
  // Constructor from a given RooAbsReal (to hold the blind value) and a set of blinding parameters
}



//_____________________________________________________________________________
RooUnblindPrecision::RooUnblindPrecision(const RooUnblindPrecision& other, const char* name) : 
  RooAbsHiddenReal(other, name), 
  _value("asym",this,other._value),
  _blindEngine(other._blindEngine) 
{
  // Copy constructor
}



//_____________________________________________________________________________
RooUnblindPrecision::~RooUnblindPrecision() 
{
  // Destructor
}



//_____________________________________________________________________________
Double_t RooUnblindPrecision::evaluate() const
{
  // Evaluate RooBlindTools unhide-precision method on blind value

  if (isHidden()) {
    // Blinding active for this event
    return _blindEngine.UnHidePrecision(_value);
  } else {
    // Blinding not active for this event
    return _value ;
  }
}




