/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooUnblindPrecision.cc,v 1.2 2002/01/16 01:35:54 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [REAL] --
//
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

#include "RooFitCore/RooArgSet.hh"
#include "RooFitModels/RooUnblindPrecision.hh"


ClassImp(RooUnblindPrecision)
;


RooUnblindPrecision::RooUnblindPrecision() : _blindEngine("") 
{
  // Default constructor
}


RooUnblindPrecision::RooUnblindPrecision(const char *name, const char *title,
					 const char *blindString, Double_t centralValue, 
					 Double_t scale, RooAbsReal& value,
					 Bool_t sin2betaMode)
  : RooAbsHiddenReal(name,title), 
  _blindEngine(blindString,RooBlindTools::full,centralValue,scale,sin2betaMode), 
  _value("value","Precision blinded value",this,value)
{  
  // Constructor from a given RooAbsReal (to hold the blind value) and a set of blinding parameters
}


RooUnblindPrecision::RooUnblindPrecision(const char *name, const char *title,
					 const char *blindString, Double_t centralValue, 
					 Double_t scale, RooAbsReal& value, RooAbsCategory& blindState,
					 Bool_t sin2betaMode)
  : RooAbsHiddenReal(name,title,blindState), 
  _blindEngine(blindString,RooBlindTools::full,centralValue,scale,sin2betaMode), 
  _value("value","Precision blinded value",this,value)
{  
  // Constructor from a given RooAbsReal (to hold the blind value) and a set of blinding parameters
}


RooUnblindPrecision::RooUnblindPrecision(const RooUnblindPrecision& other, const char* name) : 
  RooAbsHiddenReal(other, name), 
  _blindEngine(other._blindEngine), 
  _value("asym",this,other._value)
{
  // Copy constructor
}


RooUnblindPrecision::~RooUnblindPrecision() 
{
  // Destructor
}


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




