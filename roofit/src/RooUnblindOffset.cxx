/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooUnblindOffset.cc,v 1.1 2001/11/20 04:00:55 verkerke Exp $
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
// Implementation of BlindTools' offset blinding method
// A RooUnblindOffset object is a real valued function
// object, constructed from a blind value holder and a 
// set of unblinding parameters. When supplied to a PDF
// in lieu of a regular parameter, the blind value holder
// supplied to the unblinder objects will in a fit be minimized 
// to blind value corresponding to the actual minimum of the
// parameter. The transformation is chosen such that the
// the error on the blind parameters is indentical to that
// of the unblind parameter

#include "RooFitCore/RooArgSet.hh"
#include "RooFitModels/RooUnblindOffset.hh"


ClassImp(RooUnblindOffset)
;


RooUnblindOffset::RooUnblindOffset() : _blindEngine("") 
{
  // Default constructor
}


RooUnblindOffset::RooUnblindOffset(const char *name, const char *title,
					 const char *blindString, Double_t scale, RooAbsReal& cpasym)
  : RooAbsHiddenReal(name,title), _blindEngine(blindString,RooBlindTools::full,0.,scale), _value("value","Offset blinded value",this,cpasym) 
{  
  // Constructor from a given RooAbsReal (to hold the blind value) and a set of blinding parameters
}

RooUnblindOffset::RooUnblindOffset(const char *name, const char *title,
				   const char *blindString, Double_t scale, RooAbsReal& cpasym,
				   RooAbsCategory& blindState)
  : RooAbsHiddenReal(name,title,blindState),
    _blindEngine(blindString,RooBlindTools::full,0.,scale),
    _value("value","Offset blinded value",this,cpasym) 
{  
  // Constructor from a given RooAbsReal (to hold the blind value) and a set of blinding parameters
}


RooUnblindOffset::RooUnblindOffset(const RooUnblindOffset& other, const char* name) : 
  RooAbsHiddenReal(other, name), _blindEngine(other._blindEngine), _value("asym",this,other._value)
{
  // Copy constructor

}


RooUnblindOffset::~RooUnblindOffset() 
{
  // Destructor
}


Double_t RooUnblindOffset::evaluate() const
{
  // Evaluate RooBlindTools unhide-offset method on blind value
  return _blindEngine.UnHideOffset(_value);
}





