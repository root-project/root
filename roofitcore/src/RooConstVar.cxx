/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooFormulaVar.cc,v 1.23 2002/04/10 20:59:04 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [REAL] --
// RooConstVar represent a constant real-valued object


#include "RooFitCore/RooConstVar.hh"

ClassImp(RooConstVar)
  ;


RooConstVar::RooConstVar(const char *name, const char *title, Double_t value) : 
  RooAbsReal(name,title), _value(value)
{  
}


RooConstVar::RooConstVar(const RooConstVar& other, const char* name) : 
  RooAbsReal(other, name), _value(other._value)
{
  // Copy constructor
}


RooConstVar::~RooConstVar() 
{
}


void RooConstVar::writeToStream(ostream& os, Bool_t compact) const
{
  // Write object contents to stream
  os << _value ;
}

