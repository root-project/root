/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   19-Nov-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [REAL] --
//
// RooAbsHiddenReal is a base class for objects that want to
// hide their return value from interactive use, e.g. for implementations
// of parameter unblinding functions. This class overrides all printing
// methods with versions that do not reveal the objects value and it
// has a protected version of getVal()
//

#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooAbsHiddenReal.hh"


ClassImp(RooAbsHiddenReal)
;


RooAbsHiddenReal::RooAbsHiddenReal(const char *name, const char *title, const char* unit)
  : RooAbsReal(name,title,unit)
{  
  // Constructor
}


RooAbsHiddenReal::RooAbsHiddenReal(const RooAbsHiddenReal& other, const char* name) : 
  RooAbsReal(other, name)
{
  // Copy constructor
}


RooAbsHiddenReal::~RooAbsHiddenReal() 
{
  // Destructor 
}


void RooAbsHiddenReal::printToStream(ostream& os, PrintOption opt, TString indent) const
{
  // Special version of printToStream that doesn't reveal the objects value

  // Print current value and definition of formula
  os << indent << "RooAbsHiddenReal: " << GetName() << " : (value hidden) " ;
  if(!_unit.IsNull()) os << ' ' << _unit;
  printAttribList(os) ;
  os << endl ;
} 


Bool_t RooAbsHiddenReal::readFromStream(istream& is, Bool_t compact, Bool_t verbose)
{
  // No-op version of readFromStream 

  cout << "RooAbsHiddenReal::readFromStream(" << GetName() << "): not allowed" << endl ;
  return kTRUE ;
}


void RooAbsHiddenReal::writeToStream(ostream& os, Bool_t compact) const
{
  // No-op version of writeToStream 

  cout << "RooAbsHiddenReal::writeToStream(" << GetName() << "): not allowed" << endl ;
}



