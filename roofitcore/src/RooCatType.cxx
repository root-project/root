/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooCatType.cc,v 1.3 2001/04/18 20:38:02 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// RooCatType is an auxilary class for RooAbsCategory and defines a 
// a single category state. The class holds a string label and an integer 
// index value which define the state

#include <stdlib.h>
#include "RooFitCore/RooCatType.hh"

ClassImp(RooCatType)
;

void RooCatType::printToStream(ostream& os, PrintOption opt, TString indent) const {
  // Print info about this category type to the specified stream. We only have a
  // OneLine output format.

  // we don't use oneLinePrint() since GetTitle() is empty.
  os << ClassName() << "::" << GetName() << ": Value = " << getVal() << endl;
}



Bool_t RooCatType::operator==(const char* label) 
{ 
  char *endptr(0) ;
  Int_t val = strtol(label,&endptr,10) ;
  if (endptr-label==strlen(label)) {
    return operator==(val) ;
  }

  return !TString(label).CompareTo(GetName()) ; 
}
