/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooCatType.cc,v 1.2 2001/04/14 00:43:19 davidk Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

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
