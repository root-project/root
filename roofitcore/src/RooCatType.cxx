/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooCatType.cc,v 1.5 2001/07/31 05:54:18 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [CAT] --
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

