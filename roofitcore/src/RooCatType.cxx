/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooCatType.cc,v 1.1 2001/03/17 00:32:54 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#include "RooFitCore/RooCatType.hh"

ClassImp(RooCatType)
;

void RooCatType::printToStream(ostream& os, PrintOption opt, TString indent) const {
  // Print info about this category type to the specified stream. We only have a
  // OneLine output format.

  // we don't use oneLinePrint() since GetTitle() is empty.
  os << ClassName() << "::" << GetName() << ": Value = " << getVal() << endl;
}
