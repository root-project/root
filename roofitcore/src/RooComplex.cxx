/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   20-Jun-2000 DK Created initial version from RooProbDens.rdl
 *   18-Jun-2001 WV Imported from RooFitTools
 *
 * Copyright (C) 2000 Stanford University
 *****************************************************************************/

#include "RooFitCore/RooComplex.hh"
#include <iostream.h>
#include <iomanip.h>

ClassImp(RooComplex)

void RooComplex::Print() const {
  cout << *this << endl;
}

ostream& operator<<(ostream& os, const RooComplex& z)
{
  return os << "(" << z.re() << "," << z.im() << ")";
}
