/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooComplex.cc,v 1.3 2001/07/31 05:54:18 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   20-Jun-2000 DK Created initial version from RooProbDens.rdl
 *   18-Jun-2001 WV Imported from RooFitTools
 *
 * Copyright (C) 2000 Stanford University
 *****************************************************************************/

// -- CLASS DESCRIPTION [MISC] --

#include "RooFitCore/RooComplex.hh"
#include <iostream.h>
#include <iomanip.h>

ClassImp(RooComplex)

void RooComplex::Print() const {
//  WVE Solaric CC5.0 complains about this
  cout << *this << endl;
}

ostream& operator<<(ostream& os, const RooComplex& z)
{
  return os << "(" << z.re() << "," << z.im() << ")";
}
