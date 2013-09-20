/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

//////////////////////////////////////////////////////////////////////////////
// 
// BEGIN_HTML
// Class RooComplex is a simple container class for complex values
// END_HTML
//

#include "RooFit.h"

#define ROO_COMPLEX_CXX
#include "RooComplex.h"
#include "Riostream.h"
#include <iomanip>

using namespace std;

ClassImp(RooComplex)

void RooComplex::warn() const
{
    static int nwarns = 0;
    if (nwarns < 1<<12) {
	cout << "[#0] WARN: RooComplex is deprecated. Please use std::complex<Double_t> in your code instead." << std::endl;
	++nwarns;
    }
}

//_____________________________________________________________________________
void RooComplex::Print() const {
//  WVE Solaric CC5.0 complains about this
  cout << *this << endl;
}

ostream& operator<<(ostream& os, const RooComplex& z)
{
  // Print real and imaginary component on ostream
  return os << "(" << z.re() << "," << z.im() << ")";
}
