/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooComplex.cc,v 1.8 2004/11/29 12:22:17 wverkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2004, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

// -- CLASS DESCRIPTION [MISC] --

#include "RooFitCore/RooComplex.hh"
#include <iostream>
#include <iomanip>
using std::cout;
using std::endl;
using std::ostream;

ClassImp(RooComplex)

void RooComplex::Print() const {
//  WVE Solaric CC5.0 complains about this
  cout << *this << endl;
}

ostream& operator<<(ostream& os, const RooComplex& z)
{
  return os << "(" << z.re() << "," << z.im() << ")";
}
