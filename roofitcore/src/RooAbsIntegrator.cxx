/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsIntegrator.cc,v 1.12 2002/09/05 04:33:06 verkerke Exp $
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

// -- CLASS DESCRIPTION [AUX] --
// RooAbsIntegrator is the abstract interface for integrating real-valued
// functions that implement the RooAbsFunc interface.

#include "RooFitCore/RooAbsIntegrator.hh"

ClassImp(RooAbsIntegrator)
;

RooAbsIntegrator::RooAbsIntegrator(const RooAbsFunc& function) :
  _function(&function), _valid(function.isValid())
{
}
