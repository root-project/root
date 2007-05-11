/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsRootFinder.cc,v 1.10 2005/06/20 15:44:47 wverkerke Exp $
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

// -- CLASS DESCRIPTION [AUX] --
// RooAbsRootFinder is the abstract interface for finding roots of real-valued
// 1-dimensional function that implements the RooAbsFunc interface.

#include "RooFit.h"

#include "RooAbsRootFinder.h"
#include "RooAbsRootFinder.h"
#include "RooAbsFunc.h"
#include "Riostream.h"

ClassImp(RooAbsRootFinder)
;


RooAbsRootFinder::RooAbsRootFinder(const RooAbsFunc& function) :
  _function(&function), _valid(function.isValid())
{
  if(_function->getDimension() != 1) {
    cout << "RooAbsRootFinder:: cannot find roots for function of dimension "
	 << _function->getDimension() << endl;
    _valid= kFALSE;
  }
}
