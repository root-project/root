/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsIntegrator.cc,v 1.9 2001/10/08 05:20:11 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   14-Nov-2001 DK Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [AUX] --
// RooAbsRootFinder is the abstract interface for finding roots of real-valued
// 1-dimensional function that implements the RooAbsFunc interface.

#include "RooFitCore/RooAbsRootFinder.rdl"
#include "RooFitCore/RooAbsFunc.hh"
#include <iostream.h>

ClassImp(RooAbsRootFinder)
;

static const char rcsid[] =
"$Id: RooAbsIntegrator.cc,v 1.9 2001/10/08 05:20:11 verkerke Exp $";

RooAbsRootFinder::RooAbsRootFinder(const RooAbsFunc& function) :
  _function(&function), _valid(function.isValid())
{
  if(_function->getDimension() != 1) {
    cout << "RooAbsRootFinder:: cannot find roots for function of dimension "
	 << _function->getDimension() << endl;
    _valid= kFALSE;
  }
}
