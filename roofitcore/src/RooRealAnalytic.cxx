/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsFunc1D.cc,v 1.1 2001/05/02 18:08:59 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   03-Aug-2001 DK Created initial version
 *
 * Copyright (C) 2001 Stanford University
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// Lightweight interface adaptor that binds an analytic integral of a
// RooAbsReal object (specified by a code) to a set of dependent variables.

// #include "BaBar/BaBar.hh"

#include "RooFitCore/RooRealAnalytic.hh"
#include "RooFitCore/RooAbsReal.hh"

#include <assert.h>

ClassImp(RooRealAnalytic)
;

static const char rcsid[] =
"$Id: RooAbsFunc1D.cc,v 1.1 2001/05/02 18:08:59 david Exp $";

Double_t RooRealAnalytic::operator()(const Double_t xvector[]) const {
  assert(isValid());
  loadValues(xvector);
  return _func->analyticalIntegral(_code);
}
