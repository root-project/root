/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooHist.cc,v 1.3 2001/04/22 18:15:32 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   29-Apr-2001 DK Created initial version
 *
 * Copyright (C) 2001 Stanford University
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// Abstract interface for evaluating a real-valued function of one real variable
// and performing numerical algorithms on it.

#include "BaBar/BaBar.hh"

#include "RooFitCore/RooAbsFunc1D.hh"

ClassImp(RooAbsFunc1D)
;

static const char rcsid[] =
"$Id: RooHist.cc,v 1.3 2001/04/22 18:15:32 david Exp $";

RooAbsFunc1D::RooAbsFunc1D() {
}

RooAbsFunc1D::~RooAbsFunc1D() {
}

Double_t RooAbsFunc1D::integral(Double_t x1, Double_t x2, Double_t tol) const {
  return 0;
}
