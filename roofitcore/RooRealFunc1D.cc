/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooRealFunc1D.cc,v 1.1 2001/05/02 18:09:00 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   29-Apr-2001 DK Created initial version
 *
 * Copyright (C) 2001 Stanford University
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// This class performs a lightweight binding of a RooAbsReal object with
// one RooRealVar object that it depends on, and implements the abstract
// RooAbsFunc1D interface for a real-valued function of one real variable.

#include "BaBar/BaBar.hh"

#include "RooFitCore/RooRealFunc1D.hh"
#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooDataSet.hh"

ClassImp(RooRealFunc1D)
;

static const char rcsid[] =
"$Id: RooRealFunc1D.cc,v 1.1 2001/05/02 18:09:00 david Exp $";

RooRealFunc1D::RooRealFunc1D(const RooAbsReal &func, RooRealVar &x, Double_t scaleFactor,
			     const RooArgSet *normVars) :
  _funcPtr(&func), _xPtr(&x), _scale(scaleFactor), _dset(0)
{
  // Create a new binding object. The input objects are not cloned so the
  // lifetime of the newly created object is limited by their lifetimes.

  if(0 != normVars) _dset= new RooDataSet("normVars","Normalization Variables",*normVars);
}

RooRealFunc1D::~RooRealFunc1D() {
  if(_dset) delete _dset;
}

Double_t RooRealFunc1D::operator()(Double_t x) const {
  _xPtr->setVal(x);
  return _scale*_funcPtr->getVal(_dset);
}
