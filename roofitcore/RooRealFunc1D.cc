/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooRealFunc1D.cc,v 1.5 2001/08/02 21:39:11 verkerke Exp $
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

// #include "BaBar/BaBar.hh"

#include "RooFitCore/RooRealFunc1D.hh"
#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooRealIntegral.hh"

#include "TString.h"

ClassImp(RooRealFunc1D)
;

static const char rcsid[] =
"$Id: RooRealFunc1D.cc,v 1.5 2001/08/02 21:39:11 verkerke Exp $";

RooRealFunc1D::RooRealFunc1D(const RooAbsReal &func, RooRealVar &x, Double_t scaleFactor,
			     const RooArgSet *normVars) :
  _funcPtr(&func), _xPtr(&x), _scale(scaleFactor), _projected(0)
{
  // Create a new binding object. The input objects are not cloned so the
  // lifetime of the newly created object is limited by their lifetimes.

  if(0 != normVars) {
    // should we normalize over x ?
    RooArgSet vars(*normVars);
    RooAbsArg *found= vars.find(x.GetName());
    if(found) {
      // calculate our normalization factor over all vars including x
      RooRealIntegral normFunc("normFunc","normFunc",func,vars);
      _scale/= normFunc.getVal();
      // remove x from the set of vars to be projected
      vars.remove(*found);
    }
    // project out any remaining normalization variables
    if(vars.GetSize() > 0) {
      _projected= new RooRealIntegral(TString(func.GetName()).Append("Projected"),
				      TString(func.GetTitle()).Append(" (Projected)"),
				      func,vars);
      _funcPtr= _projected;
    }
  }
}

RooRealFunc1D::~RooRealFunc1D() {
  if(_projected) delete _projected;
}

Double_t RooRealFunc1D::operator()(Double_t x) const {
  _xPtr->setVal(x);
  return _scale*_funcPtr->getVal();
}
