/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooHistError.cc,v 1.2 2001/10/08 05:20:16 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   27-Apr-2001 DK Created initial version from RooMath
 *
 * Copyright (C) 2001 Stanford University
 *****************************************************************************/

// -- CLASS DESCRIPTION [PLOT] --
// RooHistError is a singleton class used to calculate the error bars
// for each bin of a RooHist object. Errors are calculated by integrating
// a specified area of a Poisson or Binomail error distribution.

#include "RooFitCore/RooHistError.hh"
#include "RooFitCore/RooBrentRootFinder.hh"

#include <iostream.h>

ClassImp(RooHistError)
  ;

static const char rcsid[] =
"$Id: RooHistError.cc,v 1.2 2001/10/08 05:20:16 verkerke Exp $";

const RooHistError &RooHistError::instance() {
  // Return a reference to a singleton object that is created the
  // first time this method is called. Only one object will be
  // constructed per ROOT session.

  static RooHistError _theInstance;
  return _theInstance;
}

RooHistError::RooHistError() {
  // Construct our singleton object.

  cout << "RooHistError: ctor" << endl;
}

Bool_t RooHistError::getPoissonInterval(Int_t n, Double_t &mu1, Double_t &mu2, Double_t nSigma) const
{
  // convert number of sigma into a confidence level
  Double_t beta= 0;

  // create a function object to use
  PoissonSum sum(n);
}

