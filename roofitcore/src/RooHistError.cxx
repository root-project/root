/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooHistError.cc,v 1.1 2001/05/02 18:09:00 david Exp $
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

#include <iostream.h>

ClassImp(RooHistError)

static const char rcsid[] =
"$Id: RooHistError.cc,v 1.1 2001/05/02 18:09:00 david Exp $";

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
