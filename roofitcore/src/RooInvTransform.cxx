/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooInvTransform.cc,v 1.1 2001/08/03 21:44:57 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   03-Aug-2001 DK Created initial version
 *
 * Copyright (C) 2001 Stanford University
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// Lightweight function object that applies a scale factor to a RooAbsFunc implementation.

// #include "BaBar/BaBar.hh"

#include "RooFitCore/RooInvTransform.hh"

#include <iostream.h>
#include <math.h>

ClassImp(RooInvTransform)
;

static const char rcsid[] =
"$Id: RooInvTransform.cc,v 1.1 2001/08/03 21:44:57 david Exp $";

RooInvTransform::RooInvTransform(const RooAbsFunc &func) :
  RooAbsFunc(func.getDimension()), _func(&func)
{
  // Apply the change of variables transformation x -> 1/x to the input
  // function and its range. The function must be one dimensional and its
  // range cannot include zero.

  if(getDimension() != 1) {
    cout << "RooInvTransform: can only be applied to a 1-dim function" << endl;
    _valid= kFALSE;
  }
}
