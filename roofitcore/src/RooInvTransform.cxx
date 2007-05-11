/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooInvTransform.cc,v 1.10 2005/06/20 15:44:54 wverkerke Exp $
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
// Lightweight function object that applies a scale factor to a RooAbsFunc implementation.


#include "RooFit.h"

#include "RooInvTransform.h"
#include "RooInvTransform.h"

#include "Riostream.h"
#include <math.h>

ClassImp(RooInvTransform)
;

RooInvTransform::RooInvTransform(const RooAbsFunc &func) :
  RooAbsFunc(func.getDimension()), _func(&func)
{
  // Apply the change of variables transformation x -> 1/x to the input
  // function and its range. The function must be one dimensional and its
  // range cannot include zero.

//   if(getDimension() != 1) {
//     cout << "RooInvTransform: can only be applied to a 1-dim function" << endl;
//     _valid= kFALSE;
//   }
}
