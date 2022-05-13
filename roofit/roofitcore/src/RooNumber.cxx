/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
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

/**
\file RooNumber.cxx
\class RooNumber
\ingroup Roofitcore

Class RooNumber implements numeric constants used by RooFit
**/

#include "RooNumber.h"

using namespace std;

ClassImp(RooNumber);
;

#ifdef HAS_NUMERIC_LIMITS

#include <numeric_limits.h>
double RooNumber::_Infinity= numeric_limits<double>::infinity();
#else

// This assumes a well behaved IEEE-754 floating point implementation.
// The next line may generate a compiler warning that can be ignored.
double RooNumber::_Infinity= 1.0e30 ;  //1./0.;

#endif


////////////////////////////////////////////////////////////////////////////////
/// Return internal infinity representation

double RooNumber::infinity()
{
  return _Infinity ;
}


////////////////////////////////////////////////////////////////////////////////
/// Return true if x is infinite by RooNumBer internal specification

Int_t RooNumber::isInfinite(double x)
{
  return (x >= +_Infinity) ? +1 : ((x <= -_Infinity) ? -1 : 0);
}

