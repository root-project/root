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

//////////////////////////////////////////////////////////////////////////////
//
// BEGIN_HTML
// Class RooNumber implements numeric constants used by RooFit
// END_HTML
//

#include "RooFit.h"
#include "RooNumber.h"

ClassImp(RooNumber)
;

#ifdef HAS_NUMERIC_LIMITS

#include <numeric_limits.h>
Double_t RooNumber::_Infinity= numeric_limits<Double_t>::infinity();
#else

// This assumes a well behaved IEEE-754 floating point implementation.
// The next line may generate a compiler warning that can be ignored.
Double_t RooNumber::_Infinity= 1.0e30 ;  //1./0.;

#endif


//_____________________________________________________________________________
Double_t RooNumber::infinity() 
{
  // Return internal infinity representation

  return _Infinity ;
}


//_____________________________________________________________________________
Int_t RooNumber::isInfinite(Double_t x) 
{
  // Return true if x is infinite by RooNumBer internal specification

  return (x >= +_Infinity) ? +1 : ((x <= -_Infinity) ? -1 : 0);
}

