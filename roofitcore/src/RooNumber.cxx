/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooNumber.cc,v 1.2 2001/08/10 22:22:54 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Aug-2001 DK Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [MISC] --

#include "RooFitCore/RooNumber.hh"

ClassImp(RooNumber)
;

#ifdef HAS_NUMERIC_LIMITS

#include <numeric_limits.h>
Double_t RooNumber::infinity= numeric_limits<Double_t>::infinity();

#else

// This assumes a well behaved IEEE-754 floating point implementation.
// The next line may generate a compiler warning that can be ignored.
Double_t RooNumber::infinity= 1.0e30 ;  //1./0.;

#endif
