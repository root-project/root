/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsFunc1D.cc,v 1.1 2001/05/02 18:08:59 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   02-Aug-2001 DK Created initial version from RooAbsFunc1D
 *
 * Copyright (C) 2001 Stanford University
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// Abstract interface for evaluating a real-valued function of one real variable
// and performing numerical algorithms on it. The purpose of this interface is
// to decouple RooAbsArg-specific implementations from numerical algorithms that
// only need a simple function evaluation interface. The domain of the function
// is assumed to be an n-dimensional box with edge coordinates specified by the
// the getMinLimit() and getMaxLimit() methods.

// #include "BaBar/BaBar.hh"

#include "RooFitCore/RooAbsFunc.hh"

ClassImp(RooAbsFunc)
;

static const char rcsid[] =
"$Id: RooAbsFunc1D.cc,v 1.1 2001/05/02 18:08:59 david Exp $";
