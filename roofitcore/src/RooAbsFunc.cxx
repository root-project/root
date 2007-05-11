/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsFunc.cc,v 1.9 2005/06/16 09:31:23 wverkerke Exp $
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
// Abstract interface for evaluating a real-valued function of one real variable
// and performing numerical algorithms on it. The purpose of this interface is
// to decouple RooAbsArg-specific implementations from numerical algorithms that
// only need a simple function evaluation interface. The domain of the function
// is assumed to be an n-dimensional box with edge coordinates specified by the
// the getMinLimit() and getMaxLimit() methods.


#include "RooFit.h"

#include "RooAbsFunc.h"
#include "RooAbsFunc.h"

ClassImp(RooAbsFunc)
;














