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
\file RooSentinel.cxx
\class RooSentinel
\ingroup Roofitcore

RooSentinel is a special purpose singleton class that terminates
all other RooFit singleton services when the process exits.

All function RooFit singleton services are created on the heap with
a static wrapper function to avoid the 'static initialization order fiasco'
but are not automatically destroyed at the end of the session. This class
installs an atexit() function that takes care of this
**/

#include "RooFit.h"

#include "RooSentinel.h"
#include "RooArgSet.h"
#include "RooRealConstant.h"
#include "RooResolutionModel.h"
#include "RooExpensiveObjectCache.h"
#include "RooDataSet.h"

Bool_t RooSentinel::_active = kFALSE ;

static void CleanUpRooFitAtExit()
{
  // Clean up function called at program termination before global objects go out of scope.
  RooArgSet::cleanup() ;
  RooDataSet::cleanup();
}



////////////////////////////////////////////////////////////////////////////////
/// Install atexit handler that calls CleanupRooFitAtExit()
/// on program termination

void RooSentinel::activate()
{
  if (!_active) {
    _active = kTRUE ;
    atexit(CleanUpRooFitAtExit) ;
  }
}


 
