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
// RooSentinel is a special purposes singleton class that terminates
// all other RooFit singleton services when the process exists. 
//
// All function RooFit singleton services are created on the heap with
// a static wrapper function to avoid the 'static initialization order fiasco'
// but are not automatically destroyed at the end of the session. This class
// installs an atexit() function that takes care of this
// END_HTML
//

#include "RooFit.h"

#include "RooSentinel.h"
#include "RooSentinel.h"
#include "RooMinuit.h"
#include "RooMsgService.h"
#include "RooNumIntConfig.h"
#include "RooNumIntFactory.h"
#include "RooNumGenFactory.h"
#include "RooNumGenConfig.h"
#include "RooNameReg.h"
#include "RooArgSet.h"
#include "RooRealConstant.h"
#include "RooResolutionModel.h"
#include "RooExpensiveObjectCache.h"
#include "RooMath.h"

Bool_t RooSentinel::_active = kFALSE ;

static void CleanUpRooFitAtExit()
{
  // Clean up function called at program termination before global objects go out of scope.
  RooMinuit::cleanup() ;
  RooMsgService::cleanup() ;
  RooNumIntConfig::cleanup() ;
  RooNumGenConfig::cleanup() ;
  RooNumIntFactory::cleanup() ;
  RooNumGenFactory::cleanup() ;
  RooNameReg::cleanup() ;
  RooArgSet::cleanup() ;
  RooRealConstant::cleanup() ;
  RooResolutionModel::cleanup() ;
  RooMath::cleanup() ;
  RooExpensiveObjectCache::cleanup() ;
}



//_____________________________________________________________________________
void RooSentinel::activate()
{
  // Install atexit handler that calls CleanupRooFitAtExit()
  // on program termination

  if (!_active) {
    _active = kTRUE ;
    atexit(CleanUpRooFitAtExit) ;
  }
}


 
