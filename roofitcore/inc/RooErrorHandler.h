/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2002, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_ERROR_HANDLER
#define ROO_ERROR_HANDLER

#include <stdlib.h>
#include <signal.h>
#include "Rtypes.h"

class RooErrorHandler
{
public:
  // Soft assert function that interrupts macro execution but doesn't kill ROOT
  static void softAssert(Bool_t condition) { if (!condition) abort() ; }

  // Soft abort function that interrupts macro execution but doesn't kill ROOT
  static void softAbort() { raise(11) ; }
} ;
 
#endif
