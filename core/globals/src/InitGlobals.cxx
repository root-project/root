// @(#)root/globals:$Id$
// Author: Fons Rademakers   12/10/2012

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
 
#include "TROOT.h"

class TInitGlobals {
public:
   TInitGlobals() { gROOT = ROOT::GetROOT(); }   // The ROOT of EVERYTHING
};
static TInitGlobals gInitGlobals;
