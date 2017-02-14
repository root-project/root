// @(#)root/base:$Id$
// Author: Fons Rademakers   9/5/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TVersionCheck
#define ROOT_TVersionCheck

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVersionCheck                                                        //
//                                                                      //
// Used to check if the shared library or plugin is compatible with     //
// the current version of ROOT.                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifdef R__CXXMODULES
#ifndef ROOT_TObject
#error "Building with modules currently requires this file to be #included through TObject.h"
#endif
#endif // R__CXXMODULES

#include "RVersion.h"

class TVersionCheck {
public:
   TVersionCheck(int versionCode);  // implemented in TSystem.cxx
};

// FIXME: Due to a modules bug: https://llvm.org/bugs/show_bug.cgi?id=31056
// our .o files get polluted with the gVersionCheck symbol despite it was not
// visible in this TU.
#ifndef R__CXXMODULES
#ifndef __CINT__
static TVersionCheck gVersionCheck(ROOT_VERSION_CODE);
#endif
#endif

#endif
