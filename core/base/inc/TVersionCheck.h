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

#ifndef ROOT_RVersion
#include "RVersion.h"
#endif

class TVersionCheck {
public:
   TVersionCheck(int versionCode);  // implemented in TSystem.cxx
};

#ifndef __CINT__
static TVersionCheck gVersionCheck(ROOT_VERSION_CODE);
#endif

#endif
