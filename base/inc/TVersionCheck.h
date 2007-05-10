// @(#)root/base:$Name:  $:$Id: TObject.h,v 1.33 2007/01/20 20:51:52 brun Exp $
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

extern int  gRootVersionCode;     // defined in TROOT
extern int *gLibraryVersion;      // defined in TSystem
extern int  gLibraryVersionIdx;   // defined in TSystem


class TVersionCheck {
public:
   TVersionCheck() {
      if (ROOT_VERSION_CODE != gRootVersionCode && gLibraryVersion)
         gLibraryVersion[gLibraryVersionIdx] = ROOT_VERSION_CODE;
   }
};

static TVersionCheck gVersionCheck;

#endif
