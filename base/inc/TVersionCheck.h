// @(#)root/base:$Name:  $:$Id: TVersionCheck.h,v 1.1 2007/05/10 15:06:21 rdm Exp $
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
#ifndef ROOT_DllImport
#include "DllImport.h"
#endif

#ifndef WIN32
R__EXTERN int  gRootVersionCode;     // defined in TROOT
R__EXTERN int *gLibraryVersion;      // defined in TSystem
R__EXTERN int  gLibraryVersionIdx;   // defined in TSystem


class TVersionCheck {
public:
   TVersionCheck() {
      if (ROOT_VERSION_CODE != gRootVersionCode && gLibraryVersion)
         gLibraryVersion[gLibraryVersionIdx] = ROOT_VERSION_CODE;
   }
};

static TVersionCheck gVersionCheck;

#endif
#endif
