// @(#)root/winnt:$Name:  $:$Id: TWin32SplashThread.h,v 1.1 2004/01/14 11:42:21 rdm Exp $
// Author: Bertrand Bellenot 14/01/04

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TWin32SplashThread
#define ROOT_TWin32SplashThread

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

///////////////////////////////////////////////////////////////////////////////
class TWin32SplashThread {
public:
   void     *fHandle;   // splash thread handle

   TWin32SplashThread(Bool_t extended);
   ~TWin32SplashThread();
};

R__EXTERN TWin32SplashThread *gSplash;

#endif
