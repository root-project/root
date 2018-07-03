
/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Windows4Root.h"
#include "TWin32SplashThread.h"

TWin32SplashThread *gSplash = 0;

extern void CreateSplash(DWORD time, bool extended);
extern void DestroySplashScreen();

////////////////////////////////////////////////////////////////////////////////
/// thread for handling Splash Screen

static DWORD WINAPI HandleSplashThread(LPVOID extended)
{
   CreateSplash(4, (Bool_t)extended);
   if (gSplash) delete gSplash;
   gSplash = 0;
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// ctor.

TWin32SplashThread::TWin32SplashThread(Bool_t extended)
{
   fHandle = 0;
   DWORD splashId = 0;
   fHandle = ::CreateThread( NULL, 0,&HandleSplashThread, (LPVOID)extended, 0, &splashId );
   gSplash = this;
}

////////////////////////////////////////////////////////////////////////////////
/// dtor

TWin32SplashThread::~TWin32SplashThread()
{
   DestroySplashScreen();
   TerminateThread(fHandle, 0);
   if (fHandle) ::CloseHandle(fHandle);
   fHandle = 0;
}

