
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

extern void CreateSplash(DWORD time, BOOL extended);

//______________________________________________________________________________
static DWORD WINAPI HandleSplashThread(LPVOID extended)
{
   // thread for handling Splash Screen

   CreateSplash(7, (Bool_t)extended);
   ::ExitThread(0);
   if (gSplash) delete gSplash;
   gSplash = 0;
   return 0;
}

//______________________________________________________________________________
TWin32SplashThread::TWin32SplashThread(Bool_t extended)
{
   //
   fHandle = 0;
   DWORD splashId = 0;
   fHandle = ::CreateThread( NULL, 0,&HandleSplashThread, (LPVOID)extended, 0, &splashId );
   gSplash = this;
}

//______________________________________________________________________________
TWin32SplashThread::~TWin32SplashThread()
{
   // dtor

   if (fHandle) ::CloseHandle(fHandle);
   fHandle = 0;
}

