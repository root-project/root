// @(#)root/thread:$Id$
// Author: Bertrand Bellenot   23/10/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TWin32Mutex                                                          //
//                                                                      //
// This class provides an interface to the Win32 mutex routines.        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0501 // needed for TryEnterCriticalSection
#endif

#include "TThread.h"
#include "TWin32Mutex.h"

ClassImp(TWin32Mutex)

//______________________________________________________________________________
TWin32Mutex::TWin32Mutex(Bool_t recursive) : TMutexImp()
{
   // Create a Win32 mutex lock.

   ::InitializeCriticalSection(&fCritSect);
}

//______________________________________________________________________________
TWin32Mutex::~TWin32Mutex()
{
   // TMutex dtor.

   ::DeleteCriticalSection(&fCritSect);
}

//______________________________________________________________________________
Int_t TWin32Mutex::Lock()
{
   // Lock the mutex.

   ::EnterCriticalSection(&fCritSect);
   return 0;
}

//______________________________________________________________________________
Int_t TWin32Mutex::TryLock()
{
   // Try locking the mutex. Returns 0 if mutex can be locked.

   if (::TryEnterCriticalSection(&fCritSect))
      return 0;
   return 1;
}

//______________________________________________________________________________
Int_t TWin32Mutex::UnLock(void)
{
   // Unlock the mutex.

   ::LeaveCriticalSection(&fCritSect);
   return 0;
}
