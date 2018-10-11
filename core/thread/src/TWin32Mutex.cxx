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

ClassImp(TWin32Mutex);

////////////////////////////////////////////////////////////////////////////////
/// Create a Win32 mutex lock.

TWin32Mutex::TWin32Mutex(Bool_t recursive) : TMutexImp()
{
   ::InitializeCriticalSection(&fCritSect);
}

////////////////////////////////////////////////////////////////////////////////
/// TMutex dtor.

TWin32Mutex::~TWin32Mutex()
{
   ::DeleteCriticalSection(&fCritSect);
}

////////////////////////////////////////////////////////////////////////////////
/// Lock the mutex.

Int_t TWin32Mutex::Lock()
{
   ::EnterCriticalSection(&fCritSect);
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Try locking the mutex. Returns 0 if mutex can be locked.

Int_t TWin32Mutex::TryLock()
{
   if (::TryEnterCriticalSection(&fCritSect))
      return 0;
   return 1;
}

////////////////////////////////////////////////////////////////////////////////
/// Unlock the mutex.

Int_t TWin32Mutex::UnLock(void)
{
   ::LeaveCriticalSection(&fCritSect);
   return 0;
}
