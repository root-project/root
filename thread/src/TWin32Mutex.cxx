// @(#)root/thread:$Name:  $:$Id: TWin32Mutex.cxx,v 1.3 2004/12/15 10:09:04 rdm Exp $
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

#include "TThread.h"
#include "TWin32Mutex.h"

ClassImp(TWin32Mutex)

//______________________________________________________________________________
TWin32Mutex::TWin32Mutex()
{
   // Create a Win32 mutex lock.

   fHMutex = ::CreateMutex(0, 0, 0);
   if (!fHMutex)
      SysError("TMutex", "CreateMutex error");
}

//______________________________________________________________________________
TWin32Mutex::~TWin32Mutex()
{
   // TMutex dtor.

  ::CloseHandle(fHMutex);
}

//______________________________________________________________________________
Int_t TWin32Mutex::Lock()
{
   // Lock the mutex.

   if (::WaitForSingleObject(fHMutex, INFINITE) != WAIT_OBJECT_0)
      return -1;
   return 0;
}

//______________________________________________________________________________
Int_t TWin32Mutex::TryLock()
{
   // Try locking the mutex. Returns 0 if mutex can be locked.

   if (::WaitForSingleObject(fHMutex, 0) == WAIT_OBJECT_0)
      return 0;
   return 1;
}

//______________________________________________________________________________
Int_t TWin32Mutex::UnLock(void)
{
   // Unlock the mutex.

   if (::ReleaseMutex(fHMutex) == 0)
      return -1;
   return 0;
}
