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

namespace {
struct TWin32MutexState : public TVirtualMutex::State {
   int fLockCount = 0;
};
}

////////////////////////////////////////////////////////////////////////////////
/// Reset the mutex state to unlocked. The state before resetting to unlocked is
/// returned and can be passed to `Restore()` later on. This function must only
/// be called while the mutex is locked.

std::unique_ptr<TVirtualMutex::State> TWin32Mutex::Reset()
{
   std::unique_ptr<TWin32MutexState> pState(new TWin32MutexState);
   if (TestBit(kIsRecursive)) {
      while (!UnLock())
         ++pState->fLockCount;
      return std::move(pState);
   }
   // Not recursive. Unlocking a non-recursive, non-robust, unlocked mutex has an
   // undefined return value - so we cannot *guarantee* that the mutex is locked.
   // But we can try.
   if (int rc = UnLock()) {
      SysError("Reset", "pthread_mutex_unlock failed with %d, "
                        "but Reset() must be called on locked mutex!",
               rc);
      return std::move(pState);
   }
   ++pState->fLockCount;
   return std::move(pState);
}

////////////////////////////////////////////////////////////////////////////////
/// Restore the mutex state to the state pointed to by `state`. This function
/// must only be called while the mutex is unlocked.

void TWin32Mutex::Restore(std::unique_ptr<TVirtualMutex::State> &&state)
{
   TWin32MutexState *pState = dynamic_cast<TWin32MutexState *>(state.get());
   if (!pState) {
      if (state) {
         SysError("Restore", "LOGIC ERROR - invalid state object!");
         return;
      }
      // No state, do nothing.
      return;
   }
   do {
      Lock();
   } while (--pState->fLockCount);
}
