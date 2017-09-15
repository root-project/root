// @(#)root/thread:$Id$
// Author: Fons Rademakers   25/06/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPosixMutex                                                          //
//                                                                      //
// This class provides an interface to the posix mutex routines.        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TThread.h"
#include "TPosixMutex.h"
#include "PosixThreadInc.h"

ClassImp(TPosixMutex);

////////////////////////////////////////////////////////////////////////////////
/// Create a posix mutex lock.

TPosixMutex::TPosixMutex(Bool_t recursive) : TMutexImp()
{
   if (recursive) {
      SetBit(kIsRecursive);

      int rc;
      pthread_mutexattr_t attr;

      rc = pthread_mutexattr_init(&attr);

      if (!rc) {
         rc = pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
         if (!rc) {
            rc = pthread_mutex_init(&fMutex, &attr);
            if (rc)
               SysError("TPosixMutex", "pthread_mutex_init error");
         } else
            SysError("TPosixMutex", "pthread_mutexattr_settype error");
      } else
         SysError("TPosixMutex", "pthread_mutex_init error");

      pthread_mutexattr_destroy(&attr);

   } else {

      int rc = pthread_mutex_init(&fMutex, 0);
      if (rc)
         SysError("TPosixMutex", "pthread_mutex_init error");

   }
}

////////////////////////////////////////////////////////////////////////////////
/// TMutex dtor.

TPosixMutex::~TPosixMutex()
{
   int rc = pthread_mutex_destroy(&fMutex);
   if (rc)
      SysError("~TPosixMutex", "pthread_mutex_destroy error");
}

////////////////////////////////////////////////////////////////////////////////
/// Lock the mutex.

Int_t TPosixMutex::Lock()
{
   return pthread_mutex_lock(&fMutex);
}

////////////////////////////////////////////////////////////////////////////////
/// Try locking the mutex. Returns 0 if mutex can be locked.

Int_t TPosixMutex::TryLock()
{
   return pthread_mutex_trylock(&fMutex);
}

////////////////////////////////////////////////////////////////////////////////
/// Unlock the mutex.

Int_t TPosixMutex::UnLock(void)
{
   return pthread_mutex_unlock(&fMutex);
}

namespace {
struct TPosixMutexState: public TVirtualMutex::State {
   int fLockCount = 0;
};
}

////////////////////////////////////////////////////////////////////////////////
/// Reset the mutex state to unlocked. The state before resetting to unlocked is
/// returned and can be passed to `Restore()` later on. This function must only
/// be called while the mutex is locked.

std::unique_ptr<TVirtualMutex::State> TPosixMutex::Reset()
{
   std::unique_ptr<TPosixMutexState> pState(new TPosixMutexState);
   if (TestBit(kIsRecursive)) {
      while (!UnLock())
         ++pState->fLockCount;
      if (!pState->fLockCount)
         SysError("Reset", "Reset() called on unlocked Mutex!");
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

void TPosixMutex::Restore(std::unique_ptr<TVirtualMutex::State> &&state)
{
   TPosixMutexState *pState = dynamic_cast<TPosixMutexState *>(state.get());
   if (!pState) {
      if (state) {
         SysError("Restore", "LOGIC ERROR - invalid state object!");
         return;
      }
      // No state, do nothing.
      return;
   }

   if (!pState->fLockCount) {
      SysError("Reset", "Restore() called with unlocked state!");
      return;
   }

   do {
      Lock();
   } while (--pState->fLockCount);
}
