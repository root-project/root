// @(#)root/thread:$Id$
// Author: Fons Rademakers   26/06/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMutex                                                               //
//                                                                      //
// This class implements mutex locks. A mutex is a mutual exclusive     //
// lock. The actual work is done via the TMutexImp class (either        //
// TPosixMutex or TWin32Mutex).                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TInterpreter.h"
#include "TMutex.h"
#include "TThreadFactory.h"
#include <errno.h>
#include "TError.h"

ClassImp(TMutex);

////////////////////////////////////////////////////////////////////////////////
/// Create a mutex lock. The actual mutex implementation will be
/// provided via the TThreadFactory.

TMutex::TMutex(Bool_t recursive)
{
   fMutexImp = gThreadFactory->CreateMutexImp(recursive);

   if (!fMutexImp)
      Error("TMutex", "could not create TMutexImp");
}

////////////////////////////////////////////////////////////////////////////////
/// Lock the mutex. Returns 0 when no error, EDEADLK when mutex was already
/// locked by this thread and this mutex is not reentrant.

Int_t TMutex::Lock()
{
   Int_t iret = fMutexImp->Lock();

   return iret;
}

////////////////////////////////////////////////////////////////////////////////
/// Try to lock mutex. Returns 0 when no error, EDEADLK when mutex was
/// already locked by this thread and this mutex is not reentrant.

Int_t TMutex::TryLock()
{
   Int_t iret = fMutexImp->TryLock();

   return iret;
}

////////////////////////////////////////////////////////////////////////////////
/// Unlock the mutex. Returns 0 when no error, EPERM when mutex was already
/// unlocked by this thread.

Int_t TMutex::UnLock()
{
   return fMutexImp->UnLock();
}

////////////////////////////////////////////////////////////////////////////////
/// Clean up of mutex.

Int_t TMutex::CleanUp()
{
   return UnLock();
}

////////////////////////////////////////////////////////////////////////////////
/// Create mutex and return pointer to it. Calling function must care
/// about proper deletion. The function is intended to be used in connection
/// with the R__LOCKGUARD2 macro for local thread protection. Since "new" is
/// used the TStorage class has to be protected by gGlobalMutex.

TVirtualMutex *TMutex::Factory(Bool_t recursive)
{
   TVirtualMutex *ret = new TMutex(recursive);
   return ret;
}
