// @(#)root/thread:$Name$:$Id$
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

ClassImp(TPosixMutex)

//______________________________________________________________________________
TPosixMutex::TPosixMutex()
{
   // Create a posix mutex lock.

#if (PthreadDraftVersion == 4)
   int rc = ERRNO(pthread_mutex_init(&fMutex, pthread_mutexattr_default));
#else
   int rc = ERRNO(pthread_mutex_init(&fMutex, 0));
#endif
   if (rc != 0)
      SysError("TMutex", "pthread_mutex_init error");
}

//______________________________________________________________________________
TPosixMutex::~TPosixMutex()
{
   // TMutex dtor.

   int rc = ERRNO(pthread_mutex_destroy(&fMutex));
   if (rc != 0)
      SysError("~TMutex", "pthread_mutex_destroy error");
}

//______________________________________________________________________________
Int_t TPosixMutex::Lock()
{
   // Lock the mutex.

   return ERRNO(pthread_mutex_lock(&fMutex));
}

//______________________________________________________________________________
Int_t TPosixMutex::TryLock()
{
   // Try locking the mutex. Returns 0 if mutex can be locked.

   return ERRNO(pthread_mutex_trylock(&fMutex));
}

//______________________________________________________________________________
Int_t TPosixMutex::UnLock(void)
{
   // Unlock the mutex.

   return ERRNO(pthread_mutex_unlock(&fMutex));
}
