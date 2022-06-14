// @(#)root/thread:$Id$
// Author: Fons Rademakers   04/01/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRWLock                                                              //
//                                                                      //
// This class implements a reader/writer lock. A rwlock allows          //
// a resource to be accessed by multiple reader threads but only        //
// one writer thread.                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TRWLock.h"

ClassImp(TRWLock);

////////////////////////////////////////////////////////////////////////////////
/// Create reader/write lock.

TRWLock::TRWLock() : fLockFree(&fMutex)
{
   fReaders = 0;
   fWriters = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Obtain a reader lock. Returns always 0.

Int_t TRWLock::ReadLock()
{
   fMutex.Lock();

   while (fWriters)
      fLockFree.Wait();

   fReaders++;

   fMutex.UnLock();

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Unlock reader lock. Returns -1 if thread was not locked,
/// 0 if everything ok.

Int_t TRWLock::ReadUnLock()
{
   fMutex.Lock();

   if (fReaders == 0) {
      fMutex.UnLock();
      return -1;
   } else {
      fReaders--;
      if (fReaders == 0)
         fLockFree.Signal();
      fMutex.UnLock();
      return 0;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Obtain a writer lock. Returns always 0.

Int_t TRWLock::WriteLock()
{
   fMutex.Lock();

   while (fWriters || fReaders)
      fLockFree.Wait();

   fWriters++;

   fMutex.UnLock();

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Unlock writer lock. Returns -1 if thread was not locked,
/// 0 if everything ok.

Int_t TRWLock::WriteUnLock()
{
   fMutex.Lock();

   if (fWriters == 0) {
      fMutex.UnLock();
      return -1;
   } else {
      fWriters = 0;
      fLockFree.Broadcast();
      fMutex.UnLock();
      return 0;
   }
}
