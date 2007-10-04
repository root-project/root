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

ClassImp(TRWLock)

//______________________________________________________________________________
TRWLock::TRWLock() : fLockFree(&fMutex)
{
   // Create reader/write lock.

   fReaders = 0;
   fWriters = 0;
}

//______________________________________________________________________________
Int_t TRWLock::ReadLock()
{
   // Obtain a reader lock. Returns always 0.

   fMutex.Lock();

   while (fWriters)
      fLockFree.Wait();

   fReaders++;

   fMutex.UnLock();

   return 0;
}

//______________________________________________________________________________
Int_t TRWLock::ReadUnLock()
{
   // Unlock reader lock. Returns -1 if thread was not locked,
   // 0 if everything ok.

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

//______________________________________________________________________________
Int_t TRWLock::WriteLock()
{
   // Obtain a writer lock. Returns always 0.

   fMutex.Lock();

   while (fWriters || fReaders)
      fLockFree.Wait();

   fWriters++;

   fMutex.UnLock();

   return 0;
}

//______________________________________________________________________________
Int_t TRWLock::WriteUnLock()
{
   // Unlock writer lock. Returns -1 if thread was not locked,
   // 0 if everything ok.

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
