// @(#)root/thread:$Name$:$Id$
// Author: Fons Rademakers   02/07/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSemaphore                                                           //
//                                                                      //
// This class implements a counting semaphore. Use a semaphore          //
// to synchronize threads.                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TSemaphore.h"

ClassImp(TSemaphore)

//______________________________________________________________________________
TSemaphore::TSemaphore(UInt_t initial) : fCond(&fMutex)
{
   // Create counting semaphore.

   fValue = initial;
}

//______________________________________________________________________________
Int_t TSemaphore::Wait()
{
   // If semaphore value is > 0 then decrement it and carry on. If it's
   // already 0 then block.

   fMutex.Lock();

   while (fValue == 0) {
      int rc = fCond.Wait();

      if (rc != 0) {
         fMutex.UnLock();
         return rc;
      }
   }

   fValue--;

   fMutex.UnLock();

   return 0;
}

//______________________________________________________________________________
Int_t TSemaphore::TryWait()
{
   // If semaphore value is > 0 then decrement it and return 0. If it's
   // already 0 then return 1.

   fMutex.Lock();

   if (fValue == 0) {
      fMutex.UnLock();
      return 1;
   }

   fValue--;

   fMutex.UnLock();

   return 0;
}

//______________________________________________________________________________
Int_t TSemaphore::Post()
{
   // If any threads are blocked in Wait(), wake one of them up. Otherwise
   // increment the value of the semaphore.

   fMutex.Lock();

   if (fValue == 0)
      fCond.Signal();

   fValue++;

   fMutex.UnLock();

   return 0;
}
