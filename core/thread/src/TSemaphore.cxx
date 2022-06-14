// @(#)root/thread:$Id$
// Author: Fons Rademakers   02/07/97 (Revised: G Ganis, Nov 2015)

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

////////////////////////////////////////////////////////////////////////////////
/// Create counting semaphore.

TSemaphore::TSemaphore(Int_t initial) : fValue(initial), fWakeups(0)
{
}

////////////////////////////////////////////////////////////////////////////////
/// If the semaphore value is > 0 then decrement it and carry on, else block,
/// waiting on the condition until it is signaled.
/// Returns always 0, for backward compatibility with the first implementation.

Int_t TSemaphore::Wait()
{
   std::unique_lock<std::mutex> lk(fMutex);
   fValue--;

   if (fValue < 0) {
      do {
         fCond.wait(lk);
      } while (fWakeups < 1);
      // We have been waken-up: decrease the related counter
      fWakeups--;
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// If the semaphore value is > 0 then decrement it and carry on, else block.
/// If millisec > 0 then a relative timeout of millisec milliseconds is applied.
/// For backward compatibility with the first implementation, millisec == 0 means
/// no timeout.
/// Returns 1 if timed-out, 0 otherwise.

Int_t TSemaphore::Wait(Int_t millisec)
{
   // For backward compatibility with the first implementation
   if (millisec <= 0) return Wait();

   Int_t rc= 0;
   std::unique_lock<std::mutex> lk(fMutex);
   fValue--;

   if (fValue < 0) {
      std::cv_status cvs = std::cv_status::timeout;
      do {
         cvs = fCond.wait_for(lk,std::chrono::milliseconds(millisec));
      } while (fWakeups < 1 && cvs != std::cv_status::timeout);
      if (cvs == std::cv_status::timeout) {
         // Give back the token ...
         fValue++;
         rc = 1;
      } else {
         // We have been waken-up: decrease the related counter
         fWakeups--;
      }
   }
   return rc;
}

////////////////////////////////////////////////////////////////////////////////
/// If the semaphore value is > 0 then decrement it and return 0. If it's
/// already 0 then return 1. This call never blocks.

Int_t TSemaphore::TryWait()
{
   std::unique_lock<std::mutex> lk(fMutex);
   if (fValue > 0) {
      fValue--;
   } else {
      return 1;
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Increment the value of the semaphore. If any threads are blocked in Wait(),
/// wakeup one of them.
/// Returns always 0, for backward compatibility with the first implementation.

Int_t TSemaphore::Post()
{
   std::unique_lock<std::mutex> lk(fMutex);
   fValue++;

   if (fValue <= 0) {
      // There were threads waiting: wake up one
      fWakeups++;
      fCond.notify_one();
   }
   return 0;
}
