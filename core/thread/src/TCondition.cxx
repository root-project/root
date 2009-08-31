// @(#)root/thread:$Id$
// Author: Fons Rademakers   01/07/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TCondition                                                           //
//                                                                      //
// This class implements a condition variable. Use a condition variable //
// to signal threads. The actual work is done via the TConditionImp     //
// class (either TPosixCondition or TWin32Condition).                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TCondition.h"
#include "TMutex.h"
#include "TThreadFactory.h"


ClassImp(TCondition)

//______________________________________________________________________________
TCondition::TCondition(TMutex *m)
{
   // Create a condition variable. The actual condition implementation
   // will be provided via the TThreadFactory. If no external mutex is
   // provided one will be created. Use GetMutex() to get this mutex
   // and use it before calling Signal() or Broadcast().

   fPrivateMutex = (m == 0);
   if (fPrivateMutex) {
      fMutex = new TMutex();
   } else {
      fMutex = m;
   }

   fConditionImp = gThreadFactory->CreateConditionImp(fMutex->fMutexImp);

   if (!fConditionImp)
      Error("TCondition", "could not create TConditionImp");
}

//______________________________________________________________________________
TCondition::~TCondition()
{
   // Clean up condition variable.

   delete fConditionImp;
   if (fPrivateMutex) delete fMutex;
}

//______________________________________________________________________________
TMutex *TCondition::GetMutex() const
{
   // Get internally created mutex. Use it to lock resources
   // before calling Signal() or Broadcast(). Returns 0 if
   // external mutex was provided in TCondition ctor.

   if (fPrivateMutex)
      return fMutex;
   return 0;
}

//______________________________________________________________________________
Int_t TCondition::Wait()
{
   // Wait to be signaled.

   if (!fConditionImp) return -1;

   Int_t iret;
   if (fPrivateMutex) fMutex->Lock();
   iret = fConditionImp->Wait();
   if (fPrivateMutex) fMutex->UnLock();
   return iret;
}

//______________________________________________________________________________
Int_t TCondition::TimedWait(ULong_t secs, ULong_t nanoSec)
{
   // Wait to be signaled or till the timer times out.
   // This method is given an absolute time since the beginning of
   // the EPOCH (use TThread::GetTime() to get this absolute time).
   // To wait for a relative time from now, use
   // TCondition::TimedWaitRelative(ULong_t ms).
   // Returns 0 if successfully signalled, 1 if time expired and -1 in
   // case of error.

   if (!fConditionImp) return -1;

   Int_t iret;
   if (fPrivateMutex) fMutex->Lock();
   iret = fConditionImp->TimedWait(secs, nanoSec);
   if (fPrivateMutex) fMutex->UnLock();
   return iret;
}

//______________________________________________________________________________
Int_t TCondition::TimedWaitRelative(ULong_t ms)
{
   // Wait to be signaled or till the timer times out.
   // This method is given a relative time from now.
   // To wait for an absolute time since the beginning of the EPOCH, use
   // TCondition::TimedWait(ULong_t secs, ULong_t nanoSec).
   // Returns 0 if successfully signalled, 1 if time expired and -1 in
   // case of error.

   if (!fConditionImp) return -1;

   ULong_t absSec, absNanoSec;
   TThread::GetTime(&absSec, &absNanoSec);

   ULong_t dsec = ms/1000;
   absSec += dsec;
   absNanoSec += (ms - dsec*1000) * 1000000;
   if (absNanoSec > 999999999) {
      absSec += 1;
      absNanoSec -= 1000000000;
   }

   return TimedWait(absSec, absNanoSec);
}
