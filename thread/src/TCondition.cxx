// @(#)root/thread:$Name:  $:$Id: TCondition.cxx,v 1.3 2004/12/10 22:27:21 rdm Exp $
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
   fMutex->fId = TThread::SelfId(); // fix the owner because lowlevel relock
   if (fPrivateMutex) fMutex->UnLock();
   return iret;
}

//______________________________________________________________________________
Int_t TCondition::TimedWait(ULong_t secs, ULong_t nanoSec)
{
   // Wait not more than secs+nanoSecs to be signaled.
   // Returns 0 if successfully signalled, 1 if time expired and -1 in
   // case of error.

   if (!fConditionImp) return -1;

   Int_t iret;
   if (fPrivateMutex) fMutex->Lock();
   iret = fConditionImp->TimedWait(secs, nanoSec);
   fMutex->fId = TThread::SelfId(); // fix the owner because lowlevel relock
   if (fPrivateMutex) fMutex->UnLock();
   return iret;
}
