// @(#)root/thread:$Name$:$Id$
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
// class (either TPosixCondition, TSolarisCondition or TNTCondition).   //
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

   TMutex *mm = m;

   fMutex = 0;
   if (!mm) { fMutex = new TMutex(); mm = fMutex; }

   fConditionImp = gThreadFactory->CreateConditionImp(mm->fMutexImp);

   if (!fConditionImp)
      Error("TCondition", "could not create TConditionImp");
}

//______________________________________________________________________________
TCondition::~TCondition()
{
   // Clean up condition variable.

   delete fConditionImp;
   delete fMutex;
}

//______________________________________________________________________________
TMutex *TCondition::GetMutex() const
{
   // Get internally created mutex. Use it to lock resources
   // before calling Signal() or Broadcast(). Returns 0 if
   // external mutex was provided in TCondition ctor.

   return fMutex;
}

//______________________________________________________________________________
Int_t TCondition::Wait()
{
   // Wait for to be signaled.

   if (!fConditionImp) return -1;

   Int_t iret;
   if (fMutex) fMutex->Lock();
   iret = fConditionImp->Wait();
   if (fMutex) fMutex->UnLock();
   return iret;
}

//______________________________________________________________________________
Int_t TCondition::TimedWait(ULong_t secs, ULong_t nanoSec)
{
   // Wait not more than secs+nanoSecs to be signaled.

   if (!fConditionImp) return -1;

   Int_t iret;
   if (fMutex) fMutex->Lock();
   iret = fConditionImp->TimedWait(secs, nanoSec);
   if (fMutex) fMutex->UnLock();
   return iret;
}
