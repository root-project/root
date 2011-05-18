// @(#)root/thread:$Id$
// Author: Bertrand Bellenot  20/10/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TWin32Condition
#define ROOT_TWin32Condition


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TWin32Condition                                                      //
//                                                                      //
// This class provides an interface to the win32 condition variable     //
// routines.                                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TConditionImp
#include "TConditionImp.h"
#endif

#include "Windows4Root.h"

#ifndef __CINT__
typedef struct
{
   int waiters_count_;
   // Number of waiting threads.

   CRITICAL_SECTION waiters_count_lock_;
   // Serialize access to <waiters_count_>.

   HANDLE sema_;
   // Semaphore used to queue up threads waiting for the condition to
   // become signaled. 

   HANDLE waiters_done_;
   // An auto-reset event used by the broadcast/signal thread to wait
   // for all the waiting thread(s) to wake up and be released from the
   // semaphore. 

   size_t was_broadcast_;
   // Keeps track of whether we were broadcasting or signaling.  This
   // allows us to optimize the code if we're just signaling.
} pthread_cond_t;
#else
struct pthread_cond_t;
#endif

class TMutexImp;
class TWin32Mutex;


class TWin32Condition : public TConditionImp {

private:
   pthread_cond_t  fCond;    // the pthread condition variable
   TWin32Mutex    *fMutex;   // mutex used around Wait() and TimedWait()

public:
   TWin32Condition(TMutexImp *m);
   virtual ~TWin32Condition();

   Int_t  Wait();
   Int_t  TimedWait(ULong_t secs, ULong_t nanoSecs = 0);
   Int_t  Signal();
   Int_t  Broadcast();

   ClassDef(TWin32Condition,0)   // Posix condition variable
};

#endif
