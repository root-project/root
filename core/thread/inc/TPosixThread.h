// @(#)root/thread:$Id$
// Author: Fons Rademakers   02/07/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPosixThread
#define ROOT_TPosixThread


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPosixThread                                                         //
//                                                                      //
// This class provides an interface to the posix thread routines.       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TThreadImp.h"

#ifndef __CINT__
#include <pthread.h>
#endif

R__EXTERN "C" unsigned int sleep(unsigned int seconds);

class TPosixThreadCleanUp;


class TPosixThread : public TThreadImp {

public:
   TPosixThread() { }
   ~TPosixThread() { }

   Int_t  Join(TThread *th, void **ret) override;
   Long_t SelfId() override;
   Int_t  Run(TThread *th, const int affinity = -1) override;

   Int_t  Kill(TThread *th) override;
   Int_t  SetCancelOff() override;
   Int_t  SetCancelOn() override;
   Int_t  SetCancelAsynchronous() override;
   Int_t  SetCancelDeferred() override;
   Int_t  CancelPoint() override;
   Int_t  CleanUpPush(void **main, void *free,void *arg) override;
   Int_t  CleanUpPop(void **main, Int_t exe) override;
   Int_t  CleanUp(void **main) override;

   Int_t  Exit(void *ret) override;

   ClassDefOverride(TPosixThread,0)  // TPosixThread class
};


class TPosixThreadCleanUp {

friend class TPosixThread;

private:
   void                *fRoutine;
   void                *fArgument;
   TPosixThreadCleanUp *fNext;

public:
   TPosixThreadCleanUp(void **main, void *routine, void *arg);
   ~TPosixThreadCleanUp() { }
};

#endif
