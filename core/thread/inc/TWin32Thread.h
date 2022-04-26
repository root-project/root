// @(#)root/thread:$Id$
// Author: Bertrand Bellenot  20/10/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TWin32Thread
#define ROOT_TWin32Thread


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TWin32Thread                                                         //
//                                                                      //
// This class provides an interface to the Win32 thread routines.       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TThreadImp.h"

#include "Windows4Root.h"

class TWin32ThreadCleanUp;

class TWin32Thread : public TThreadImp {

public:
   TWin32Thread() { }
   ~TWin32Thread() { }

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

   Int_t  Exit(void *ret);

   ClassDefOverride(TWin32Thread,0)  // TWin32Thread class
};


class TWin32ThreadCleanUp {

friend class TWin32Thread;

private:
   void                *fRoutine;
   void                *fArgument;
   TWin32ThreadCleanUp *fNext;

public:
   TWin32ThreadCleanUp(void **main,void *routine,void *arg);
   ~TWin32ThreadCleanUp() { }
};

#endif
