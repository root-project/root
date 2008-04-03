// @(#)root/thread:$Id$
// Author: Fons Rademakers   01/07/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPosixCondition
#define ROOT_TPosixCondition


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPosixCondition                                                      //
//                                                                      //
// This class provides an interface to the posix condition variable     //
// routines.                                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TConditionImp
#include "TConditionImp.h"
#endif

#ifndef __CINT__
#include <pthread.h>
#else
struct pthread_cond_t;
#endif

class TMutexImp;
class TPosixMutex;


class TPosixCondition : public TConditionImp {

private:
   pthread_cond_t  fCond;    // the pthread condition variable
   TPosixMutex    *fMutex;   // mutex used around Wait() and TimedWait()

public:
   TPosixCondition(TMutexImp *m);
   virtual ~TPosixCondition();

   Int_t  Wait();
   Int_t  TimedWait(ULong_t secs, ULong_t nanoSecs = 0);
   Int_t  Signal();
   Int_t  Broadcast();

   ClassDef(TPosixCondition,0)   // Posix condition variable
};

#endif
