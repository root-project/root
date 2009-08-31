// @(#)root/thread:$Id$
// Author: Fons Rademakers   25/06/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPosixMutex
#define ROOT_TPosixMutex


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPosixMutex                                                          //
//                                                                      //
// This class provides an interface to the posix mutex routines.        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMutexImp
#include "TMutexImp.h"
#endif

#ifndef __CINT__
#include <pthread.h>
#else
struct pthread_mutex_t;
#endif

class TPosixMutex : public TMutexImp {

friend class TPosixCondition;

private:
   pthread_mutex_t  fMutex;   // the pthread mutex

public:
   TPosixMutex(Bool_t recursive=kFALSE);
   virtual ~TPosixMutex();

   Int_t  Lock();
   Int_t  UnLock();
   Int_t  TryLock();

   ClassDef(TPosixMutex,0)  // Posix mutex lock
};

#endif
