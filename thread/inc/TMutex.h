// @(#)root/thread:$Name:  $:$Id: TMutex.h,v 1.1.1.1 2000/05/16 17:00:48 rdm Exp $
// Author: Fons Rademakers   26/06/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMutex
#define ROOT_TMutex


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMutex                                                               //
//                                                                      //
// This class implements mutex locks. A mutex is a mutual exclusive     //
// lock. The actual work is done via the TMutexImp class (either        //
// TPosixMutex, TSolarisMutex or TNTMutex).                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TVirtualMutex
#include "TVirtualMutex.h"
#endif
#ifndef ROOT_TThread
#include "TThread.h"
#endif
#ifndef ROOT_TMutexImp
#include "TMutexImp.h"
#endif


class TMutex : public TVirtualMutex {

friend class TCondition;

private:
   TMutexImp  *fMutexImp;   // pointer to mutex implementation
   UInt_t      fId;         // id of thread which locked mutex
   Int_t       fRef;        // reference count in case of recursive locking by same thread

public:
   TMutex(Bool_t recursive = kFALSE);
   virtual ~TMutex() { delete fMutexImp; }

   Int_t  Lock() ;
   Int_t  TryLock() ;
   Int_t  UnLock() ;
   Int_t  CleanUp();

   ClassDef(TMutex,0)  // Mutex lock class
};

#endif
