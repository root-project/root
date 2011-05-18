// @(#)root/base:$Id$
// Author: Fons Rademakers   14/07/2002

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TVirtualMutex
#define ROOT_TVirtualMutex


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualMutex                                                        //
//                                                                      //
// This class implements a mutex interface. The actual work is done via //
// TMutex which is available as soon as the thread library is loaded.   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TVirtualMutex;

// Global mutex set in TThread::Init
R__EXTERN TVirtualMutex *gGlobalMutex;

class TVirtualMutex : public TObject {

public:
   TVirtualMutex(Bool_t /* recursive */ = kFALSE) { }
   virtual ~TVirtualMutex() { }

   virtual Int_t Lock() = 0;
   virtual Int_t TryLock() = 0;
   virtual Int_t UnLock() = 0;
   virtual Int_t CleanUp() = 0;
   Int_t Acquire() { return Lock(); }
   Int_t Release() { return UnLock(); }

   virtual TVirtualMutex *Factory(Bool_t /*recursive*/ = kFALSE) = 0;

   ClassDef(TVirtualMutex,0)  // Virtual mutex lock class
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TLockGuard                                                           //
//                                                                      //
// This class provides mutex resource management in a guaranteed and    //
// exception safe way. Use like this:                                   //
// {                                                                    //
//    TLockGuard guard(mutex);                                          //
//    ... // do something                                               //
// }                                                                    //
// when guard goes out of scope the mutex is unlocked in the TLockGuard //
// destructor. The exception mechanism takes care of calling the dtors  //
// of local objects so it is exception safe.                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TLockGuard {

private:
   TVirtualMutex *fMutex;

   TLockGuard(const TLockGuard&);             // not implemented
   TLockGuard& operator=(const TLockGuard&);  // not implemented

public:
   TLockGuard(TVirtualMutex *mutex)
     : fMutex(mutex) { if (fMutex) fMutex->Lock(); }
   virtual ~TLockGuard() { if (fMutex) fMutex->UnLock(); }

   ClassDef(TLockGuard,0)  // Exception safe locking/unlocking of mutex
};

// Zero overhead macros in case not compiled with thread support
#if defined (_REENTRANT) || defined (WIN32)
#define R__LOCKGUARD(mutex) TLockGuard _R__UNIQUE_(R__guard)(mutex)
#define R__LOCKGUARD2(mutex)                             \
   if (gGlobalMutex && !mutex) {                         \
      gGlobalMutex->Lock();                              \
      if (!mutex)                                        \
         mutex = gGlobalMutex->Factory(kTRUE);           \
      gGlobalMutex->UnLock();                            \
   }                                                     \
   R__LOCKGUARD(mutex)
#else
#define R__LOCKGUARD(mutex)  if (mutex) { }
#define R__LOCKGUARD2(mutex) if (mutex) { }
#endif

#endif
