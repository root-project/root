// @(#)root/base:$Name:  $:$Id: TVirtualMutex.h,v 1.1 2002/02/14 16:12:52 rdm Exp $
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


class TVirtualMutex : public TObject {

public:
   TVirtualMutex(Bool_t /* recursive */ = kFALSE) { }
   virtual ~TVirtualMutex() { }

   virtual Int_t Lock() { return 0; }
   virtual Int_t TryLock() { return 0; }
   virtual Int_t UnLock() { return 0; }
   virtual Int_t CleanUp() { return 0; }
   Int_t Acquire() { return Lock(); }
   Int_t Release() { return UnLock(); }

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

public:
   TLockGuard(TVirtualMutex *mutex)
                         { fMutex = mutex; if (fMutex) fMutex->Lock(); }
   virtual ~TLockGuard() { if (fMutex) fMutex->UnLock(); }

   ClassDef(TLockGuard,0)  // Exception safe locking/unlocking of mutex
};


R__EXTERN TVirtualMutex *gContainerMutex;
R__EXTERN TVirtualMutex *gCINTMutex;

// Zero overhead macros in case not compiled with thread support
#ifdef _REENTRANT
#define R__LOCKGUARD(mutex) TLockGuard R__guard(mutex)
#else
#define R__LOCKGUARD
#endif

#endif
