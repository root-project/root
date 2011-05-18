// @(#)root/thread:$Id$
// Author: Bertrand Bellenot  20/10/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TWin32Mutex
#define ROOT_TWin32Mutex


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TWin32Mutex                                                          //
//                                                                      //
// This class provides an interface to the Win32 mutex routines.        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMutexImp
#include "TMutexImp.h"
#endif

#include "Windows4Root.h"

#ifdef __CINT__
struct CRITICAL_SECTION;
#endif

class TWin32Mutex : public TMutexImp {

friend class TWin32Condition;

private:
   CRITICAL_SECTION fCritSect;

public:
   TWin32Mutex(Bool_t recursive=kFALSE);
   virtual ~TWin32Mutex();

   Int_t  Lock();
   Int_t  UnLock();
   Int_t  TryLock();

   ClassDef(TWin32Mutex,0)  // Win32 mutex lock
};

#endif
