// @(#)root/thread:$Name:  $:$Id: TWin32Mutex.h,v 1.1.1.1 2000/05/16 17:00:48 rdm Exp $
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

class TWin32Mutex : public TMutexImp {

friend class TWin32Condition;
   
private:

   HANDLE fHMutex;

public:
   TWin32Mutex();
   virtual ~TWin32Mutex();

   Int_t  Lock();
   Int_t  UnLock();
   Int_t  TryLock();

   ClassDef(TWin32Mutex,0)  // Win32 mutex lock
};

#endif
