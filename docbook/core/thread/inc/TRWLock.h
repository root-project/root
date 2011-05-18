// @(#)root/thread:$Id$
// Author: Fons Rademakers   04/01/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRWLock
#define ROOT_TRWLock


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRWLock                                                              //
//                                                                      //
// This class implements a reader/writer lock. A rwlock allows          //
// a resource to be accessed by multiple reader threads but only        //
// one writer thread.                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TMutex
#include "TMutex.h"
#endif
#ifndef ROOT_TCondition
#include "TCondition.h"
#endif


class TRWLock : public TObject {

private:
   Int_t        fReaders;   // number of readers
   Int_t        fWriters;   // number of writers
   TMutex       fMutex;     // rwlock mutex
   TCondition   fLockFree;  // rwlock condition variable

   TRWLock(const TRWLock &);           // not implemented
   TRWLock& operator=(const TRWLock&); // not implemented

public:
   TRWLock();
   virtual ~TRWLock() { }

   Int_t  ReadLock();
   Int_t  ReadUnLock();
   Int_t  WriteLock();
   Int_t  WriteUnLock();

   ClassDef(TRWLock,0)  // Reader/writer lock
};

#endif
