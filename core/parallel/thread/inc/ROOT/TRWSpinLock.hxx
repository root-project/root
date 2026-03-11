// @(#)root/thread:$Id$
// Authors: Enric Tejedor CERN  12/09/2016
//          Philippe Canal FNAL 12/09/2016

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRWSpinLock
#define ROOT_TRWSpinLock

#include "TSpinMutex.hxx"

#include <atomic>
#include <condition_variable>

namespace ROOT {
class TRWSpinLock {
private:
   std::atomic<int> fReaders;           ///<! Number of readers
   std::atomic<int> fReaderReservation; ///<! A reader wants access
   std::atomic<int> fWriterReservation; ///<! A writer wants access
   std::atomic<bool> fWriter;           ///<! Is there a writer?
   ROOT::TSpinMutex fMutex;             ///<! RWlock internal mutex
   std::condition_variable_any fCond;   ///<! RWlock internal condition variable

public:
   ////////////////////////////////////////////////////////////////////////
   /// Regular constructor.
   TRWSpinLock() : fReaders(0), fReaderReservation(0), fWriterReservation(0), fWriter(false) {}

   void ReadLock();
   void ReadUnLock();
   void WriteLock();
   void WriteUnLock();
};

class TRWSpinLockReadGuard {
private:
   TRWSpinLock &fLock;

public:
   TRWSpinLockReadGuard(TRWSpinLock &lock);
   ~TRWSpinLockReadGuard();
};

class TRWSpinLockWriteGuard {
private:
   TRWSpinLock &fLock;

public:
   TRWSpinLockWriteGuard(TRWSpinLock &lock);
   ~TRWSpinLockWriteGuard();
};

} // end of namespace ROOT

#endif
