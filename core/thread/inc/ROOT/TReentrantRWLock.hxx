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
#include <thread>
#include <unordered_map>

namespace ROOT {
   template <typename MutexT = ROOT::TSpinMutex>
   class TReentrantRWLock {
   private:
      using ReaderColl_t = std::unordered_map<std::thread::id,int>;

      std::atomic<int>             fReaders; ///<! Number of readers
      std::atomic<int>             fReaderReservation; ///<! A reader wants access
      std::atomic<int>             fWriterReservation; ///<! A writer wants access
      std::atomic<bool>            fWriter;  ///<! Is there a writer?
      MutexT                       fMutex;   ///<! RWlock internal mutex
      std::condition_variable_any  fCond;    ///<! RWlock internal condition variable
      std::thread::id              fWriterThread; ///<! Holder of the write lock
      size_t                       fWriteRecurse; ///<! Number of re-entry in the lock by the same thread.
      ReaderColl_t                 fReadersCount; ///<! Set of reader thread ids

   public:
      ////////////////////////////////////////////////////////////////////////
      /// Regular constructor.
      TReentrantRWLock() : fReaders(0), fReaderReservation(0), fWriterReservation(0), fWriter(false), fWriteRecurse(0) {}

      void ReadLock();
      void ReadUnLock();
      void WriteLock();
      void WriteUnLock();
   };
} // end of namespace ROOT

#endif
