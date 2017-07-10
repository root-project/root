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

#include "ThreadLocalStorage.h"
#include "TSpinMutex.hxx"

#include <atomic>
#include <condition_variable>
#include <thread>
#include <unordered_map>

namespace ROOT {
namespace Internal {
struct UniqueLockRecurseCount {
   struct LocalCounts {
      int fReadersCount = 0;
      bool fIsWriter = false;
   };
   size_t fWriteRecurse = 0; ///<! Number of re-entry in the lock by the same thread.

   UniqueLockRecurseCount();

   using local_t = LocalCounts*;

   local_t GetLocal() {
      TTHREAD_TLS_DECL(LocalCounts, gLocal);
      return &gLocal;
   }

   void IncrementReadCount(local_t &local) { ++(local->fReadersCount); }

   template <typename MutexT>
   void IncrementReadCount(local_t &local, MutexT &) { IncrementReadCount(local); }

   void DecrementReadCount(local_t &local) { --(local->fReadersCount); }

   template <typename MutexT>
   void DecrementReadCount(local_t &local, MutexT &) { DecrementReadCount(local); }

   bool IsNotCurrentWriter(local_t &local) { return !local->fIsWriter; }

   void SetIsWriter(local_t &local)
   {
      // if (fWriteRecurse == std::numeric_limits<decltype(fWriteRecurse)>::max()) {
      //    ::Fatal("TRWSpinLock::WriteLock", "Too many recursions in TRWSpinLock!");
      // }
      ++fWriteRecurse;
      local->fIsWriter = true;
   }

   void DecrementWriteCount() { --fWriteRecurse; }

   void ResetIsWriter(local_t &local) { local->fIsWriter = false; }

   size_t GetLocalReadersCount(local_t &local) { return local->fReadersCount; }
};

struct RecurseCounts {
   using ReaderColl_t = std::unordered_map<std::thread::id, int>;
   size_t fWriteRecurse; ///<! Number of re-entry in the lock by the same thread.

   std::thread::id fWriterThread; ///<! Holder of the write lock
   ReaderColl_t fReadersCount;    ///<! Set of reader thread ids

   using local_t = std::thread::id;

   local_t GetLocal() { return std::this_thread::get_id(); }

   void IncrementReadCount(local_t &local) { ++(fReadersCount[local]); }

   template <typename MutexT>
   void IncrementReadCount(local_t &local, MutexT &mutex)
   {
      std::unique_lock<MutexT> lock(mutex);
      IncrementReadCount(local);
   }

   void DecrementReadCount(local_t &local) { --(fReadersCount[local]); }

   template <typename MutexT>
   void DecrementReadCount(local_t &local, MutexT &mutex)
   {
      std::unique_lock<MutexT> lock(mutex);
      DecrementReadCount(local);
   }

   bool IsNotCurrentWriter(local_t &local) { return fWriterThread != local; }

   void SetIsWriter(local_t &local)
   {
      // if (fWriteRecurse == std::numeric_limits<decltype(fWriteRecurse)>::max()) {
      //    ::Fatal("TRWSpinLock::WriteLock", "Too many recursions in TRWSpinLock!");
      // }
      ++fWriteRecurse;
      fWriterThread = local;
   }

   void DecrementWriteCount() { --fWriteRecurse; }

   void ResetIsWriter(local_t & /* local */) { fWriterThread = std::thread::id(); }

   size_t GetLocalReadersCount(local_t &local) { return fReadersCount[local]; }


};
} // Internal

template <typename MutexT = ROOT::TSpinMutex, typename RecurseCountsT = Internal::RecurseCounts>
class TReentrantRWLock {
private:

   std::atomic<int> fReaders;           ///<! Number of readers
   std::atomic<int> fReaderReservation; ///<! A reader wants access
   std::atomic<int> fWriterReservation; ///<! A writer wants access
   std::atomic<bool> fWriter;           ///<! Is there a writer?
   MutexT fMutex;                       ///<! RWlock internal mutex
   std::condition_variable_any fCond;   ///<! RWlock internal condition variable

   RecurseCountsT fRecurseCounts;        ///<! Trackers for re-entry in the lock by the same thread.

   // size_t fWriteRecurse;                ///<! Number of re-entry in the lock by the same thread.

   // std::thread::id fWriterThread; ///<! Holder of the write lock
   // ReaderColl_t fReadersCount;    ///<! Set of reader thread ids

public:
   ////////////////////////////////////////////////////////////////////////
   /// Regular constructor.
   TReentrantRWLock() : fReaders(0), fReaderReservation(0), fWriterReservation(0), fWriter(false) {}

   void ReadLock();
   void ReadUnLock();
   void WriteLock();
   void WriteUnLock();

   };
} // end of namespace ROOT

#endif
