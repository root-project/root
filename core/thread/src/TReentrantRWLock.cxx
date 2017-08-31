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

/** \class TReentrantRWLock
    \brief An implementation of a reentrant read-write lock with a
           configurable internal mutex/lock (default Spin Lock).

This class provides an implementation of a rreentrant ead-write lock
that uses an internal lock and a condition variable to synchronize
readers and writers when necessary.

The implementation allows a single reader to take the write lock without
releasing the reader lock.  It also allows the writer to take a read lock.
In other word, the lock is re-entrant for both reading and writing.

The implementation tries to make faster the scenario when readers come
and go but there is no writer. In that case, readers will not pay the
price of taking the internal spin lock.

Moreover, this RW lock tries to be fair with writers, giving them the
possibility to claim the lock and wait for only the remaining readers,
thus preventing starvation.
*/

#include "ROOT/TReentrantRWLock.hxx"
#include "ROOT/TSpinMutex.hxx"
#include "TMutex.h"
#include "TError.h"

using namespace ROOT;

Internal::UniqueLockRecurseCount::UniqueLockRecurseCount()
{
   static bool singleton = false;
   if (singleton) {
      ::Fatal("UniqueLockRecurseCount Ctor", "Only one TReentrantRWLock using a UniqueLockRecurseCount is allowed.");
   }
   singleton = true;
}


////////////////////////////////////////////////////////////////////////////
/// Acquire the lock in read mode.
template <typename MutexT, typename RecurseCountsT>
void TReentrantRWLock<MutexT, RecurseCountsT>::ReadLock()
{
   ++fReaderReservation;

   // if (fReaders == std::numeric_limits<decltype(fReaders)>::max()) {
   //    ::Fatal("TRWSpinLock::WriteLock", "Too many recursions in TRWSpinLock!");
   // }

   auto local = fRecurseCounts.GetLocal();

   if (!fWriter) {
      // There is no writer, go freely to the critical section
      ++fReaders;
      --fReaderReservation;

      fRecurseCounts.IncrementReadCount(local, fMutex);
   } else {
      // A writer claimed the RW lock, we will need to wait on the
      // internal lock
      --fReaderReservation;

      std::unique_lock<MutexT> lock(fMutex);

      // Wait for writers, if any
      if (fWriter && fRecurseCounts.IsNotCurrentWriter(local)) fCond.wait(lock, [this] { return !fWriter; });

      fRecurseCounts.IncrementReadCount(local);

      // This RW lock now belongs to the readers
      ++fReaders;

      lock.unlock();
   }
}

//////////////////////////////////////////////////////////////////////////
/// Release the lock in read mode.
template <typename MutexT, typename RecurseCountsT>
void TReentrantRWLock<MutexT, RecurseCountsT>::ReadUnLock()
{
   auto local = fRecurseCounts.GetLocal();

   --fReaders;
   if (fWriterReservation && fReaders == 0) {
      // We still need to lock here to prevent interleaving with a writer
      std::lock_guard<MutexT> lock(fMutex);

      fRecurseCounts.DecrementReadCount(local);

      // Make sure you wake up a writer, if any
      // Note: spurrious wakeups are okay, fReaders
      // will be checked again in WriteLock
      fCond.notify_all();
   } else {

      fRecurseCounts.DecrementReadCount(local, fMutex);
   }
}

//////////////////////////////////////////////////////////////////////////
/// Acquire the lock in write mode.
template <typename MutexT, typename RecurseCountsT>
void TReentrantRWLock<MutexT, RecurseCountsT>::WriteLock()
{
   ++fWriterReservation;

   std::unique_lock<MutexT> lock(fMutex);

   auto local = fRecurseCounts.GetLocal();

   // Release this thread's reader lock(s)
   auto readerCount = fRecurseCounts.GetLocalReadersCount(local);

   fReaders -= readerCount;

   // Wait for other writers, if any
   if (fWriter && fRecurseCounts.IsNotCurrentWriter(local)) {
      if (readerCount && fReaders == 0) {
         // we decrease fReaders to zero, let's wake up the
         // other writer.
         fCond.notify_all();
      }
      fCond.wait(lock, [this] { return !fWriter; });
   }

   // Claim the lock for this writer
   fWriter = true;
   fRecurseCounts.SetIsWriter(local);

   // Wait until all reader reservations finish
   while (fReaderReservation) {
   };

   // Wait for remaining readers
   fCond.wait(lock, [this] { return fReaders == 0; });

   // Restore this thread's reader lock(s)
   fReaders += readerCount;

   --fWriterReservation;

   lock.unlock();
}

//////////////////////////////////////////////////////////////////////////
/// Release the lock in write mode.
template <typename MutexT, typename RecurseCountsT>
void TReentrantRWLock<MutexT, RecurseCountsT>::WriteUnLock()
{
   // We need to lock here to prevent interleaving with a reader
   std::lock_guard<MutexT> lock(fMutex);

   if (!fWriter || fRecurseCounts.fWriteRecurse == 0) {
      Error("TRWSpinLock::WriteUnLock", "Write lock already released for %p", this);
      return;
   }

   fRecurseCounts.DecrementWriteCount();

   if (!fRecurseCounts.fWriteRecurse) {
      fWriter = false;

      auto local = fRecurseCounts.GetLocal();
      fRecurseCounts.ResetIsWriter(local);

      // Notify all potential readers/writers that are waiting
      fCond.notify_all();
   }
}

namespace {
template <typename MutexT, typename RecurseCountsT>
struct TReentrantRWLockState: public TVirtualMutex::State {
   RecurseCountsT fRecurseCounts;
};
}

template <typename MutexT, typename RecurseCountsT>
std::unique_ptr<TVirtualMutex::State> TReentrantRWLock<MutexT, RecurseCountsT>::Reset()
{
   std::unique_ptr<TReentrantRWLockState<MutexT, RecurseCountsT>> pState;
   // Do something.
   ::Fatal("Reset()", "Not implemented, contact pcanal@fnal.gov");
   return std::move(pState);
}

template <typename MutexT, typename RecurseCountsT>
void TReentrantRWLock<MutexT, RecurseCountsT>::Restore(std::unique_ptr<TVirtualMutex::State> &&)
{
   // Do something.
   ::Fatal("Restore()", "Not implemented, contact pcanal@fnal.gov");
}

namespace ROOT {
template class TReentrantRWLock<ROOT::TSpinMutex, ROOT::Internal::RecurseCounts>;
template class TReentrantRWLock<TMutex, ROOT::Internal::RecurseCounts>;

template class TReentrantRWLock<ROOT::TSpinMutex, ROOT::Internal::UniqueLockRecurseCount>;
template class TReentrantRWLock<TMutex, ROOT::Internal::UniqueLockRecurseCount>;
}
