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

/** \class TRWSpinLock
    \brief An implementation of a read-write lock with an internal spin lock.

This class provides an implementation of a read-write lock that uses an
internal spin lock and a condition variable to synchronize readers and
writers when necessary.

The implementation tries to make faster the scenario when readers come
and go but there is no writer. In that case, readers will not pay the
price of taking the internal spin lock.

Moreover, this RW lock tries to be fair with writers, giving them the
possibility to claim the lock and wait for only the remaining readers,
thus preventing starvation.
*/

#include "ROOT/TRWSpinLock.hxx"

using namespace ROOT;

////////////////////////////////////////////////////////////////////////////
/// Acquire the lock in read mode.
void TRWSpinLock::ReadLock()
{
  ++fReaderReservation;
   if (!fWriter) {
      // There is no writer, go freely to the critical section
      ++fReaders;
      --fReaderReservation;
   } else {
      // A writer claimed the RW lock, we will need to wait on the
      // internal lock
      --fReaderReservation;

      std::unique_lock<ROOT::TSpinMutex> lock(fMutex);

      // Wait for writers, if any
      fCond.wait(lock, [this]{ return !fWriter; });

      // This RW lock now belongs to the readers
      ++fReaders;

      lock.unlock();
   }
}

//////////////////////////////////////////////////////////////////////////
/// Release the lock in read mode.
void TRWSpinLock::ReadUnLock()
{
   --fReaders;
   if (fWriterReservation && fReaders == 0) {
      // We still need to lock here to prevent interleaving with a writer
      std::lock_guard<ROOT::TSpinMutex> lock(fMutex);

      // Make sure you wake up a writer, if any
      // Note: spurrious wakeups are okay, fReaders
      // will be checked again in WriteLock
      fCond.notify_all();
   }
}

//////////////////////////////////////////////////////////////////////////
/// Acquire the lock in write mode.
void TRWSpinLock::WriteLock()
{
   ++fWriterReservation;

   std::unique_lock<ROOT::TSpinMutex> lock(fMutex);

   // Wait for other writers, if any
   fCond.wait(lock, [this]{ return !fWriter; });

   // Claim the lock for this writer
   fWriter = true;

   // Wait until all reader reservations finish
   while(fReaderReservation) {};

   // Wait for remaining readers
   fCond.wait(lock, [this]{ return fReaders == 0; });

   --fWriterReservation;

   lock.unlock();
}

//////////////////////////////////////////////////////////////////////////
/// Release the lock in write mode.
void TRWSpinLock::WriteUnLock()
{
   // We need to lock here to prevent interleaving with a reader
   std::lock_guard<ROOT::TSpinMutex> lock(fMutex);

   fWriter = false;

   // Notify all potential readers/writers that are waiting
   fCond.notify_all();
}


TRWSpinLockReadGuard::TRWSpinLockReadGuard(TRWSpinLock &lock) : fLock(lock)
{
   fLock.ReadLock();
}

TRWSpinLockReadGuard::~TRWSpinLockReadGuard()
{
   fLock.ReadUnLock();
}

TRWSpinLockWriteGuard::TRWSpinLockWriteGuard(TRWSpinLock &lock) : fLock(lock)
{
   fLock.WriteLock();
}

TRWSpinLockWriteGuard::~TRWSpinLockWriteGuard()
{
   fLock.WriteUnLock();
}


