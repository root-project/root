// @(#)root/thread:$Id$
// Author: Fons Rademakers   26/06/97

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRWMutexImp                                                          //
//                                                                      //
// This class implements the TVirtualRWMutex interface,                 //
// based on TRWSpinLock.                                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TRWMutexImp.h"
#include "ROOT/TSpinMutex.hxx"
#include "TMutex.h"

namespace ROOT {

////////////////////////////////////////////////////////////////////////////////
/// Take the Read Lock of the mutex.

template <typename MutexT, typename RecurseCountsT>
TVirtualRWMutex::Hint_t *TRWMutexImp<MutexT, RecurseCountsT>::ReadLock()
{
   return fMutexImp.ReadLock();
}

////////////////////////////////////////////////////////////////////////////////
/// Take the Write Lock of the mutex.

template <typename MutexT, typename RecurseCountsT>
TVirtualRWMutex::Hint_t *TRWMutexImp<MutexT, RecurseCountsT>::WriteLock()
{
   return fMutexImp.WriteLock();
}

////////////////////////////////////////////////////////////////////////////////
/// Release the read lock of the mutex

template <typename MutexT, typename RecurseCountsT>
void TRWMutexImp<MutexT, RecurseCountsT>::ReadUnLock(TVirtualRWMutex::Hint_t *hint)
{
   fMutexImp.ReadUnLock(hint);
}

////////////////////////////////////////////////////////////////////////////////
/// Release the read lock of the mutex

template <typename MutexT, typename RecurseCountsT>
void TRWMutexImp<MutexT, RecurseCountsT>::WriteUnLock(TVirtualRWMutex::Hint_t *hint)
{
   fMutexImp.WriteUnLock(hint);
}

////////////////////////////////////////////////////////////////////////////////
/// Create mutex and return pointer to it.

template <typename MutexT, typename RecurseCountsT>
TVirtualRWMutex *TRWMutexImp<MutexT, RecurseCountsT>::Factory(Bool_t /*recursive = kFALSE*/)
{
   return new TRWMutexImp();
}

////////////////////////////////////////////////////////////////////////////////
/// Restore the mutex state to `state`. This function must only be called while
/// the mutex is locked. Returns the DeltaState between now and the resulting
/// state (i.e. lock count before state), such that the difference can be
/// re-applied using `Apply()`.
/// In pseudo-code:
///     delta = current_lock_count - earlierState.lock_count;
///     current_lock_count -= delta;
///     return delta;

template <typename MutexT, typename RecurseCountsT>
std::unique_ptr<TVirtualRWMutex::StateDelta>
TRWMutexImp<MutexT, RecurseCountsT>::Rewind(const TVirtualRWMutex::State &earlierState)
{
   return fMutexImp.Rewind(earlierState);
}

////////////////////////////////////////////////////////////////////////////////
/// Apply the mutex state delta.
/// In pseudo-code:
///     current_lock_count += delta;

template <typename MutexT, typename RecurseCountsT>
void TRWMutexImp<MutexT, RecurseCountsT>::Apply(std::unique_ptr<TVirtualRWMutex::StateDelta> &&delta)
{
   fMutexImp.Apply(std::move(delta));
}

////////////////////////////////////////////////////////////////////////////////
/// Get the mutex state *before* the current lock was taken. This function must
/// only be called while the mutex is locked.

template <typename MutexT, typename RecurseCountsT>
std::unique_ptr<TVirtualRWMutex::State>
TRWMutexImp<MutexT, RecurseCountsT>::GetStateBefore()
{
   return fMutexImp.GetStateBefore();
}

template class TRWMutexImp<TMutex>;
template class TRWMutexImp<ROOT::TSpinMutex>;
template class TRWMutexImp<std::mutex>;
template class TRWMutexImp<TMutex, ROOT::Internal::UniqueLockRecurseCount>;
template class TRWMutexImp<ROOT::TSpinMutex, ROOT::Internal::UniqueLockRecurseCount>;

#ifdef R__HAS_TBB
template class TRWMutexImp<std::mutex, ROOT::Internal::RecurseCountsTBB>;
template class TRWMutexImp<std::mutex, ROOT::Internal::RecurseCountsTBBUnique>;
#endif

} // End of namespace ROOT
