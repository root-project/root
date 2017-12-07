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
/// Reset the mutex state to unlocked. The state before resetting to unlocked is
/// returned and can be passed to `Restore()` later on. This function must only
/// be called while the mutex is locked.

template <typename MutexT, typename RecurseCountsT>
std::unique_ptr<TVirtualMutex::State> TRWMutexImp<MutexT, RecurseCountsT>::Reset()
{
   return fMutexImp.Reset();
}

////////////////////////////////////////////////////////////////////////////////
/// Restore the mutex state to the state pointed to by `state`. This function
/// must only be called while the mutex is unlocked.

template <typename MutexT, typename RecurseCountsT>
void TRWMutexImp<MutexT, RecurseCountsT>::Restore(std::unique_ptr<TVirtualMutex::State> &&state)
{
   fMutexImp.Restore(std::move(state));
}

template class TRWMutexImp<TMutex>;
template class TRWMutexImp<ROOT::TSpinMutex>;
template class TRWMutexImp<std::mutex>;
template class TRWMutexImp<TMutex, ROOT::Internal::UniqueLockRecurseCount>;
template class TRWMutexImp<ROOT::TSpinMutex, ROOT::Internal::UniqueLockRecurseCount>;

} // End of namespace ROOT