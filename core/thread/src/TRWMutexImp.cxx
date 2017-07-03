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

////////////////////////////////////////////////////////////////////////////////
/// Take the Read Lock of the mutex.

template <typename MutexT, typename RecurseCountsT>
void TRWMutexImp<MutexT, RecurseCountsT>::ReadLock()
{
   fMutexImp.ReadLock();
}

////////////////////////////////////////////////////////////////////////////////
/// Take the Write Lock of the mutex.

template <typename MutexT, typename RecurseCountsT>
void TRWMutexImp<MutexT, RecurseCountsT>::WriteLock()
{
   fMutexImp.WriteLock();
}

////////////////////////////////////////////////////////////////////////////////
/// Release the read lock of the mutex

template <typename MutexT, typename RecurseCountsT>
void TRWMutexImp<MutexT, RecurseCountsT>::ReadUnLock()
{
   fMutexImp.ReadUnLock();
}

////////////////////////////////////////////////////////////////////////////////
/// Release the read lock of the mutex

template <typename MutexT, typename RecurseCountsT>
void TRWMutexImp<MutexT, RecurseCountsT>::WriteUnLock()
{
   fMutexImp.WriteUnLock();
}

////////////////////////////////////////////////////////////////////////////////
/// Create mutex and return pointer to it.

template <typename MutexT, typename RecurseCountsT>
TVirtualRWMutex *TRWMutexImp<MutexT, RecurseCountsT>::Factory(Bool_t /*recursive = kFALSE*/)
{
   return new TRWMutexImp();
}

template class TRWMutexImp<TMutex>;
template class TRWMutexImp<ROOT::TSpinMutex>;
template class TRWMutexImp<TMutex, ROOT::Internal::UniqueLockRecurseCount>;
template class TRWMutexImp<ROOT::TSpinMutex, ROOT::Internal::UniqueLockRecurseCount>;