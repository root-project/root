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

template <typename MutexT>
void TRWMutexImp<MutexT>::ReadLock()
{
   fMutexImp.ReadLock();
}

////////////////////////////////////////////////////////////////////////////////
/// Take the Write Lock of the mutex.

template <typename MutexT>
void TRWMutexImp<MutexT>::WriteLock()
{
   fMutexImp.WriteLock();
}

////////////////////////////////////////////////////////////////////////////////
/// Release the read lock of the mutex

template <typename MutexT>
void TRWMutexImp<MutexT>::ReadUnLock()
{
   fMutexImp.ReadUnLock();
}

////////////////////////////////////////////////////////////////////////////////
/// Release the read lock of the mutex

template <typename MutexT>
void TRWMutexImp<MutexT>::WriteUnLock()
{
   fMutexImp.WriteUnLock();
}

////////////////////////////////////////////////////////////////////////////////
/// Create mutex and return pointer to it.

template <typename MutexT>
TVirtualRWMutex *TRWMutexImp<MutexT>::Factory(Bool_t /*recursive = kFALSE*/)
{
   return new TRWMutexImp();
}

template class TRWMutexImp<TMutex>;
template class TRWMutexImp<ROOT::TSpinMutex>;