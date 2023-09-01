// @(#)root/thread:$Id$
// Author: Fons Rademakers   01/07/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPosixCondition                                                      //
//                                                                      //
// This class provides an interface to the posix condition variable     //
// routines.                                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TPosixCondition.h"
#include "TPosixMutex.h"
#include "PosixThreadInc.h"

#include <errno.h>


ClassImp(TPosixCondition);

////////////////////////////////////////////////////////////////////////////////
/// Create Condition variable. Ctor must be given a pointer to an
/// existing mutex. The condition variable is then linked to the mutex,
/// so that there is an implicit unlock and lock around Wait() and
/// TimedWait().

TPosixCondition::TPosixCondition(TMutexImp *m)
{
   fMutex = (TPosixMutex *) m;

   int rc = pthread_cond_init(&fCond, 0);

   if (rc)
      SysError("TPosixCondition", "pthread_cond_init error");
}

////////////////////////////////////////////////////////////////////////////////
/// TCondition dtor.

TPosixCondition::~TPosixCondition()
{
   int rc = pthread_cond_destroy(&fCond);

   if (rc)
      SysError("~TPosixCondition", "pthread_cond_destroy error");
}

////////////////////////////////////////////////////////////////////////////////
/// Wait for the condition variable to be signalled. The mutex is
/// implicitely released before waiting and locked again after waking up.
/// If Wait() is called by multiple threads, a signal may wake up more
/// than one thread. See POSIX threads documentation for details.

Int_t TPosixCondition::Wait()
{
   return pthread_cond_wait(&fCond, &(fMutex->fMutex));
}

////////////////////////////////////////////////////////////////////////////////
/// TimedWait() is given an absolute time to wait until. To wait for a
/// relative time from now, use TThread::GetTime(). See POSIX threads
/// documentation for why absolute times are better than relative.
/// Returns 0 if successfully signalled, 1 if time expired.

Int_t TPosixCondition::TimedWait(ULong_t secs, ULong_t nanoSecs)
{
   timespec rqts = { (Long_t)secs, (Long_t)nanoSecs };

   int rc = pthread_cond_timedwait(&fCond, &(fMutex->fMutex), &rqts);

   if (rc == ETIMEDOUT)
      rc = 1;

   return rc;
}

////////////////////////////////////////////////////////////////////////////////
/// If one or more threads have called Wait(), Signal() wakes up at least
/// one of them, possibly more. See POSIX threads documentation for details.

Int_t TPosixCondition::Signal()
{
   return pthread_cond_signal(&fCond);
}


////////////////////////////////////////////////////////////////////////////////
/// Broadcast is like signal but wakes all threads which have called Wait().

Int_t TPosixCondition::Broadcast()
{
   return pthread_cond_broadcast(&fCond);
}
