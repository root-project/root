// @(#)root/thread:$Name:  $:$Id: TPosixThread.cxx,v 1.1.1.1 2000/05/16 17:00:48 rdm Exp $
// Author: Fons Rademakers   02/07/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPosixThread                                                         //
//                                                                      //
// This class provides an interface to the posix thread routines.       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TPosixThread.h"


ClassImp(TPosixThread)


//______________________________________________________________________________
Int_t TPosixThread::Run(TThread *th)
{
   int det;
   pthread_t id;
   pthread_attr_t *attr = new pthread_attr_t;

   pthread_attr_init(attr);

   // Set detach state
   det = (th->fDetached) ? PTHREAD_CREATE_DETACHED : PTHREAD_CREATE_JOINABLE;

   pthread_attr_setdetachstate(attr, det);
   int ierr = pthread_create(&id, attr, &TThread::Fun,th);
   th->fId = id;
   if (attr) pthread_attr_destroy(attr);
   return ierr;
}

//______________________________________________________________________________
Int_t TPosixThread::Join(Long_t jid, void **ret)
{
   return pthread_join((pthread_t) jid, ret);
}

//______________________________________________________________________________
Int_t TPosixThread::Exit(void *ret)
{
   pthread_exit(ret);
   return 0;
}

//______________________________________________________________________________
Int_t TPosixThread::Kill(TThread *th)
{
   return pthread_cancel((pthread_t) th->fId);
}

//______________________________________________________________________________
Int_t TPosixThread::SetCancelOff()
{
   return pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, 0);
}

//______________________________________________________________________________
Int_t TPosixThread::SetCancelOn()
{
   return pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, 0);
}

//______________________________________________________________________________
Int_t TPosixThread::SetCancelAsynchronous()
{
   return pthread_setcanceltype (PTHREAD_CANCEL_ASYNCHRONOUS, 0);
}

//______________________________________________________________________________
Int_t TPosixThread::SetCancelDeferred()
{
   return pthread_setcanceltype (PTHREAD_CANCEL_DEFERRED, 0);
}

//______________________________________________________________________________
Int_t TPosixThread::CancelPoint()
{
   int istate;
   pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, &istate );
   pthread_testcancel();
   pthread_setcancelstate(istate, 0);

   return 0;
}

//______________________________________________________________________________
Int_t TPosixThread::CleanUpPush(void **main, void *free,void *arg)
{
   // pthread_cleanup_push(free, arg);
   if (!free) fprintf(stderr, "CleanUpPush ***ERROR*** Routine=NULL\n");
   new TPosixThreadCleanUp(main,free,arg);
   return 0;
}

//______________________________________________________________________________
Int_t TPosixThread::CleanUpPop(void **main,Int_t exe)
{
   //  pthread_cleanup_pop(exe); // happy pthread future

   if (!*main) return 1;
   TPosixThreadCleanUp *l = (TPosixThreadCleanUp*)(*main);
   if (!l->fRoutine) fprintf(stderr,"CleanUpPop ***ERROR*** Routine=NULL\n");
   if (exe && l->fRoutine) ((void (*)(void*))(l->fRoutine))(l->fArgument);
   *main = l->fNext;  delete l;
   return 0;
}

//______________________________________________________________________________
Int_t TPosixThread::CleanUp(void **main)
{
   fprintf(stderr," CleanUp %lx\n",(Long_t)*main);
   while(!CleanUpPop(main,1)) { }
   return 0;
}

//______________________________________________________________________________
Long_t TPosixThread::SelfId()
{
   return pthread_self();
}

//______________________________________________________________________________
Int_t TPosixThread::Sleep(ULong_t secs, ULong_t nanos)
{
   return sleep(secs);
}

//______________________________________________________________________________
Int_t TPosixThread::GetTime(ULong_t *absSec, ULong_t *absNanoSec)
{
   return 0;
}


//   Clean Up section. PTHREAD implementations of cleanup after cancel are
//   too different and often too bad. Temporary I invent my own bicycle.
//                                                              V.Perev.

//______________________________________________________________________________
TPosixThreadCleanUp::TPosixThreadCleanUp(void **main, void *routine, void *arg)
{
   fNext = (TPosixThreadCleanUp*)*main;
   fRoutine = routine; fArgument = arg;
   *main  = this;
}
