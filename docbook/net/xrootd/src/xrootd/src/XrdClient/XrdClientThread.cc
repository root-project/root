//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdClientThread                                                      //
//                                                                      //
// An user friendly thread wrapper                                      //
// Author: F.Furano (INFN, 2005)                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//           $Id$

const char *XrdClientThreadCVSID = "$Id$";

#include <pthread.h>
#include <signal.h>

#include "XrdClient/XrdClientThread.hh"

//_____________________________________________________________________________
void * XrdClientThreadDispatcher(void * arg)
{
   // This function is launched by the thread implementation. Its purpose
   // is to call the actual thread body, passing to it the original arg and
   // a pointer to the thread object which launched it.

   XrdClientThread::XrdClientThreadArgs *args = (XrdClientThread::XrdClientThreadArgs *)arg;

   args->threadobj->SetCancelDeferred();
   args->threadobj->SetCancelOn();

   if (args->threadobj->ThreadFunc)
      return args->threadobj->ThreadFunc(args->arg, args->threadobj);

   return 0;

}

//_____________________________________________________________________________
int XrdClientThread::MaskSignal(int snum, bool block)
{
   // Modify masking for signal snum: if block is true the signal is blocked,
   // else is unblocked. If snum <= 0 (default) all the allowed signals are
   // blocked / unblocked.
#ifndef WIN32
   sigset_t mask;
   int how = block ? SIG_BLOCK : SIG_UNBLOCK;
   if (snum <= 0)
      sigfillset(&mask);
      else sigaddset(&mask, snum);
   return pthread_sigmask(how, &mask, 0);
#else
   return 0;
#endif
}


