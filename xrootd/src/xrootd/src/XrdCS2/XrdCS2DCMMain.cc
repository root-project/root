/******************************************************************************/
/*                                                                            */
/*                      X r d C S 2 D C M M a i n . c c                       */
/*                                                                            */
/* (c) 2006 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//           $Id$

const char *XrdCS2DCMMainCVSID = "$Id$";

/* This is the Castor2 end-point stager interface. The syntax is:

   XrdCS2d [options] <mdir>

   options: [-d] [-l <lfile>] [-q <lim>]

Where:
   -d     Turns on debugging mode

   -l     Specifies location of the log file. This may also come from the
          XrdOucLOGFILE environmental variable.
          By default, error messages go to standard error.

   -q     Maximum number of outstanding stagein requests. Default is 100.

<mdir>    Is the directory to be used to record persistent information.

*/

/******************************************************************************/
/*                         i n c l u d e   f i l e s                          */
/******************************************************************************/
  
#include <unistd.h>
#include <ctype.h>
#include <errno.h>
#include <iostream.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdio.h>
#include <sys/param.h>

#include "Xrd/XrdScheduler.hh"
#include "Xrd/XrdTrace.hh"

#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysLogger.hh"
#include "XrdSys/XrdSysPthread.hh"

#include "XrdCS2/XrdCS2DCM.hh"
  
/******************************************************************************/
/*                      G l o b a l   V a r i a b l e s                       */
/******************************************************************************/

       XrdCS2DCM          XrdCS2d;

       XrdScheduler       XrdSched;

       XrdSysLogger       XrdLogger;

       XrdSysError        XrdLog(&XrdLogger, "XrdCS2");

       XrdOucTrace        XrdTrace(&XrdLog);

/******************************************************************************/
/*            E x t e r n a l   T h r e a d   I n t e r f a c e s             */
/******************************************************************************/
  
void *mainEvent(void *parg)
{
   XrdCS2d.doEvents();

   return (void *)0;
}
  
void *udpEvent(void *parg)
{
   XrdCS2d.doMessages();

   return (void *)0;
}

/******************************************************************************/
/*                                  m a i n                                   */
/******************************************************************************/
  
int main(int argc, char *argv[])
{
   sigset_t myset;
   pthread_t tid;
   const char *wp;
   int retc;

// Turn off sigpipe and host a variety of others before we start any threads
//
   signal(SIGPIPE, SIG_IGN);  // Solaris optimization
   sigemptyset(&myset);
   sigaddset(&myset, SIGPIPE);
   sigaddset(&myset, SIGCHLD);
   pthread_sigmask(SIG_BLOCK, &myset, NULL);

// Set the default stack size here
//
   if (sizeof(long) > 4) XrdSysThread::setStackSize((size_t)1048576);
      else               XrdSysThread::setStackSize((size_t)786432);

// Perform configuration
//
   if (XrdCS2d.Configure(argc, argv)) wp = "completed.";
      else wp = "failed.";
   XrdLog.Emsg("Config", "XrdCS2d initialization ", wp);
   if (*wp == 'f') _exit(1);

// Start the event thread
//
   if ((retc = XrdSysThread::Run(&tid, mainEvent, (void *)0,
                            XRDSYSTHREAD_BIND, "Event handler")))
      {XrdLog.Emsg("main", retc, "create event thread"); _exit(3);}

// Start the UDP event thread
//
   if ((retc = XrdSysThread::Run(&tid, udpEvent, (void *)0,
                                     XRDSYSTHREAD_BIND, "UDP event handler")))
      {XrdLog.Emsg("main", retc, "create udp event thread"); _exit(3);}

// At this point we should be able to accept new requests.
//
   XrdCS2d.doRequests();

// We should never get here
//
   pthread_exit(0);
}
