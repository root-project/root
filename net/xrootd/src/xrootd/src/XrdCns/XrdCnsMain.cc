/******************************************************************************/
/*                                                                            */
/*                         X r d G n s M a i n . c c                          */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//           $Id$

const char *XrdGnsMainCVSID = "$Id$";

/* This is the Global Name Space interface. The syntax is:

   XrdGnsd [options] xroot://<host[:port]>/[/prefix]

   options: [-a <apath>] [-d] [-l <lfile>] [-q <lim>]

Where:
   -a     The admin path where the event log is placed and where named
          sockets are created. If not specified, the admin path comes from
          the XRDADMINPATH env variable. Otherwise, /tmp is used.

   -d     Turns on debugging mode

   -l     Specifies location of the log file. This may also come from the
          XrdOucLOGFILE environmental variable.
          By default, error messages go to standard error.

   -q     Maximum number of event entries. Default is 1023.

<host>    Is the hostname of the server managing the cluster name space.

*/

/******************************************************************************/
/*                         i n c l u d e   f i l e s                          */
/******************************************************************************/
  
#include <unistd.h>
#include <ctype.h>
#include <errno.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdio.h>
#include <sys/param.h>

#include "Xrd/XrdScheduler.hh"
#include "Xrd/XrdTrace.hh"

#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSys/XrdSysLogger.hh"
#include "XrdSys/XrdSysPthread.hh"

#include "XrdCns/XrdCnsDaemon.hh"
  
/******************************************************************************/
/*                      G l o b a l   V a r i a b l e s                       */
/******************************************************************************/

       XrdCnsDaemon       XrdCnsd;

       XrdScheduler       XrdSched;

       XrdSysLogger       XrdLogger;

       XrdSysError        XrdLog(&XrdLogger, "XrdCns");

       XrdOucTrace        XrdTrace(&XrdLog);

/******************************************************************************/
/*                                  m a i n                                   */
/******************************************************************************/
  
int main(int argc, char *argv[])
{
   sigset_t myset;
   const char *wp;

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
   if (XrdCnsd.Configure(argc, argv)) wp = "completed.";
      else wp = "failed.";
   XrdLog.Emsg("Config", "Cnsd initialization ", wp);
   if (*wp == 'f') _exit(1);

// At this point we should be able to accept new requests.
//
   XrdCnsd.doRequests();

// We should never get here
//
   pthread_exit(0);
}
