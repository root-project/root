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

const char *XrdCnsMainCVSID = "$Id$";

/* This is the Cluster Name Space interface. The syntax is:

   XrdCnsd [options] [[xroot://]<host[:port]>[/[/prefix]] . . .]

   options: [-a <apath>] [-b <bpath>] [-B <bpath>] [-c] [-d] [-e <epath>]

            [-i <tspec>] [-I <tspec>] [-l <lfile>] [-p <port>] [-q <lim>] [-R]
Where:
   -a     The admin path where the event log is placed and where named
          sockets are created. If not specified, the admin path comes from
          the XRDADMINPATH env variable. Otherwise, /tmp is used. This option
          is valid only for command line use.

   -b     The archive (i.e., backup) path to use. If not specified, no backup
          is done. Data is  written to "<bpath>/cns/<host>". By default, the
          backups are written to each redirector. By prefixing <bpath> with
          <host[:port]:> then backups are written to the specified host:port.
          If <port> is omitted the the specified or default -p value is used.
          Note that this backup can be used to create an inventory file.

   -B     Same as -b *except* that only the inventory is maintained (i.e., no
          composite name space is created).

   -c     Specified the config file name. By defaults this comes from the envar
          XRDCONFIGFN set by the underlying xrootd. Note that if -R is specified
          then -c must be specified as there is no underlying xrootd.

   -d     Turns on debugging mode. Valid only via command line.

   -D     Sets the client library debug value. Specify an number -2 to 3.

   -e     The directory where the event logs are to be written. By default
          this is whatever <apath> becomes.

   -i     The interval between forced log archiving. Default is 20m (minutes).

   -I     The time interval, in seconds, between checks for the inventory file.

   -l     Specifies location of the log file. This may also come from the
          XRDLOGDIR environmental variable. Valid only via command line.
          By default, error messages go to standard error.

   -L     The local root (ignored except when -R specified).

   -N     The name2name library and parms (ignored except when -R specified).

   -p     The default port number to use for the xrootd that can be used to
          create/maintain the name space as well as hold archived logs. The
          number 1095 is used bt default.

   -q     Maximum number of log records before the log is closed and archived.
          Specify 1 to 1024. The default if 512.

   -R     Run is stand-alone mode and recreate the name space and, perhaps,
          the inventory file.

<host>    Is the hostname of the server managing the cluster name space. You
          may specify more than one if they are replicated. The default is to
          use the hosts specified via the "all.manager" directive.
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

#include "Xrd/XrdTrace.hh"

#include "XrdCns/XrdCnsConfig.hh"
#include "XrdCns/XrdCnsDaemon.hh"

#include "XrdOuc/XrdOucStream.hh"

#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSys/XrdSysLogger.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdSys/XrdSysTimer.hh"
  
/******************************************************************************/
/*                      G l o b a l   V a r i a b l e s                       */
/******************************************************************************/

namespace XrdCns
{
extern XrdCnsConfig       Config;

extern XrdCnsDaemon       XrdCnsd;

       XrdSysError        MLog(0,"Cns_");

       XrdOucTrace        XrdTrace(&MLog);
}

/******************************************************************************/
/*            E x t e r n a l   T h r e a d   I n t e r f a c e s             */
/******************************************************************************/

namespace XrdCns
{
void *MLogWorker(void *parg)
{
   time_t midnite = XrdSysTimer::Midnight() + 86400;

// Just blab out the midnight herald
//
   while(1)
        {XrdSysTimer::Wait(midnite - time(0));
         MLog.Say(0, "XrdCnsd - Cluster Name Space Daemon");
         midnite += 86400;
        }
   return (void *)0;
}
}
using namespace XrdCns;

/******************************************************************************/
/*                                  m a i n                                   */
/******************************************************************************/
  
int main(int argc, char *argv[])
{
   XrdSysLogger MLogger;
   XrdOucStream stdinEvents;    // STDIN fed events
   sigset_t myset;
   char *xrdLogD = 0;

// Establish message routing
//
   MLog.logger(&MLogger);

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

// Process the options and arguments
//
   if (!Config.Configure(argc, argv)) exit(1);

// Construct the logfile path and bind it
//
   if (Config.logfn || (xrdLogD = getenv("XRDLOGDIR")))
      {pthread_t tid;
       char buff[2048];
       int retc;
       if (Config.logfn) strcpy(buff, Config.logfn);
          else {strcpy(buff, xrdLogD); strcat(buff, "cnsdlog");}
       if (Config.logKeep) MLogger.setKeep(Config.logKeep);
       MLogger.Bind(buff, 24*60*60);
       MLog.logger(&MLogger);
       if ((retc = XrdSysThread::Run(&tid, MLogWorker, (void *)0,
                                 XRDSYSTHREAD_BIND, "Midnight runner")))
          MLog.Emsg("Main", retc, "create midnight runner");
      }

// Complete configuration. We do it this way so that we can easily run this
// either as a plug-in or as a command.
//
   if (!Config.Configure()) _exit(1);

// At this point we should be able to accept new requests
//
   stdinEvents.Attach(STDIN_FILENO, 32*1024);
   XrdCnsd.getEvents(stdinEvents, "xrootd");

// We should never get here
//
   _exit(8);
}
