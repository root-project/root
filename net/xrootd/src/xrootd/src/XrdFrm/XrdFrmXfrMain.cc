/******************************************************************************/
/*                                                                            */
/*                      X r d F r m X f r M a i n . c c                       */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

/* This is the "main" part of the frm_xfragent & frm_xfrd commands.
*/

/* This is the "main" part of the frm_migrd command. Syntax is:
*/
static const char *XrdFrmOpts  = ":bc:dfhk:l:n:s:Tv";
static const char *XrdFrmUsage =

  " [-b] [-c <cfgfn>] [-d] [-f] [-k {num | sz{k|m|g}] [-l <lfile>] [-n name] [-s pidfile] [-T] [-v]\n";
/*
Where:

   -b     Run as a true daemon in the bacground (only for xfrd).

   -c     The configuration file. The default is '/opt/xrootd/etc/xrootd.cf'

   -d     Turns on debugging mode.

   -f     Fix orphaned files (i.e., lock and pin) by removing them.

   -k     Keeps num log files or no more that sz log files.

   -l     Specifies location of the log file. This may also come from the
          XrdOucLOGFILE environmental variable.
          By default, error messages go to standard error.

   -n     The instance name.

   -T     Runs in test mode (no actual migration will occur).

   -v     Verbose mode, typically prints each file details.
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

#include "XrdFrm/XrdFrmConfig.hh"
#include "XrdFrm/XrdFrmTrace.hh"
#include "XrdFrm/XrdFrmXfrAgent.hh"
#include "XrdFrm/XrdFrmXfrDaemon.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSys/XrdSysLogger.hh"
#include "XrdSys/XrdSysPthread.hh"

using namespace XrdFrm;
  
/******************************************************************************/
/*                      G l o b a l   V a r i a b l e s                       */
/******************************************************************************/

       XrdFrmConfig       XrdFrm::Config(XrdFrmConfig::ssXfr,
                                         XrdFrmOpts, XrdFrmUsage);

// The following is needed to resolve symbols for objects included from xrootd
//
       XrdOucTrace       *XrdXrootdTrace;
       XrdSysError        XrdLog(0, "");
       XrdOucTrace        XrdTrace(&Say);

/******************************************************************************/
/*                                  m a i n                                   */
/******************************************************************************/
  
int main(int argc, char *argv[])
{
   XrdSysLogger Logger;
   extern int mainConfig();
   sigset_t myset;
   char *pP;

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

// If we are named frm_pstg then we are runnng in agent-mode
//
    if (!(pP = rindex(argv[0], '/'))) pP = argv[0];
       else pP++;
   if (strncmp("frm_xfrd", pP, 8)) Config.isAgent = 1;


// Perform configuration
//
   Say.logger(&Logger);
   XrdLog.logger(&Logger);
   if (!Config.Configure(argc, argv, &mainConfig)) exit(4);

// Fill out the dummy symbol to avoid crashes
//
   XrdXrootdTrace = new XrdOucTrace(&Say);

// All done, simply exit based on our persona
//
   exit(Config.isAgent ? XrdFrmXfrAgent::Start() : XrdFrmXfrDaemon::Start());
}

/******************************************************************************/
/*                            m a i n C o n f i g                             */
/******************************************************************************/
  
int mainConfig()
{
// Initialize the daemon, depending on who we are to be
//
   return (Config.isAgent ? 0 : !XrdFrmXfrDaemon::Init());
}
