/******************************************************************************/
/*                                                                            */
/*                      X r d C n s S s i M a i n . c c                       */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//           $Id$

const char *XrdCnsSsiMainCVSID = "$Id$";

/* This is the Cluster Name Space utility. The syntax is:

   cns_ssi {diff | list | updt} [options] <path>

   options: [-f] [-h] [-m] [-n] [-p] [-s] [-S] [-v] [-l <lfile>]

Where:

   <path> The archive (i.e., backup) path to use. All sub-directories in
          <path> of the form "<path>/cns/<host>" are considered.

   -l     list: Equivalent to specifying '-h -m -n -p -s'.

   -m     list: Displays the file mode.

   -n     list: Displays the space name.

   -p     list: Displays the physical location.

   -s     list: Displays the file size.

   -S     list: Same as -s but displays size in k, m, g, or t.

   -l     updt: Specifies location of the log file. This may also come from the
          XRDLOGDIR environmental variable. By default, messages go to stderr.

   -v     updt: Increases the verbosity of messages.
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

#include "XrdCns/XrdCnsSsi.hh"
#include "XrdCns/XrdCnsSsiCfg.hh"
#include "XrdCns/XrdCnsSsiSay.hh"

#include "XrdOuc/XrdOucTList.hh"

#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysLogger.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdSys/XrdSysTimer.hh"

/******************************************************************************/
/*                      G l o b a l   V a r i a b l e s                       */
/******************************************************************************/

namespace XrdCns
{
extern XrdCnsSsiCfg       Config;

       XrdSysError        MLog(0,"Cns_");

       XrdCnsSsiSay       Say(&MLog);
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
   XrdOucTList *tP;
   sigset_t myset;
   char *hP;
   int rc = 0;

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
   Say.setV(Config.Verbose);

// Construct the logfile path and bind it (command line only)
//
   if (!Config.logFN && (Config.logFN = getenv("XRDLOGDIR")))
      {pthread_t tid;
       char buff[2048];
       int retc;
       strcpy(buff, Config.logFN); strcat(buff, "cnsssilog");
       MLogger.Bind(buff, 24*60*60);
       MLog.logger(&MLogger);
       if ((retc = XrdSysThread::Run(&tid, MLogWorker, (void *)0,
                                 XRDSYSTHREAD_BIND, "Midnight runner")))
          MLog.Emsg("Main", retc, "create midnight runner");
      }

// Process the request
//
   while((tP = Config.dirList))
        {hP = tP->text + tP->val;
         if (Config.Xeq == 'l') rc |= XrdCnsSsi::List(hP, tP->text);
            else {int i = XrdCnsSsi::Updt(hP, tP->text);
                  if (i) Say.M("Unable to update ", hP, " inventory.");
                  rc |= i;
                 }
         Config.dirList = tP->next;
         delete tP;
        }

// All done
//
   exit(rc);
}
