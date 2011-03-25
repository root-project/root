/******************************************************************************/
/*                                                                            */
/*                     X r d F r m P u r g M a i n . c c                      */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//           $Id$

const char *XrdFrmPurgMainCVSID = "$Id$";

/* This is the "main" part of the frm_purge command. Syntax is:
*/
static const char *XrdFrmOpts  = ":bc:dfhk:l:n:O:s:Tv";
static const char *XrdFrmUsage =

  " [-b] [-c <cfgfile>] [-d] [-f] [-k {num | sz{k|m|g}] [-l <lfile>] [-n name]"
  " [-O free[,hold]] [-s pidfile] [-T] [-v] [<spaces>] [<paths>]\n";
/*
Where:

   -b     Run as a true daemon process in the background.

   -c     The configuration file. The default is '/opt/xrootd/etc/xrootd.cf'

   -d     Turns on debugging mode.

   -f     Fix orphaned files (i.e., lock and pin) by removing them.

   -k     Keeps num log files or no more that sz log files.

   -l     Specifies location of the log file. This may also come from the
          XrdOucLOGFILE environmental variable.
          By default, error messages go to standard error.

   -n     The instance name.

   -O     Run this one time only as a command. The parms are:
          {free% | sz{k|m|g}[,hold]

   -T     Runs in test mode (no actual purge will occur).

   -v     Verbose mode, typically prints each file purged and other details.

   o-t-a  The one-time-args run this as a command only once. The args direct
          the purging process. These may only be specified when -O specified.

          Syntax is: [space] path | space [path]
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
#include "XrdFrm/XrdFrmPurge.hh"
#include "XrdFrm/XrdFrmTrace.hh"
#include "XrdNet/XrdNetOpts.hh"
#include "XrdNet/XrdNetSocket.hh"
#include "XrdOuc/XrdOucUtils.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSys/XrdSysLogger.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdSys/XrdSysTimer.hh"

using namespace XrdFrm;
  
/******************************************************************************/
/*                      G l o b a l   V a r i a b l e s                       */
/******************************************************************************/

       XrdFrmConfig       XrdFrm::Config(XrdFrmConfig::ssPurg,
                                         XrdFrmOpts, XrdFrmUsage);

// The following is needed to resolve symbols for objects included from xrootd
//
       XrdOucTrace       *XrdXrootdTrace;
       XrdSysError        XrdLog(0, "");
       XrdOucTrace        XrdTrace(&Say);

/******************************************************************************/
/*                     T h r e a d   I n t e r f a c e s                      */
/******************************************************************************/
  
void *mainServer(void *parg)
{
//  int udpFD = *static_cast<int *>(parg);
//  XrdFrmPurge::Server(udpFD);
    return (void *)0;
}

/******************************************************************************/
/*                                  m a i n                                   */
/******************************************************************************/
  
int main(int argc, char *argv[])
{
   XrdSysLogger Logger;
   extern int mainConfig();
   sigset_t myset;

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
   Say.logger(&Logger);
   XrdLog.logger(&Logger);
   if (!Config.Configure(argc, argv, &mainConfig)) exit(4);

// Fill out the dummy symbol to avoid crashes
//
   XrdXrootdTrace = new XrdOucTrace(&Say);

// Display configuration (defered because mum might have been in effect)
//
   if (!Config.isOTO || Config.Verbose) XrdFrmPurge::Display();

// Now simply poke the server every so often
//
   if (Config.isOTO) XrdFrmPurge::Purge();
      else do {if (Config.StopPurge)
                  {int n = 0;
                   struct stat buf;
                   while(!stat(Config.StopPurge, &buf))
                        {if (!n--)
                            {Say.Emsg("PurgMain", Config.StopPurge,
                                      "exists; purging suspended."); n = 12;}
                         XrdSysTimer::Snooze(5);
                        }
                  }
               XrdFrmPurge::Purge();
               XrdSysTimer::Snooze(Config.WaitPurge);
              } while(1);

// All done
//
   exit(0);
}

/******************************************************************************/
/*                            m a i n C o n f i g                             */
/******************************************************************************/
  
int mainConfig()
{
   XrdFrmConfig::Policy *pP = Config.dfltPolicy.Next;
   XrdFrmConfig::VPInfo *vP = Config.VPList;
   XrdNetSocket *udpSock;
   pthread_t tid;
   int retc, udpFD;

// If test is in effect, remove the fix flag
//
   if (Config.Test) Config.Fix = 0;

// Go through the policy list and add each policy
//
   while((pP = Config.dfltPolicy.Next))
        {if (!XrdFrmPurge::Policy(pP->Sname))
            XrdFrmPurge::Policy(pP->Sname, pP->minFree, pP->maxFree,
                                pP->Hold,  pP->Ext);
         Config.dfltPolicy.Next = pP->Next;
         delete pP;
        }

// Make sure we have a public policy
//
   if (!XrdFrmPurge::Policy("public"))
       XrdFrmPurge::Policy("public", Config.dfltPolicy.minFree,
                               Config.dfltPolicy.maxFree,
                               Config.dfltPolicy.Hold,
                               Config.dfltPolicy.Ext);

// Now add any missing policies (we need one for every space)
//
   while(vP)
      {if (!XrdFrmPurge::Policy(vP->Name))
          XrdFrmPurge::Policy(vP->Name, Config.dfltPolicy.minFree,
                                  Config.dfltPolicy.maxFree,
                                  Config.dfltPolicy.Hold,
                                  Config.dfltPolicy.Ext);
       vP = vP->Next;
      }

// Enable the appropriate spaces and over-ride config value
//
   if (!XrdFrmPurge::Init(Config.spacList, Config.cmdFree, Config.cmdHold))
      return 1;

// We are done if this is a one-time-only call
//
   if (Config.isOTO) return 0;

// Get a UDP socket for the server
//
   if (!(udpSock = XrdNetSocket::Create(&Say, Config.AdminPath,
                   "purg.udp", Config.AdminMode, XRDNET_UDPSOCKET))) return 1;
      else {udpFD = udpSock->Detach(); delete udpSock;}

// Start the Server thread
//
   if ((retc = XrdSysThread::Run(&tid, mainServer, (void *)&udpFD,
                                  XRDSYSTHREAD_BIND, "Server")))
      {Say.Emsg("main", retc, "create server thread"); return 1;}

// All done
//
   return 0;
}
