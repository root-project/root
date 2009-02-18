/******************************************************************************/
/*                                                                            */
/*                     X r d F r m P r e S t a g e . c c                      */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//           $Id$

const char *XrdFrmPreStageCVSID = "$Id$";

/* This is the "main" part of the frm_PreStage command. Syntax is:
*/
static const char *XrdFrmOpts  = ":c:dk:l:n:s";
static const char *XrdFrmUsage =

  " [-c <cfgfile>] [-d] [-k {num | sz{k|m|g}] [-l <lfile>] [-n name] [-s]\n";
/*
Where:

   -c     The configuration file. The default is '/opt/xrootd/etc/xrootd.cf'

   -d     Turns on debugging mode.

   -k     Keeps num log files or no more that sz log files.

   -l     Specifies location of the log file. This may also come from the
          XrdOucLOGFILE environmental variable.
          By default, error messages go to standard error.

   -s     Prints transfer statistics for each successful file transfer.

   -n     The instance name.
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
#include "XrdFrm/XrdFrmPstg.hh"
#include "XrdFrm/XrdFrmPstgReq.hh"
#include "XrdFrm/XrdFrmPstgXfr.hh"
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

       XrdSysLogger       XrdFrm::Logger;

       XrdSysError        XrdFrm::Say(&Logger, "");

       XrdOucTrace        XrdFrm::Trace(&Say);

       XrdFrmConfig       XrdFrm::Config(XrdFrmConfig::ssPstg, 
                                         XrdFrmOpts, XrdFrmUsage);

       XrdFrmPstg         XrdFrm::PreStage;

// The following is needed to resolve symbols for objects included from xrootd
//
       XrdOucTrace       *XrdXrootdTrace;
       XrdSysError        XrdLog(&Logger, "");
       XrdOucTrace        XrdTrace(&Say);

/******************************************************************************/
/*                     T h r e a d   I n t e r f a c e s                      */
/******************************************************************************/
  
void *mainServer(void *parg)
{
    int udpFD = *static_cast<int *>(parg);
    PreStage.Server(udpFD);
    return (void *)0;
}
  
void *mainStage(void *parg)
{
    PreStage.Server_Stage();
    return (void *)0;
}
  
void *mainXfer(void *parg)
{   XrdFrmPstgXfr *xP = new XrdFrmPstgXfr;
    xP->Start();
    return (void *)0;
}

/******************************************************************************/
/*                                  m a i n                                   */
/******************************************************************************/
  
int main(int argc, char *argv[])
{
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

// If we are named frm_pstga then we are runnng in agent-mode
//
    if (!(pP = rindex(argv[0], '/'))) pP = argv[0];
       else pP++;
   if (!strcmp("frm_pstga", pP)) Config.isAgent = 1;

// Perform configuration
//
   if (!Config.Configure(argc, argv, &mainConfig)) exit(4);

// Fill out the dummy symbol to avoid crashes
//
   XrdXrootdTrace = new XrdOucTrace(&Say);

// If we are running in agent mode scadadle to that (it's simple)
//
   if (Config.isAgent) exit(PreStage.Agent(Config.c2sFN));

// Now simply poke the server every so often
//
   while(1)
        {PreStage.Server_Driver(1);
         XrdSysTimer::Snooze(Config.WaitTime);
        }

// We get here is we failed to initialize
//
   exit(255);
}

/******************************************************************************/
/*                            m a i n C o n f i g                             */
/******************************************************************************/
  
int mainConfig()
{
   XrdNetSocket *udpSock;
   struct sockaddr *sockP;
   pthread_t tid;
   char buff[2048], *qPath;
   int retc, i, n, udpFD;

// Make the queue path
//
   if ((qPath = Config.qPath))
      {if ((retc = XrdOucUtils::makePath(qPath, Config.AdminMode)))
          {Say.Emsg("Config", retc, "create queue directory", qPath);
           return 1;
          }
      } else qPath = Config.AdminPath;

// Make sure we are the only ones running (daemon only)
//
   if (!Config.isAgent)
      {sprintf(buff, "%spstgd.lock", qPath);
       if (!XrdFrmPstgReq::Unique(buff)) return 1;
      }

// Initialize the request queues if all went well
//
   for (i = 0; i <= XrdFrmPstgReq::maxPrty; i++)
       {sprintf(buff, "%spstgQ.%d", qPath, i);
        rQueue[i] = new XrdFrmPstgReq(buff);
        if (!rQueue[i]->Init()) return 1;
       }

// Develop the clint/server UDP path name
//
   strcpy(buff, Config.AdminPath);
   strcat(buff,"pstg.udp");
   if (XrdNetSocket::socketAddr(&Say, buff, &sockP, n)) return 1;
      else {free(sockP); Config.c2sFN = strdup(buff);}

// We are done if this is an agent
//
   if (Config.isAgent) return 0;

// Start the required number of transfer threads
//
   n = Config.xfrMax;
   if (!XrdFrmPstgXfr::Init()) return 1;
   while(n--)
        {if ((retc = XrdSysThread::Run(&tid, mainXfer, (void *)0,
                                       XRDSYSTHREAD_BIND, "Xfr Agent")))
            {Say.Emsg("main", retc, "create xfr thread"); return 1;}
        }

// Start the Staging thread
//
   if ((retc = XrdSysThread::Run(&tid, mainStage, (void *)0,
                                 XRDSYSTHREAD_BIND, "Stager")))
      {Say.Emsg("main", retc, "create stager thread"); return 1;}

// Get a UDP socket for the server
//
   if (!(udpSock = XrdNetSocket::Create(&Say, Config.AdminPath,
                   "pstg.udp", Config.AdminMode, XRDNET_UDPSOCKET))) return 1;
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
