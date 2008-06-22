/******************************************************************************/
/*                                                                            */
/*                            X r d M a i n . c c                             */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//           $Id$

const char *XrdMainCVSID = "$Id$";

/* This is the XRootd server. The syntax is:

   xrootd [options]

   options: [-b] [-c <fname>] [-d] [-h] [-l <fname>] [-p <port>] [<oth>]

Where:
   -b     forces background execution.

   -c     specifies the configuration file. This may also come from the
          XrdCONFIGFN environmental variable.

   -d     Turns on debugging mode (equivalent to xrd.trace all)

   -h     Displays usage line and exits.

   -l     Specifies location of the log file. This may also come from the
          XrdOucLOGFILE environmental variable or from the oofs layer. By
          By default, error messages go to standard error.

   -p     Is the port to use either as a service name or an actual port number.
          The default port is 1094.

   <oth>  Are other protocol specific options.

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

#include "Xrd/XrdBuffer.hh"
#include "Xrd/XrdConfig.hh"
#include "Xrd/XrdInet.hh"
#include "Xrd/XrdLink.hh"
#include "Xrd/XrdProtLoad.hh"
#include "Xrd/XrdScheduler.hh"
#define  TRACELINK newlink
#include "Xrd/XrdTrace.hh"

#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSys/XrdSysLogger.hh"
#include "XrdSys/XrdSysPthread.hh"
  
/******************************************************************************/
/*                      G l o b a l   V a r i a b l e s                       */
/******************************************************************************/

       XrdConfig          XrdConf;

extern int                XrdNetTCPlep;   // Defined by config
extern XrdInet           *XrdNetTCP[];    // Defined by config
       XrdInet           *XrdNetADM = 0;

       XrdScheduler       XrdSched;

       XrdSysLogger       XrdLogger;

       XrdSysError        XrdLog(&XrdLogger, "Xrd");

       XrdSysThread      *XrdThread;

       XrdOucTrace        XrdTrace(&XrdLog);

/******************************************************************************/
/*            E x t e r n a l   T h r e a d   I n t e r f a c e s             */
/******************************************************************************/
  
void *mainAccept(void *parg)
{  XrdInet *myNet = (XrdInet *)parg;
   int myPort = (myNet == XrdNetTCP[XrdProtLoad::ProtoMax]
                       ?  -(myNet->Port()) : myNet->Port());
   XrdProtLoad ProtSelect(myPort);
   XrdLink *newlink;

   while(1) if ((newlink = myNet->Accept(XRDNET_NODNTRIM)))
               {newlink->setProtocol((XrdProtocol *)&ProtSelect);
                XrdSched.Schedule((XrdJob *)newlink);
               }

   return (void *)0;
}

/******************************************************************************/
/*                             m a i n A d m i n                              */
/******************************************************************************/
  
void *mainAdmin(void *parg)
{
   XrdLink *newlink;
// static XrdProtocol_Admin  ProtAdmin;
   int ProtAdmin;

// At this point we should be able to accept new connections
//
   while(1) if ((newlink = XrdNetADM->Accept()))
               {newlink->setProtocol((XrdProtocol *)&ProtAdmin);
                XrdSched.Schedule((XrdJob *)newlink);
               }
   return (void *)0;
}

/******************************************************************************/
/*                                  m a i n                                   */
/******************************************************************************/
  
int main(int argc, char *argv[])
{
   sigset_t myset;
   pthread_t tid;
   char buff[128];
   int i, retc;

// Turn off sigpipe and host a variety of others before we start any threads
//
   signal(SIGPIPE, SIG_IGN);  // Solaris optimization
   sigemptyset(&myset);
   sigaddset(&myset, SIGPIPE);
   sigaddset(&myset, SIGUSR1);
   sigaddset(&myset, SIGUSR2);
#ifdef SIGRTMAX
   sigaddset(&myset, SIGRTMAX);
   sigaddset(&myset, SIGRTMAX-1);
#endif
   sigaddset(&myset, SIGCHLD);
   pthread_sigmask(SIG_BLOCK, &myset, NULL);

// Set the default stack size here
//
   if (sizeof(long) > 4) XrdSysThread::setStackSize((size_t)1048576);
      else               XrdSysThread::setStackSize((size_t)786432);

// Process configuration file
//
   if (XrdConf.Configure(argc, argv)) _exit(1);

// Start the admin thread if an admin network is defined
//
   if (XrdNetADM && (retc = XrdSysThread::Run(&tid, mainAdmin, (void *)0,
                            XRDSYSTHREAD_BIND, "Admin handler")))
      {XrdLog.Emsg("main", retc, "create admin thread"); _exit(3);}

// At this point we should be able to accept new connections. Spawn a
// thread for each network except the first. The main thread will handle
// that network as some implementations require a main active thread.
//
   for (i = 1; i <= XrdProtLoad::ProtoMax; i++)
       if (XrdNetTCP[i])
          {sprintf(buff, "Port %d handler", XrdNetTCP[i]->Port());
           if ((retc = XrdSysThread::Run(&tid, mainAccept,
                                         (void *)XrdNetTCP[i],
                                         XRDSYSTHREAD_BIND, strdup(buff))))
              {sprintf(buff, "create port %d handler", XrdNetTCP[i]->Port());
               XrdLog.Emsg("main", retc, buff);
               _exit(3);
              }
          }

// Finally, start accepting connections on the main port
//
   mainAccept((void *)XrdNetTCP[0]);

// We should never get here
//
   pthread_exit(0);
}
