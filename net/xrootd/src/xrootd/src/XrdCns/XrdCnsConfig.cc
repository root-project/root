/******************************************************************************/
/*                                                                            */
/*                       X r d C n s C o n f i g . c c                        */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

const char *XrdCnsConfigCVSID = "$Id$";

#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <sys/types.h>

#include "Xrd/XrdTrace.hh"
#include "Xrd/XrdScheduler.hh"
#include "XrdNet/XrdNetDNS.hh"
#include "XrdNet/XrdNetOpts.hh"
#include "XrdNet/XrdNetSocket.hh"

#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSys/XrdSysLogger.hh"
#include "XrdOuc/XrdOucUtils.hh"
#include "XrdSys/XrdSysTimer.hh"

#include "XrdCns/XrdCnsDaemon.hh"
#include "XrdCns/XrdCnsEvent.hh"

/******************************************************************************/
/*           G l o b a l   C o n f i g u r a t i o n   O b j e c t            */
/******************************************************************************/

extern XrdScheduler      XrdSched;

extern XrdCnsDaemon      XrdCnsd;

extern XrdSysError       XrdLog;

extern XrdSysLogger      XrdLogger;

extern XrdOucTrace       XrdTrace;

/******************************************************************************/
/*            E x t e r n a l   T h r e a d   I n t e r f a c e s             */
/******************************************************************************/
  
void *CnsEvents(void *parg)
{
   XrdOucStream *esP = static_cast<XrdOucStream *>(parg);
   XrdCnsd.getEvents(*esP);

   return (void *)0;
}

/******************************************************************************/
/*                         L o c a l   C l a s s e s                          */
/******************************************************************************/

class XrdLogWorker : XrdJob
{
public:

     void DoIt() {XrdLog.Say(0, "XrdCnsd - Cluster Name Space Daemon");
                  midnite += 86400;
                  XrdSched.Schedule((XrdJob *)this, midnite);
                 }

          XrdLogWorker() : XrdJob("midnight runner")
                         {midnite = XrdSysTimer::Midnight() + 86400;
                          XrdSched.Schedule((XrdJob *)this, midnite);
                         }
         ~XrdLogWorker() {}
private:
time_t midnite;
};

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdCnsDaemon::XrdCnsDaemon(void)
{
   myName   = XrdNetDNS::getHostName();
}
  
/******************************************************************************/
/*                             C o n f i g u r e                              */
/******************************************************************************/
  
int XrdCnsDaemon::Configure(int argc, char **argv)
{
/*
  Function: Establish configuration at start up time.

  Input:    None.

  Output:   1 upon success or 0 otherwise.
*/
   XrdNetSocket *EventSock;
   pthread_t tid;
   int n, QLim = 1024, retc, NoGo = 0;
   char c, pfxPath[2048], buff[2048], *aPath = 0, *logfn = 0;
   extern char *optarg;
   extern int optind, opterr;

// Process the options
//
   opterr = 0;
   if (argc > 1 && '-' == *argv[1]) 
      while ((c = getopt(argc,argv,"a:dl:q:")) && ((unsigned char)c != 0xff))
     { switch(c)
       {
       case 'a': aPath = optarg;
                 break;
       case 'd': XrdTrace.What = TRACE_ALL;
                 XrdSysThread::setDebug(&XrdLog);
                 break;
       case 'l': if (logfn) free(logfn);
                 logfn = strdup(optarg);
                 break;
       case 'q': if (!(QLim = atoi(optarg)))
                    {XrdLog.Emsg("Config", "Invalid -q value -",optarg);NoGo=1;}
                 break;
       default:  if (index("lq", (int)(*(argv[optind-1]+1))))
                    XrdLog.Emsg("Config", argv[optind-1], "parameter not specified.");
                    else XrdLog.Emsg("Config", "Invalid option -", argv[optind-1]);
                 NoGo = 1;
       }
     }

// Make sure we have a host/port here
//
   if (optind >= argc)
      {XrdLog.Emsg("Config", "routing not specified."); NoGo = 1;}
      else {strcpy(pfxPath, argv[optind]);
            if (!strncmp("xroot://", pfxPath, 8)
            ||  !strncmp( "root://", pfxPath, 7))
               {n = strlen(pfxPath);
                if (pfxPath[n-1] != '/') 
                   {strcat(pfxPath, "/");
                    XrdLog.Emsg("Config","Slash added; routing is now:", pfxPath);
                   }
               }
           }

// Bind the log file if we have one
//
   if (logfn)
      {XrdLogger.Bind(logfn, 24*60*60);
       free(logfn);
       new XrdLogWorker();
      }

// Put out the herald
//
   XrdLog.Emsg("Config", "XrdCnsd initialization started.");
   if (NoGo) return 0;

// Get the directory where the meta information is to go
//
   if (!aPath) if (!(aPath = getenv("XRDADMINPATH"))) aPath = (char *)"/tmp/";
   strcpy(buff, aPath);
   n = strlen(buff);
   if (buff[n-1] != '/') {buff[n] = '/'; buff[n+1] = '\0';}
   aPath = strdup(buff);
   XrdLog.Emsg("Config", "Admin path set to", aPath);

// Create the admin directory if it does not exists
//
   if ((retc = XrdOucUtils::makePath(aPath,0770)))
      {XrdLog.Emsg("Config", retc, "create admin directory", aPath);
       return 0;
      }

// Check if we should continue
//
   if (NoGo) return 0;

// Start the Scheduler
//
   XrdSched.Start();

// Initialize event handling
//
   if (!XrdCnsEvent::Init(aPath, pfxPath, QLim)) return 0;

// Create our notification path (r/w for us and our group)
//
   if (!(EventSock = XrdNetSocket::Create(&XrdLog, aPath, "XrdCnsd.events",
                                   0660, XRDNET_FIFO))) return 0;
      else {int eFD = EventSock->Detach();
            delete EventSock;
            fifoEvents.Attach(eFD, 32*1024);
           }

// Start the fifo event thread
//
   if ((retc = XrdSysThread::Run(&tid, CnsEvents, (void *)&fifoEvents,
                                 XRDSYSTHREAD_BIND, "Fifo event handler")))
      {XrdLog.Emsg("Config", retc, "create fifo event thread"); return 0;}

// Attach standard-in to our stream
//
   stdinEvents.Attach(STDIN_FILENO, 32*1024);

// Start the stdin event thread
//
   if ((retc = XrdSysThread::Run(&tid, CnsEvents, (void *)&stdinEvents,
                                 XRDSYSTHREAD_BIND, "STDIN event handler")))
      {XrdLog.Emsg("Config", retc, "create stdin event thread"); return 0;}

// All done
//
   return !NoGo;
}
