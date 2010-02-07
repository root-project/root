/******************************************************************************/
/*                                                                            */
/*                    X r d C n s L o g S e r v e r . c c                     */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//          $Id$

const char *XrdCnsLogServerCVSID = "$Id$";
  
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "Xrd/XrdTrace.hh"

#include "XrdOss/XrdOssPath.hh"
#include "XrdOuc/XrdOucTList.hh"
#include "XrdCns/XrdCnsConfig.hh"
#include "XrdCns/XrdCnsLogClient.hh"
#include "XrdCns/XrdCnsLogFile.hh"
#include "XrdCns/XrdCnsLogRec.hh"
#include "XrdCns/XrdCnsLogServer.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysPthread.hh"
  
/******************************************************************************/
/*                        G l o b a l   O b j e c t s                         */
/******************************************************************************/
  
namespace XrdCns
{
extern XrdCnsConfig Config;

extern XrdSysError  MLog;

extern XrdOucTrace  XrdTrace;
}

using namespace XrdCns;


/******************************************************************************/
/*                     T h r e a d   I n t e r f a c e s                      */
/******************************************************************************/
  
namespace XrdCns
{
void *StartLogServer(void *parg)
{
   XrdCnsLogServer *lsP = static_cast<XrdCnsLogServer *>(parg);
   lsP->Run();
   MLog.Emsg("Run", "Fatal log server error; terminating!");
   _exit(8);
   return (void *)0;
}
}

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdCnsLogServer::XrdCnsLogServer()
{
// Construct our logfile path
//
   strcpy(logDir, Config.ePath);
   logFN = logDir + strlen(Config.ePath);
}

/******************************************************************************/
/*                                  I n i t                                   */
/******************************************************************************/
  
int XrdCnsLogServer::Init(XrdOucTList *rList)
{
   struct stat Stat;
   pthread_t tid;
   int rc, aOK = 1;

// First, inform the log file the maximum true records we will add
//
   XrdCnsLogFile::maxRecs(Config.qLim);

// If this is a command line recreate then just serially do the recreates
//
   if (Config.Opts & XrdCnsConfig::optRecr)
      {while(rList)
            {Client = new XrdCnsLogClient(rList, 0);
             if (!Client->Init()) aOK = 0;
             delete Client;
             rList = rList->next;
            }
        return aOK;
       }

// First see if we have a pending log file
//
   strcpy(logFN, "cns.log");
   if (stat(logDir, &Stat))
      {if (errno != ENOENT)
          {MLog.Emsg("Init", errno, "stat file", logDir); return 0;}
       Stat.st_size = 0;
      }

// If this is an empty log file, then remove it, otherwise end file it
//
   if (Stat.st_size != 0)
      {XrdCnsLogFile myLogFile(logDir);
       if (!myLogFile.Open(0, Stat.st_size) || !myLogFile.Eol()) return 0;
      }
   unlink(logDir);

// We now can create required log clients and initialize them
//
   *logFN = '\0';
   while(rList)
        {Client = new XrdCnsLogClient(rList, Client);
         if (!Client->Init())
            {MLog.Emsg("Init", "Initialization for",rList->text,"failed.");
             aOK = 0;
            }
         rList = rList->next;
        }

// Now activate a new log and start the clients
//
   if (aOK)
      {strcpy(logFN, "cns.log");
       logFile = new XrdCnsLogFile(logDir);
       if (!logFile->Open(0)) aOK = 0;
          else {if (!Client->Activate(logFile)) aOK = 0;
                   else Client->Start();
               }
      }

// Now start the server
//
   if ((rc = XrdSysThread::Run(&tid, StartLogServer, (void *)this,
                                 XRDSYSTHREAD_BIND, "Log server")))
      {MLog.Emsg("Start", rc, "create log server thread"); aOK = 0;}

// All done
//
   return aOK;
}

/******************************************************************************/
/* Private                       M a s s a g e                                */
/******************************************************************************/
  
void XrdCnsLogServer::Massage(XrdCnsLogRec *lrP)
{
   const char *cP;
   char lfnBuff[MAXPATHLEN+1], pfnBuff[MAXPATHLEN+1], lnkBuff[MAXPATHLEN+1];
   char *cgiP, *mP;
   int lnkbsz = sizeof(lnkBuff);

// Get the pfn for the lfn
//
   strcpy(lfnBuff, lrP->Lfn1());
   if ((cgiP = index(lfnBuff, '?'))) *cgiP = '\0';
   if (!Config.LocalPath(lfnBuff, pfnBuff, sizeof(pfnBuff))) return;

// Now get space information
//
   cP = XrdOssPath::Extract(pfnBuff, lnkBuff, lnkbsz);

// Check if we actually have a true mount point
//
   if (lnkBuff[1]) mP = lnkBuff;
      else {Config.MountPath(lfnBuff, pfnBuff, sizeof(pfnBuff));
            mP = pfnBuff;
           }

// Set information in the create record
//
   lrP->setData(cP, mP);
}

/******************************************************************************/
/*                                   R u n                                    */
/******************************************************************************/
  
void XrdCnsLogServer::Run()
{
   static const char *TraceID = "ServerRun";
   XrdCnsLogFile *lfP;
   XrdCnsLogRec  *lrP;
   char lrT;
   int  nRecs;

// All we need to do is transfer log records from the queue to the log file
// and periodically close out the log and start over. Timing marks are
// periodically placed in the queue that cause a nil-pointer to be returned.
// We honor them if we actually have something in the log file. In any case,
// We must activate the new file before we close out the old file to make sure
// log clients have a logfile that they can actually process.
//
do{nRecs = Config.qLim; lfP = logFile;

   while(nRecs && (lrP = XrdCnsLogRec::Get(lrT)))
        {if (lrP->Type() == XrdCnsLogRec::lrCreate) Massage(lrP);
         lfP->Add(lrP); lrP->Recycle();
         nRecs--;
        }

   if (nRecs != Config.qLim)
      {TRACE(DEBUG, "Closing out " <<(Config.qLim-nRecs) <<" log records.");
       lfP->Eol();
       if (!lfP->Unlink()) break;
       logFile = new XrdCnsLogFile(logDir);
       if (!logFile->Open() || !Client->Activate(logFile)) break;
       delete lfP;
      }
  } while(1);

// At the moment we don't really have a recovery strategy
//
   MLog.Emsg("Run", "Fatal error occurred; terminating. . .");
}
