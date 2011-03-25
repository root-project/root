/******************************************************************************/
/*                                                                            */
/*                      X r d F r m M i g r a t e . c c                       */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <utime.h>
#include <sys/param.h>
#include <sys/types.h>

#include "XrdOss/XrdOss.hh"
#include "XrdOss/XrdOssPath.hh"
#include "XrdOuc/XrdOucNSWalk.hh"
#include "XrdOuc/XrdOucTList.hh"
#include "XrdFrm/XrdFrmFiles.hh"
#include "XrdFrm/XrdFrmConfig.hh"
#include "XrdFrm/XrdFrmMigrate.hh"
#include "XrdFrm/XrdFrmRequest.hh"
#include "XrdFrm/XrdFrmTrace.hh"
#include "XrdFrm/XrdFrmTransfer.hh"
#include "XrdFrm/XrdFrmXfrQueue.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdSys/XrdSysPlatform.hh"
#include "XrdSys/XrdSysTimer.hh"

using namespace XrdFrm;

/******************************************************************************/
/*                        S t a t i c   M e m b e r s                         */
/******************************************************************************/

XrdFrmFileset    *XrdFrmMigrate::fsDefer = 0;

int               XrdFrmMigrate::numMig = 0;
  
/******************************************************************************/
/* Private:                          A d d                                    */
/******************************************************************************/
  
void XrdFrmMigrate::Add(XrdFrmFileset *sP)
{
   EPNAME("Add");
   const char *Why;
   time_t xTime;

// Check to see if the file is really eligible for purging
//
   if ((Why = Eligible(sP, xTime)))
      {DEBUG(sP->basePath() <<"cannot be migrated; " <<Why);
       delete sP;
       return;
      }

// Add the file to the migr queue or the defer queue based on mod time
//
   if (xTime < Config.IdleHold) Defer(sP);
      else Queue(sP);
}
  
/******************************************************************************/
/* Private:                      A d v a n c e                                */
/******************************************************************************/

int XrdFrmMigrate::Advance()
{
   XrdFrmFileset *fP;
   int xTime;

// Try to re-add everything in this queue
//
   while(fsDefer)
        {xTime = static_cast<int>(time(0) - fsDefer->baseFile()->Stat.st_mtime);
         if (xTime < Config.IdleHold) break;
         fP = fsDefer; fsDefer = fsDefer->Next;
         if (fP->Refresh(1,0)) Add(fP);
            else delete fP;
        }

// Return number of seconds to next advance event
//
   return fsDefer ? Config.IdleHold - xTime : 0;
}
  
/******************************************************************************/
/* Private:                        D e f e r                                  */
/******************************************************************************/

void XrdFrmMigrate::Defer(XrdFrmFileset *sP)
{
   XrdFrmFileset *fP = fsDefer, *pfP = 0;
   time_t mTime = sP->baseFile()->Stat.st_mtime;

// Insert this entry into the defer queue in ascending mtime order
//
   while(fP && fP->baseFile()->Stat.st_mtime < mTime)
        {pfP = fP; fP = fP->Next;}

// Chain in the fileset
//
   sP->Next = fP;
   if (pfP) pfP->Next = sP;
      else  fsDefer   = sP;
}

/******************************************************************************/
/*                               D i s p l a y                                */
/******************************************************************************/

void XrdFrmMigrate::Display()
{
   XrdFrmConfig::VPInfo *vP = Config.pathList;
   XrdOucTList *tP;

// Type header
//
   Say.Say("=====> ", "Migrate configuration:");

// Display what we will scan
//
   while(vP)
        {Say.Say("=====> ", "Scanning ", (vP->Val?"r/w: ":"r/o: "), vP->Name);
         tP = vP->Dir;
         while(tP) {Say.Say("=====> ", "Excluded ", tP->text); tP = tP->next;}
         vP = vP->Next;
        }
}
  
/******************************************************************************/
/* Private:                     E l i g i b l e                               */
/******************************************************************************/
  
const char *XrdFrmMigrate::Eligible(XrdFrmFileset *sP, time_t &xTime)
{
   XrdOucNSWalk::NSEnt *baseFile = sP->baseFile();
   XrdOucNSWalk::NSEnt *failFile = sP->failFile();
   time_t mTimeBF, mTimeLK, nowTime = time(0);
   const char *eTxt;

// File is inelegible if lockfile mtime is zero (i.e., an mstore placeholder)
//
   mTimeLK = static_cast<time_t>(sP->cpyInfo.Attr.cpyTime);
   if (!mTimeLK) return "migration defered";

// File is ineligible if it has not changed since last migration
//
   mTimeBF = baseFile->Stat.st_mtime;
   if (mTimeLK >= mTimeBF) return "file unchanged";

// File is ineligible if it has a fail file that is still recent
//
   if (failFile && (eTxt=XrdFrmTransfer::checkFF(sP->failPath()))) return eTxt;

// Migration may need to be defered if the file has been modified too recently
// (caller will check)
//
   xTime = static_cast<int>(nowTime - mTimeBF);

// File can be migrated
//
   return 0;
}

/******************************************************************************/
/*                               M i g r a t e                                */
/******************************************************************************/
  
void *XrdMigrateStart(void *parg)
{
    XrdFrmMigrate::Migrate(0);
    return (void *)0;
}
  
void XrdFrmMigrate::Migrate(int doinit)
{
   XrdFrmFileset *fP;
   char buff[80];
   int migWait, wTime;

// If we have not initialized yet, start a thread to handle this
//
   if (doinit)
      {pthread_t tid;
       int retc;
       if ((retc = XrdSysThread::Run(&tid, XrdMigrateStart, (void *)0,
                                     XRDSYSTHREAD_BIND, "migration scan")))
          Say.Emsg("Migrate", retc, "create migrtion thread");
       return;
      }

// Start the migration sequence, first do a name space scan which will trigger
// all eligible migrations and defer any that need to wait. We then drain the
// defer queue and wait for the next period to start.
//
do{migWait = Config.WaitMigr; numMig = 0;
   Scan();
   while((wTime = Advance()))
        {if ((migWait -= wTime) <= 0) break;
            else  XrdSysTimer::Snooze(wTime);
        }
   while(fsDefer) {fP = fsDefer; fsDefer = fsDefer->Next; delete fP;}
   sprintf(buff, "%d file%s selected for transfer.",numMig,(numMig==1?"":"s"));
   Say.Emsg("Migrate", buff);
   if (migWait > 0) XrdSysTimer::Snooze(migWait);
  } while(1);
}

/******************************************************************************/
/* Private:                        Q u e u e                                  */
/******************************************************************************/

void XrdFrmMigrate::Queue(XrdFrmFileset *sP)
{
   static int reqID = 0;
   XrdFrmRequest myReq;

// Convert the fileset to a request element
//
   memset(&myReq, 0, sizeof(myReq));
   strlcpy(myReq.User, Config.myProg, sizeof(myReq.User));
   sprintf(myReq.ID, "Internal%d", reqID++);
   myReq.Options = XrdFrmRequest::Migrate;
   myReq.addTOD  = static_cast<long long>(time(0));
   if (Config.LogicalPath(sP->basePath(), myReq.LFN, sizeof(myReq.LFN)))
      {XrdFrmXfrQueue::Add(&myReq, 0, XrdFrmRequest::migQ); numMig++;}

// All done
//
   delete sP;
}
  
/******************************************************************************/
/* Private:                         S c a n                                   */
/******************************************************************************/
  
void XrdFrmMigrate::Scan()
{
   static const int Opts = XrdFrmFiles::Recursive | XrdFrmFiles::CompressD
                         | XrdFrmFiles::NoAutoDel;
   static time_t lastHP = time(0), nowT = time(0);

   XrdFrmConfig::VPInfo *vP = Config.pathList;
   XrdFrmFileset *sP;
   XrdFrmFiles   *fP;
   char buff[128];
   int ec = 0, Bad = 0, aFiles = 0, bFiles = 0;

// Purge that bad file table evey 24 hours to keep complaints down
//
   if (nowT - lastHP >= 86400) {XrdFrmFileset::Purge(); lastHP = nowT;}

// Indicate scan started
//
   VMSG("Scan", "Name space scan started. . .");

// Process each directory
//
   do {fP = new XrdFrmFiles(vP->Name, Opts, vP->Dir);
       while((sP = fP->Get(ec,1)))
            {aFiles++;
             if (sP->Screen()) Add(sP);
                else {delete sP; bFiles++;}
            }
       if (ec) Bad = 1;
       delete fP;
      } while((vP = vP->Next));

// Indicate scan ended
//
   sprintf(buff, "%d file%s with %d error%s", aFiles, (aFiles != 1 ? "s":""),
                                              bFiles, (bFiles != 1 ? "s":""));
   VMSG("Scan", "Name space scan ended;", buff);

// Issue warning if we encountered errors
//
   if (Bad) Say.Emsg("Scan", "Errors encountered while scanning for "
                             "migratable files.");
}
