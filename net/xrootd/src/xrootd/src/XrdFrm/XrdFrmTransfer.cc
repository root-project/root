/******************************************************************************/
/*                                                                            */
/*                     X r d F r m T r a n s f e r . c c                      */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include <string.h>
#include <strings.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <utime.h>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/stat.h>

#include "XrdFrm/XrdFrmCID.hh"
#include "XrdFrm/XrdFrmConfig.hh"
#include "XrdFrm/XrdFrmMonitor.hh"
#include "XrdFrm/XrdFrmReqFile.hh"
#include "XrdFrm/XrdFrmRequest.hh"
#include "XrdFrm/XrdFrmTrace.hh"
#include "XrdFrm/XrdFrmTransfer.hh"
#include "XrdFrm/XrdFrmXfrJob.hh"
#include "XrdFrm/XrdFrmXfrQueue.hh"
#include "XrdFrm/XrdFrmXAttr.hh"
#include "XrdNet/XrdNetCmsNotify.hh"
#include "XrdOss/XrdOss.hh"
#include "XrdOuc/XrdOucEnv.hh"
#include "XrdOuc/XrdOucMsubs.hh"
#include "XrdOuc/XrdOucProg.hh"
#include "XrdOuc/XrdOucSxeq.hh"
#include "XrdOuc/XrdOucXAttr.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysPlatform.hh"

using namespace XrdFrm;

/******************************************************************************/
/*                         L o c a l   C l a s s e s                          */
/******************************************************************************/

struct XrdFrmTranArg
{
XrdOucEnv   *theEnv;
XrdOucProg  *theCmd;
XrdOucMsubs *theVec;
char        *theSrc;
char        *theDst;
char        *theINS;
char         theMDP[8];

            XrdFrmTranArg(XrdOucEnv *Env)
                         : theEnv(Env), theSrc(0), theDst(0), theINS(0)
                           {theMDP[0] = '0'; theMDP[1] = 0;}
           ~XrdFrmTranArg() {}
};

struct XrdFrmTranChk
{      struct stat           *Stat;
       int                    lkfd;
       int                    lkfx;

       XrdFrmTranChk(struct stat *sP) : Stat(sP), lkfd(-1), lkfx(0) {}
      ~XrdFrmTranChk() {if (lkfd >= 0) close(lkfd);}
};
  
/******************************************************************************/
/*                               S t a t i c s                                */
/******************************************************************************/
  
XrdSysMutex               XrdFrmTransfer::pMutex;
XrdOucHash<char>          XrdFrmTransfer::pTab;

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdFrmTransfer::XrdFrmTransfer()
{
   int i;

// Construct program objects
//
   for (i = 0; i < 4; i++)
       xfrCmd[i] = (Config.xfrCmd[i].theVec ? new XrdOucProg(&Say) : 0);
}

/******************************************************************************/
/* Public:                       c h e c k F F                                */
/******************************************************************************/
  
const char *XrdFrmTransfer::checkFF(const char *Path)
{
   EPNAME("checkFF");
   struct stat buf;

// Check for a fail file
//
   if (!stat(Path, &buf))
      {if (buf.st_ctime+Config.FailHold >= time(0))
          return "request previously failed";
       if (Config.Test) {DEBUG("would have removed '" <<Path <<"'");}
          else {Config.ossFS->Unlink(Path, XRDOSS_isPFN);
                DEBUG("removed '" <<Path <<"'");
               }
      }

// Return all is well
//
   return 0;
}

/******************************************************************************/
/*                                 F e t c h                                  */
/******************************************************************************/
  
const char *XrdFrmTransfer::Fetch()
{
   EPNAME("Fetch");
   static const mode_t fMode = S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH;
   static const int crOpts = (O_CREAT|O_TRUNC)<<8|XRDOSS_mkpath;

   XrdOucEnv myEnv(xfrP->reqData.Opaque?xfrP->reqData.LFN+xfrP->reqData.Opaque:0);
   XrdFrmTranArg cmdArg(&myEnv);
   struct stat pfnStat;
   time_t xfrET;
   const char *eTxt;
   char lfnpath[MAXPATHLEN+8], *Lfn, Rfn[MAXPATHLEN+256], *theSrc;
   int iXfr, lfnEnd, rc, isURL = 0;
   long long fSize = 0;

// The remote source is either the url-lfn or a translated lfn
//
   if ((isURL = xfrP->reqData.LFO)) theSrc = xfrP->reqData.LFN;
      else {if (!Config.RemotePath(xfrP->reqData.LFN, Rfn, sizeof(Rfn)))
                return "lfn2rfn failed";
            theSrc = Rfn;
            isURL = (*Rfn != '/');
           }

// Check if we can actually handle this transfer
//
   if (isURL)
      {if (xfrCmd[2]) iXfr = 2;
          else return "url copies not configured";
      } else {
       if (xfrCmd[0]) iXfr = 0;
          else return "non-url copies not configured";
      }

// Check for a fail file
//
   if ((eTxt = ffCheck())) return eTxt;

// Check if the file exists
//
   if (!stat(xfrP->PFN, &pfnStat))
      {DEBUG(xfrP->PFN <<" exists; not fetched.");
       return 0;
      }

// Construct the file name to which to we originally transfer the data. This is
// the lfn if we do not pre-allocate files and "lfn.anew" otherwise.
//
   Lfn = (xfrP->reqData.LFN)+xfrP->reqData.LFO;
   lfnEnd = strlen(Lfn);
   strlcpy(lfnpath, Lfn, sizeof(lfnpath)-8);
   if (Config.xfrCmd[iXfr].Opts & Config.cmdAlloc)
      {strcpy(&lfnpath[lfnEnd], ".anew");
       strcpy(&xfrP->PFN[xfrP->pfnEnd], ".anew");
      }

// Setup the command
//
   cmdArg.theCmd = xfrCmd[iXfr];
   cmdArg.theVec = Config.xfrCmd[iXfr].theVec;
   cmdArg.theSrc = theSrc;
   cmdArg.theDst = xfrP->PFN;
   cmdArg.theINS = xfrP->reqData.iName;
   if (!SetupCmd(&cmdArg)) return "incoming transfer setup failed";

// If the copycmd needs a placeholder in the filesystem for this transfer, we
// must create one. We first remove any existing "anew" file because we will
// over-write it. The create process will create a lock file if need be. 
// However, we can ignore it as we are the only ones actually using it.
//
   if (Config.xfrCmd[iXfr].Opts & Config.cmdAlloc)
      {Config.ossFS->Unlink(lfnpath);
       rc = Config.ossFS->Create(xfrP->reqData.User,lfnpath,fMode,myEnv,crOpts);
       if (rc)
          {Say.Emsg("Fetch", rc, "create placeholder for", lfnpath);
           return "create failed";
          }
      }

// Now run the command to get the file and make sure the file is there
// If it is, make sure that if a lock file exists its date/time is greater than
// the file we just fetched; then rename it to be the correct name.
//
   xfrET = time(0);
   if (!(rc = cmdArg.theCmd->Run()))
      {if ((rc = stat(xfrP->PFN, &pfnStat)))
          Say.Emsg("Fetch", lfnpath, "fetched but not found!");
          else {fSize  = pfnStat.st_size;
                if (Config.xfrCmd[iXfr].Opts & Config.cmdAlloc)
                   FetchDone(lfnpath, rc, pfnStat.st_mtime);
               }
      }

// Clean up if we failed otherwise tell the cmsd that we have a new file
//
   xfrP->PFN[xfrP->pfnEnd] = '\0';
   if (rc)
      {Config.ossFS->Unlink(lfnpath);
       ffMake(rc == -2);
       if (rc == -2) {xfrP->RetCode = 2; return "file not found";}
       return "fetch failed";
      }
   if (Config.cmsPath) Config.cmsPath->Have(Lfn);

// We completed successfully, see if we need to do statistics
//
   if ((Config.xfrCmd[iXfr].Opts & Config.cmdStats) || Config.monStage
   ||  (Trace.What & TRACE_Debug))
      {time_t eNow = time(0);
       int inqT, xfrT;
       inqT = static_cast<int>(xfrET - time_t(xfrP->reqData.addTOD));
       if ((xfrT = static_cast<int>(eNow - xfrET)) <= 0) xfrT = 1;
       if ((Config.xfrCmd[iXfr].Opts & Config.cmdStats) 
       ||  (Trace.What & TRACE_Debug))
          {char sbuff[80];
           sprintf(sbuff, "Got: %lld qt: %d xt: %d up: ",fSize,inqT,xfrT);
           lfnpath[lfnEnd] = '\0';
           Say.Say(0, sbuff, xfrP->reqData.User, " ", lfnpath);
          }
       if (Config.monStage)
          {snprintf(lfnpath+lfnEnd, sizeof(lfnpath)-lfnEnd-1,
                    "\n&tod=%lld&sz=%lld&qt=%d&tm=%d",
                    static_cast<long long>(eNow), fSize, inqT, xfrT);
           XrdFrmMonitor::Map(XROOTD_MON_MAPSTAG,xfrP->reqData.User,lfnpath);
          }
     }

// All done
//
   return 0;
}

/******************************************************************************/
  
const char *XrdFrmTransfer::FetchDone(char *lfnpath, int &rc, time_t lktime)
{

// If we are running in new mode, update file attributes
//
   rc = 0;
   if (Config.runNew)
      {XrdOucXAttr<XrdFrmXAttrCpy> cpyInfo;
       cpyInfo.Attr.cpyTime = static_cast<long long>(lktime);
       if ((rc = cpyInfo.Set(xfrP->PFN)))
          Say.Emsg("Fetch", rc, "set copy time xattr on", xfrP->PFN);
      }

// Check for a lock file and if we have one, reset it's time or delete it
//
   if (Config.runOld)
      {struct stat lkfStat;
       strcpy(&xfrP->PFN[xfrP->pfnEnd+5], ".lock");
       if (!stat(xfrP->PFN, &lkfStat))
          {if (Config.runNew && !rc) unlink(xfrP->PFN);
              else {struct utimbuf tbuff;
                    tbuff.actime = tbuff.modtime = lktime;
                    if ((rc = utime(xfrP->PFN, &tbuff)))
                       Say.Emsg("Fetch", rc, "set utime on", xfrP->PFN);
                   }
          }
      }

// Now rename the lfn to be what it needs to be in the end
//
   if (!rc && (rc=Config.ossFS->Rename(lfnpath,xfrP->reqData.LFN)))
      Say.Emsg("Fetch", rc, "rename", lfnpath);

// Done
//
   return (rc ? "Failed" : 0);
}

/******************************************************************************/
/* Private:                      f f C h e c k                                */
/******************************************************************************/
  
const char *XrdFrmTransfer::ffCheck()
{
   const char *eTxt;

   strcpy(&xfrP->PFN[xfrP->pfnEnd], ".fail");
   eTxt = checkFF(xfrP->PFN);
   xfrP->PFN[xfrP->pfnEnd] = '\0';
   if (eTxt) xfrP->RetCode = 1;
   return eTxt;
}
  
/******************************************************************************/
/* Private:                       f f M a k e                                 */
/******************************************************************************/
  
void XrdFrmTransfer::ffMake(int nofile)
{
   static const mode_t fMode = S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH;
   int myFD;

// Create a fail file and if failure is due to "file not found" set the mtime
// to 2 so that the oss layer picks up the same error in the future.
//
   strcpy(&xfrP->PFN[xfrP->pfnEnd], ".fail");
   myFD = open(xfrP->PFN, O_CREAT, fMode);
   if (myFD >= 0)
      {close(myFD);
       if (nofile)
          {struct utimbuf tbuff;
           tbuff.actime = time(0); tbuff.modtime = 2;
           utime(xfrP->PFN, &tbuff);
          }
      }
   xfrP->PFN[xfrP->pfnEnd] = '\0';
}
  
/******************************************************************************/
/*                                  I n i t                                   */
/******************************************************************************/
  
void *InitXfer(void *parg)
{   XrdFrmTransfer *xP = new XrdFrmTransfer;
    xP->Start();
    return (void *)0;
}
  
int XrdFrmTransfer::Init()
{
   pthread_t tid;
   int retc, n;

// Initialize the cluster identification object first
//
   CID.Init(Config.QPath);

// Initialize the transfer queue first
//
   if (!XrdFrmXfrQueue::Init()) return 0;

// Start the required number of transfer threads
//
   n = Config.xfrMax;
   while(n--)
        {if ((retc = XrdSysThread::Run(&tid, InitXfer, (void *)0,
                                       XRDSYSTHREAD_BIND, "transfer")))
            {Say.Emsg("main", retc, "create xfr thread"); return 0;}
        }

// All done
//
   return 1;
}

/******************************************************************************/
/* Private:                     S e t u p C m d                               */
/******************************************************************************/
  
int XrdFrmTransfer::SetupCmd(XrdFrmTranArg *argP)
{
   char *pdata[XrdOucMsubs::maxElem + 2], *cP;
   int   pdlen[XrdOucMsubs::maxElem + 2], i, k, n;

   XrdOucMsubsInfo 
              Info(xfrP->reqData.User, argP->theEnv, Config.the_N2N,
                   xfrP->reqData.LFN+xfrP->reqData.LFO,
                   argP->theSrc, xfrP->reqData.Prty,
                   xfrP->reqData.Options & XrdFrmRequest::makeRW?O_RDWR:O_RDONLY,
                   argP->theMDP, xfrP->reqData.ID, xfrP->PFN, argP->theDst);

// We must establish the cluster and instance name if we have one
//
   if (argP->theINS && argP->theEnv)
      {CID.Get(argP->theINS, CMS_CID, argP->theEnv);
       argP->theEnv->Put(XRD_INS, argP->theINS);
      }

// Substitute in the parameters
//
   k = argP->theVec->Subs(Info, pdata, pdlen);

// Catenate all of the arguments
//
   *cmdBuff = '\0'; n = sizeof(cmdBuff) - 4; cP = cmdBuff;
   for (i = 0; i < k; i++)
       {n -= pdlen[i];
        if (n < 0)
           {Say.Emsg("Setup",E2BIG,"build command line for", xfrP->reqData.LFN);
            return 0;
           }
        strcpy(cP, pdata[i]); cP += pdlen[i];
       }

// Now setup the command
//
   return (argP->theCmd->Setup(cmdBuff, &Say) == 0);
}

/******************************************************************************/
/* Public:                         S t a r t                                  */
/******************************************************************************/
  
void XrdFrmTransfer::Start()
{
   EPNAME("Transfer");  // Wrong but looks better
   const char *Msg;

// Prime I/O queue selection

// Endless loop looking for transfer jobs
//
   while(1)
        {xfrP = XrdFrmXfrQueue::Get();

         DEBUG(xfrP->Type <<" starting " <<xfrP->reqData.LFN
               <<" for " <<xfrP->reqData.User);

         Msg = (xfrP->qNum & XrdFrmRequest::outQ ? Throw() : Fetch());
         if (Msg && !(xfrP->RetCode)) xfrP->RetCode = 1;
         xfrP->PFN[xfrP->pfnEnd] = 0;

         if (xfrP->RetCode || Config.Verbose)
            {char buff1[80], buff2[80];
             sprintf(buff1, "%s for %s ", xfrP->RetCode ? "failed" : "complete",
                                          xfrP->reqData.User);
             if (xfrP->RetCode == 0) *buff2 = 0;
                else sprintf(buff2, "; %s", (Msg ? Msg : "reason unknown"));
             Say.Say(0, xfrP->Type, buff1, xfrP->reqData.LFN,buff2);
            } else {
             DEBUG(xfrP->Type
                  <<(xfrP->RetCode ? " failed   " : " complete ")
                  << xfrP->reqData.LFN <<" rc=" <<xfrP->RetCode
                  <<' ' <<(Msg ? Msg : ""));
            }

         XrdFrmXfrQueue::Done(xfrP, Msg);
        }
}

/******************************************************************************/
/* Private:                      T r a c k D C                                */
/******************************************************************************/
  
int XrdFrmTransfer::TrackDC(char *Lfn, char *Mdp, char *Rfn)
{
   char *FName, *Slash, *Slush = 0, *begRfn = Rfn;
   int n = -1;

// If this is a url, then don't back space into the url part
//
   if (*Rfn != '/'
   &&  (Slash = index(Rfn, '/'))     && *(Slash+1) == '/'
   &&  (Slash = index(Slash+2, '/')) && *(Slash+1) == '/') begRfn = Slash+1;

// Discard the filename component
//
   if (!(FName = rindex(begRfn, '/')) || FName == begRfn) return 0;
   *FName = 0; Slash = Slush = FName;

// Try to find the created directory path
//
   pMutex.Lock();
   while(Slash != begRfn && !pTab.Find(Rfn))
        {do {Slash--;} while(Slash != begRfn && *Slash != '/');
         if (Slush) *Slush = '/';
         *Slash = 0; Slush = Slash;
         n++;
        }
   pMutex.UnLock();

// Compute offset of uncreated part
//
   *Slash = '/';
   if (Slash == begRfn) n = 0;
      else n = (n >= 0 ? Slash - begRfn : FName - begRfn);
   sprintf(Mdp, "%d", n);

// All done
//
   return n;
}
  
/******************************************************************************/
  
int XrdFrmTransfer::TrackDC(char *Rfn)
{
   char *Slash;

// Trim off the trailing end
//
   if (!(Slash = rindex(Rfn, '/')) || Slash == Rfn) return 0;
   *Slash = 0;

// The path has been added, do insert it into the table of created paths
//
   pMutex.Lock();
   pTab.Add(Rfn, 0, 0, Hash_data_is_key);
   pMutex.UnLock();
   *Slash = '/';
   return 0;
}
  
/******************************************************************************/
/*                                 T h r o w                                  */
/******************************************************************************/
  
const char *XrdFrmTransfer::Throw()
{
   XrdOucEnv myEnv(xfrP->reqData.Opaque?xfrP->reqData.LFN+xfrP->reqData.Opaque:0);
   XrdFrmTranArg cmdArg(&myEnv);
   struct stat begStat, endStat;
   XrdFrmTranChk Chk(&begStat);
   time_t xfrET;
   const char *eTxt;
   char Rfn[MAXPATHLEN+256], *lfnpath = xfrP->reqData.LFN, *theDest;
   int isMigr = xfrP->reqData.Options & XrdFrmRequest::Migrate;
   int iXfr, isURL, rc, mDP = -1;

// The remote source is either the url-lfn or a translated lfn
//
   if ((isURL = xfrP->reqData.LFO)) theDest = xfrP->reqData.LFN;
      else {if (!Config.RemotePath(xfrP->reqData.LFN, Rfn, sizeof(Rfn)))
                return "lfn2rfn failed";
            theDest = Rfn;
            isURL = (*Rfn != '/');
           }

// Check if we can actually handle this transfer
//
   if (isURL)
      {if (xfrCmd[3]) iXfr = 3;
          else return "url copies not configured";
      } else {
       if (xfrCmd[1]) iXfr = 1;
          else return "non-url copies not configured";
      }

// Check if the file exists
//
   if (stat(xfrP->PFN, &begStat)) return (xfrP->reqFQ ? "file not found" : 0);

// Check for a fail file
//
   if ((eTxt = ffCheck())) return eTxt;

// If this is an mss migration request, then recheck if the file can and
// need to be migrated based on the lock file. This also obtains a directory
// lock and lock file lock, as needed. If the file need not be migrated but
// should be purge, we will get a null string error.
//
   if (isMigr && (eTxt = ThrowOK(&Chk)))
      {if (*eTxt) return eTxt;
       if (!(xfrP->reqData.Options & XrdFrmRequest::Purge)) return "logic error";
       Throwaway();
       return 0;
      }

// Setup the command, including directory tracking, as needed
//
   cmdArg.theCmd = xfrCmd[iXfr];
   cmdArg.theVec = Config.xfrCmd[iXfr].theVec;
   cmdArg.theDst = theDest;
   cmdArg.theSrc = xfrP->PFN;
   cmdArg.theINS = xfrP->reqData.iName;
   if (Config.xfrCmd[iXfr].Opts & Config.cmdMDP)
      mDP = TrackDC(lfnpath+xfrP->reqData.LFO, cmdArg.theMDP, Rfn);
   if (!SetupCmd(&cmdArg)) return "outgoing transfer setup failed";

// Now run the command to put the file. If the command fails and this is a
// migration request, cretae a fail file if one does not exist.
//
   xfrET = time(0);
   if ((rc = cmdArg.theCmd->Run()))
      {if (isMigr) ffMake(rc == 2);
       return "copy failed";
      }

// Track directory creations if we need to track them
//
   if (mDP >= 0) TrackDC(Rfn);

// Obtain state of the file after the copy
//
   if (stat(xfrP->PFN, &endStat))
      {Say.Emsg("Throw", lfnpath, "transfered but not found!");
       return "unable to verify copy";
      }

// Make sure the file was not modified during the copy. This is an error for
// queued requests but internally generated requests will simply be retried.
//
   if (begStat.st_mtime != endStat.st_mtime
    || begStat.st_size  != endStat.st_size)
      {Say.Emsg("Throw", lfnpath, "modified during transfer!");
       return "file modified during copy";
      }

// Purge the file if so wanted. Otherwise, if this is a migration request,
// make sure that if a lock file exists its date/time is equal to the file
// we just copied to prevent the file from being copied again (we have a lock).
//
   if (xfrP->reqData.Options & XrdFrmRequest::Purge) Throwaway();
      else if (isMigr) ThrowDone(&Chk, endStat.st_mtime);

// Do statistics if so wanted
//
   if ((Config.xfrCmd[iXfr].Opts & Config.cmdStats) 
   ||  (Trace.What & TRACE_Debug))
      {int inqT, xfrT;
       long long Fsize = endStat.st_size;
       char sbuff[80];
       inqT = static_cast<int>(xfrET - time_t(xfrP->reqData.addTOD));
       if ((xfrT = static_cast<int>(time(0) - xfrET)) <= 0) xfrT = 1;
       sprintf(sbuff, "Put: %lld qt: %d xt: %d up: ",Fsize,inqT,xfrT);
       Say.Say(0, sbuff, xfrP->reqData.User, " ", xfrP->reqData.LFN);
     }

// All done
//
   return 0;
}

/******************************************************************************/
/* Private:                    T h r o w a w a y                              */
/******************************************************************************/

void XrdFrmTransfer::Throwaway()
{
   EPNAME("Throwaway");

// Purge the file. We do this via the pfn but also indicate we want all
// migration support suffixes removed it they exist.
//
   if (Config.Test) {DEBUG("Would have removed '" <<xfrP->PFN <<"'");}
      else {Config.ossFS->Unlink(xfrP->PFN, XRDOSS_isPFN|XRDOSS_isMIG);
            DEBUG("removed '" <<xfrP->PFN <<"'");
            if (Config.cmsPath) Config.cmsPath->Gone(xfrP->PFN);
           }
}
  
/******************************************************************************/
/* Private:                    T h r o w D o n e                              */
/******************************************************************************/
  
void XrdFrmTransfer::ThrowDone(XrdFrmTranChk *cP, time_t endTime)
{

// Update file attributes if we are running in new mode, otherwise do
//
   if (Config.runNew)
      {XrdOucXAttr<XrdFrmXAttrCpy> cpyInfo;
       cpyInfo.Attr.cpyTime = static_cast<long long>(endTime);
       if (cpyInfo.Set(xfrP->PFN, cP->lkfd))
          Say.Emsg("Throw", "Unable to set copy time xattr for", xfrP->PFN);
          else if (cP->lkfx)
                  {strcpy(&xfrP->PFN[xfrP->pfnEnd], ".lock");
                   unlink(xfrP->PFN);
                   xfrP->PFN[xfrP->pfnEnd] = '\0';
                  }
      } else {
       struct stat Stat;
       strcpy(&xfrP->PFN[xfrP->pfnEnd], ".lock");
       if (!stat(xfrP->PFN, &Stat))
          {struct utimbuf tbuff;
           tbuff.actime = tbuff.modtime = endTime;
           if (utime(xfrP->PFN, &tbuff))
              Say.Emsg("Throw", errno, "set utime for", xfrP->PFN);
          }
       xfrP->PFN[xfrP->pfnEnd] = '\0';
      }
}
  
/******************************************************************************/
/* Private:                      T h r o w O K                                */
/******************************************************************************/
  
const char *XrdFrmTransfer::ThrowOK(XrdFrmTranChk *cP)
{
   class fdClose
        {public:
         int Num;
             fdClose() : Num(-1) {}
            ~fdClose() {if (Num >= 0) close(Num);}
        } fnFD;

   XrdOucXAttr<XrdFrmXAttrCpy> cpyInfo;
   struct stat lokStat;
   int statRC;

// Check if the file is in use by checking if we got an exclusive lock
//
   if ((fnFD.Num = open(xfrP->PFN, O_RDWR)) < 0) return "unable to open file";
   fcntl(fnFD.Num, F_SETFD, FD_CLOEXEC);
   if (XrdOucSxeq::Serialize(fnFD.Num,XrdOucSxeq::noWait)) return "file in use";

// Get the info on the lock file (enabled if old mode is in effect
//
   if (Config.runOld)
      {strcpy(&xfrP->PFN[xfrP->pfnEnd], ".lock");
       statRC = stat(xfrP->PFN, &lokStat);
       xfrP->PFN[xfrP->pfnEnd] = '\0';
      } else statRC = 1;
   if (statRC && !Config.runNew) return "missing lock file";

// If running in new mode then we must get the extened attribute for this file
// unless we got the lock file time which takes precendence.
//
   if (Config.runNew)
      {if (!statRC)
          cpyInfo.Attr.cpyTime = static_cast<long long>(lokStat.st_mtime);
          else if (cpyInfo.Get(xfrP->PFN, fnFD.Num) <= 0)
                  return "unable to get copy time xattr";
      }

// Verify the information
//
   if (cpyInfo.Attr.cpyTime >= static_cast<long long>(cP->Stat->st_mtime))
      {if (xfrP->reqData.Options & XrdFrmRequest::Purge) return "";
       return "already migrated";
      }

// Keep the lock on the base file until we are through. No one is allowed to
// modify this file until we have migrate it.
//
   cP->lkfd = fnFD.Num;
   cP->lkfx = statRC == 0;
   fnFD.Num = -1;
   return 0;
}
