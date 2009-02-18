/******************************************************************************/
/*                                                                            */
/*                      X r d F r m P s t g X f r . c c                       */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

#include <string.h>
#include <strings.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <utime.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "XrdFrm/XrdFrmConfig.hh"
#include "XrdFrm/XrdFrmPstgReq.hh"
#include "XrdFrm/XrdFrmPstgXfr.hh"
#include "XrdFrm/XrdFrmTrace.hh"
#include "XrdNet/XrdNetMsg.hh"
#include "XrdOss/XrdOss.hh"
#include "XrdOuc/XrdOucEnv.hh"
#include "XrdOuc/XrdOucMsubs.hh"
#include "XrdOuc/XrdOucProg.hh"
#include "XrdOuc/XrdOucTList.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysTimer.hh"
#include "XrdSys/XrdSysPlatform.hh"
#include "XrdXrootd/XrdXrootdMonitor.hh"

using namespace XrdFrm;

/******************************************************************************/
/*                         L o c a l   C l a s s e s                          */
/******************************************************************************/

struct XrdFrmPstgXrq
{      XrdFrmPstgXrq         *Next;
       XrdOucTList           *NoteList;
       XrdFrmPstgReq::Request reqData;
       static const int       PFNSZ = 1024;
       char                   PFN[PFNSZ];
       int                    pfnEnd;
       int                    Slot;
       XrdFrmPstgXrq() {}
};
  
/******************************************************************************/
/*                               S t a t i c s                                */
/******************************************************************************/
  
XrdSysMutex               XrdFrmPstgXfr::hMutex;
XrdOucHash<XrdFrmPstgXrq> XrdFrmPstgXfr::hTab;

XrdSysMutex               XrdFrmPstgXfr::qMutex;
XrdSysSemaphore           XrdFrmPstgXfr::qReady(0);
XrdSysSemaphore           XrdFrmPstgXfr::qAvail(0);

XrdFrmPstgXrq            *XrdFrmPstgXfr::First = 0;
XrdFrmPstgXrq            *XrdFrmPstgXfr::Last  = 0;
XrdFrmPstgXrq            *XrdFrmPstgXfr::Free  = 0;

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdFrmPstgXfr::XrdFrmPstgXfr()
{

// Construct a program object
//
   xfrCmd = new XrdOucProg(&Say);
}

/******************************************************************************/
/*                                  I n i t                                   */
/******************************************************************************/
  
int XrdFrmPstgXfr::Init()
{
   XrdFrmPstgXrq *xP;
   int n = Config.xfrMax*2;

// Create twice as many free queue elements as we have xfr agents
//
   while(n--)
        {xP = new XrdFrmPstgXrq; 
         xP->Next = Free;
         Free     = xP;
         qAvail.Post();
        }

// All done
//
   return 1;
}

/******************************************************************************/
/* Private:                       N o t i f y                                 */
/******************************************************************************/
  
int XrdFrmPstgXfr::Notify(XrdFrmPstgReq::Request *rP, int rc, const char *msg)
{
   static const char *isFile = "file:///";
   static const int   lnFile = 8;
   static const char *isUDP  = "udp://";
   static const int   lnUDP  = 6;
   char msgbuff[4096];
   int n;

// Check if message really needs to be sent
//
   if ((!rc && !(rP->Options & XrdFrmPstgReq::msgSucc))
   ||  ( rc && !(rP->Options & XrdFrmPstgReq::msgFail))) return 0;

// Check for file destination
//
   if (!strncmp(rP->Notify, isFile, lnFile))
      {if (rc) n = sprintf(msgbuff, "stage %s %s %s\n", 
                          (rc > 1 ? "ENOENT" : "BAD"), rP->LFN, msg);
          else n = sprintf(msgbuff, "stage OK %s\n", rP->LFN);
       Send2File((rP->Notify)+lnFile, msgbuff, n);
       return 0;
      }

// Check for udp destination
//
   if (!strncmp(rP->Notify, isUDP,  lnUDP))
      {char *txtP, *dstP = (rP->Notify)+lnUDP;
       if ((txtP = index(dstP, '/'))) *txtP++ = '\0';
          else txtP = (char *)"";
       n = sprintf(msgbuff, "%s %s %s %s", (rc ? "unprep" : "ready"),
                            rP->ID, txtP, rP->LFN);
       Send2UDP(dstP, msgbuff, n);
       return 0;
      }

// Issue warning as we don't yet support mail or tcp notifications
//
   Say.Emsg("Notify", "Unsupported notification path '", rP->Notify, "'.");
   return 0;
}

/******************************************************************************/
/* Public:                         Q u e u e                                  */
/******************************************************************************/
  
int XrdFrmPstgXfr::Queue(XrdFrmPstgReq::Request *rP, int Slot)
{
   EPNAME("Queue");
   XrdFrmPstgXrq *xP;
   struct stat buf;
   char lclpath[XrdFrmPstgXrq::PFNSZ];

// First check if this request is active or pending
//
   hMutex.Lock();
   if ((xP = hTab.Find(rP->LFN)))
      {if (rP->Options & (XrdFrmPstgReq::msgSucc | XrdFrmPstgReq::msgFail)
       &&  strcmp(xP->reqData.Notify, rP->Notify))
          {XrdOucTList *tP = new XrdOucTList(rP->Notify, 0, xP->NoteList);
           xP->NoteList = tP;
          }
       hMutex.UnLock();
       DEBUG(rP->LFN <<" already in progress.");
       rQueue[Slot]->Del(rP);
       return 0;
      }
   hMutex.UnLock();

// Obtain the local name
//
   if (!Config.LocalPath(rP->LFN, lclpath, sizeof(lclpath)-16))
      {rQueue[Slot]->Del(rP);
       return Notify(rP, 1, "Unable to generate pfn");
      }

// Check if the file exists
//
   if (!stat(lclpath, &buf))
      {DEBUG(lclpath <<" exists; staging skipped.");
       rQueue[Slot]->Del(rP);
       return Notify(rP, 0);
      }

// Obtain a queue slot
//
   do {qMutex.Lock();
       if ((xP = Free)) break;
       qMutex.UnLock();
       qAvail.Wait();
      } while(!xP);
   Free = xP->Next;
   qMutex.UnLock();

// Initialize the slot
//
   xP->Next     = 0;
   xP->NoteList = 0;
   xP->Slot     = Slot;
   xP->reqData  = *rP;
   strcpy(xP->PFN, lclpath);
   xP->pfnEnd = strlen(lclpath);

// Add this to the table of requests
//
   hMutex.Lock();
   hTab.Add(xP->reqData.LFN, xP, 0, Hash_keep);
   hMutex.UnLock();

// Place request in the transfer queue
//
   qMutex.Lock();
   if (Last) {Last->Next = xP; Last = xP;}
      else    Last = First = xP;
   qMutex.UnLock();
   qReady.Post();

// All done
//
   return 1;
}

/******************************************************************************/
/* Private:                    S e n d 2 F i l e                              */
/******************************************************************************/
  
void XrdFrmPstgXfr::Send2File(char *Dest, char *Msg, int Mln)
{
   EPNAME("Notify");
   int FD;

// Do some debugging
//
   DEBUG("sending '" <<Msg <<"' via " <<Dest);

// Open the file
//
   if ((FD = open(Dest, O_WRONLY)) < 0)
      {Say.Emsg("Notify", errno, "send notification via", Dest); return;}

// Write the message
//
   if (write(FD, Msg, Mln) < 0)
      Say.Emsg("Notify", errno, "send notification via", Dest);
   close(FD);
}

/******************************************************************************/
/* Private:                     S e n d 2 U D P                               */
/******************************************************************************/

void XrdFrmPstgXfr::Send2UDP(char *Dest, char *Msg, int Mln)
{
   EPNAME("Notify");
   static XrdNetMsg Relay(&Say, 0);

// Do some debugging
//
   DEBUG("sending '" <<Msg <<"' via " <<Dest);
  
// Send off the message
//
   Relay.Send(Msg, Mln, Dest);
}

/******************************************************************************/
/*                                 S t a g e                                  */
/******************************************************************************/
  
const char *XrdFrmPstgXfr::Stage(XrdFrmPstgXrq *xP, int &retcode)
{
   EPNAME("Stage");
   static const mode_t fMode = S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH;
   static const int holdTime = 3*60*60;
   static const int crOpts = (O_CREAT|O_EXCL)<<8|XRDOSS_mkpath;
   XrdOucEnv myEnv(xP->reqData.Opaque?xP->reqData.LFN+xP->reqData.Opaque:0);
   struct stat buf;
   time_t eNow, wrTime, xfrET, inqET;
   char lfnpath[1280];
   int myFD, lfnEnd, rc;
   long long fSize = 0;

// Check if the file exists
//
   retcode = 0;
   if (!stat(xP->PFN, &buf))
      {DEBUG(xP->PFN <<" exists; not staged.");
       return 0;
      }

// Check for a fail file
//
   strcpy(&xP->PFN[xP->pfnEnd], ".fail");
   rc = stat(xP->PFN, &buf);
   if (!rc)
      {if (buf.st_ctime < time(0)+holdTime)
          {retcode = 1; xP->PFN[xP->pfnEnd] = '\0';
           return "request previously failed";
          }
       unlink(xP->PFN);
      }
   xP->PFN[xP->pfnEnd] = '\0';

// Now check for the stop file before commencing any further
//
   if (Config.StopFile)
      {rc = 0;
       while(!stat(Config.StopFile, &buf))
            {if (!rc--)
                {DEBUG("Stop file " <<Config.StopFile
                       <<" exists; staging suspended.");
                 rc = 12;
                }
             XrdSysTimer::Snooze(5);
            }
      }

// Construct the file to which to we originally tranfer the data
//
   lfnEnd = strlen(xP->reqData.LFN);
   strlcpy(lfnpath, xP->reqData.LFN, sizeof(lfnpath)-16);
   strcpy(&lfnpath[lfnEnd], ".anew");
   strcpy(&xP->PFN[xP->pfnEnd], ".anew");

// Setup the command
//
   if (!StageCmd(xP, &myEnv))
      {retcode = 1; return "xfr setup failed";}

// We now need a placeholder in the filesystem for this transfer
//
   Config.ossFS->Unlink(lfnpath);
   rc = Config.ossFS->Create(xP->reqData.User, lfnpath, fMode, myEnv, crOpts);
   if (rc)
      {Say.Emsg("Stage", rc, "create placeholder for", lfnpath);
       retcode = 1; return "create failed";
      }

// Now run the command to get the file and make sure the file is there
// If it is, make sure that if a lock file exists it date/time is greater than
// the file we just staged in; then rename it to be the correct name.
//
   xfrET = time(0);
   if (!(rc = xfrCmd->Run()))
      {if (stat(xP->PFN, &buf))
         {Say.Emsg("stage", lfnpath, "staged but not found!"); rc = 1;}
          else {wrTime = buf.st_mtime;
                fSize  = buf.st_size;
                strcpy(&xP->PFN[xP->pfnEnd+5], ".lock");
                if (!stat(xP->PFN, &buf))
                   {struct utimbuf tbuff;
                    tbuff.actime = tbuff.modtime = wrTime+1;
                    utime(xP->PFN, &tbuff);
                   }
                if ((rc = Config.ossFS->Rename(lfnpath, xP->reqData.LFN)))
                   {Say.Emsg("Stage", rc, "rename", lfnpath); rc = 1;}
               }
      }

// Clean up if we failed otherwise record the action as needed
//
   if (rc)
      {Config.ossFS->Unlink(lfnpath);
       strcpy(&xP->PFN[xP->pfnEnd], ".fail");
       if ((myFD = open(xP->PFN, O_CREAT, fMode)) >= 0) close(myFD);
       xP->PFN[xP->pfnEnd] = '\0';
       if (retcode == -2) {retcode = 2; return "file not found";}
       retcode = 1;
       return "stage failed";
      } else {
        int inqT, xfrT;
        eNow = time(0);
        xfrET = eNow - xfrET; inqET = eNow - time_t(xP->reqData.addTOD);
        if (xfrET <= 0) xfrET = 1;
        inqT = static_cast<int>(inqET); xfrT = static_cast<int>(xfrET);
        DEBUG("sz=" <<fSize <<" xt=" <<xfrT <<" qt=" <<inqT
                    <<"up=" <<xP->reqData.User <<' ' <<lfnpath);
        if (Config.monStage)
           {snprintf(lfnpath+lfnEnd, sizeof(lfnpath)-lfnEnd-1,
                     "\n&qt=%d&sz=%lld&tm=%d",inqT, fSize, xfrT);
            XrdXrootdMonitor::Map(XROOTD_MON_MAPSTAG,xP->reqData.User,lfnpath);
           }
      }

// All done
//
   xP->PFN[xP->pfnEnd] = '\0';
   return 0;
}

/******************************************************************************/
/* Private:                     S t a g e C m d                               */
/******************************************************************************/
  
int XrdFrmPstgXfr::StageCmd(XrdFrmPstgXrq *xP, XrdOucEnv *theEnv)
{
   char *pdata[XrdOucMsubs::maxElem + 2], *cP;
   int   pdlen[XrdOucMsubs::maxElem + 2], i, k, n;

   XrdOucMsubsInfo 
              Info(xP->reqData.User, theEnv, Config.the_N2N, xP->reqData.LFN, 0,
                   xP->reqData.Prty,
                   xP->reqData.Options & XrdFrmPstgReq::stgRW ? O_RDWR:O_RDONLY,
                   StageOpt(xP), xP->reqData.ID, xP->PFN);

// Substitute in the parameters
//
   k = Config.xfrVec->Subs(Info, pdata, pdlen);

// Catenate all of the arguments
//
   *cmdBuff = '\0'; n = sizeof(cmdBuff) - 4; cP = cmdBuff;
   for (i = 0; i < k; i++)
       {n -= pdlen[i];
        if (n < 0)
           {Say.Emsg("Stage",E2BIG,"build command line for", xP->reqData.LFN);
            return 0;
           }
        strcpy(cP, pdata[i]); cP += pdlen[i];
       }

// Now setup the command
//
   return (xfrCmd->Setup(cmdBuff, &Say) == 0);
}

/******************************************************************************/
/* Private:                     S t a g e O p t                               */
/******************************************************************************/
  
const char *XrdFrmPstgXfr::StageOpt(XrdFrmPstgXrq *xP)
{

    if (xP->reqData.Options & XrdFrmPstgReq::msgFail
    &&  xP->reqData.Options & XrdFrmPstgReq::msgSucc) return "fn";

    if (xP->reqData.Options & XrdFrmPstgReq::msgSucc) return "nq";

    return "f";
}

/******************************************************************************/
/* Public:                         S t a r t                                  */
/******************************************************************************/
  
void XrdFrmPstgXfr::Start()
{
   EPNAME("Stage");  // Wong but looks better
   XrdFrmPstgXrq *xP;
   XrdOucTList *tP;
   const char *Msg;
   int rc;

// Endless loop looking for requests
//
   while(1)
        {do {qReady.Wait();
             qMutex.Lock();
             if ((xP = First)) {if (!(First = xP->Next)) Last = 0;}
             qMutex.UnLock();
            } while(!xP);

         DEBUG("starting " <<xP->PFN <<" for " <<xP->reqData.User);
         Msg = Stage(xP, rc);
         DEBUG("complete " <<xP->PFN <<" rc=" <<rc <<' ' <<(Msg ? Msg : ""));

         hMutex.Lock(); hTab.Del(xP->reqData.LFN); hMutex.UnLock();

         do {Notify(&(xP->reqData), rc, Msg);
             if ((tP = xP->NoteList))
                {strcpy(xP->reqData.Notify, tP->text);
                 xP->NoteList = tP->next;
                 delete tP;
                }
            } while(tP);

         rQueue[xP->Slot]->Del(&(xP->reqData));
        }
}
