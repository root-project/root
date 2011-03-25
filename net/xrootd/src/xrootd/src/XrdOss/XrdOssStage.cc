/******************************************************************************/
/*                                                                            */
/*                        X r d O s s S t a g e . c c                         */
/*                                                                            */
/*                                                                            */
/* (C) 2006 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*                DE-AC03-76-SFO0515 with the Deprtment of Energy             */
/******************************************************************************/

/* The XrdOssStage() routine is responsible for getting data from a remote
   location to the local filesystem. The current implementation invokes a
   shell script to perform the "staging".

   This routine is thread-safe if compiled with:
   AIX: -D_THREAD_SAFE
   SUN: -D_REENTRANT
*/

#include <unistd.h>
#include <errno.h>
#include <strings.h>
#include <signal.h>
#include <stdio.h>
#include <time.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <sys/stat.h>
#include <sys/wait.h>

#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSys/XrdSysPlatform.hh"
#include "XrdOss/XrdOssApi.hh"
#include "XrdOss/XrdOssError.hh"
#include "XrdOss/XrdOssOpaque.hh"
#include "XrdOss/XrdOssStage.hh"
#include "XrdOuc/XrdOuca2x.hh"
#include "XrdOuc/XrdOucEnv.hh"
#include "XrdOuc/XrdOucMsubs.hh"
#include "XrdOuc/XrdOucName2Name.hh"
#include "XrdOuc/XrdOucProg.hh"
#include "XrdOuc/XrdOucReqID.hh"
#include "XrdFrm/XrdFrmProxy.hh"

/******************************************************************************/
/*            G l o b a l s   a n d   S t a t i c   O b j e c t s             */
/******************************************************************************/

extern XrdSysError OssEroute;
 
XrdSysMutex          XrdOssStage_Req::StageMutex;
XrdSysSemaphore      XrdOssStage_Req::ReadyRequest;
XrdOssStage_Req      XrdOssStage_Req::StageQ((XrdOssStage_Req *)0);

#define XRDOSS_FAIL_FILE (char *)".fail"

/******************************************************************************/
/*                    E x t e r n a l   F u n c t i o n s                     */
/******************************************************************************/
  
extern unsigned long XrdOucHashVal(const char *KeyVal);

int XrdOssScrubScan(const char *key, char *cip, void *xargp) {return 0;}

/******************************************************************************/
/*                        o o s s _ F i n d _ P r t y                         */
/******************************************************************************/
  
int XrdOssFind_Prty(XrdOssStage_Req *req, void *carg)
{
    int prty = *(int *)carg;
    return (req->prty >= prty);
}
  
/******************************************************************************/
/*                         o o s s _ F i n d _ R e q                          */
/******************************************************************************/

int XrdOssFind_Req(XrdOssStage_Req *req, void *carg)
{
    XrdOssStage_Req *xreq = (XrdOssStage_Req *)carg;
    return (req->hash == xreq->hash) && !strcmp(req->path, xreq->path);
}

/******************************************************************************/
/*                                 S t a g e                                  */
/******************************************************************************/
  
int XrdOssSys::Stage(const char *Tid, const char *fn, XrdOucEnv &env, 
                     int Oflag, mode_t Mode, unsigned long long Popts)
{
// Use the appropriate method here: queued staging or real-time staging
//
   return (StageRealTime ? Stage_RT(Tid, fn, env, Popts)
                         : Stage_QT(Tid, fn, env, Oflag, Mode));
}

/******************************************************************************/
/*                              S t a g e _ Q T                               */
/******************************************************************************/
  
int XrdOssSys::Stage_QT(const char *Tid, const char *fn, XrdOucEnv &env, 
                        int Oflag, mode_t Mode)
{
   static XrdOucReqID ReqID(static_cast<int>(getpid()),(char *)"localhost",
                                          (unsigned long)0xef000001);
   static XrdSysMutex      PTMutex;
   static XrdOucHash<char> PTable;
   static time_t nextScrub = xfrkeep + time(0);
   char *Found, *pdata[XrdOucMsubs::maxElem + 2];
   int pdlen[XrdOucMsubs::maxElem + 2];
   time_t cTime, mTime, tNow = time(0);

// If there is a fail file and the error occured within the hold time,
// fail the request. Otherwise, try it again. This avoids tight loops.
//
   if ((cTime = HasFile(fn, XRDOSS_FAIL_FILE, &mTime))
   && xfrhold && (tNow - cTime) < xfrhold)
      return (mTime != 2 ? -XRDOSS_E8009 : -ENOENT);

// If enough time has gone by between the last scrub, do it now
//
   if (nextScrub < tNow) 
      {PTMutex.Lock(); 
       if (nextScrub < tNow) 
          {PTable.Apply(XrdOssScrubScan, (void *)0);
           nextScrub = xfrkeep + tNow;
          }
       PTMutex.UnLock();
      }

// Check if this file is already being brought in. If so, return calculated
// wait time for this file.
//
   PTMutex.Lock();
   Found = PTable.Add(fn, 0, xfrkeep, Hash_data_is_key);
   PTMutex.UnLock();
   if (Found) return CalcTime();

// Check if we should use our built-in frm interface
//
   if (StageFrm)
      {char idbuff[64];
       ReqID.ID(idbuff, sizeof(idbuff));
       int n;
       return (n = StageFrm->Add('+', fn, env.Env(n), Tid, idbuff,
                   StageEvents, StageAction)) ? n : CalcTime();
      }

// If a stagemsg template was not defined; use our default template
//
   if (!StageSnd)
      {char idbuff[64], usrbuff[512];
       ReqID.ID(idbuff, sizeof(idbuff));
       if (!StageFormat)
      {pdata[0] = (char *)"+ ";  pdlen[0] = 2;}
else  {pdlen[0] = getID(Tid,env,usrbuff,sizeof(usrbuff)); pdata[0] = usrbuff;}
       pdata[1] = idbuff;        pdlen[1] = strlen(idbuff);  // Request ID
       pdata[2] = (char *)" ";   pdlen[2] = 1;
       pdata[3] = StageEvents;   pdlen[3] = StageEvSize;     // notification
       pdata[4] = (char *)" ";   pdlen[4] = 1;
       pdata[5] = (char *)"0 ";  pdlen[5] = 2;               // prty
       pdata[6] = StageAction;   pdlen[6] = StageActLen;     // action
       pdata[7] = (char *)fn;    pdlen[7] = strlen(fn);
       pdata[8] = (char *)"\n";  pdlen[8] = 1;
       pdata[9] = 0;             pdlen[9] = 0;
       if (StageProg->Feed((const char **)pdata, pdlen)) return -XRDOSS_E8025;
      } else {
       XrdOucMsubsInfo Info(Tid, &env, lcl_N2N, fn, 0, 
                            Mode, Oflag, StageAction, "n/a");
       int k = StageSnd->Subs(Info, pdata, pdlen);
       pdata[k]   = (char *)"\n"; pdlen[k++] = 1;
       pdata[k]   = 0;            pdlen[k]   = 0;
       if (StageProg->Feed((const char **)pdata, pdlen)) return -XRDOSS_E8025;
      }

// All done
//
   return CalcTime();
}

/******************************************************************************/
/*                              S t a g e _ R T                               */
/******************************************************************************/
  
int XrdOssSys::Stage_RT(const char *Tid, const char *fn, XrdOucEnv &env,
                        unsigned long long Popts)
{
    extern int XrdOssFind_Prty(XrdOssStage_Req *req, void *carg);
    XrdSysMutexHelper StageAccess(XrdOssStage_Req::StageMutex);
    XrdOssStage_Req req, *newreq, *oldreq;
    struct stat statbuff;
    extern int XrdOssFind_Req(XrdOssStage_Req *req, void *carg);
    char actual_path[MAXPATHLEN+1], *remote_path;
    char *val;
    int rc, prty;

// If there is no stagecmd then return an error
//
   if (!StageCmd) return -XRDOSS_E8006;

// Set up the minimal new request structure
//
   req.hash = XrdOucHashVal(fn);
   req.path = strdup(fn);

// Check if this file is already being brought in. If it's in the chain but
// has an error associated with it. If the error window is still in effect,
// check if a fail file exists. If one does exist, fail the request. If it
// doesn't exist or if the window has expired, delete the error element and
// retry the request. This keeps us from getting into tight loops.
//
   if ((oldreq = XrdOssStage_Req::StageQ.fullList.Apply(XrdOssFind_Req,(void *)&req)))
      {if (!(oldreq->flags & XRDOSS_REQ_FAIL)) return CalcTime(oldreq);
       if (oldreq->sigtod > time(0) && HasFile(fn, XRDOSS_FAIL_FILE))
          return (oldreq->flags & XRDOSS_REQ_ENOF ? -ENOENT : -XRDOSS_E8009);
       delete oldreq;
      }

// Generate remote path
//
   if (rmt_N2N)
      if ((rc = rmt_N2N->lfn2rfn(fn, actual_path, sizeof(actual_path)))) 
         return rc;
         else remote_path = actual_path;
      else remote_path = (char *)fn;

// Obtain the size of this file, if possible. Note that an exposure exists in
// that a request for the file may come in again before we have the size. This
// is ok, it just means that we'll be off in our time estimate
//
   if (Popts & XRDEXP_NOCHECK) statbuff.st_size = 1024*1024*1024;
      else {StageAccess.UnLock();
            if ((rc = MSS_Stat(remote_path, &statbuff))) return rc;
            StageAccess.Lock(&XrdOssStage_Req::StageMutex);
           }

// Create a new request
//
   if (!(newreq = new XrdOssStage_Req(req.hash, fn)))
       return OssEroute.Emsg("Stage",-ENOMEM,"create req for",fn);

// Add this request to the list of requests
//
   XrdOssStage_Req::StageQ.fullList.Insert(&(newreq->fullList));

// Recalculate the cumalitive pending stage queue and
//
   newreq->size = statbuff.st_size;
   pndbytes += statbuff.st_size;

// Calculate the system priority
//
   if (!(val = env.Get(OSS_SYSPRTY))) prty = OSS_USE_PRTY;
      else if (XrdOuca2x::a2i(OssEroute,"system prty",val,&prty,0)
           || prty > OSS_MAX_PRTY) return -XRDOSS_E8010;
           else prty = prty << 8;

// Calculate the user priority
//
   if (OptFlags & XrdOss_USRPRTY && (val = env.Get(OSS_USRPRTY)))
      {if (XrdOuca2x::a2i(OssEroute,"user prty",val,&rc,0)
       || rc > OSS_MAX_PRTY) return -XRDOSS_E8010;
       prty |= rc;
      }

// Queue the request at the right position and signal an xfr thread
//
   if ((oldreq = XrdOssStage_Req::StageQ.pendList.Apply(XrdOssFind_Prty,(void *)&prty)))
                           oldreq->pendList.Insert(&newreq->pendList);
      else XrdOssStage_Req::StageQ.pendList.Insert(&newreq->pendList);
   XrdOssStage_Req::ReadyRequest.Post();

// Return the estimated time to arrival
//
   return CalcTime(newreq);
}
  
/******************************************************************************/
/*                              S t a g e _ I n                               */
/******************************************************************************/
  
void *XrdOssSys::Stage_In(void *carg)
{
    XrdOucDLlist<XrdOssStage_Req> *rnode;
    XrdOssStage_Req              *req;
    int rc, alldone = 0;
    time_t etime;

      // Wait until something shows up in the ready queue and process
      //
   do   {XrdOssStage_Req::ReadyRequest.Wait();

      // Obtain exclusive control over the queues
      //
         XrdOssStage_Req::StageMutex.Lock();

      // Check if we really have something in the queue
      //
         if (XrdOssStage_Req::StageQ.pendList.Singleton())
            {XrdOssStage_Req::StageMutex.UnLock();
             continue;
            }

      // Remove the last entry in the queue
      //
         rnode = XrdOssStage_Req::StageQ.pendList.Prev();
         req   = rnode->Item();
         rnode->Remove();
         req->flags |= XRDOSS_REQ_ACTV;

      // Account for bytes being moved
      //
         pndbytes -= req->size;
         stgbytes += req->size;

      // Bring in the file (don't hold the stage lock while doing so)
      //
         XrdOssStage_Req::StageMutex.UnLock();
         etime = time(0);
         rc = GetFile(req);
         etime = time(0) - etime;
         XrdOssStage_Req::StageMutex.Lock();

      // Account for resources and adjust xfr rate
      //
         stgbytes -= req->size;
         if (!rc)
            {if (etime > 1) 
                {xfrspeed=((xfrspeed*(totreqs+1))+(req->size/etime))/(totreqs+1);
                 if (xfrspeed < 512000) xfrspeed = 512000;
                }
             totreqs++;          // Successful requests
             totbytes += req->size;
             delete req;
            }
            else {req->flags &= ~XRDOSS_REQ_ACTV;
                  req->flags |= (rc == 2 ? XRDOSS_REQ_ENOF : XRDOSS_REQ_FAIL);
                  req->sigtod = xfrhold + time(0);
                  badreqs++;
                 }

      // Check if we should continue or be terminated and unlock staging
      //
         if ((alldone = (xfrthreads < xfrtcount)))
            xfrtcount--;
         XrdOssStage_Req::StageMutex.UnLock();

         } while (!alldone);

// Notmally we would never get here
//
   return (void *)0;
}
  
  
/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                              C a l c T i m e                               */
/******************************************************************************/
  
int XrdOssSys::CalcTime()
{

// For queued staging we have no good way to estimate the time, as of yet.
// So, return 60 seconds. Note that the following code, which is far more
// elaborate, rarely returns the right estimate anyway.
//
   return (StageAsync ? -EINPROGRESS : 60);
}

int XrdOssSys::CalcTime(XrdOssStage_Req *req) // StageMutex lock held!
{
    unsigned long long tbytes = req->size + stgbytes/2;
    int xfrtime, numq = 1;
    time_t now;
    XrdOssStage_Req *rqp = req;

// Return an EINP{ROG if we are doing async staging
//
   if (StageAsync) return -EINPROGRESS;

// If the request is active, recalculate the time based on previous estimate
//
   if (req->flags & XRDOSS_REQ_ACTV) 
      {if ((xfrtime = req->sigtod - time(0)) > xfrovhd) return xfrtime;
          else return (xfrovhd < 4 ? 2 : xfrovhd / 2);
      }

// Calculate the number of pending bytes being transfered plus 1/2 of the
// current number of bytes being transfered
//
    while ((rqp=(rqp->pendList.Next()->Item()))) {tbytes += rqp->size; numq++;}

// Calculate when this request should be completed
//
    now = time(0);
    req->sigtod = tbytes / xfrspeed + numq * xfrovhd + now;

// Calculate the time it will take to get this file
//
   if ((xfrtime = req->sigtod - now) <= xfrovhd) return xfrovhd+3;
   return xfrtime;
}
  
/******************************************************************************/
/*                               G e t F i l e                                */
/******************************************************************************/

int XrdOssSys::GetFile(XrdOssStage_Req *req)
{
   char rfs_fn[MAXPATHLEN+1];
   char lfs_fn[MAXPATHLEN+1];
   int retc;

// Convert the local filename and generate the corresponding remote name.
//
   if ( (retc =  GenLocalPath(req->path, lfs_fn)) ) return retc;
   if ( (retc = GenRemotePath(req->path, rfs_fn)) ) return retc;

// Run the command to get the file
//
   if ((retc = StageProg->Run(rfs_fn, lfs_fn)))
      {OssEroute.Emsg("Stage", retc, "stage", req->path);
       return (retc == 2 ? -ENOENT : -XRDOSS_E8009);
      }

// All went well
//
   return 0;
}

/******************************************************************************/
/*                                 g e t I D                                  */
/******************************************************************************/
  
int XrdOssSys::getID(const char *Tid, XrdOucEnv &Env, char *buff, int bsz)
{
   char *bP;
   int n;

// The buffer always starts with a '+'
//
   *buff = '+'; bP = buff+1; bsz -= 3;

// Get the trace id
//
   if (Tid && (n = strlen(Tid)) <= bsz) {strcpy(bP, Tid); bP += n;}

// Insert space
//
   *bP++ = ' '; *bP = '\0';
   return bP - buff;
}

/******************************************************************************/
/*                               H a s F i l e                                */
/******************************************************************************/
  
time_t XrdOssSys::HasFile(const char *fn, const char *fsfx, time_t *mTime)
{
    struct stat statbuff;
    int fnlen;
    char path[MAXPATHLEN+1];
    char *pp = path;

// Copy the path with possible conversion
//
   if (GenLocalPath(fn, path)) return 0;

// Add the suffix
//
   fnlen = strlen(path);
   if ((fnlen + strlen(fsfx)) >= sizeof(path)) return 0;
   pp += fnlen;
   strcpy(pp, fsfx);

// Now check if the file actually exists
//
   if (stat(path, &statbuff)) return 0;
   if (mTime) *mTime = statbuff.st_mtime;
   return statbuff.st_ctime;
}
