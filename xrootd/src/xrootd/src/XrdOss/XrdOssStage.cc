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

//         $Id$

const char *XrdOssStageCVSID = "$Id$";

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
#include <iostream.h>
#include <signal.h>
#include <stdio.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>

#include "XrdOss/XrdOssApi.hh"
#include "XrdOss/XrdOssCache.hh"
#include "XrdOss/XrdOssError.hh"
#include "XrdOss/XrdOssLock.hh"
#include "XrdOss/XrdOssOpaque.hh"
#include "XrdOuc/XrdOuca2x.hh"
#include "XrdOuc/XrdOucEnv.hh"
#include "XrdOuc/XrdOucMsubs.hh"
#include "XrdOuc/XrdOucName2Name.hh"
#include "XrdOuc/XrdOucProg.hh"
#include "XrdOuc/XrdOucReqID.hh"

/******************************************************************************/
/*           G l o b a l   E r r o r   R o u t i n g   O b j e c t            */
/******************************************************************************/

extern XrdSysError OssEroute;
 
/******************************************************************************/
/*             H a s h   C o m p u t a t i o n   F u n c t i o n              */
/******************************************************************************/
  
extern unsigned long XrdOucHashVal(const char *KeyVal);

/******************************************************************************/
/*              O t h e r   E x t e r n a l   F u n c t i o n s               */
/******************************************************************************/
  
int XrdOssScrubScan(const char *key, char *cip, void *xargp) {return 0;}

/******************************************************************************/
/*                                 S t a g e                                  */
/******************************************************************************/
  
int XrdOssSys::Stage(const char *Tid, const char *fn, XrdOucEnv &env, 
                     int Oflag, mode_t Mode)
{
// Use the appropriate method here: queued staging or real-time staging
//
   return (StageRealTime ? Stage_RT(Tid, fn, env)
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
   int rc, pdlen[XrdOucMsubs::maxElem + 2];
   time_t tNow = time(0);

// If there is a fail file and the error occured within the hold time,
// fail the request. Otherwise, try it again. This avoids tight loops.
//
   if ((rc = HasFile(fn, XRDOSS_FAIL_FILE))
   && xfrhold && (tNow - rc) < xfrhold)  return -XRDOSS_E8009;

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

// If a stagemsg template was not defined; use our default template
//
   if (!StageSnd)
      {char idbuff[64];
       ReqID.ID(idbuff, sizeof(idbuff));
       pdata[0] = (char *)"+ ";  pdlen[0] = 2;
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
       XrdOucMsubsInfo Info(Tid, &env, lcl_N2N, fn, 0, Mode, Oflag);
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
  
int XrdOssSys::Stage_RT(const char *Tid, const char *fn, XrdOucEnv &env)
{
    extern int XrdOssFind_Prty(XrdOssCache_Req *req, void *carg);
    XrdOssCache_Req req, *newreq, *oldreq;
    XrdOssCache_Lock CacheAccess; // Obtains & releases the cache lock
    struct stat statbuff;
    extern int XrdOssFind_Req(XrdOssCache_Req *req, void *carg);
    char actual_path[XrdOssMAX_PATH_LEN+1], *remote_path;
    char *val;
    int rc, prty;

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
   if ((oldreq = StageQ.fullList.Apply(XrdOssFind_Req, (void *)&req)))
      {if (!(oldreq->flags & XRDOSS_REQ_FAIL)) return CalcTime(oldreq);
       if (oldreq->sigtod > time(0) && HasFile(fn, XRDOSS_FAIL_FILE))
          return -XRDOSS_E8009;
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
   CacheAccess.UnLock();
   if ((rc = MSS_Stat(remote_path, &statbuff))) return rc;
   CacheAccess.Lock();

// Create a new request
//
   if (!(newreq = new XrdOssCache_Req(req.hash, fn)))
       return OssEroute.Emsg("XrdOssStage",-ENOMEM,"create req for",fn);

// Add this request to the list of requests
//
   StageQ.fullList.Insert(&(newreq->fullList));

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
   if (OptFlags & XrdOss_USRPRTY)
      if ((val = env.Get(OSS_USRPRTY))
      && (XrdOuca2x::a2i(OssEroute,"user prty",val,&rc,0)
          || rc > OSS_MAX_PRTY)) return -XRDOSS_E8010;
         else prty |= rc;

// Queue the request at the right position and signal an xfr thread
//
   if ((oldreq = StageQ.pendList.Apply(XrdOssFind_Prty, (void *)&prty)))
          oldreq->pendList.Insert(&newreq->pendList);
      else StageQ.pendList.Insert(&newreq->pendList);
   ReadyRequest.Post();

// Return the estimated time to arrival
//
   return CalcTime(newreq);
}
  
/******************************************************************************/
/*                              S t a g e _ I n                               */
/******************************************************************************/
  
void *XrdOssSys::Stage_In(void *carg)
{
    XrdOucDLlist<XrdOssCache_Req> *rnode;
    XrdOssCache_Req              *req;
    int rc, alldone = 0;
    time_t etime;

      // Wait until something shows up in the ready queue and process
      //
   do   {ReadyRequest.Wait();

      // Obtain exclusive control over the queues
      //
         CacheContext.Lock();

      // Check if we really have something in the queue
      //
         if (StageQ.pendList.Singleton())
            {CacheContext.UnLock();
             continue;
            }

      // Remove the last entry in the queue
      //
         rnode = StageQ.pendList.Prev();
         req   = rnode->Item();
         rnode->Remove();
         req->flags |= XRDOSS_REQ_ACTV;

      // Account for bytes being moved
      //
         pndbytes -= req->size;
         stgbytes += req->size;

      // Bring in the file (don't hold the cache lock while doing so)
      //
         CacheContext.UnLock();
         etime = time(0);
         rc = GetFile(req);
         etime = time(0) - etime;
         CacheContext.Lock();

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
                  req->flags |=  XRDOSS_REQ_FAIL;
                  req->sigtod = xfrhold + time(0);
                  badreqs++;
                 }

      // Check if we should continue or be terminated and unlock the cache
      //
         if ((alldone = (xfrthreads < xfrtcount)))
            xfrtcount--;
         CacheContext.UnLock();

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

int XrdOssSys::CalcTime(XrdOssCache_Req *req) // CacheContext lock held!
{
    unsigned long long tbytes = req->size + stgbytes/2;
    int xfrtime, numq = 1;
    time_t now;
    XrdOssCache_Req *rqp = req;

// Return an EINP{ROG if we are doing async staging
//
   if (StageAsync) return -EINPROGRESS;

// If the request is active, recalculate the time based on previous estimate
//
   if (req->flags & XRDOSS_REQ_ACTV) 
      if ((xfrtime = req->sigtod - time(0)) > xfrovhd) return xfrtime;
         else return (xfrovhd < 4 ? 2 : xfrovhd / 2);

// Calculate the number of pending bytes being transfered plus 1/2 of the
// current number of bytes being transfered
//
    while ((rqp=(rqp->pendList.Next()->Item()))) {tbytes += rqp->size; numq++;}

// Calculate when this request should be completed
//
    now = time(0);
    req->sigtod = tbytes / xfrspeed + numq * xfrovhd + now;

// Calculate the time it will take to get this file into the cache
//
   if ((xfrtime = req->sigtod - now) <= xfrovhd) return xfrovhd+3;
   return xfrtime;
}
  
/******************************************************************************/
/*                               G e t F i l e                                */
/******************************************************************************/

int XrdOssSys::GetFile(XrdOssCache_Req *req)
{
   char rfs_fn[XrdOssMAX_PATH_LEN+1];
   char lfs_fn[XrdOssMAX_PATH_LEN+1];
   int retc;

// Convert the local filename and generate the corresponding remote name.
//
   if ( (retc =  GenLocalPath(req->path, lfs_fn)) ) return retc;
   if ( (retc = GenRemotePath(req->path, rfs_fn)) ) return retc;

// Run the command to get the file
//
   if ((retc = StageProg->Run(rfs_fn, lfs_fn)))
      {OssEroute.Emsg("Stage", retc, "stage", req->path);
       return -XRDOSS_E8009;
      }

// All went well
//
   return 0;
}

/******************************************************************************/
/*                               H a s F i l e                                */
/******************************************************************************/
  
time_t XrdOssSys::HasFile(const char *fn, const char *fsfx)
{
    struct stat statbuff;
    int fnlen;
    char path[XrdOssMAX_PATH_LEN+1];
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
   return (stat(path, &statbuff) ? 0 : statbuff.st_ctime);
}
