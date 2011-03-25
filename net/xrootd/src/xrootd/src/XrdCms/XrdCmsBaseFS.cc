/******************************************************************************/
/*                                                                            */
/*                       X r d C m s B a s e F S . c c                        */
/*                                                                            */
/* (c) 2011 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include <errno.h>
#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <stdio.h>
  
#include "XProtocol/YProtocol.hh"
#include "XProtocol/XPtypes.hh"

#include "XrdCms/XrdCmsBaseFS.hh"
#include "XrdCms/XrdCmsConfig.hh"
#include "XrdCms/XrdCmsPrepare.hh"
#include "XrdCms/XrdCmsTrace.hh"

#include "XrdOss/XrdOss.hh"

#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysTimer.hh"

using namespace XrdCms;

/******************************************************************************/
/*            E x t e r n a l   T h r e a d   I n t e r f a c e s             */
/******************************************************************************/

void *XrdCmsBasePacer(void *carg)
      {((XrdCmsBaseFS *)carg)->Pacer();
       return (void *)0;
      }

void *XrdCmsBaseRunner(void *carg)
      {((XrdCmsBaseFS *)carg)->Runner();
       return (void *)0;
      }

/******************************************************************************/
/* Private:                       B y p a s s                                 */
/******************************************************************************/

int XrdCmsBaseFS::Bypass()
{
   static XrdSysTimer Window;

// If we are not timing requests, we can bypass (typically checked beforehand)
//
   if (!theQ.rLimit) return 1;

// If this is a fixed rate queue then we cannot bypass
//
   if (Fixed) return 0;

// Check if we can reset the number of requests that can be issued inline. We
// do this to bypass the queue unless until we get flooded by requests.
//
   theQ.Mutex.Lock();
   if (!theQ.rLeft && !theQ.pqFirst)
      {unsigned long Interval = 0;
       Window.Report(Interval);
       if (Interval >= 450)
          {theQ.rLeft = theQ.rAgain;
           Window.Reset();
           cerr <<"BYPASS " <<Interval <<"ms left=" <<theQ.rLeft <<endl;
          }
      }

// At this point we may or may not have freebies left
//
   if (theQ.rLeft > 0)
      {theQ.rLeft--; theQ.Mutex.UnLock();
       return 1;
      }

// This request must be queued
//
   theQ.Mutex.UnLock();
   return 0;
}
  
/******************************************************************************/
/* Public:                        E x i s t s                                 */
/******************************************************************************/
  
int XrdCmsBaseFS::Exists(XrdCmsRRData &Arg, XrdCmsPInfo &Who, int noLim)
{
   int aOK, fnPos;

// If we cannot do this locally, then we need to forward the request but only
// if we have a route. Otherwise, just indicate that queueing is necessary.
//
   if (!lclStat)
      {aOK = (!theQ.rLimit || noLim || (!Fixed && Bypass()));
       if (Who.rovec) Queue(Arg, Who, -(Arg.PathLen-1), !aOK);
       return 0;
      }

// If directory checking is enabled, find where the directory component ends 
// and then check if we even have this directory.
//
   if (dmLife)
      {for (fnPos=Arg.PathLen-2; fnPos >= 0 && Arg.Path[fnPos] != '/'; fnPos--);
       if (fnPos > 0 && !hasDir(Arg.Path, fnPos)) return -1;
      } else fnPos = 0;

// If we are not limiting requests, or not limiting everyone and this is not
// a meta-manager, or we are not timing requests and can skip the queue; then
// issue the fstat() inline and report back the result.
//
   if (!theQ.rLimit || noLim || (Fixed && Bypass()))
      return Exists(Arg.Path, fnPos);

// We can't do this now, so forcibly queue the request
//
   if (Who.rovec) Queue(Arg, Who, fnPos, 1);
   return 0;
}

/******************************************************************************/

int XrdCmsBaseFS::Exists(char *Path, int fnPos, int UpAT)
{
   EPNAME("Exists");
   static struct dMoP dirMiss = {0}, dirPres = {1};
   struct stat buf;
   int Opts = (UpAT ? XRDOSS_resonly|XRDOSS_updtatm : XRDOSS_resonly);

// If directory checking is enabled, find where the directory component ends 
// if so requested.
//
   if (fnPos < 0 && dmLife)
      {for (fnPos = -(fnPos+1); fnPos >= 0 && Path[fnPos] != '/'; fnPos--);
       if (fnPos > 0 && !hasDir(Path, fnPos)) return -1;
      }

// Issue stat() via oss plugin. If it succeeds, return result.
//
   if (!Config.ossFS->Stat(Path, &buf, Opts))
      {if ((buf.st_mode & S_IFMT) == S_IFREG)
          return (buf.st_mode & S_ISUID ? CmsHaveRequest::Pending
                                        : CmsHaveRequest::Online);

       return (buf.st_mode & S_IFMT) == S_IFDIR ? CmsHaveRequest::Online : -1;
      }

// The entry does not exist but if we are a staging server then it may be in
// the prepare queue in which case we must say that it is pending arrival.
//
   if (Config.DiskSS && PrepQ.Exists(Path)) return CmsHaveRequest::Pending;

// The entry does not exist. Check if the directory exists and if not, put it
// in our directory missing table so we don't keep hitting this directory.
// This is disabled by default and enabled by the cms.dfs directive.
//
   if (fnPos > 0 && dmLife)
      {struct dMoP *xVal = &dirMiss;
       int xLife = dmLife;
       Path[fnPos] = '\0';
       if (!Config.ossFS->Stat(Path, &buf, XRDOSS_resonly))
          {xLife = dpLife; xVal = &dirPres;}
       fsMutex.Lock();
       fsDirMP.Rep(Path, xVal, xLife, Hash_keepdata);
       fsMutex.UnLock();
       DEBUG("add " <<xLife <<(xVal->Present ? " okdir " : " nodir ") <<Path);
       Path[fnPos] = '/';
      }
   return -1;
}

/******************************************************************************/
/* Private:                       h a s D i r                                 */
/******************************************************************************/
  
int XrdCmsBaseFS::hasDir(char *Path, int fnPos)
{
   struct dMoP *dP;
   int Have;

// Strip to directory and check if we have it
//
   Path[fnPos] = '\0';
   fsMutex.Lock();
   Have = ((dP = fsDirMP.Find(Path)) ? dP->Present : 1);
   fsMutex.UnLock();
   Path[fnPos] = '/';
   return Have;
}

/******************************************************************************/
/*                                  I n i t                                   */
/******************************************************************************/
  
void XrdCmsBaseFS::Init(int Opts, int DMLife, int DPLife)
{

// Set values.
//
   dmLife  = DMLife;
   dpLife  = DPLife ? DPLife : DMLife * 10;
   Server  = (Opts & Servr) != 0;
   lclStat = (Opts & Cntrl) != 0 || Server;
   preSel  = (Opts & Immed) == 0;
   dfsSys  = (Opts & DFSys) != 0;
}

/******************************************************************************/
/*                                 L i m i t                                  */
/******************************************************************************/
  
void XrdCmsBaseFS::Limit(int rLim, int Qmax)
{

// Establish the limits
//
   if (rLim < 0) {theQ.rAgain=theQ.rLeft = -1; rLim = -rLim;    Fixed = 1;}
      else {theQ.rAgain = theQ.rLeft = (rLim > 1 ? rLim/2 : 1); Fixed = 0;}
   theQ.rLimit = (rLim <= 1000 ? rLim : 0);
   if (Qmax > 0) theQ.qMax = Qmax;
      else if (!(theQ.qMax = theQ.rLimit*2 + theQ.rLimit/2)) theQ.qMax = 1;
}

/******************************************************************************/
/*                                 P a c e r                                  */
/******************************************************************************/
  
void XrdCmsBaseFS::Pacer()
{
   XrdCmsBaseFR *rP;
   int inQ, rqRate = 1000/theQ.rLimit;

// Process requests at the given rate
//
do{theQ.pqAvail.Wait();
   theQ.Mutex.Lock(); inQ = 1;
   while((rP = theQ.pqFirst))
        {if (!(theQ.pqFirst = rP->Next)) {theQ.pqLast = 0; inQ = 0;}
         theQ.Mutex.UnLock();
         if (rP->PDirLen > 0 && !hasDir(rP->Path, rP->PDirLen))
            {delete rP; continue;}
         theQ.Mutex.Lock();
         if (theQ.rqFirst) {theQ.rqLast->Next = rP; theQ.rqLast = rP;}
            else {theQ.rqFirst  = theQ.rqLast = rP; theQ.rqAvail.Post();}
         theQ.Mutex.UnLock();
         XrdSysTimer::Wait(rqRate);
         if (!inQ) break;
         theQ.Mutex.Lock();
        }
   if (inQ) theQ.Mutex.UnLock();
  } while(1);
}
  
/******************************************************************************/
/*                                 Q u e u e                                  */
/******************************************************************************/

void XrdCmsBaseFS::Queue(XrdCmsRRData &Arg, XrdCmsPInfo &Who,
                         int fnpos, int Force)
{
   EPNAME("Queue");
   static int noMsg = 1;
   XrdCmsBaseFR *rP;
   int Msg, n, prevHWM;

// If we can bypass the queue and execute this now. Avoid the grabbing the buff.
//
   if (!Force)
      {XrdCmsBaseFR myReq(&Arg, Who, fnpos);
       Xeq(&myReq);
       return;
      }

// Queue this request for callback after an appropriate time.
// We will also steal the underlying data buffer from the Arg.
//
   DEBUG("inq " <<theQ.qNum <<" pace " <<Arg.Path);
   rP = new XrdCmsBaseFR(Arg, Who, fnpos);

// Add the element to the queue
//
   theQ.Mutex.Lock();
   n = ++theQ.qNum; prevHWM = theQ.qHWM;
   if ((Msg = (n > prevHWM))) theQ.qHWM = n;
   if (theQ.pqFirst) {theQ.pqLast->Next = rP; theQ.pqLast = rP;}
      else {theQ.pqFirst = theQ.pqLast  = rP; theQ.pqAvail.Post();}
   theQ.Mutex.UnLock();

// Issue a warning message if we have an excessive number of requests queued
//
   if (n > theQ.qMax && Msg && (n-prevHWM > 3 || noMsg))
      {int Pct = n/theQ.qMax;
       char Buff[80];
       noMsg = 0;
       sprintf(Buff, "Queue overrun %d%%; %d requests now queued.", Pct, n);
       Say.Emsg("Pacer", Buff);
      }
}
  
/******************************************************************************/
/*                                R u n n e r                                 */
/******************************************************************************/
  
void XrdCmsBaseFS::Runner()
{
   XrdCmsBaseFR *rP;
   int inQ;

// Process requests at the given rate
//
do{theQ.rqAvail.Wait();
   theQ.Mutex.Lock(); inQ = 1;
   while((rP = theQ.rqFirst))
        {if (!(theQ.rqFirst = rP->Next)) {theQ.rqLast = 0; inQ = 0;}
         theQ.qNum--;
         theQ.Mutex.UnLock();
         Xeq(rP); delete rP;
         if (!inQ) break;
         theQ.Mutex.Lock();
        }
   if (inQ) theQ.Mutex.UnLock();
  } while(1);
}

/******************************************************************************/
/*                                 S t a r t                                  */
/******************************************************************************/
  
void XrdCmsBaseFS::Start()
{
   EPNAME("Start");
   void *Me = (void *)this;
   pthread_t tid;

// Issue some debugging here so we know how we are starting up
//
   DEBUG("Srv=" <<int(Server) <<" dfs=" <<int(dfsSys) <<" lcl=" <<int(lclStat)
         <<" Pre=" <<int(preSel) <<" dmLife=" <<dmLife <<' ' <<dpLife);
   DEBUG("Lim=" <<theQ.rLimit <<' ' <<theQ.rAgain <<" fix=" <<int(Fixed)
         <<" Qmax=" <<theQ.qMax);

// Set the passthru option if we can't do this locally and have no limit
//
   Punt = (!theQ.rLimit && !lclStat);

// If we need to throttle we will need two threads for the queue. The first is
// the pacer thread that feeds the runner thread at a fixed rate.
//
   if (theQ.rLimit)
      {if (XrdSysThread::Run(&tid, XrdCmsBasePacer,  Me, 0, "fsQ pacer")
       ||  XrdSysThread::Run(&tid, XrdCmsBaseRunner, Me, 0, "fsQ runner"))
          {Say.Emsg("cmsd", errno, "start baseFS queue handler");
           theQ.rLimit = 0;
          }
      }
}

/******************************************************************************/
/* Pricate:                          X e q                                    */
/******************************************************************************/

void XrdCmsBaseFS::Xeq(XrdCmsBaseFR *rP)
{
   int rc;
  
// If we are not doing local stat calls, callback indicating a forward is needed
//
   if (!lclStat)
      {if (cBack) (*cBack)(rP, 0);
       return;
      }

// Check if we can avoid doing a stat()
//
   if (dmLife && rP->PDirLen > 0 && !hasDir(rP->Path, rP->PDirLen))
      {if (cBack) (*cBack)(rP, -1);
       return;
      }

// If we have exceeded the queue limit and this is a meta-manager request
// then just deep-six it. Local requests must complete
//
   if (theQ.qNum > theQ.qMax)
      {Say.Emsg("Xeq", "Queue limit exceeded; ignoring lkup for", rP->Path);
       return;
      }

// Perform a local stat() and if we don't have the file
//
   rc = Exists(rP->Path, rP->PDirLen);
   if (cBack) (*cBack)(rP, rc);
}
