/******************************************************************************/
/*                                                                            */
/*                       X r d B w m H a n d l e . c c                        */
/*                                                                            */
/* (c) 2008 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

const char *XrdBwmHandleCVSID = "$Id$";

#include <stdio.h>
#include <string.h>

#include "XrdBwm/XrdBwmHandle.hh"
#include "XrdBwm/XrdBwmLogger.hh"
#include "XrdBwm/XrdBwmTrace.hh"
#include "XrdSfs/XrdSfsInterface.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysPlatform.hh"

#include "XProtocol/XProtocol.hh"

/******************************************************************************/
/*                        S t a t i c   O b j e c t s                         */
/******************************************************************************/
  
XrdBwmLogger   *XrdBwmHandle::Logger = 0;
XrdBwmPolicy   *XrdBwmHandle::Policy = 0;
XrdBwmHandle   *XrdBwmHandle::Free = 0;
unsigned int    XrdBwmHandle::numQueued = 0;

extern XrdSysError BwmEroute;

/******************************************************************************/
/*                         L o c a l   C l a s s e s                          */
/******************************************************************************/
  
class XrdBwmHandleCB : public XrdOucEICB, public XrdOucErrInfo
{
public:

static
XrdBwmHandleCB *Alloc()
                  {XrdBwmHandleCB *mP;
                   xMutex.Lock();
                   if (!(mP = Free)) mP = new XrdBwmHandleCB;
                      else Free = mP->Next;
                   xMutex.UnLock();
                   return mP;
                  }

void  Done(int &Results, XrdOucErrInfo *eInfo)
                  {xMutex.Lock();
                   Next = Free;
                   Free = this;
                   xMutex.UnLock();
                  }

int   Same(unsigned long long arg1, unsigned long long arg2) {return 0;}

      XrdBwmHandleCB() : Next(0) {}
     ~XrdBwmHandleCB() {}

private:
       XrdBwmHandleCB *Next;
static XrdSysMutex     xMutex;
static XrdBwmHandleCB *Free;
};

XrdSysMutex     XrdBwmHandleCB::xMutex;
XrdBwmHandleCB *XrdBwmHandleCB::Free = 0;
  
/******************************************************************************/
/*                     E x t e r n a l   L i n k a g e s                      */
/******************************************************************************/
  
void *XrdBwmHanXeq(void *pp)
{
     return XrdBwmHandle::Dispatch();
}

/******************************************************************************/
/*                    c l a s s   X r d B w m H a n d l e                     */
/******************************************************************************/
/******************************************************************************/
/*                              A c t i v a t e                               */
/******************************************************************************/

#define tident Parms.Tident
  
int XrdBwmHandle::Activate(XrdOucErrInfo &einfo)
{
   EPNAME("Activate");
   XrdSysMutexHelper myHelper(hMutex);
   char *rBuff;
   int  rSize, rc;

// Check the status of this request.
//
   if (Status != Idle)
      {if (Status == Scheduled)
          einfo.setErrInfo(kXR_inProgress, "Request already scheduled.");
          else einfo.setErrInfo(kXR_InvalidRequest, "Visa already issued.");
       return SFS_ERROR;
      }

// Try to schedule this request.
//
   qTime = time(0);
   rBuff = einfo.getMsgBuff(rSize);
   if (!(rc = Policy->Schedule(rBuff, rSize, Parms))) return SFS_ERROR;

// If resource immediately available, let client run
//
   if (rc > 0)
      {rHandle = rc;
       Status  = Dispatched;
       rTime   = time(0);
       ZTRACE(sched,"Run " <<Parms.Lfn <<' ' <<Parms.LclNode
                    <<(Parms.Direction==XrdBwmPolicy::Incomming?" <- ":" -> ")
                    <<Parms.RmtNode);
       einfo.setErrCode(strlen(rBuff));
       return (*rBuff ? SFS_DATA : SFS_OK);
      }

// Request was queued. We need to hold on to this so we can issue an async
// response later when the resource becomes available.
//
   rHandle = -rc;
   ErrCB = einfo.getErrCB(ErrCBarg);
   einfo.setErrCB((XrdOucEICB *)&myEICB);
   Status = Scheduled;
   refHandle(rHandle, this);
   ZTRACE(sched, "inQ " <<Parms.Lfn <<' ' <<Parms.LclNode
                <<(Parms.Direction==XrdBwmPolicy::Incomming?" <- ":" -> ")
                <<Parms.RmtNode);

// Indicate that client needs to wait
//
   return SFS_STARTED;
}
#undef tident

/******************************************************************************/
/* static public                A l l o c   # 1                               */
/******************************************************************************/
  
XrdBwmHandle *XrdBwmHandle::Alloc(const char *theUsr,  const char *thePath,
                                  const char *LclNode, const char *RmtNode,
                                  int Incomming)
{
   XrdBwmHandle *hP = Alloc();

// Initialize the hanlde
//
   if (hP)
      {hP->Parms.Tident    = theUsr;           // Always available
       hP->Parms.Lfn       = strdup(thePath);
       hP->Parms.LclNode   = strdup(LclNode);
       hP->Parms.RmtNode   = strdup(RmtNode);
       hP->Parms.Direction = (Incomming ? XrdBwmPolicy::Incomming
                                        : XrdBwmPolicy::Outgoing);
       hP->Status          = Idle;
       hP->qTime           = 0;
       hP->rTime           = 0;
       hP->xSize           = 0;
       hP->xTime           = 0;
      }

// All done
//
   return hP;
}

/******************************************************************************/
/* private                      A l l o c   # 2                               */
/******************************************************************************/
  
XrdBwmHandle *XrdBwmHandle::Alloc(XrdBwmHandle *old_hP)
{
   static const int minAlloc = 4096/sizeof(XrdBwmHandle);
   static XrdSysMutex aMutex;
   XrdBwmHandle *hP;

// No handle currently in the table. Get a new one off the free list or
// return one to the free list.
//
   aMutex.Lock();
   if (old_hP) {old_hP->Next = Free; Free = old_hP; hP = 0;}
     else {if (!Free && (hP = new XrdBwmHandle[minAlloc]))
              {int i = minAlloc; while(i--) {hP->Next = Free; Free = hP; hP++;}}
           if ((hP = Free)) Free = hP->Next;
          }
   aMutex.UnLock();

   return hP;
}
  
/******************************************************************************/
/*                              D i s p a t c h                               */
/******************************************************************************/

#define tident hP->Parms.Tident
  
void *XrdBwmHandle::Dispatch()
{
   EPNAME("Dispatch");
   XrdBwmHandleCB *erP = XrdBwmHandleCB::Alloc();
   XrdBwmHandle   *hP;
   char *RespBuff;
   int   RespSize, readyH, Result, Err;

// Dispatch ready requests in an endless loop
//
   do {

// Setup buffer
//
   RespBuff = erP->getMsgBuff(RespSize); 
   *RespBuff = '\0';
   erP->setErrCode(0);

// Get next ready request and test if it ended with an error
//
   if ((Err = (readyH = Policy->Dispatch(RespBuff, RespSize)) < 0))
      readyH = -readyH;

// Find the matching handle
//
   if (!(hP = refHandle(readyH)))
      {sprintf(RespBuff, "%d", readyH);
       BwmEroute.Emsg("Dispatch", "Lost handle from", RespBuff);
       if (!Err) Policy->Done(readyH);
       continue;
      }

// Lock the handle and make sure it can be dispatched
//
   hP->hMutex.Lock();
   if (hP->Status != Scheduled)
      {BwmEroute.Emsg("Dispatch", "ref to unscheduled handle",
                      hP->Parms.Tident, hP->Parms.Lfn);
       if (!Err) Policy->Done(readyH);
      } else {
       hP->myEICB.Wait(); hP->rTime = time(0);
       erP->setErrCB((XrdOucEICB *)erP, hP->ErrCBarg);
       if (Err) {hP->Status = Idle; Result = SFS_ERROR;}
          else  {hP->Status = Dispatched;
                 erP->setErrCode(strlen(RespBuff));
                 Result = (*RespBuff ? SFS_DATA : SFS_OK);
                }
       ZTRACE(sched,(Err?"Err ":"Run ") <<hP->Parms.Lfn <<' ' <<hP->Parms.LclNode
             <<(hP->Parms.Direction == XrdBwmPolicy::Incomming ? " <- ":" -> ")
             <<hP->Parms.RmtNode);
       hP->ErrCB->Done(Result, (XrdOucErrInfo *)erP);
       erP = XrdBwmHandleCB::Alloc();
      }
    hP->hMutex.UnLock();
   } while(1);

// Keep the compiler happy
//
   return (void *)0;
}

#undef tident

/******************************************************************************/
/* private                     r e f H a n d l e                              */
/******************************************************************************/
  
XrdBwmHandle *XrdBwmHandle::refHandle(int refID, XrdBwmHandle *hP)
{
   static XrdSysMutex tMutex;
   static struct {XrdBwmHandle *First;
                  XrdBwmHandle *Last;
                 }              hTab[256] = {{0,0}};
   XrdBwmHandle *pP = 0;
   int i = refID % 256;

// If we have a handle passed, add the handle to the table
//
   tMutex.Lock();
   if (hP)
      {hP->Next = 0;
       if (hTab[i].Last) {hTab[i].Last->Next = hP; hTab[i].Last = hP;}
          else {hTab[i].First = hTab[i].Last = hP; hP->Next = 0;}
       numQueued++;
      } else {
       hP = hTab[i].First;
       while(hP && hP->rHandle != refID) {pP = hP; hP = hP->Next;}
       if (hP)
          {if (pP) pP->Next = hP->Next;
              else hTab[i].First = hP->Next;
           if (hTab[i].Last == hP) hTab[i].Last = pP;
           numQueued--;
          }
      }
    tMutex.UnLock();

// All done.
//
   return hP;
}

/******************************************************************************/
/* public                         R e t i r e                                 */
/******************************************************************************/

// The handle must be locked upon entry! It is unlocked upon exit.

void XrdBwmHandle::Retire()
{
   XrdSysMutexHelper myHelper(hMutex);

// Get the global lock as the links field can only be manipulated with it.
// If not idle, cancel the resource. If scheduled, remove it from the table.
//
   if (Status != Idle) 
      {Policy->Done(rHandle);
       if (Status == Scheduled && !refHandle(rHandle, this))
          BwmEroute.Emsg("Retire", "Lost handle to", Parms.Tident, Parms.Lfn);
       Status = Idle; rHandle = 0;
      }

// If we have a logger, then log this event
//
   if (Logger && qTime)
      {XrdBwmLogger::Info myInfo;
       myInfo.Tident  = Parms.Tident;
       myInfo.Lfn     = Parms.Lfn;
       myInfo.lclNode = Parms.LclNode;
       myInfo.rmtNode = Parms.RmtNode;
       myInfo.ATime   = qTime;
       myInfo.BTime   = rTime;
       myInfo.CTime   = time(0);
       myInfo.Size    = xSize;
       myInfo.ESec    = xTime;
       myInfo.Flow    = (Parms.Direction == XrdBwmPolicy::Incomming ? 'I':'O');
       Policy->Status(myInfo.numqIn, myInfo.numqOut, myInfo.numqXeq);
       Logger->Event(myInfo);
      }

// Free storage appendages and recycle handle
//
   if (Parms.Lfn)     {free(Parms.Lfn);     Parms.Lfn = 0;}
   if (Parms.LclNode) {free(Parms.LclNode); Parms.LclNode = 0;}
   if (Parms.RmtNode) {free(Parms.RmtNode); Parms.RmtNode = 0;}
   Alloc(this);
}

/******************************************************************************/
/*                             s e t P o l i c y                              */
/******************************************************************************/
  
int XrdBwmHandle::setPolicy(XrdBwmPolicy *pP, XrdBwmLogger *lP)
{
   pthread_t tid;
   int rc, startThread = (Policy == 0);

// Set the policy and then start a thread to do dispatching if we have none
//
   Policy = pP;
   if (startThread)
      if ((rc = XrdSysThread::Run(&tid, XrdBwmHanXeq, (void *)0,
                                  0, "Handle Dispatcher")))
         {BwmEroute.Emsg("setPolicy", rc, "create handle dispatch thread");
          return 1;
         }

// All done
//
   Logger = lP;
   return 0;
}
