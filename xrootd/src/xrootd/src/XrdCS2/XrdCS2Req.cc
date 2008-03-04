/******************************************************************************/
/*                                                                            */
/*                           X r d S 2 R e q . c c                            */
/*                                                                            */
/* (c) 2006 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//          $Id$

const char *XrdCS2ReqCVSID = "$Id$";

#include <stdlib.h>
#include <string.h>

#include "XrdCS2/XrdCS2Req.hh"
#define XRDOLBTRACETYPE ->
#include "XrdOlb/XrdOlbTrace.hh"
#include "XrdOlb/XrdOlbXmi.hh"
#include "XrdSys/XrdSysPlatform.hh"
 
/******************************************************************************/
/*                               G l o b a l s                                */
/******************************************************************************/
  
XrdSysError           *XrdCS2Req::eDest;
XrdOucTrace           *XrdCS2Req::Trace;
XrdOucName2Name       *XrdCS2Req::N2N = 0;
XrdSysMutex            XrdCS2Req::myMutex;
XrdSysSemaphore        XrdCS2Req::mySem_R;
XrdSysSemaphore        XrdCS2Req::mySem_W;
XrdCS2Req             *XrdCS2Req::nextFree   =  0;
int                    XrdCS2Req::numFree    =  0;
int                    XrdCS2Req::numinQ_R   =  0;
int                    XrdCS2Req::numinQ_W   =  0;
char                   XrdCS2Req::isWaiting_R=  0;
char                   XrdCS2Req::isWaiting_W=  0;
XrdCS2Req             *XrdCS2Req::STab[Slots]= {0};

/******************************************************************************/
/*                                 A l l o c                                  */
/******************************************************************************/
  
XrdCS2Req *XrdCS2Req::Alloc(XrdOlbReq *reqP, const char *Path, int as_W)
{
   XrdCS2Req *rp;
   int rc;

// Allocate a request object. If we have no memory, tell the requester
// to try again in a minute.
//
   myMutex.Lock();
   if (nextFree) 
      {rp = nextFree;
       nextFree = rp->Next;
       numFree--;
      }
      else if (!(rp = new XrdCS2Req())) 
              {myMutex.UnLock();
               reqP->Reply_Wait(retryTime);
               return (XrdCS2Req *)0;
              }
   myMutex.UnLock();

// Initialize the path which may be a mapping from the lfn to the MSS name
//
   if (N2N) rc = N2N->lfn2rfn(Path, rp->thePath, PathSize);
      else if (strlcpy(rp->thePath, Path, PathSize) >= PathSize) rc = ENAMETOOLONG;
              else rc = 0;

// Make sure the initialization succeeded
//
   if (rc) {reqP->Reply_Error("Unable to map lfn to Castor2 name.");
            eDest->Emsg("CS2Req", rc, "map lfn to Castor2 name.");
            rp->Recycle();
            return 0;
           }

// Complete initialization and return the object
//
   rp->olbReq  = reqP;
   rp->Next    = 0;
   rp->Same    = 0;
   rp->myLock  = 0;
   rp->is_W    = as_W;
   return rp;
}

/******************************************************************************/
/*                                 Q u e u e                                  */
/******************************************************************************/
  
void XrdCS2Req::Queue()
{
   EPNAME("CS2Queue");
   unsigned long hval = SlotNum(thePath);

// Add this request to the slot table
//
   if (!myLock) myMutex.Lock();
   if (!STab[hval]) STab[hval] = this;
      else {XrdCS2Req *prp = 0, *rp = STab[hval];
            while(rp) if (strcmp(rp->thePath, thePath)) {prp = rp; rp = rp->Next;}
                         else {Same = rp->Same; rp->Same = this;}
            if (prp) prp->Next = this;
           }
    if (is_W) {if (isWaiting_W) {isWaiting_W = 0; mySem_W.Post(); numinQ_W++;}}
       else   {if (isWaiting_R) {isWaiting_R = 0; mySem_R.Post(); numinQ_R++;}}

// Do some debugging
//
   DEBUG((is_W ? numinQ_W : numinQ_R) <<" in queue; added path=" <<thePath);
   myMutex.UnLock();
}

/******************************************************************************/
/*                               R e c y c l e                                */
/******************************************************************************/
  
XrdCS2Req *XrdCS2Req::Recycle()
{
   XrdCS2Req *reqP = Same;

// Put this object on the free queue or delete it
//
   if (numFree >= maxFree) 
      {if (myLock) myMutex.UnLock();
       delete this;
      } else {
       if (!myLock) myMutex.Lock();
       Next = nextFree;
       nextFree = this;
       numFree++;
       myMutex.UnLock();
      }
   return reqP;
}

/******************************************************************************/
/*                                R e m o v e                                 */
/******************************************************************************/
  
XrdCS2Req *XrdCS2Req::Remove(const char *Path)
{
   EPNAME("CS2Req");
   unsigned long hval = SlotNum(Path);
   XrdCS2Req *rp, *prp = 0;

// Find the request in the slot table
//
   myMutex.Lock();
   if ((rp = STab[hval]))
      {while(rp && strcmp(Path, rp->thePath)) {prp = rp; rp = rp->Next;}
       if (rp)
          {if (prp) prp->Next = rp->Next;
              else  STab[hval]= rp->Next;
           if (rp->is_W) {numinQ_W--; if (numinQ_W < 0) numinQ_W = 0;}
              else       {numinQ_R--; if (numinQ_R < 0) numinQ_R = 0;}
          }
      }
   myMutex.UnLock();

// Document the miss
//
   if (!rp) eDest->Emsg("CS2Req", "Did not find request for", Path);
      else {DEBUG((rp->is_W?numinQ_W:numinQ_R) <<" in Q; rmv path=" <<rp->thePath);}
   return rp;
}

/******************************************************************************/
/*                                   S e t                                    */
/******************************************************************************/
  
void XrdCS2Req::Set(XrdOlbXmiEnv *Env)
{

// Copy out needed stuff from the environment
//
   eDest = Env->eDest;         // -> Error message handler
   Trace = Env->Trace;         // -> Trace handler
   N2N   = Env->Name2Name;     // -> lfn mapper
}

/******************************************************************************/
/*                               S l o t N u m                                */
/******************************************************************************/
  
unsigned int XrdCS2Req::SlotNum(const char *Path)
{
   int plen = strlen(Path);
   unsigned long long temp1 = 0, temp2;

// Develop a hash
//
   while(plen >= (int)sizeof(temp2))
        {memcpy(&temp2, Path, sizeof(temp2));
         temp1 ^= temp2; Path += sizeof(temp2); plen -= sizeof(temp2);
        }
   if (plen) {temp2 = 0; memcpy(&temp2, Path, sizeof(temp2)); temp1 ^= temp2;}

// Return the slot number
//
   return static_cast<unsigned int>(temp1 % Slots);
}

/******************************************************************************/
/*                              W a i t 4 Q _ R                               */
/******************************************************************************/
  
int XrdCS2Req::Wait4Q_R()
{
   int temp;

// Wait until something is in the queue
//
   myMutex.Lock();
   while(!numinQ_R)
        {isWaiting_R = 1;
         myMutex.UnLock();
         mySem_R.Wait();
         myMutex.Lock();
        };
   temp = numinQ_R;
   myMutex.UnLock();
   return temp;
}

/******************************************************************************/
/*                              W a i t 4 Q _ W                               */
/******************************************************************************/
  
int XrdCS2Req::Wait4Q_W()
{
   int temp;

// Wait until something is in the queue
//
   myMutex.Lock();
   while(!numinQ_W)
        {isWaiting_W = 1;
         myMutex.UnLock();
         mySem_W.Wait();
         myMutex.Lock();
        };
   temp = numinQ_W;
   myMutex.UnLock();
   return temp;
}
