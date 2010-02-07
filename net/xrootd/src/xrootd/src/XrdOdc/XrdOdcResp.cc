/******************************************************************************/
/*                                                                            */
/*                         X r d O d c R e s p . c c                          */
/*                                                                            */
/* (c) 2006 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//          $Id$

const char *XrdOdcRespCVSID = "$Id$";

#include <stdlib.h>
#include <string.h>
  
#include "XrdOdc/XrdOdcConfig.hh"
#include "XrdOdc/XrdOdcMsg.hh"
#include "XrdOdc/XrdOdcResp.hh"
#include "XrdOdc/XrdOdcTrace.hh"
#include "XrdOuc/XrdOucErrInfo.hh"
#include "XrdSfs/XrdSfsInterface.hh"
 
/******************************************************************************/
/*                               G l o b a l s                                */
/******************************************************************************/
  
XrdSysMutex             XrdOdcResp::myMutex;
XrdOdcResp             *XrdOdcResp::nextFree =  0;
int                     XrdOdcResp::numFree  =  0;
int                     XrdOdcResp::RepDelay =  5;

extern XrdOucTrace OdcTrace;
 
/******************************************************************************/
/*                                 A l l o c                                  */
/******************************************************************************/
  
XrdOdcResp *XrdOdcResp::Alloc(XrdOucErrInfo *erp, int msgid)
{
   XrdOdcResp *rp;

// Allocate a response object. We must be assured that the semaphore count
// is zero. This will be true for freshly allocated objects. For reused
// objects we will need to run down the count to zero as multiple calls
// to sem_init may produced undefined behaviour.
//
   myMutex.Lock();
   if (nextFree) 
      {rp = nextFree;
       nextFree = rp->next; 
       numFree--;
       rp->SyncCB.Init();
      }
      else if (!(rp = new XrdOdcResp())) 
              {myMutex.UnLock();
               return (XrdOdcResp *)0;
              }
   myMutex.UnLock();

// Initialize it. We also replace the callback object pointer with a pointer
// to the synchronization semaphore as we have taken over the object and must
// provide a callback synchronization path for the caller.
//
   strlcpy(rp->UserID, erp->getErrUser(), sizeof(rp->UserID));
   rp->ErrCB = erp->getErrCB(rp->ErrCBarg);
   erp->setErrCB((XrdOucEICB *)&rp->SyncCB);
   rp->myID   = msgid;
   rp->next   = 0;

// Return the response object
//
   return rp;
}

/******************************************************************************/
/*                               R e c y c l e                                */
/******************************************************************************/
  
void XrdOdcResp::Recycle()
{

// Put this object on the free queue
//
   if (numFree >= maxFree) delete this;
      else {myMutex.Lock();
            next = nextFree;
            nextFree = this;
            numFree++;
            myMutex.UnLock();
           }
}

/******************************************************************************/
/*                                 R e p l y                                  */
/******************************************************************************/
  
void XrdOdcResp::Reply(const char *Man, char *msg)
{
   EPNAME("Reply")
   XrdOucEICB *theCB;
   int Result, msgval;
   char *colon, *opaque;

// If there is no callback object, ignore this call. Eventually, we may wish
// to simulate a callback but this is rather complicated.
//
   if (!ErrCB)
      {DEBUG("No callback object for user " <<UserID <<" msgid=" <<myID
             <<' ' <<Man);
       Recycle();
       return;
      }

// Get the values for the callback. Unfortunately, we need to repeat message
// processing. One day we will consolidate tis code between Finder and Msg.
//
        if (!strncmp(msg, "!try", 4))
           {msg += 5; Result = SFS_REDIRECT;
            while(*msg && (' ' == *msg)) msg++;
            if (!(colon = index(msg, (int)':'))) msgval = 0;
               else {msgval = atoi(colon+1);
                     if (!(opaque = index(colon, (int)'?'))) *colon = '\0';
                     else {*opaque = '\0'; *colon = '?';
                           memmove(colon+1, opaque+1, strlen(opaque+1)+1);
                          }
                    }
            TRACE(Redirect, UserID <<" redirected to " <<msg
                  <<':' <<msgval <<" by " << Man);
           }
   else if (!strncmp(msg, "!wait", 5))
           {msg += 6; msgval = 0;
            while(*msg && (' ' == *msg)) msg++;
            if (!(Result = atoi(msg))) Result = RepDelay;
            *msg = '\0';
            TRACE(Redirect, UserID <<" asked to wait "
                  <<Result <<" by " << Man);
           }
   else if (!strncmp(msg, "!data", 5))
           {msg += 6;
            while(*msg && (' ' == *msg)) msg++;
            Result = SFS_DATA;
            msgval = (*msg ? strlen(msg)+1 : 0);
            TRACE(Redirect, UserID <<" given text data '"
                  <<msg <<"' by " << Man);
           }
   else if (!strncmp(msg, "?err", 4))
           {msg += 5; Result = SFS_ERROR; msgval = 0;
            while(*msg && (' ' == *msg)) msg++;
            TRACE(Redirect, UserID <<" given error msg '"
                  <<msg <<"' by " << Man);
           }
   else if (!strncmp(msg, "!err", 4))
           {msg += 5; Result = SFS_ERROR;
            while(*msg && (' ' == *msg)) msg++;
            char *ecode = msg;
            while(*msg && (' ' != *msg)) msg++;
            if (*msg) {*msg++ = '\0'; while(*msg && (' ' == *msg)) msg++;}
            msgval = XrdOdcMsg::mapError(ecode);
            TRACE(Redirect, UserID <<" given error " <<msgval <<" msg '"
                  <<msg <<"' by " << Man);
           }
   else    {Result = SFS_ERROR; msgval = 0; 
            msg = (char *)"Redirector protocol error";
            TRACE(Redirect, UserID <<" given error msg '"
                  <<msg <<"' due to " << Man);
           }

// Copy the data into our object to allow the callback to run asynchrnously
//
   setErrInfo(msgval, msg);

// Before invoking the callback we must be assured that the waitresp response
// has been sent to the client. We do this by waiting on a semaphore which is
// posted *after* the waitresp response is sent.
//
   SyncCB.Wait();

// Invoke the callback; telling it to call us back for recycling
//
   theCB = ErrCB;
   ErrCB = (XrdOucEICB *)this;
   theCB->Done(Result, (XrdOucErrInfo *)this);
}

/******************************************************************************/
/*                           X r d O d c R e s p Q                            */
/******************************************************************************/
/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdOdcRespQ::XrdOdcRespQ()
{
   memset(mqTab, 0, sizeof(mqTab));
}

/******************************************************************************/
/*                                   A d d                                    */
/******************************************************************************/
  
void XrdOdcRespQ::Add(XrdOdcResp *rp)
{
   int i;

// Compute index and either add or chain the entry
//
   i = rp->myID % mqSize;
   myMutex.Lock();
   rp->next = (mqTab[i] ? mqTab[i] : 0);
   mqTab[i] = rp;
   myMutex.UnLock();
}

/******************************************************************************/
/*                                 P u r g e                                  */
/******************************************************************************/
  
void XrdOdcRespQ::Purge()
{
   XrdOdcResp *rp;
   int i;

   myMutex.Lock();
   for (i = 0; i < mqSize; i++)
       {while ((rp = mqTab[i])) {mqTab[i] = rp->next; delete rp;}}
   myMutex.UnLock();
}

/******************************************************************************/
/*                                   R e m                                    */
/******************************************************************************/
  
XrdOdcResp *XrdOdcRespQ::Rem(int msgid)
{
   int i;
   XrdOdcResp *rp, *pp = 0;

// Compute the index and find the entry
//
   i = msgid % mqSize;
   myMutex.Lock();
   rp = mqTab[i];
   while(rp && rp->myID != msgid) {pp = rp; rp = rp->next;}

// Remove the entry if we found it
//
   if (rp) {if (pp) pp->next = rp->next;
               else mqTab[i] = rp->next;
           }

// Return what we found
//
   myMutex.UnLock();
   return rp;
}
