/******************************************************************************/
/*                                                                            */
/*                          X r d O d c M s g . c c                           */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

const char *XrdOdcMsgCVSID = "$Id$";

#include <stdlib.h>
  
#include "XrdOdc/XrdOdcMsg.hh"
#include "XrdOdc/XrdOdcTrace.hh"
#include "XrdOuc/XrdOucErrInfo.hh"
 
/******************************************************************************/
/*                               G l o b a l s                                */
/******************************************************************************/
  
int         XrdOdcMsg::nextid   =  0;

XrdOdcMsg  *XrdOdcMsg::msgTab   =  0;
XrdOdcMsg  *XrdOdcMsg::nextfree =  0;

XrdSysMutex XrdOdcMsg::FreeMsgQ;

extern XrdOucTrace OdcTrace;

#define XRDODC_MIDMASK 1023
#define XRDODC_MAXMSGS 1024
#define XRDODC_MIDINCR 1024
#define XRDODC_INCMASK 0x3ffffc00
 
/******************************************************************************/
/*                                 A l l o c                                  */
/******************************************************************************/
  
// Returns the message object locked!

XrdOdcMsg *XrdOdcMsg::Alloc(XrdOucErrInfo *erp)
{
   XrdOdcMsg *mp;
   int       lclid;

// Allocate a message object
//
   FreeMsgQ.Lock();
   if (nextfree) {mp = nextfree; nextfree = mp->next;}
      else {FreeMsgQ.UnLock(); return (XrdOdcMsg *)0;}
   lclid = nextid = (nextid + XRDODC_MIDINCR) & XRDODC_INCMASK;
   FreeMsgQ.UnLock();

// Initialize it
//
   mp->Hold.Lock();
   mp->id      = (mp->id & XRDODC_MIDMASK) | lclid;
   mp->Resp    = erp;
   mp->next    = 0;
   mp->inwaitq = 1;

// Return the message object
//
   return mp;
}
 
/******************************************************************************/
/*                                  I n i t                                   */
/******************************************************************************/
  
int XrdOdcMsg::Init()
{
   int i;
   XrdOdcMsg *msgp;

// Allocate the fixed number of msg blocks. These will never be freed!
//
   if (!(msgp = new XrdOdcMsg[XRDODC_MAXMSGS]())) return 1;
   msgTab = &msgp[0];
   nextid = XRDODC_MAXMSGS;

// Place all of the msg blocks on the free list
//
  for (i = 0; i < XRDODC_MAXMSGS; i++)
     {msgp->next = nextfree; nextfree = msgp; msgp->id = i; msgp++;}

// All done
//
   return 0;
}

/******************************************************************************/
/*                              m a p E r r o r                               */
/******************************************************************************/
  
int XrdOdcMsg::mapError(const char *ecode)
{
   if (!strcmp("ENOENT", ecode))       return ENOENT;
   if (!strcmp("EPERM", ecode))        return EPERM;
   if (!strcmp("EACCES", ecode))       return EACCES;
   if (!strcmp("EIO", ecode))          return EIO;
   if (!strcmp("ENOMEM", ecode))       return ENOMEM;
   if (!strcmp("ENOSPC", ecode))       return ENOSPC;
   if (!strcmp("ENAMETOOLONG", ecode)) return ENAMETOOLONG;
   if (!strcmp("ENETUNREACH", ecode))  return ENETUNREACH;
   if (!strcmp("ENOTBLK", ecode))      return ENOTBLK;
   if (!strcmp("EISDIR", ecode))       return EISDIR;
   return EINVAL;
}

/******************************************************************************/
/*                               R e c y c l e                                */
/******************************************************************************/
  
// Message object lock *must* be held by the caller upon entry!

void XrdOdcMsg::Recycle()
{
   static XrdOucErrInfo dummyResp;

// Remove this from he wait queue and substitute a safe resp object. We do
// this because a reply may be pending and will post when we release the lock
//
   inwaitq = 0; 
   Resp = &dummyResp;
   Hold.UnLock();

// Place message object on re-usable queue
//
   FreeMsgQ.Lock();
   next = nextfree; 
   nextfree = this; 
   FreeMsgQ.UnLock();
}

/******************************************************************************/
/*                                 R e p l y                                  */
/******************************************************************************/
  
int XrdOdcMsg::Reply(int msgid, char *msg)
{
   EPNAME("Reply")
   XrdOdcMsg *mp;
   int retc;

// Find the appropriate message
//
   if (!(mp = XrdOdcMsg::RemFromWaitQ(msgid)))
      {DEBUG("Reply to non-existent message; id=" <<msgid);
       return 0;
      }

// Determine the error code
//
        if (!strncmp(msg, "!try", 4))
           {msg += 5;
            retc = -EREMOTE;
            while(*msg && (' ' == *msg)) msg++;
           }
   else if (*msg == '+')
           {msg += 1;
            retc = -EINPROGRESS;
           }
   else if (!strncmp(msg, "!wait", 5))
           {msg += 6;
            retc = -EAGAIN;
            while(*msg && (' ' == *msg)) msg++;
           }
   else if (!strncmp(msg, "!data", 5))
           {msg += 6;
            retc = -EALREADY;
            while(*msg && (' ' == *msg)) msg++;
           }
   else if (!strncmp(msg, "?err", 4))
           {msg += 5;
            retc = -EINVAL;
            while(*msg && (' ' == *msg)) msg++;
           }
   else if (!strncmp(msg, "!err", 4))
           {msg += 5;
            while(*msg && (' ' == *msg)) msg++;
            char *ecode = msg;
            while(*msg && (' ' != *msg)) msg++;
            if (*msg) {*msg++ = '\0'; while(*msg && (' ' == *msg)) msg++;}
            retc = -mapError(ecode);
           }
   else retc = -EINVAL;

// Make sure the reply is not too long
//
   if (strlen(msg) >= XrdOucEI::Max_Error_Len)
      {DEBUG("Truncated: " <<msg);
       msg[XrdOucEI::Max_Error_Len-1] = '\0';
      }

// Reply and return
//
   mp->Resp->setErrInfo(retc, msg);
   mp->Hold.Signal();
   mp->Hold.UnLock();
   return 1;
}

/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                          R e m F r o m W a i t Q                           */
/******************************************************************************/

// RemFromWaitQ() returns the msg object with the object locked! The caller
//                must unlock the object.
  
XrdOdcMsg *XrdOdcMsg::RemFromWaitQ(int msgid)
{
   int msgnum;

// Locate the message object (the low order bits index it)
//
  msgnum = msgid & XRDODC_MIDMASK;
  msgTab[msgnum].Hold.Lock();
  if (!msgTab[msgnum].inwaitq || msgTab[msgnum].id != msgid)
     {msgTab[msgnum].Hold.UnLock(); return (XrdOdcMsg *)0;}
  msgTab[msgnum].inwaitq = 0;
  return &msgTab[msgnum];
}
