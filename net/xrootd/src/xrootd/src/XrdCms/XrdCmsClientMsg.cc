/******************************************************************************/
/*                                                                            */
/*                    X r d C m s C l i e n t M s g . c c                     */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

// Vased on: XrdCmsClientMsg.cc,v 1.14 2006/09/26 07:49:14 abh

const char *XrdCmsClientMsgCVSID = "$Id$";

#include <stdlib.h>
  
#include "XProtocol/YProtocol.hh"
#include "XrdCms/XrdCmsClientMsg.hh"
#include "XrdCms/XrdCmsParser.hh"
#include "XrdCms/XrdCmsTrace.hh"
#include "XrdOuc/XrdOucErrInfo.hh"
#include "XrdNet/XrdNetBuffer.hh"

using namespace XrdCms;
 
/******************************************************************************/
/*                               G l o b a l s                                */
/******************************************************************************/
  
int               XrdCmsClientMsg::nextid   =  0;
int               XrdCmsClientMsg::numinQ   =  0;

XrdCmsClientMsg  *XrdCmsClientMsg::msgTab   =  0;
XrdCmsClientMsg  *XrdCmsClientMsg::nextfree =  0;

XrdSysMutex       XrdCmsClientMsg::FreeMsgQ;

/******************************************************************************/
/*                                 A l l o c                                  */
/******************************************************************************/
  
// Returns the message object locked!

XrdCmsClientMsg *XrdCmsClientMsg::Alloc(XrdOucErrInfo *erp)
{
   XrdCmsClientMsg *mp;
   int       lclid;

// Allocate a message object
//
   FreeMsgQ.Lock();
   if (nextfree) {mp = nextfree; nextfree = mp->next;}
      else {FreeMsgQ.UnLock(); return (XrdCmsClientMsg *)0;}
   lclid = nextid = (nextid + MidIncr) & IncMask;
   numinQ++;
   FreeMsgQ.UnLock();

// Initialize it
//
   mp->Hold.Lock();
   mp->id      = (mp->id & MidMask) | lclid;
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
  
int XrdCmsClientMsg::Init()
{
   int i;
   XrdCmsClientMsg *msgp;

// Allocate the fixed number of msg blocks. These will never be freed!
//
   if (!(msgp = new XrdCmsClientMsg[MaxMsgs]())) return 1;
   msgTab = &msgp[0];
   nextid = MaxMsgs;

// Place all of the msg blocks on the free list
//
  for (i = 0; i < MaxMsgs; i++)
     {msgp->next = nextfree; nextfree = msgp; msgp->id = i; msgp++;}

// All done
//
   return 0;
}

/******************************************************************************/
/*                               R e c y c l e                                */
/******************************************************************************/
  
// Message object lock *must* be held by the caller upon entry!

void XrdCmsClientMsg::Recycle()
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
   if (numinQ >= 0) numinQ--;
   FreeMsgQ.UnLock();
}

/******************************************************************************/
/*                                 R e p l y                                  */
/******************************************************************************/
  
int XrdCmsClientMsg::Reply(const char *Man, CmsRRHdr &hdr, XrdNetBuffer *buff)
{
   EPNAME("Reply")
   XrdCmsClientMsg *mp;

// Find the appropriate message
//
   if (!(mp = XrdCmsClientMsg::RemFromWaitQ(hdr.streamid)))
      {DEBUG("to non-existent message; id=" <<hdr.streamid);
       return 0;
      }

// Decode the response
//
   mp->Result = XrdCmsParser::Decode(Man, hdr, buff->data, buff->dlen,
                                     (XrdOucErrInfo *)(mp->Resp));

// Signal a reply and return
//
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
  
XrdCmsClientMsg *XrdCmsClientMsg::RemFromWaitQ(int msgid)
{
   int msgnum;

// Locate the message object (the low order bits index it)
//
  msgnum = msgid & MidMask;
  msgTab[msgnum].Hold.Lock();
  if (!msgTab[msgnum].inwaitq || msgTab[msgnum].id != msgid)
     {msgTab[msgnum].Hold.UnLock(); return (XrdCmsClientMsg *)0;}
  msgTab[msgnum].inwaitq = 0;
  return &msgTab[msgnum];
}
