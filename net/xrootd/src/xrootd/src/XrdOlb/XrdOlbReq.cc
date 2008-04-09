/******************************************************************************/
/*                                                                            */
/*                          X r d O l b R e q . c c                           */
/*                                                                            */
/* (c) 2006 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

const char *XrdOlbReqCVSID = "$Id$";

#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/uio.h>
  
#include "XrdOlb/XrdOlbReq.hh"
#include "XrdOlb/XrdOlbRRQ.hh"
#include "XrdOlb/XrdOlbServer.hh"
#include "XrdOlb/XrdOlbTrace.hh"
#include "XrdOlb/XrdOlbRTable.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XProtocol/XProtocol.hh"

using namespace XrdOlb;

/******************************************************************************/
/*                        C o n s t r u c t o r   # 1                         */
/******************************************************************************/
  
XrdOlbReq::XrdOlbReq(XrdOlbServer *sp, XrdOlbRRQInfo *ip)
{
   ServerP = sp;
   InfoP   = ip;
   ReqNum  = 0;
   ReqSnum = 0;
   ReqSins = 0;
}

/******************************************************************************/
/*                        C o n s t r u c t o r   # 2                         */
/******************************************************************************/
  
XrdOlbReq::XrdOlbReq(unsigned int rn, XrdOlbRRQInfo *ip)
{
   InfoP   = 0;
   ServerP = 0;
   ReqNum  = rn;
   ReqSnum = ip->Rnum;
   ReqSins = ip->Rinst;
}

/******************************************************************************/
/*                           R e p l y _ E r r o r                            */
/******************************************************************************/
  
void XrdOlbReq::Reply_Error(const char *emsg, int elen)
{
   struct iovec iov[4]; // Reply() fills in the last element

// Construct the message (element 0 filled in by Reply())
//
   iov[1].iov_base = (char *)" ?err ";
   iov[1].iov_len  = 6;
   iov[2].iov_base = (char *)emsg;
   iov[2].iov_len  = (elen ? elen : strlen(emsg));

// Send off the reply (this object may be deleted so make a fast exit)
//
   Reply(iov, 3);
}
 
/******************************************************************************/
/*                           R e p l y _ E r r o r                            */
/******************************************************************************/
  
void XrdOlbReq::Reply_Error(const char *ecode, const char *emsg, int elen)
{
   struct iovec iov[6]; // Reply() fills in the last element

// Construct the message (element 0 filled in by Reply())
//
   iov[1].iov_base = (char *)" !err ";
   iov[1].iov_len  = 6;
   iov[2].iov_base = (char *)ecode;
   iov[2].iov_len  = strlen(ecode);
   iov[3].iov_base = (char *)" ";
   iov[3].iov_len  = 1;
   iov[4].iov_base = (char *)emsg;
   iov[4].iov_len  = (elen ? elen : strlen(emsg));

// Send off the reply (this object may be deleted so make a fast exit)
//
   Reply(iov, 5);
}

/******************************************************************************/
/*                              R e p l y _ O K                               */
/******************************************************************************/
  
void XrdOlbReq::Reply_OK()
{
   struct iovec iov[3]; // Reply() fills in the last element

// Construct the message (element 0 filled in by Reply())
//
   iov[1].iov_base = (char *)" !data";
   iov[1].iov_len  = 6;

// Send off the reply (this object may be deleted so make a fast exit)
//
   Reply(iov, 2);
}
  
/******************************************************************************/
  
void XrdOlbReq::Reply_OK(const char *data, int dlen)
{
   struct iovec iov[4]; // Reply() fills in the last element

// Construct the message (element 0 filled in by Reply())
//
   iov[1].iov_base = (char *)" !data ";
   iov[1].iov_len  = 7;
   iov[2].iov_base = (char *)data;
   iov[2].iov_len  = (dlen ? dlen : strlen(data));

// Send off the reply (this object may be deleted so make a fast exit)
//
   Reply(iov, 3);
}
 
/******************************************************************************/

void XrdOlbReq::Reply_OK(struct stat &buf)
{
   char sbuff[256];

   Reply_OK(sbuff, StatGen(buf, sbuff)-1);
}

/******************************************************************************/
/*                        R e p l y _ R e d i r e c t                         */
/******************************************************************************/
  
void XrdOlbReq::Reply_Redirect(const char *sname, 
                               const char *lcgi, const char *ocgi)
{
   struct iovec iov[8]; // Reply() fills in the last element
   int iovnum;

// Construct the message (element 0 filled in by Reply())
//
   iov[1].iov_base = (char *)" !try ";
   iov[1].iov_len  = 6;
   iov[2].iov_base = (char *)sname;
   iov[2].iov_len  = strlen(sname);

// Now we need to see if we have any cgi info to pass
//
   if (!lcgi && !ocgi) iovnum = 3;
      else {if (ocgi)
               {iov[3].iov_base = (char *)"?";
                iov[3].iov_len  = 1;
                iov[4].iov_base = (char *)ocgi;
                iov[4].iov_len  = strlen(ocgi);
                if (lcgi)
                   {iov[5].iov_base = (char *)"?";
                    iov[5].iov_len  = 1;
                    iov[6].iov_base = (char *)lcgi;
                    iov[6].iov_len  = strlen(lcgi);
                     iovnum = 7;
                   } iovnum = 5;
               } else {
                iov[3].iov_base = (char *)"??";
                iov[3].iov_len  = 2;
                iov[4].iov_base = (char *)lcgi;
                iov[4].iov_len  = strlen(lcgi);
                iovnum = 5;
               }
           }

// Send off the reply (this object may be deleted so make a fast exit)
//
   Reply(iov, iovnum);
}

/******************************************************************************/
/*                            R e p l y _ W a i t                             */
/******************************************************************************/
  
void XrdOlbReq::Reply_Wait(int sec)
{
   struct iovec iov[4]; // Reply() fills in the last element
   char buff[32];

// Construct the message (element 0 filled in by Reply())
//
   iov[1].iov_base = (char *)" !wait ";
   iov[1].iov_len  = 7;
   iov[2].iov_base = buff;
   iov[2].iov_len  = sprintf(buff, "%d", sec);

// Send off the reply
//
   Reply(iov, 3);
}

/******************************************************************************/
/*                        R e p l y _ W a i t R e s p                         */
/******************************************************************************/
  
XrdOlbReq *XrdOlbReq::Reply_WaitResp(int sec)
{
   static XrdSysMutex rnMutex;
   static unsigned int RequestNum = 0;
          struct iovec iov[3];  // Reply() fills in the last element
          char buff[32];
          unsigned int rnum;
          XrdOlbReq *newReq;

// If this is already a waitresp object then we cannot do this again. So,
// just return a null pointer indicating an invalid call.
//
   if (!ServerP) return (XrdOlbReq *)0;

// Generate a request number unless no reply is needed
//
   if (InfoP->ID[1] || InfoP->ID[0] != '0')
      {rnMutex.Lock();
       RequestNum++;
       rnum = RequestNum;
       rnMutex.UnLock();
     } else rnum = 0;

// Construct a new request object. This object will be used to actually effect
// the reply. We need to do this because the server may disappear before we
// actually reply. In which case the reply gets deep-sixed.
//
   newReq = new XrdOlbReq(rnum, InfoP);

// Construct the message (element 0 filled in by Reply()) and send it
//
   if (rnum)
      {iov[1].iov_base = (char *)buff;
       iov[1].iov_len  = sprintf(buff, " +%d", rnum);
       Reply(iov, 2);
      }

// Return an object to affect an asynchronous reply
//
   return newReq;
}

/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
  
#define XRDXROOTD_STAT_CLASSNAME XrdOlbReq
#include "XrdXrootd/XrdXrootdStat.icc"

/******************************************************************************/
/*                               n o R e p l y                                */
/******************************************************************************/
  
void XrdOlbReq::noReply()
{
   static int nrNum = 255;

// We always issue a message about double object use otherwise issue warning
// as this is indicative of an improper configuration.
//
   if (ReqSnum < 0)
      Say.Emsg("Req", "Attempted reply to twice to a 2way async request.");
      else {nrNum++;
            if (!(nrNum & 255)) Say.Emsg("Req", 
                                "Attempted reply to a 1way request; "
                                "probable incorrect ofs forward directive.");
           }
}

/******************************************************************************/
/*                                 R e p l y                                  */
/******************************************************************************/
  
void XrdOlbReq::Reply(struct iovec *iov, int iovnum)
{
   EPNAME("Reply");
   XrdOlbServer *sp;
   char buff[32];

// Check if we really are supposed to reply

// Insert a trailing newline character
//
   iov[iovnum].iov_base = (char *)"\n";
   iov[iovnum].iov_len  = 1;
   iovnum++;

// Reply format differs depending on whether this is a sync or async reply
//
   if (ServerP)
      {if (InfoP->ID[1] || InfoP->ID[0] != '0')
         {iov[0].iov_base = (char *)InfoP->ID;
          iov[0].iov_len  = strlen(InfoP->ID);
          ServerP->Send(iov, iovnum);
         } else noReply();
       return;
      }

// This a true async callback
//
   if (!ReqNum) {noReply(); return;}
   iov[0].iov_base = buff;
   iov[0].iov_len  = sprintf(buff, "%d >", ReqNum);
   iov[1].iov_base = (static_cast<char *>(iov[1].iov_base)+1);
   iov[1].iov_len--;

// Async replies are more complicated here since we must find the server using
// a logical address that may no longer be valid.
//
   RTable.Lock();
   if ((sp = RTable.Find(ReqSnum, ReqSins))) sp->Send(iov, iovnum);
      else {DEBUG("Async resp " <<ReqNum <<' ' <<iov[1].iov_base <<" discarded; server gone");}
   RTable.UnLock();

// Only one async response is allowed. Mark this object unusable
//
   ReqNum  = 0;
   ReqSnum = -1;
}
