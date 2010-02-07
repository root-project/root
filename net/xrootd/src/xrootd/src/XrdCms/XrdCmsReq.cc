/******************************************************************************/
/*                                                                            */
/*                          X r d C m s R e q . c c                           */
/*                                                                            */
/* (c) 2006 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

// Original Version: 1.4 2007/07/31 02:25:16 abh

const char *XrdCmsReqCVSID = "$Id$";

#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <netinet/in.h>
#include <inttypes.h>
  
#include "XrdCms/XrdCmsNode.hh"
#include "XrdCms/XrdCmsReq.hh"
#include "XrdCms/XrdCmsRRQ.hh"
#include "XrdCms/XrdCmsRTable.hh"
#include "XrdCms/XrdCmsTrace.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysPlatform.hh"
#include "XrdSys/XrdSysPthread.hh"

#include "XProtocol/XProtocol.hh"
#include "XProtocol/YProtocol.hh"

using namespace XrdCms;

/******************************************************************************/
/*                        C o n s t r u c t o r   # 1                         */
/******************************************************************************/
  
XrdCmsReq::XrdCmsReq(XrdCmsNode *nP, unsigned int reqid, char adv)
{
   NodeP   = nP;
   ReqID   = reqid;
   ReqNnum = nP->getSlot();
   ReqNins = nP->Inst();
   ReqAdv  = adv;
}

/******************************************************************************/
/*                        C o n s t r u c t o r   # 2                         */
/******************************************************************************/
  
XrdCmsReq::XrdCmsReq(XrdCmsReq *Req, unsigned int rn)
{
   NodeP   = 0;
   ReqID   = rn;
   ReqNnum = Req->ReqNnum;
   ReqNins = Req->ReqNins;
}

/******************************************************************************/
/*                           R e p l y _ E r r o r                            */
/******************************************************************************/
  
void XrdCmsReq::Reply_Error(const char *emsg, int elen)
{

// Make sure that elen includes a null byte
//
   if (!elen) elen = strlen(emsg)+1;
      else if (emsg[elen]) elen++;

// Send off the reply
//
   Reply(kYR_error, kYR_EINVAL, emsg, elen);
}
 
/******************************************************************************/
/*                           R e p l y _ E r r o r                            */
/******************************************************************************/
  
void XrdCmsReq::Reply_Error(const char *ecode, const char *emsg, int elen)
{
   unsigned int eval;

// Translate the error name
//
        if (!strcmp("ENOENT", ecode))       eval = kYR_ENOENT;
   else if (!strcmp("EPERM", ecode))        eval = kYR_EPERM;
   else if (!strcmp("EACCES", ecode))       eval = kYR_EACCES;
   else if (!strcmp("EIO", ecode))          eval = kYR_EIO;
   else if (!strcmp("ENOMEM", ecode))       eval = kYR_ENOMEM;
   else if (!strcmp("ENOSPC", ecode))       eval = kYR_ENOSPC;
   else if (!strcmp("ENAMETOOLONG", ecode)) eval = kYR_ENAMETOOLONG;
   else if (!strcmp("ENETUNREACH", ecode))  eval = kYR_ENETUNREACH;
   else if (!strcmp("ENOTBLK", ecode))      eval = kYR_ENOTBLK;
   else if (!strcmp("EISDIR", ecode))       eval = kYR_EISDIR;
   else                                     eval = kYR_EINVAL;

// Make sure that elen includes a null byte
//
   if (!elen) elen = strlen(emsg)+1;
      else if (emsg[elen]) elen++;

// Send off the reply
//
   Reply(kYR_error, eval, emsg, elen);
}

/******************************************************************************/
  
void XrdCmsReq::Reply_Error(int ecode, const char *emsg, int elen)
{
   unsigned int eval;

// Translate the error name
//
   switch(ecode)
         {case ENOENT:       eval = kYR_ENOENT;
          case EPERM:        eval = kYR_EPERM;
          case EACCES:       eval = kYR_EACCES;
          case EIO:          eval = kYR_EIO;
          case ENOMEM:       eval = kYR_ENOMEM;
          case ENOSPC:       eval = kYR_ENOSPC;
          case ENAMETOOLONG: eval = kYR_ENAMETOOLONG;
          case ENETUNREACH:  eval = kYR_ENETUNREACH;
          case ENOTBLK:      eval = kYR_ENOTBLK;
          case EISDIR:       eval = kYR_EISDIR;
          default:           eval = kYR_EINVAL;
         };

// Make sure that elen includes a null byte
//
   if (!elen) elen = strlen(emsg)+1;
      else if (emsg[elen]) elen++;

// Send off the reply
//
   Reply(kYR_error, eval, emsg, elen);
}

/******************************************************************************/
/*                              R e p l y _ O K                               */
/******************************************************************************/
  
void XrdCmsReq::Reply_OK()
{

// Send off the reply (this object may be deleted so make a fast exit)
//
   Reply(kYR_data, 0, "", 1);
}
  
/******************************************************************************/
  
void XrdCmsReq::Reply_OK(const char *data, int dlen)
{

// Make sure that elen includes a null byte
//
   if (!dlen) dlen = strlen(data)+1;
      else if (data[dlen]) dlen++;

// Send off the reply (this object may be deleted so make a fast exit)
//
   Reply(kYR_data, 0, data, dlen);
}
 
/******************************************************************************/

void XrdCmsReq::Reply_OK(struct stat &buf)
{
   char sbuff[256];

   Reply_OK(sbuff, StatGen(buf, sbuff));
}

/******************************************************************************/
/*                        R e p l y _ R e d i r e c t                         */
/******************************************************************************/
  
void XrdCmsReq::Reply_Redirect(const char *sname, 
                               const char *lcgi, const char *ocgi)
{
   char hbuff[256], *colon;
   const char *hP = hbuff;
   int hlen, Port;

// Find the port number in the host name
//
   if (!(colon = (char *) index(sname, ':'))) 
      {Port = 0;
       hP = sname;
      } else {
       Port = atoi(colon+1);
       hlen = colon-sname+1;
       if (hlen >= (int)sizeof(hbuff)) hlen = sizeof(hbuff);
       strlcpy(hbuff, sname, hlen);
      }

// Send off the request
//
   Reply_Redirect(hP, Port, lcgi, ocgi);
}

/******************************************************************************/
  
void XrdCmsReq::Reply_Redirect(const char *sname, int Port,
                               const char *lcgi, const char *ocgi)
{
   struct iovec iov[8];
   int iovnum, hlen = strlen(sname);

// Fill out the iovec
//
   iov[1].iov_base = (char *)sname;
   iov[1].iov_len  = strlen(sname);
   hlen = iov[1].iov_len;

// Now we need to see if we have any cgi info to pass
//
   if (!lcgi && !ocgi) iovnum = 2;
      else {if (ocgi)
               {iov[2].iov_base = (char *)"?";
                iov[2].iov_len  = 1;
                iov[3].iov_base = (char *)ocgi;
                iov[3].iov_len  = strlen(ocgi);
                hlen += iov[3].iov_len + 1;
                if (lcgi)
                   {iov[4].iov_base = (char *)"?";
                    iov[4].iov_len  = 1;
                    iov[5].iov_base = (char *)lcgi;
                    iov[5].iov_len  = strlen(lcgi);
                    hlen += iov[5].iov_len + 1;
                          iovnum = 6;
                   } else iovnum = 4;
               } else {
                iov[2].iov_base = (char *)"??";
                iov[2].iov_len  = 2;
                iov[3].iov_base = (char *)lcgi;
                iov[3].iov_len  = strlen(lcgi);
                hlen += iov[3].iov_len + 2;
                iovnum = 4;
               }
           }

// Make sure that last iov element and hlen includes the terminating null byte
//
   iov[iovnum-1].iov_len++; hlen++;

// Send off the reply
//
   Reply(kYR_redirect, (unsigned int)Port, 0, hlen, iov, iovnum);
}

/******************************************************************************/
/*                            R e p l y _ W a i t                             */
/******************************************************************************/
  
void XrdCmsReq::Reply_Wait(int sec)
{

// Send off the reply
//
   Reply(kYR_wait, (unsigned int)sec, "", 1);
}

/******************************************************************************/
/*                        R e p l y _ W a i t R e s p                         */
/******************************************************************************/
  
XrdCmsReq *XrdCmsReq::Reply_WaitResp(int sec)
{
   static XrdSysMutex rnMutex;
   static unsigned int RequestNum = 0;
          unsigned int rnum;
          XrdCmsReq *newReq;

// If this is already a waitresp object then we cannot do this again. So,
// just return a null pointer indicating an invalid call.
//
   if (!NodeP) return (XrdCmsReq *)0;

// Generate a request number unless no reply is needed
//
   if (ReqID)
      {rnMutex.Lock();
       RequestNum++;
       rnum = RequestNum;
       rnMutex.UnLock();
     } else rnum = 0;

// Construct a new request object. This object will be used to actually effect
// the reply. We need to do this because the server may disappear before we
// actually reply. In which case the reply gets deep-sixed.
//
   newReq = new XrdCmsReq(this, rnum);

// Reply to the requestor mapping our ID to their ID
//
   if (rnum)
      {
       Reply(kYR_waitresp, rnum);
      }

// Return an object to affect an asynchronous reply
//
   return newReq;
}

/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
  
#define XRDXROOTD_STAT_CLASSNAME XrdCmsReq
#include "XrdXrootd/XrdXrootdStat.icc"

/******************************************************************************/
/*                               n o R e p l y                                */
/******************************************************************************/
  
void XrdCmsReq::noReply()
{
   static int nrNum = 255;

// We always issue a message about double object use otherwise issue warning
// as this is indicative of an improper configuration.
//
   if (ReqNnum < 0)
      Say.Emsg("Req", "Attempted reply to twice to a 2way async request.");
      else {nrNum++;
            if (!(nrNum & 255)) Say.Emsg("Req", 
                                "Attempted reply to a 1way request; "
                                "probably incorrect ofs forward directive.");
           }
}

/******************************************************************************/
/*                                 R e p l y                                  */
/******************************************************************************/
  
void XrdCmsReq::Reply(       int    respCode, unsigned int respVal,
                      const  char  *respData, int respLen,
                      struct iovec *iov,      int iovnum)
{
   EPNAME("Reply");
   CmsResponse Resp = {{ReqID, respCode, 0, 0}, htonl(respVal)};
   struct iovec myiov[2], *iovP;
   XrdCmsNode *nP;

// Set the actual data length
//
   Resp.Hdr.datalen = htons(static_cast<short>(respLen+sizeof(int)));

// Complete iovec
//
   if (iov)
      { iov->iov_base = (char *)&Resp;
        iov->iov_len  = sizeof(Resp);
        iovP = iov;
      } else {
       myiov[0].iov_base = (char *)&Resp;
       myiov[0].iov_len  = sizeof(Resp);
       if (respData)
          {myiov[1].iov_base = (char *)respData;
           myiov[1].iov_len  = respLen;
           iovnum = 2;
          } else iovnum = 1;
       iovP = myiov;
      }

// Reply format differs depending on whether this is a sync or async reply
//
   if (NodeP)
      {if (ReqID) NodeP->Send(iovP, iovnum);
          else noReply();
       return;
      }

// Async replies are more complicated here since we must find the server using
// a logical address that may no longer be valid.
//
   RTable.Lock();
   if ((nP = RTable.Find(ReqNnum, ReqNins)))
      {Resp.Hdr.modifier |= CmsResponse::kYR_async;
       nP->Send(iovP, iovnum);
      }
      else {DEBUG("Async resp " <<ReqID <<" discarded; server gone");}
   RTable.UnLock();

// Only one async response is allowed. Mark this object unusable
//
   ReqID   = 0;
   ReqNnum = -1;
}
