/******************************************************************************/
/*                                                                            */
/*                  X r d X r o o t d R e s p o n s e . c c                   */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//      $Id$

const char *XrdXrootdResponseCVSID = "$Id$";
 
#include <sys/types.h>
#include <netinet/in.h>
#include <inttypes.h>
#include <string.h>

#include "Xrd/XrdLink.hh"
#include "XrdXrootd/XrdXrootdResponse.hh"
#include "XrdXrootd/XrdXrootdTrace.hh"
  
/******************************************************************************/
/*                               G l o b a l s                                */
/******************************************************************************/
  
extern XrdOucTrace *XrdXrootdTrace;

const char *XrdXrootdResponse::TraceID = "Response";

/******************************************************************************/
/*                         L o c a l   D e f i n e s                          */
/******************************************************************************/

#define TRACELINK Link
  
/******************************************************************************/
/*                                  P u s h                                   */
/******************************************************************************/

int XrdXrootdResponse::Push(void *data, int dlen)
{
    kXR_int32 DLen = static_cast<kXR_int32>(htonl(dlen));
    RespIO[1].iov_base = (caddr_t)&DLen;
    RespIO[1].iov_len  = sizeof(dlen);
    RespIO[2].iov_base = (caddr_t)data;
    RespIO[2].iov_len  = dlen;

    TRACES(RSP, "pushing " <<dlen <<" data bytes");

    if (Link->Send(&RespIO[1], 2, sizeof(kXR_int32) + dlen) < 0)
       return Link->setEtext("send failure");
    return 0;
}

int XrdXrootdResponse::Push()
{
    static int null = 0;
    TRACES(RSP, "pushing " <<sizeof(kXR_int32) <<" data bytes");
    if (Link->Send((char *)&null, sizeof(kXR_int32)) < 0)
       return Link->setEtext("send failure");
    return 0;
}

/******************************************************************************/
/*                                  S e n d                                   */
/******************************************************************************/

int XrdXrootdResponse::Send()
{

    Resp.status = static_cast<kXR_unt16>(htons(kXR_ok));
    Resp.dlen   = 0;
    TRACES(RSP, "sending OK");

    if (Link->Send((char *)&Resp, sizeof(Resp)) < 0)
       return Link->setEtext("send failure");
    return 0;
}

/******************************************************************************/

int XrdXrootdResponse::Send(const char *msg)
{

    Resp.status        = static_cast<kXR_unt16>(htons(kXR_ok));
    RespIO[1].iov_base = (caddr_t)msg;
    RespIO[1].iov_len  = strlen(msg)+1;
    Resp.dlen          = static_cast<kXR_int32>(htonl(RespIO[1].iov_len));
    TRACES(RSP, "sending OK: " <<msg);

    if (Link->Send(RespIO, 2, sizeof(Resp) + RespIO[1].iov_len) < 0)
       return Link->setEtext("send failure");
    return 0;
}

/******************************************************************************/

int XrdXrootdResponse::Send(XResponseType rcode, void *data, int dlen)
{

    Resp.status        = static_cast<kXR_unt16>(htons(rcode));
    RespIO[1].iov_base = (caddr_t)data;
    RespIO[1].iov_len  = dlen;
    Resp.dlen          = static_cast<kXR_int32>(htonl(dlen));
    TRACES(RSP, "sending " <<dlen <<" data bytes; status=" <<rcode);

    if (Link->Send(RespIO, 2, sizeof(Resp) + dlen) < 0)
       return Link->setEtext("send failure");
    return 0;
}

/******************************************************************************/

int XrdXrootdResponse::Send(XResponseType rcode, int info, const char *data)
{
    kXR_int32 xbuf = static_cast<kXR_int32>(htonl(info));
    int dlen;

    Resp.status        = static_cast<kXR_unt16>(htons(rcode));
    RespIO[1].iov_base = (caddr_t)(&xbuf);
    RespIO[1].iov_len  = sizeof(xbuf);
    RespIO[2].iov_base = (caddr_t)data;
    RespIO[2].iov_len  = dlen = strlen(data);
    Resp.dlen          = static_cast<kXR_int32>(htonl((dlen+sizeof(xbuf))));

    TRACES(RSP, "sending " <<dlen <<" data bytes; status=" <<rcode);

    if (Link->Send(RespIO, 3, sizeof(Resp) + dlen) < 0)
       return Link->setEtext("send failure");
    return 0;
}

/******************************************************************************/

int XrdXrootdResponse::Send(void *data, int dlen)
{

    Resp.status        = static_cast<kXR_unt16>(htons(kXR_ok));
    RespIO[1].iov_base = (caddr_t)data;
    RespIO[1].iov_len  = dlen;
    Resp.dlen          = static_cast<kXR_int32>(htonl(dlen));
    TRACES(RSP, "sending " <<dlen <<" data bytes; status=0");

    if (Link->Send(RespIO, 2, sizeof(Resp) + dlen) < 0)
       return Link->setEtext("send failure");
    return 0;
}

/******************************************************************************/

int XrdXrootdResponse::Send(struct iovec *IOResp, int iornum, int iolen)
{
    int i, dlen = 0;

    if (iolen < 0) for (i = 1; i < iornum; i++) dlen += IOResp[i].iov_len;
       else dlen = iolen;

    Resp.status        = static_cast<kXR_unt16>(htons(kXR_ok));
    IOResp[0].iov_base = RespIO[0].iov_base;
    IOResp[0].iov_len  = RespIO[0].iov_len;
    Resp.dlen          = static_cast<kXR_int32>(htonl(dlen));
    TRACES(RSP, "sending " <<dlen <<" data bytes; status=0");

    if (Link->Send(IOResp, iornum, sizeof(Resp) + dlen) < 0)
       return Link->setEtext("send failure");
    return 0;
}

/******************************************************************************/

int XrdXrootdResponse::Send(XErrorCode ecode, const char *msg)
{
    int dlen;
    kXR_int32 erc = static_cast<kXR_int32>(htonl(ecode));

    Resp.status        = static_cast<kXR_unt16>(htons(kXR_error));
    RespIO[1].iov_base = (char *)&erc;
    RespIO[1].iov_len  = sizeof(erc);
    RespIO[2].iov_base = (caddr_t)msg;
    RespIO[2].iov_len  = strlen(msg)+1;
                dlen   = sizeof(erc) + RespIO[2].iov_len;
    Resp.dlen          = static_cast<kXR_int32>(htonl(dlen));
    TRACES(EMSG, "sending err " <<ecode <<": " <<msg);

    if (Link->Send(RespIO, 3, sizeof(Resp) + dlen) < 0)
       return Link->setEtext("send failure");
    return 0;
}
 
/******************************************************************************/

int XrdXrootdResponse::Send(int fdnum, long long offset, int dlen)
{
   struct XrdLink::sfVec myVec[2];

// We are only called should sendfile be enabled for this response
//
   Resp.status = static_cast<kXR_unt16>(htons(kXR_ok));
   Resp.dlen   = static_cast<kXR_int32>(htonl(dlen));

// Fill out the sendfile vector
//
   myVec[0].buffer = (char *)&Resp;
   myVec[0].sendsz = sizeof(Resp);
   myVec[0].fdnum  = -1;
   myVec[1].offset = static_cast<off_t>(offset);
   myVec[1].sendsz = dlen;
   myVec[1].fdnum  = fdnum;

// Send off the request
//
    TRACES(RSP, "sendfile " <<dlen <<" data bytes; status=0");
    if (Link->Send(myVec, 2) < 0)
       return Link->setEtext("sendfile failure");
    return 0;
}

/******************************************************************************/

int XrdXrootdResponse::Send(XrdXrootdReqID &ReqID, 
                            XResponseType   Status,
                            struct iovec   *IOResp, 
                            int             iornum, 
                            int             iolen)
{
   static const kXR_unt16 Xattn = static_cast<kXR_unt16>(htons(kXR_attn));
   static const kXR_int32 Xarsp = static_cast<kXR_int32>(htonl(kXR_asynresp));

// We would have used struct ServerResponseBody_Attn_asynresp but the silly
// imbedded 4096 char array causes grief when computing lengths.
//
   struct {ServerResponseHeader atnHdr;
           kXR_int32            act;
           kXR_int32            rsvd;  // Same as char[4]
           ServerResponseHeader theHdr;
          } asynResp;

   static const int sfxLen = sizeof(asynResp) - sizeof(asynResp.atnHdr);

   XrdLink           *Link;
   unsigned char      theSID[2];
   int                theFD, rc;
   unsigned int       theInst;

// Fill out the header with constant information
//
   asynResp.atnHdr.streamid[0] = '\0';
   asynResp.atnHdr.streamid[1] = '\0';
   asynResp.atnHdr.status      = Xattn;
   asynResp.act                = Xarsp;
   asynResp.rsvd               = 0;

// Complete the io vector to send this response
//
   IOResp[0].iov_base = (char *)&asynResp;
   IOResp[0].iov_len  = sizeof(asynResp);           // 0

// Insert the status code
//
    asynResp.theHdr.status = static_cast<kXR_unt16>(htons(Status));

// We now insert the length of the delayed response and the full response
//
   asynResp.theHdr.dlen = static_cast<kXR_int32>(htonl(iolen));
   iolen += sfxLen;
   asynResp.atnHdr.dlen = static_cast<kXR_int32>(htonl(iolen));
   iolen += sizeof(ServerResponseHeader);

// Decode the destination
//
   ReqID.getID(theSID, theFD, theInst);

// Map the destination to an endpoint, and send the response
//
   if ((Link = XrdLink::fd2link(theFD, theInst)))
      {Link->setRef(1);
       if (Link->isInstance(theInst))
          {asynResp.theHdr.streamid[0] = theSID[0];
           asynResp.theHdr.streamid[1] = theSID[1];
           rc = Link->Send(IOResp, iornum, iolen);
          } else rc = -1;
       Link->setRef(-1);
       return (rc < 0 ? -1 : 0);
      }
   return -1;
}
  
/******************************************************************************/
/*                                   S e t                                    */
/******************************************************************************/

void XrdXrootdResponse::Set(unsigned char *stream)
{
   static char hv[] = "0123456789abcdef";
   char *outbuff;
   int i;

   Resp.streamid[0] = stream[0];
   Resp.streamid[1] = stream[1];

   if (TRACING((TRACE_REQ|TRACE_RSP)))
      {outbuff = trsid;
       for (i = 0; i < (int)sizeof(Resp.streamid); i++)
           {*outbuff++ = hv[(stream[i] >> 4) & 0x0f];
            *outbuff++ = hv[ stream[i]       & 0x0f];
            }
       *outbuff++ = ' '; *outbuff = '\0';
      }
}
