// @(#)root/proofd:$Name:  $:$Id: XrdProofdResponse.cxx,v 1.7 2006/11/09 23:23:32 rdm Exp $
// Author: Gerardo Ganis  12/12/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofdResponse                                                    //
//                                                                      //
// Authors: G. Ganis, CERN, 2005                                        //
//                                                                      //
// Utility class to handle replies to clients.                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <sys/types.h>
#include <netinet/in.h>
#include <inttypes.h>
#include <string.h>

#include "Xrd/XrdLink.hh"
#include "XrdProofdResponse.h"

// Tracing utils
#include "XrdProofdTrace.h"
extern XrdOucTrace *XrdProofdTrace;
const char *XrdProofdResponse::fgTraceID = " : Response";
#define TRACEID fTraceID.c_str()
#define TRSID ((const char *)fTrsid)

// Local define
#define TRACELINK fLink

//______________________________________________________________________________
int XrdProofdResponse::Send()
{
   // Auxilliary Send method

   if (!fLink) {
      TRACE(RSP,"XrdProofdResponse:: link is undefined! ");
      return 0;
   }
   XrdOucMutexHelper mh(fMutex);

   fResp.status = static_cast<kXR_unt16>(htons(kXR_ok));
   fResp.dlen   = 0;
   TRACES(RSP, (int *)fLink<< ": sending OK");

   if (fLink->Send((char *)&fResp, sizeof(fResp)) < 0)
      return fLink->setEtext("send failure");
   return 0;
}

//______________________________________________________________________________
int XrdProofdResponse::Send(XResponseType rcode)
{
   // Auxilliary Send method

   if (!fLink) {
      TRACE(RSP,"XrdProofdResponse:: link is undefined! ");
      return 0;
   }
   XrdOucMutexHelper mh(fMutex);

   fResp.status        = static_cast<kXR_unt16>(htons(rcode));
   fResp.dlen          = 0;
   TRACES(RSP, (int *)fLink<< ": sending OK; status = "<<rcode);

   if (fLink->Send((char *)&fResp, sizeof(fResp)) < 0)
      return fLink->setEtext("send failure");
   return 0;
}

//______________________________________________________________________________
int XrdProofdResponse::Send(const char *msg)
{
   // Auxilliary Send method

   if (!fLink) {
      TRACE(RSP,"XrdProofdResponse:: link is undefined! ");
      return 0;
   }
   XrdOucMutexHelper mh(fMutex);

   fResp.status        = static_cast<kXR_unt16>(htons(kXR_ok));
   fRespIO[1].iov_base = (caddr_t)msg;
   fRespIO[1].iov_len  = strlen(msg)+1;
   fResp.dlen          = static_cast<kXR_int32>(htonl(fRespIO[1].iov_len));
   TRACES(RSP,(int *)fLink<<": sending OK: " <<msg);

   if (fLink->Send(fRespIO, 2, sizeof(fResp) + fRespIO[1].iov_len) < 0)
      return fLink->setEtext("send failure");
   return 0;
}

//______________________________________________________________________________
int XrdProofdResponse::Send(XResponseType rcode, void *data, int dlen)
{
   // Auxilliary Send method

   if (!fLink) {
      TRACE(RSP,"XrdProofdResponse:: link is undefined! ");
      return 0;
   }
   XrdOucMutexHelper mh(fMutex);

   fResp.status        = static_cast<kXR_unt16>(htons(rcode));
   fRespIO[1].iov_base = (caddr_t)data;
   fRespIO[1].iov_len  = dlen;
   fResp.dlen          = static_cast<kXR_int32>(htonl(dlen));
   TRACES(RSP,(int *)fLink<< ": sending " <<dlen <<" data bytes; status=" <<rcode);

   if (fLink->Send(fRespIO, 2, sizeof(fResp) + dlen) < 0)
      return fLink->setEtext("send failure");
   return 0;
}

//______________________________________________________________________________
int XrdProofdResponse::Send(XResponseType rcode, int info, char *data)
{
   // Auxilliary Send method

   if (!fLink) {
      TRACE(RSP,"XrdProofdResponse:: link is undefined! ");
      return 0;
   }
   XrdOucMutexHelper mh(fMutex);

   kXR_int32 xbuf = static_cast<kXR_int32>(htonl(info));
   int dlen = 0;
   int nn = 2;

   fResp.status        = static_cast<kXR_unt16>(htons(rcode));
   fRespIO[1].iov_base = (caddr_t)(&xbuf);
   fRespIO[1].iov_len  = sizeof(xbuf);
   if (data) {
      nn = 3;
      fRespIO[2].iov_base = (caddr_t)data;
      fRespIO[2].iov_len  = dlen = strlen(data);
      TRACES(RSP,(int *)fLink<<": sending " <<dlen <<" data bytes; status=" <<rcode);
   } else {
      TRACES(RSP,(int *)fLink<<": sending info: " <<info <<"; status=" <<rcode);
   }
   fResp.dlen          = static_cast<kXR_int32>(htonl((dlen+sizeof(xbuf))));

   if (fLink->Send(fRespIO, nn, sizeof(fResp) + dlen) < 0)
      return fLink->setEtext("send failure");
   return 0;
}

//______________________________________________________________________________
int XrdProofdResponse::Send(XResponseType rcode, XProofActionCode acode,
                            void *data, int dlen )
{
   // Auxilliary Send method

   if (!fLink) {
      TRACE(RSP,"XrdProofdResponse:: link is undefined! ");
      return 0;
   }
   XrdOucMutexHelper mh(fMutex);

   kXR_int32 xbuf = static_cast<kXR_int32>(htonl(acode));
   int nn = 2;

   fResp.status        = static_cast<kXR_unt16>(htons(rcode));
   fRespIO[1].iov_base = (caddr_t)(&xbuf);
   fRespIO[1].iov_len  = sizeof(xbuf);
   if (data) {
      nn = 3;
      fRespIO[2].iov_base = (caddr_t)data;
      fRespIO[2].iov_len  = dlen;
      TRACES(RSP,(int *)fLink<<": sending " <<dlen <<
             " data bytes; status=" <<rcode<<"; action="<<acode);
   } else {
      TRACES(RSP,(int *)fLink<<": sending action code=" <<acode <<
             "; status=" <<rcode);
   }
   fResp.dlen          = static_cast<kXR_int32>(htonl((dlen+sizeof(xbuf))));

   if (fLink->Send(fRespIO, nn, sizeof(fResp) + dlen) < 0)
      return fLink->setEtext("send failure");
   return 0;
}

//______________________________________________________________________________
int XrdProofdResponse::Send(XResponseType rcode, XProofActionCode acode,
                            kXR_int32 cid, void *data, int dlen )
{
   // Auxilliary Send method

   if (!fLink) {
      TRACE(RSP,"XrdProofdResponse:: link is undefined! ");
      return 0;
   }
   XrdOucMutexHelper mh(fMutex);

   kXR_int32 xbuf = static_cast<kXR_int32>(htonl(acode));
   kXR_int32 xcid = static_cast<kXR_int32>(htonl(cid));
   int hlen = sizeof(xbuf) + sizeof(xcid);
   int nn = 3;

   fResp.status        = static_cast<kXR_unt16>(htons(rcode));
   fRespIO[1].iov_base = (caddr_t)(&xbuf);
   fRespIO[1].iov_len  = sizeof(xbuf);
   fRespIO[2].iov_base = (caddr_t)(&xcid);
   fRespIO[2].iov_len  = sizeof(xcid);
   if (data) {
      nn = 4;
      fRespIO[3].iov_base = (caddr_t)data;
      fRespIO[3].iov_len  = dlen;
      TRACES(RSP,(int *)fLink<<": sending " <<dlen << " data bytes;"<<
             " status=" <<rcode<<"; action="<<acode<<
             "; cid=" <<cid);
   } else {
      TRACES(RSP,(int *)fLink<<": sending action code=" <<acode <<
             "; status=" <<rcode<<"; cid=" <<cid);
   }
   fResp.dlen          = static_cast<kXR_int32>(htonl((dlen+hlen)));

   if (fLink->Send(fRespIO, nn, sizeof(fResp) + dlen) < 0)
      return fLink->setEtext("send failure");
   return 0;
}

//______________________________________________________________________________
int XrdProofdResponse::Send(XResponseType rcode, XProofActionCode acode,
                            int info )
{
   // Auxilliary Send method

   if (!fLink) {
      TRACE(RSP,"XrdProofdResponse:: link is undefined! ");
      return 0;
   }
   XrdOucMutexHelper mh(fMutex);

   kXR_int32 xbuf = static_cast<kXR_int32>(htonl(acode));
   kXR_int32 xinf = static_cast<kXR_int32>(htonl(info));
   int hlen = sizeof(xbuf) + sizeof(xinf);

   fResp.status        = static_cast<kXR_unt16>(htons(rcode));
   fRespIO[1].iov_base = (caddr_t)(&xbuf);
   fRespIO[1].iov_len  = sizeof(xbuf);
   fRespIO[2].iov_base = (caddr_t)(&xinf);
   fRespIO[2].iov_len  = sizeof(xinf);
   TRACES(RSP,(int *)fLink<<": sending info=" <<info <<
          "; status=" <<rcode<<"; action="<<acode);
   fResp.dlen          = static_cast<kXR_int32>(htonl((hlen)));

   if (fLink->Send(fRespIO, 3, sizeof(fResp)) < 0)
      return fLink->setEtext("send failure");
   return 0;
}

//______________________________________________________________________________
int XrdProofdResponse::Send(kXR_int32 int1, kXR_int32 int2, void *data, int dlen )
{
   // Auxilliary Send method

   if (!fLink) {
      TRACE(RSP,"XrdProofdResponse:: link is undefined! ");
      return 0;
   }
   XrdOucMutexHelper mh(fMutex);

   kXR_int32 i1 = static_cast<kXR_int32>(htonl(int1));
   kXR_int32 i2 = static_cast<kXR_int32>(htonl(int2));
   int ilen = sizeof(i1) + sizeof(i2);
   int nn = 3;

   fResp.status        = static_cast<kXR_unt16>(htons(kXR_ok));
   fRespIO[1].iov_base = (caddr_t)(&i1);
   fRespIO[1].iov_len  = sizeof(i1);
   fRespIO[2].iov_base = (caddr_t)(&i2);
   fRespIO[2].iov_len  = sizeof(i2);
   if (data) {
      nn = 4;
      fRespIO[3].iov_base = (caddr_t)data;
      fRespIO[3].iov_len  = dlen;
      TRACES(RSP,(int *)fLink<<": sending " <<dlen << " data bytes;"<<
             " int1=" <<int1<<"; int2="<<int2);
   } else {
      TRACES(RSP,(int *)fLink<<": sending int1=" <<int1 <<"; int2=" <<int2);
   }
   fResp.dlen          = static_cast<kXR_int32>(htonl((dlen+ilen)));

   if (fLink->Send(fRespIO, nn, sizeof(fResp) + dlen) < 0)
      return fLink->setEtext("send failure");
   return 0;
}

//______________________________________________________________________________
int XrdProofdResponse::Send(kXR_int32 int1, void *data, int dlen )
{
   // Auxilliary Send method

   if (!fLink) {
      TRACE(RSP,"XrdProofdResponse:: link is undefined! ");
      return 0;
   }
   XrdOucMutexHelper mh(fMutex);

   kXR_int32 i1 = static_cast<kXR_int32>(htonl(int1));
   int ilen = sizeof(i1);
   int nn = 2;

   fResp.status        = static_cast<kXR_unt16>(htons(kXR_ok));
   fRespIO[1].iov_base = (caddr_t)(&i1);
   fRespIO[1].iov_len  = sizeof(i1);
   if (data) {
      nn = 3;
      fRespIO[2].iov_base = (caddr_t)data;
      fRespIO[2].iov_len  = dlen;
      TRACES(RSP,(int *)fLink<<": sending " <<dlen << " data bytes;"<<
             " int1=" <<int1);
   } else {
      TRACES(RSP,(int *)fLink<<": sending int1=" <<int1);
   }
   fResp.dlen          = static_cast<kXR_int32>(htonl((dlen+ilen)));

   if (fLink->Send(fRespIO, nn, sizeof(fResp) + dlen) < 0)
      return fLink->setEtext("send failure");
   return 0;
}

//______________________________________________________________________________
int XrdProofdResponse::Send(void *data, int dlen)
{
   // Auxilliary Send method

   if (!fLink) {
      TRACE(RSP,"XrdProofdResponse:: link is undefined! ");
      return 0;
   }
   XrdOucMutexHelper mh(fMutex);

   fResp.status        = static_cast<kXR_unt16>(htons(kXR_ok));
   fRespIO[1].iov_base = (caddr_t)data;
   fRespIO[1].iov_len  = dlen;
   fResp.dlen          = static_cast<kXR_int32>(htonl(dlen));
   TRACES(RSP,(int *)fLink<< ": sending " <<dlen <<" data bytes; status=0");

   if (fLink->Send(fRespIO, 2, sizeof(fResp) + dlen) < 0)
      return fLink->setEtext("send failure");
   return 0;
}

//______________________________________________________________________________
int XrdProofdResponse::Send(struct iovec *IOResp, int iornum, int iolen)
{
   // Auxilliary Send method

   if (!fLink) {
      TRACE(RSP,"XrdProofdResponse:: link is undefined! ");
      return 0;
   }
   XrdOucMutexHelper mh(fMutex);

   int i, dlen = 0;

   if (iolen < 0) for (i = 1; i < iornum; i++) dlen += IOResp[i].iov_len;
   else dlen = iolen;

   fResp.status        = static_cast<kXR_unt16>(htons(kXR_ok));
   IOResp[0].iov_base = fRespIO[0].iov_base;
   IOResp[0].iov_len  = fRespIO[0].iov_len;
   fResp.dlen          = static_cast<kXR_int32>(htonl(dlen));
   TRACES(RSP,(int *)fLink<< ": sending " <<dlen <<" data bytes; status=0");

   if (fLink->Send(IOResp, iornum, sizeof(fResp) + dlen) < 0)
      return fLink->setEtext("send failure");
   return 0;
}

//______________________________________________________________________________
int XrdProofdResponse::Send(XErrorCode ecode, const char *msg)
{
   // Auxilliary Send method

   if (!fLink) {
      TRACE(RSP,"XrdProofdResponse:: link is undefined! ");
      return 0;
   }
   XrdOucMutexHelper mh(fMutex);

   int dlen;
   kXR_int32 erc = static_cast<kXR_int32>(htonl(ecode));

   fResp.status        = static_cast<kXR_unt16>(htons(kXR_error));
   fRespIO[1].iov_base = (char *)&erc;
   fRespIO[1].iov_len  = sizeof(erc);
   fRespIO[2].iov_base = (caddr_t)msg;
   fRespIO[2].iov_len  = strlen(msg)+1;
   dlen   = sizeof(erc) + fRespIO[2].iov_len;
   fResp.dlen          = static_cast<kXR_int32>(htonl(dlen));
   TRACES(RSP,(int *)fLink<< ": sending err " <<ecode <<": " <<msg);

   if (fLink->Send(fRespIO, 3, sizeof(fResp) + dlen) < 0)
      return fLink->setEtext("send failure");
   return 0;
}

//______________________________________________________________________________
int XrdProofdResponse::Send(XPErrorCode ecode, const char *msg)
{
   // Auxilliary Send method

   if (!fLink) {
      TRACE(RSP,"XrdProofdResponse:: link is undefined! ");
      return 0;
   }
   XrdOucMutexHelper mh(fMutex);

   int dlen;
   kXR_int32 erc = static_cast<kXR_int32>(htonl(ecode));

   fResp.status        = static_cast<kXR_unt16>(htons(kXR_error));
   fRespIO[1].iov_base = (char *)&erc;
   fRespIO[1].iov_len  = sizeof(erc);
   fRespIO[2].iov_base = (caddr_t)msg;
   fRespIO[2].iov_len  = strlen(msg)+1;
   dlen   = sizeof(erc) + fRespIO[2].iov_len;
   fResp.dlen          = static_cast<kXR_int32>(htonl(dlen));
   TRACES(RSP,(int *)fLink<< ": sending err " <<ecode <<": " <<msg);

   if (fLink->Send(fRespIO, 3, sizeof(fResp) + dlen) < 0)
      return fLink->setEtext("send failure");
   return 0;
}

//______________________________________________________________________________
void XrdProofdResponse::Set(unsigned char *stream)
{
   // Auxilliary Set method

   static char hv[] = "0123456789abcdef";
   char *outbuff;
   int i;
   XrdOucMutexHelper mh(fMutex);

   fResp.streamid[0] = stream[0];
   fResp.streamid[1] = stream[1];

   if (TRACING(REQ) || TRACING(RSP)) {
      outbuff = fTrsid;
      for (i = 0; i < (int)sizeof(fResp.streamid); i++) {
         *outbuff++ = hv[(stream[i] >> 4) & 0x0f];
         *outbuff++ = hv[ stream[i]       & 0x0f];
      }
      *outbuff++ = ' ';
      *outbuff = '\0';
   }
}

//______________________________________________________________________________
void XrdProofdResponse::Set(unsigned short sid)
{
   // Auxilliary Set method

   static char hv[] = "0123456789abcdef";
   char *outbuff;
   int i;
   unsigned char stream[2];
   XrdOucMutexHelper mh(fMutex);

   memcpy((void *)&stream[0], (const void *)&sid, sizeof(sid));

   fResp.streamid[0] = stream[0];
   fResp.streamid[1] = stream[1];

   if (TRACING(REQ) || TRACING(RSP)) {
      outbuff = fTrsid;
      for (i = 0; i < (int)sizeof(fResp.streamid); i++) {
         *outbuff++ = hv[(stream[i] >> 4) & 0x0f];
         *outbuff++ = hv[ stream[i]       & 0x0f];
      }
      *outbuff++ = ' ';
      *outbuff = '\0';
   }
}

//______________________________________________________________________________
void XrdProofdResponse::GetSID(unsigned short &sid)
{
   // Get stream ID (to be able to restore it later
   XrdOucMutexHelper mh(fMutex);

   memcpy((void *)&sid, (void *)&fResp.streamid[0], sizeof(sid));
}
