// @(#)root/proofd:$Id$
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

#include "XrdProofdAux.h"
#include "XrdProofdProtocol.h"
#include "XrdProofdResponse.h"

// Tracing utils
#include "XrdProofdTrace.h"

// Notification macro
#define XPRNOTIFY(m,e) \
   if (rc != 0) { \
      TRACER(this, XERR, m << ": " << e); \
   } else { \
      if (TRACING(RSP)) { \
         if (e.length() > 0) { \
            TRACER(this, RSP, m << " (" << e <<")"); \
         } else { \
            TRACER(this, RSP, m); \
         } \
      } \
   }
// Tracing condition
#define XPRTRACING(a) ((a != 0) || (TRACING(RSP)))
// Check link macro
#define CHECKLINK \
   if (!fLink) { \
      TRACE(XERR, "link is undefined! "); \
      return 0; \
   } else if (fLink->FDnum() < 0) { \
      TRACE(XERR, "link descriptor invalid for link "<<fLink<<"! ("<< fLink->FDnum()<<")"); \
      return 0; \
   }

//______________________________________________________________________________
int XrdProofdResponse::Send()
{
   // Auxilliary Send method
   XPDLOC(RSP, "Response::Send:1")

   CHECKLINK;

   int rc = 0;
   XrdOucString tmsg, emsg;
   {  XrdSysMutexHelper mh(fMutex);

      fResp.status = static_cast<kXR_unt16>(htons(kXR_ok));
      fResp.dlen   = 0;

      // Send over
      rc = LinkSend((char *)&fResp, sizeof(fResp), emsg);
   }
   XPRNOTIFY("sending OK", emsg);
   return rc;
}

//______________________________________________________________________________
int XrdProofdResponse::Send(XResponseType rcode)
{
   // Auxilliary Send method
   XPDLOC(RSP, "Response::Send:2")

   CHECKLINK;

   int rc = 0;
   XrdOucString tmsg, emsg;
   {  XrdSysMutexHelper mh(fMutex);

      fResp.status        = static_cast<kXR_unt16>(htons(rcode));
      fResp.dlen          = 0;

      // Send over
      rc = LinkSend((char *)&fResp, sizeof(fResp), emsg);
      if (XPRTRACING(rc)) tmsg.form("sending OK: status = %d", rcode);
   }
   XPRNOTIFY(tmsg, emsg);
   return rc;
}

//______________________________________________________________________________
int XrdProofdResponse::Send(const char *msg)
{
   // Auxilliary Send method
   XPDLOC(RSP, "Response::Send:3")

   CHECKLINK;

   int rc = 0;
   XrdOucString tmsg, emsg;
   {  XrdSysMutexHelper mh(fMutex);

      fResp.status        = static_cast<kXR_unt16>(htons(kXR_ok));
      fRespIO[1].iov_base = (caddr_t)msg;
      fRespIO[1].iov_len  = strlen(msg)+1;
      fResp.dlen          = static_cast<kXR_int32>(htonl(fRespIO[1].iov_len));

      // Send over
      rc = LinkSend(fRespIO, 2, sizeof(fResp) + fRespIO[1].iov_len, emsg);
      if (XPRTRACING(rc)) tmsg.form("sending OK: %s", msg);
   }
   XPRNOTIFY(tmsg, emsg);
   return rc;
}

//______________________________________________________________________________
int XrdProofdResponse::Send(XResponseType rcode, void *data, int dlen)
{
   // Auxilliary Send method
   XPDLOC(RSP, "Response::Send:4")

   CHECKLINK;

   int rc = 0;
   XrdOucString tmsg, emsg;
   {  XrdSysMutexHelper mh(fMutex);

      fResp.status        = static_cast<kXR_unt16>(htons(rcode));
      fRespIO[1].iov_base = (caddr_t)data;
      fRespIO[1].iov_len  = dlen;
      fResp.dlen          = static_cast<kXR_int32>(htonl(dlen));

      // Send over
      rc = LinkSend(fRespIO, 2, sizeof(fResp) + dlen, emsg);
      if (XPRTRACING(rc)) tmsg.form("sending %d data bytes; status=%d", dlen, rcode);
   }
   XPRNOTIFY(tmsg, emsg);
   return rc;
}

//______________________________________________________________________________
int XrdProofdResponse::Send(XResponseType rcode, int info, char *data)
{
   // Auxilliary Send method
   XPDLOC(RSP, "Response::Send:5")

   CHECKLINK;

   int rc = 0;
   XrdOucString tmsg, emsg;
   {  XrdSysMutexHelper mh(fMutex);

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
      }
      fResp.dlen          = static_cast<kXR_int32>(htonl((dlen+sizeof(xbuf))));

      // Send over
      rc = LinkSend(fRespIO, nn, sizeof(fResp) + dlen, emsg);
      if (XPRTRACING(rc)) {
         if (data)
            tmsg.form("sending %d data bytes; info=%d; status=%d", dlen, info, rcode);
         else
            tmsg.form("sending info=%d; status=%d", info, rcode);
      }
   }
   XPRNOTIFY(tmsg, emsg);
   return rc;
}

//______________________________________________________________________________
int XrdProofdResponse::Send(XResponseType rcode, XProofActionCode acode,
                            void *data, int dlen )
{
   // Auxilliary Send method
   XPDLOC(RSP, "Response::Send:6")

   CHECKLINK;

   int rc = 0;
   XrdOucString tmsg, emsg;
   {  XrdSysMutexHelper mh(fMutex);

      kXR_int32 xbuf = static_cast<kXR_int32>(htonl(acode));
      int nn = 2;

      fResp.status        = static_cast<kXR_unt16>(htons(rcode));
      fRespIO[1].iov_base = (caddr_t)(&xbuf);
      fRespIO[1].iov_len  = sizeof(xbuf);
      if (data) {
         nn = 3;
         fRespIO[2].iov_base = (caddr_t)data;
         fRespIO[2].iov_len  = dlen;
      }
      fResp.dlen = static_cast<kXR_int32>(htonl((dlen+sizeof(xbuf))));

      // Send over
      rc = LinkSend(fRespIO, nn, sizeof(fResp) + dlen, emsg);
      if (XPRTRACING(rc)) {
         tmsg.form("sending %d data bytes; status=%d; action=%d",
                   dlen, rcode, acode);
      } else {
         tmsg.form("sending status=%d; action=%d", rcode, acode);
      }
   }
   XPRNOTIFY(tmsg, emsg);
   return rc;
}

//______________________________________________________________________________
int XrdProofdResponse::Send(XResponseType rcode, XProofActionCode acode,
                            kXR_int32 cid, void *data, int dlen )
{
   // Auxilliary Send method
   XPDLOC(RSP, "Response::Send:7")

   CHECKLINK;

   int rc = 0;
   XrdOucString tmsg, emsg;
   {  XrdSysMutexHelper mh(fMutex);

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
      }
      fResp.dlen = static_cast<kXR_int32>(htonl((dlen+hlen)));

      // Send over
      rc = LinkSend(fRespIO, nn, sizeof(fResp) + dlen, emsg);
      if (XPRTRACING(rc)) {
         tmsg.form("sending %d data bytes; status=%d; action=%d; cid=%d",
                   dlen, rcode, acode, cid);
      } else {
         tmsg.form("sending status=%d; action=%d; cid=%d", rcode, acode, cid);
      }
   }
   XPRNOTIFY(tmsg, emsg);
   return rc;
}

//______________________________________________________________________________
int XrdProofdResponse::Send(XResponseType rcode, XProofActionCode acode,
                            int info )
{
   // Auxilliary Send method
   XPDLOC(RSP, "Response::Send:8")

   CHECKLINK;

   int rc = 0;
   XrdOucString tmsg, emsg;
   {  XrdSysMutexHelper mh(fMutex);

      kXR_int32 xbuf = static_cast<kXR_int32>(htonl(acode));
      kXR_int32 xinf = static_cast<kXR_int32>(htonl(info));
      int hlen = sizeof(xbuf) + sizeof(xinf);

      fResp.status        = static_cast<kXR_unt16>(htons(rcode));
      fRespIO[1].iov_base = (caddr_t)(&xbuf);
      fRespIO[1].iov_len  = sizeof(xbuf);
      fRespIO[2].iov_base = (caddr_t)(&xinf);
      fRespIO[2].iov_len  = sizeof(xinf);
      fResp.dlen = static_cast<kXR_int32>(htonl((hlen)));

      // Send over
      rc = LinkSend(fRespIO, 3, sizeof(fResp), emsg);
      if (XPRTRACING(rc))
         tmsg.form("sending info=%d; status=%d; action=%d", info, rcode, acode);
   }
   XPRNOTIFY(tmsg, emsg);
   return rc;
}

//______________________________________________________________________________
int XrdProofdResponse::SendI(kXR_int32 int1, kXR_int16 int2, kXR_int16 int3,
                            void *data, int dlen )
{
   // Auxilliary Send method
   XPDLOC(RSP, "Response::SendI:1")

   CHECKLINK;

   int rc = 0;
   XrdOucString tmsg, emsg;
   {  XrdSysMutexHelper mh(fMutex);

      kXR_int32 i1 = static_cast<kXR_int32>(htonl(int1));
      kXR_int16 i2 = static_cast<kXR_int16>(htons(int2));
      kXR_int16 i3 = static_cast<kXR_int16>(htons(int3));
      int ilen = sizeof(i1) + sizeof(i2) + sizeof(i3);
      int nn = 4;

      fResp.status        = static_cast<kXR_unt16>(htons(kXR_ok));
      fRespIO[1].iov_base = (caddr_t)(&i1);
      fRespIO[1].iov_len  = sizeof(i1);
      fRespIO[2].iov_base = (caddr_t)(&i2);
      fRespIO[2].iov_len  = sizeof(i2);
      fRespIO[3].iov_base = (caddr_t)(&i3);
      fRespIO[3].iov_len  = sizeof(i3);
      if (data) {
         nn = 5;
         fRespIO[4].iov_base = (caddr_t)data;
         fRespIO[4].iov_len  = dlen;
      }
      fResp.dlen = static_cast<kXR_int32>(htonl((dlen+ilen)));

      // Send over
      rc = LinkSend(fRespIO, nn, sizeof(fResp) + dlen, emsg);
      if (XPRTRACING(rc)) {
         tmsg.form("sending %d data bytes; int1=%d; int2=%d; int3=%d",
                    dlen, int1, int2, int3);
      } else {
         tmsg.form("sending int1=%d; int2=%d; int3=%d", int1, int2, int3);
      }
   }
   XPRNOTIFY(tmsg, emsg);
   return rc;
}

//______________________________________________________________________________
int XrdProofdResponse::SendI(kXR_int32 int1, kXR_int32 int2, void *data, int dlen )
{
   // Auxilliary Send method
   XPDLOC(RSP, "Response::SendI:2")

   CHECKLINK;

   int rc = 0;
   XrdOucString tmsg, emsg;
   {  XrdSysMutexHelper mh(fMutex);

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
      }
      fResp.dlen = static_cast<kXR_int32>(htonl((dlen+ilen)));

      // Send over
      rc = LinkSend(fRespIO, nn, sizeof(fResp) + dlen, emsg);
      if (XPRTRACING(rc)) {
         tmsg.form("sending %d data bytes; int1=%d; int2=%d",
                   dlen, int1, int2);
      } else {
         tmsg.form("sending int1=%d; int2=%d", int1, int2);
      }
   }
   XPRNOTIFY(tmsg, emsg);
   return rc;
}

//______________________________________________________________________________
int XrdProofdResponse::SendI(kXR_int32 int1, void *data, int dlen )
{
   // Auxilliary Send method
   XPDLOC(RSP, "Response::SendI:3")

   CHECKLINK;

   int rc = 0;
   XrdOucString tmsg, emsg;
   {  XrdSysMutexHelper mh(fMutex);

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
      }
      fResp.dlen          = static_cast<kXR_int32>(htonl((dlen+ilen)));

      // Send over
      rc = LinkSend(fRespIO, nn, sizeof(fResp) + dlen, emsg);
      if (XPRTRACING(rc)) {
         tmsg.form("sending %d data bytes; int1=%d", dlen, int1);
      } else {
         tmsg.form("sending int1=%d", int1);
      }
   }
   XPRNOTIFY(tmsg, emsg);
   return rc;
}

//______________________________________________________________________________
int XrdProofdResponse::Send(void *data, int dlen)
{
   // Auxilliary Send method
   XPDLOC(RSP, "Response::Send:9")

   CHECKLINK;

   int rc = 0;
   XrdOucString tmsg, emsg;
   {  XrdSysMutexHelper mh(fMutex);

      fResp.status        = static_cast<kXR_unt16>(htons(kXR_ok));
      fRespIO[1].iov_base = (caddr_t)data;
      fRespIO[1].iov_len  = dlen;
      fResp.dlen          = static_cast<kXR_int32>(htonl(dlen));

      // Send over
      rc = LinkSend(fRespIO, 2, sizeof(fResp) + dlen, emsg);
      if (XPRTRACING(rc)) tmsg.form("sending %d data bytes; status=0", dlen);
   }
   XPRNOTIFY(tmsg, emsg);
   return rc;
}

//______________________________________________________________________________
int XrdProofdResponse::Send(struct iovec *IOResp, int iornum, int iolen)
{
   // Auxilliary Send method
   XPDLOC(RSP, "Response::Send:10")

   CHECKLINK;

   int rc = 0;
   XrdOucString tmsg, emsg;
   {  XrdSysMutexHelper mh(fMutex);

      int i, dlen = 0;
      if (iolen < 0) {
         for (i = 1; i < iornum; i++)
            dlen += IOResp[i].iov_len;
      } else {
         dlen = iolen;
      }

      fResp.status        = static_cast<kXR_unt16>(htons(kXR_ok));
      IOResp[0].iov_base = fRespIO[0].iov_base;
      IOResp[0].iov_len  = fRespIO[0].iov_len;
      fResp.dlen          = static_cast<kXR_int32>(htonl(dlen));

      // Send over
      rc = LinkSend(fRespIO, iornum, sizeof(fResp) + dlen, emsg);
      if (XPRTRACING(rc)) tmsg.form("sending %d data bytes; status=0", dlen);
   }
   XPRNOTIFY(tmsg, emsg);
   return rc;
}

//______________________________________________________________________________
int XrdProofdResponse::Send(XErrorCode ecode, const char *msg)
{
   // Auxilliary Send method
   XPDLOC(RSP, "Response::Send:11")

   CHECKLINK;

   int rc = 0;
   XrdOucString tmsg, emsg;
   {  XrdSysMutexHelper mh(fMutex);

      int dlen;
      kXR_int32 erc = static_cast<kXR_int32>(htonl(ecode));

      fResp.status        = static_cast<kXR_unt16>(htons(kXR_error));
      fRespIO[1].iov_base = (char *)&erc;
      fRespIO[1].iov_len  = sizeof(erc);
      fRespIO[2].iov_base = (caddr_t)msg;
      fRespIO[2].iov_len  = strlen(msg)+1;
      dlen   = sizeof(erc) + fRespIO[2].iov_len;
      fResp.dlen          = static_cast<kXR_int32>(htonl(dlen));

      // Send over
      rc = LinkSend(fRespIO, 3, sizeof(fResp) + dlen, emsg);
      if (XPRTRACING(rc)) tmsg.form("sending err %d: %s", ecode, msg);
   }
   XPRNOTIFY(tmsg, emsg);
   return rc;
}

//______________________________________________________________________________
int XrdProofdResponse::Send(XPErrorCode ecode, const char *msg)
{
   // Auxilliary Send method
   XPDLOC(RSP, "Response::Send:12")

   CHECKLINK;

   int rc = 0;
   XrdOucString tmsg, emsg;
   {  XrdSysMutexHelper mh(fMutex);

      int dlen;
      kXR_int32 erc = static_cast<kXR_int32>(htonl(ecode));

      fResp.status        = static_cast<kXR_unt16>(htons(kXR_error));
      fRespIO[1].iov_base = (char *)&erc;
      fRespIO[1].iov_len  = sizeof(erc);
      fRespIO[2].iov_base = (caddr_t)msg;
      fRespIO[2].iov_len  = strlen(msg)+1;
      dlen   = sizeof(erc) + fRespIO[2].iov_len;
      fResp.dlen          = static_cast<kXR_int32>(htonl(dlen));

      // Send over
      rc = LinkSend(fRespIO, 3, sizeof(fResp) + dlen, emsg);
      if (XPRTRACING(rc)) tmsg.form("sending err %d: %s", ecode, msg);
   }
   XPRNOTIFY(tmsg, emsg);
   return rc;
}

//______________________________________________________________________________
int XrdProofdResponse::LinkSend(const char *buff, int len, XrdOucString &emsg)
{
   // Method actually sending the buffer(s) over the link.
   // The link is closed in case of error, because we cannot use it anymore
   // and the counter part needs to reconnect.
   // Return 0 on success, -1 on failure.

   int rc = 0;

   // If we fail we close the link, and ask the client to reconnect
   if ((rc = fLink->Send(buff, len)) < 0) {
      emsg.form("problems sending %d bytes", len);
      fLink->Close();
   }

   // Done
   return ((rc < 0) ? fLink->setEtext("send failure") : 0);
}

//______________________________________________________________________________
int XrdProofdResponse::LinkSend(const struct iovec *iov,
                                int iocnt, int, XrdOucString &emsg)
{
   // Method actually sending the buffer(s) over the link.
   // Functionality a la 'writev' is simulated by segmenting the sending.
   // This allows to avoid a recovery problem with 'writev'.
   // Return 0 on success, -1 on failure.

   int rc = 0;

   // If we fail we close the link, and ask the client to reconnect
   if ((rc = fLink->Send(iov, iocnt, 0)) < 0) {
      int bytes = 0;
      for (int i = 0; i < iocnt; i++) bytes += iov[i].iov_len;
      emsg.form("problems sending %d bytes (writev)", bytes);
      fLink->Close();
   }

   // Done
   return ((rc < 0) ? fLink->setEtext("send (writev) failure") : 0);
}

//______________________________________________________________________________
void XrdProofdResponse::Set(unsigned char *stream)
{
   // Auxilliary Set method

   XrdSysMutexHelper mh(fMutex);

   fResp.streamid[0] = stream[0];
   fResp.streamid[1] = stream[1];

   SetTrsid();
}

//______________________________________________________________________________
void XrdProofdResponse::Set(unsigned short sid)
{
   // Auxilliary Set method

   unsigned char stream[2];

   {  XrdSysMutexHelper mh(fMutex);

      memcpy((void *)&stream[0], (const void *)&sid, sizeof(sid));

      fResp.streamid[0] = stream[0];
      fResp.streamid[1] = stream[1];
   }
   SetTrsid();
}

//______________________________________________________________________________
void XrdProofdResponse::GetSID(unsigned short &sid)
{
   // Get stream ID (to be able to restore it later

   {  XrdSysMutexHelper mh(fMutex);
      memcpy((void *)&sid, (void *)&fResp.streamid[0], sizeof(sid));
   }
}

//______________________________________________________________________________
void XrdProofdResponse::Set(XrdLink *l)
{
   // Set the link to be used by this response
   XPDLOC(RSP, "Response::Set")

   {  XrdSysMutexHelper mh(fMutex);
      fLink = l;
   }
   GetSID(fSID);
   if (fLink) {
      if (fLink->FDnum() < 0) {
         TRACE(XERR, "link descriptor invalid for link "<<fLink<<"! ("<< fLink->FDnum()<<")");
      } else {
         TRACE(DBG,"using link "<<fLink<<", descriptor:"<<fLink->FDnum());
      }
   } else {
      TRACE(XERR,"link is undefined!");
   }
}

//______________________________________________________________________________
void XrdProofdResponse::SetTraceID()
{
   // Auxilliary set method
   XPDLOC(RSP, "Response::SetTraceID")

   {  XrdSysMutexHelper mh(fMutex);
      if (fLink && fTag.length() > 0) {
         fTraceID.form("%s%s: %s: ", fTrsid, fLink->ID, fTag.c_str());
      } else if (fLink) {
         fTraceID.form("%s%s: ", fTrsid, fLink->ID);
      } else if (fTag.length() > 0) {
         fTraceID.form("%s%s: ", fTrsid, fTag.c_str());
      } else {
         fTraceID.form("%s: ", fTrsid);
      }
   }

   TRACE(DBG,"trace set to '"<<fTraceID<<"'")
}

//______________________________________________________________________________
void XrdProofdResponse::SetTrsid()
{
   // Auxilliary Set method

   static char hv[] = "0123456789abcdef";

   int i;
   char *outbuff = fTrsid;
   for (i = 0; i < (int)sizeof(fResp.streamid); i++) {
      *outbuff++ = hv[(fResp.streamid[i] >> 4) & 0x0f];
      *outbuff++ = hv[ fResp.streamid[i]       & 0x0f];
   }
   *outbuff++ = ' ';
   *outbuff = '\0';
}
