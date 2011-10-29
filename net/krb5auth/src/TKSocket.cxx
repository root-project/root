// @(#)root/krb5auth:$Id$
// Author: Maarten Ballintijn   27/10/2003

#include <stdlib.h>
#include <errno.h>
#include <sys/types.h>
#include <netinet/in.h>

#include "TKSocket.h"
#include "TSocket.h"
#include "TError.h"


extern "C" {
// missing from "krb5.h"
extern int krb5_net_read(/*IN*/ krb5_context context, int fd,
                         /*OUT*/ char *buf,/*IN*/ int len);

extern int krb5_net_write(/*IN*/ krb5_context context, int fd,
                          const char *buf, int len);
}

#ifdef __APPLE__
#define SOCKET int
#define SOCKET_ERRNO errno
#define SOCKET_EINTR EINTR
#define SOCKET_READ(a,b,c) read(a,b,c)
#define SOCKET_WRITE(a,b,c) write(a,b,c)
/*
 * lib/krb5/os/net_read.c
 *
 * Copyright 1987, 1988, 1990 by the Massachusetts Institute of Technology.
 * All Rights Reserved.
 *
 * Export of this software from the United States of America may
 *   require a specific license from the United States Government.
 *   It is the responsibility of any person or organization contemplating
 *   export to obtain such a license before exporting.
 *
 * WITHIN THAT CONSTRAINT, permission to use, copy, modify, and
 * distribute this software and its documentation for any purpose and
 * without fee is hereby granted, provided that the above copyright
 * notice appear in all copies and that both that copyright notice and
 * this permission notice appear in supporting documentation, and that
 * the name of M.I.T. not be used in advertising or publicity pertaining
 * to distribution of the software without specific, written prior
 * permission.  Furthermore if you modify this software you must label
 * your software as modified software and not distribute it in such a
 * fashion that it might be confused with the original M.I.T. software.
 * M.I.T. makes no representations about the suitability of
 * this software for any purpose.  It is provided "as is" without express
 * or implied warranty.
 *
 */

/*
 * krb5_net_read() reads from the file descriptor "fd" to the buffer
 * "buf", until either 1) "len" bytes have been read or 2) cannot
 * read anymore from "fd".  It returns the number of bytes read
 * or a read() error.  (The calling interface is identical to
 * read(2).)
 *
 * XXX must not use non-blocking I/O
 */

int krb5_net_read(krb5_context /*context*/, int fd, register char *buf, register int len)
{
   int cc, len2 = 0;

   do {
      cc = SOCKET_READ((SOCKET)fd, buf, len);
      if (cc < 0) {
         if (SOCKET_ERRNO == SOCKET_EINTR)
            continue;

         /* XXX this interface sucks! */
	      errno = SOCKET_ERRNO;

         return(cc);          /* errno is already set */
      } else if (cc == 0) {
         return(len2);
      } else {
         buf += cc;
         len2 += cc;
         len -= cc;
      }
   } while (len > 0);
   return(len2);
}
/*
 * lib/krb5/os/net_write.c
 *
 * Copyright 1987, 1988, 1990 by the Massachusetts Institute of Technology.
 * All Rights Reserved.
 *
 * Export of this software from the United States of America may
 *   require a specific license from the United States Government.
 *   It is the responsibility of any person or organization contemplating
 *   export to obtain such a license before exporting.
 *
 * WITHIN THAT CONSTRAINT, permission to use, copy, modify, and
 * distribute this software and its documentation for any purpose and
 * without fee is hereby granted, provided that the above copyright
 * notice appear in all copies and that both that copyright notice and
 * this permission notice appear in supporting documentation, and that
 * the name of M.I.T. not be used in advertising or publicity pertaining
 * to distribution of the software without specific, written prior
 * permission.  Furthermore if you modify this software you must label
 * your software as modified software and not distribute it in such a
 * fashion that it might be confused with the original M.I.T. software.
 * M.I.T. makes no representations about the suitability of
 * this software for any purpose.  It is provided "as is" without express
 * or implied warranty.
 *
 */

/*
 * krb5_net_write() writes "len" bytes from "buf" to the file
 * descriptor "fd".  It returns the number of bytes written or
 * a write() error.  (The calling interface is identical to
 * write(2).)
 *
 * XXX must not use non-blocking I/O
 */

int krb5_net_write(krb5_context /*context*/, int fd, register const char *buf, int len)
{
   int cc;
   register int wrlen = len;
   do {
      cc = SOCKET_WRITE((SOCKET)fd, buf, wrlen);
      if (cc < 0) {
         if (SOCKET_ERRNO == SOCKET_EINTR)
            continue;

         /* XXX this interface sucks! */
         errno = SOCKET_ERRNO;

         return(cc);
      } else {
         buf += cc;
         wrlen -= cc;
      }
   } while (wrlen > 0);
   return(len);
}
#endif

ClassImp(TKSocket)


krb5_context TKSocket::fgContext = 0;
krb5_ccache TKSocket::fgCCDef = 0;
krb5_principal TKSocket::fgClient = 0;

//______________________________________________________________________________
TKSocket::TKSocket(TSocket *s)
   : fSocket(s), fServer(0), fAuthContext(0)
{
   // Constructor

}

//______________________________________________________________________________
TKSocket::~TKSocket()
{
   // Destructor

   krb5_free_principal(fgContext, fServer);
   krb5_auth_con_free(fgContext, fAuthContext);
   delete fSocket;
}

//______________________________________________________________________________
TKSocket *TKSocket::Connect(const char *server, Int_t port)
{
   // Connect to 'server' on 'port'

   Int_t rc;

   if (fgContext == 0) {
      rc = krb5_init_context(&fgContext);
      if (rc != 0) {
         ::Error("TKSocket::Connect","while initializing krb5 (%d), %s",
                 rc, error_message(rc));
         return 0;
      }

      rc = krb5_cc_default(fgContext, &fgCCDef);
      if (rc != 0) {
         ::Error("TKSocket::Connect","while getting default credential cache (%d), %s",
                 rc, error_message(rc));
         krb5_free_context(fgContext); fgContext = 0;
         return 0;
      }

      rc = krb5_cc_get_principal(fgContext, fgCCDef, &fgClient);
      if (rc != 0) {
         ::Error("TKSocket::Connect","while getting client principal from %s (%d), %s",
                 krb5_cc_get_name(fgContext,fgCCDef), rc, error_message(rc));
         krb5_cc_close(fgContext,fgCCDef); fgCCDef = 0;
         krb5_free_context(fgContext); fgContext = 0;
         return 0;
      }
   }

   TSocket  *s = new TSocket(server, port);

   if (!s->IsValid()) {
      ::SysError("TKSocket::Connect","Cannot connect to %s:%d", server, port);
      delete s;
      return 0;
   }

   TKSocket *ks = new TKSocket(s);

   rc = krb5_sname_to_principal(fgContext, server, "host", KRB5_NT_SRV_HST, &ks->fServer);
   if (rc != 0) {
      ::Error("TKSocket::Connect","while getting server principal (%d), %s",
              rc, error_message(rc));
      delete ks;
      return 0;
   }

   krb5_data cksum_data;
   cksum_data.data = StrDup(server);
   cksum_data.length = strlen(server);

   krb5_error *err_ret;
   krb5_ap_rep_enc_part *rep_ret;

   int sock = ks->fSocket->GetDescriptor();
   rc = krb5_sendauth(fgContext, &ks->fAuthContext, (krb5_pointer) &sock,
                      (char *)"KRB5_TCP_Python_v1.0", fgClient, ks->fServer,
                      AP_OPTS_MUTUAL_REQUIRED,
                      &cksum_data,
                      0,           /* no creds, use ccache instead */
                      fgCCDef, &err_ret, &rep_ret, 0);

   delete [] cksum_data.data;

   if (rc != 0) {
      ::Error("TKSocket::Connect","while sendauth (%d), %s",
              rc, error_message(rc));
      delete ks;
      return 0;
   }

   return ks;
}

//______________________________________________________________________________
Int_t TKSocket::BlockRead(char *&buf, EEncoding &type)
{
   // Read block on information from server. The result is stored in buf.
   // The number of read bytes is returned; -1 is returned in case of error.

   Int_t rc;
   Desc_t desc;
   Int_t fd = fSocket->GetDescriptor();

   rc = krb5_net_read(fgContext, fd, (char *)&desc, sizeof(desc));
   if (rc == 0) errno = ECONNABORTED;

   if (rc <= 0) {
      SysError("BlockRead","reading descriptor (%d), %s",
               rc, error_message(rc));
      return -1;
   }

   type = static_cast<EEncoding>(ntohs(desc.fType));

   krb5_data enc;
   enc.length = ntohs(desc.fLength);
   enc.data = new char[enc.length+1];

   rc = krb5_net_read(fgContext, fd, enc.data, enc.length);
   enc.data[enc.length] = 0;

   if (rc == 0) errno = ECONNABORTED;

   if (rc <= 0) {
      SysError("BlockRead","reading data (%d), %s",
               rc, error_message(rc));
      delete [] enc.data;
      return -1;
   }

   krb5_data out;
   switch (type) {
   case kNone:
      buf = enc.data;
      rc = enc.length;
      break;
   case kSafe:
      rc = krb5_rd_safe(fgContext, fAuthContext, &enc, &out, 0);
      break;
   case kPriv:
      rc = krb5_rd_priv(fgContext, fAuthContext, &enc, &out, 0);
      break;
   default:
      Error("BlockWrite","unknown encoding type (%d)", type);
      return -1;
   }

   if (type != kNone) {
      // copy data to buffer that is new'ed
      buf = new char[out.length+1];
      memcpy(buf, out.data, out.length);
      buf[out.length] = 0;
      free(out.data);
      delete [] enc.data;
      rc = out.length;
   }

   return rc;
}

//______________________________________________________________________________
Int_t TKSocket::BlockWrite(const char *buf, Int_t length, EEncoding type)
{
   // Block-send 'length' bytes to server from 'buf'.

   Desc_t desc;
   krb5_data in;
   krb5_data enc;
   Int_t rc;
   in.data = const_cast<char*>(buf);
   in.length = length;

   switch (type) {
   case kNone:
      enc.data = in.data;
      enc.length = in.length;
      break;
   case kSafe:
      rc = krb5_mk_safe(fgContext, fAuthContext, &in, &enc, 0);
      break;
   case kPriv:
      rc = krb5_mk_priv(fgContext, fAuthContext, &in, &enc, 0);
      break;
   default:
      Error("BlockWrite","unknown encoding type (%d)", type);
      return -1;
   }

   desc.fLength = htons(enc.length);
   desc.fType = htons(type);

   Int_t fd = fSocket->GetDescriptor();
   rc = krb5_net_write(fgContext, fd, (char *)&desc, sizeof(desc));
   if (rc <= 0) {
      Error("BlockWrite","writing descriptor (%d), %s",
            rc, error_message(rc));
      return -1;
   }

   rc = krb5_net_write(fgContext, fd, (char *)enc.data, enc.length);
   if (rc <= 0) {
      Error("BlockWrite","writing data (%d), %s",
            rc, error_message(rc));
      return -1;
   }

   if (type != kNone) free(enc.data);

   return rc;
}
