// @(#)root/krb5auth:$Name:$:$Id:$
// Author: Johannes Muelmenstaedt  17/03/2002

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/* Parts of this file are copied from the MIT krb5 distribution and
 * are subject to the following license:
 *
 * Copyright 1990,1991 by the Massachusetts Institute of Technology.
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

#include <errno.h>
#include <signal.h>
#include <string.h>

#include <krb5.h>
extern "C" int krb5_net_read(krb5_context, int, char *, int); // ow ow ow!

#include "TSocket.h"
#include "TAuthenticate.h"
#include "TError.h"
#include "TSystem.h"


Int_t Krb5Authenticate(TSocket *, TString &);

class Krb5AuthInit {
public:
   Krb5AuthInit() { TAuthenticate::SetKrb5AuthHook(&Krb5Authenticate); }
};
static Krb5AuthInit krb5auth_init;


//______________________________________________________________________________
Int_t Krb5Authenticate(TSocket *socket, TString &user)
{
   // Kerberos v5 authentication code. Returns 0 in case authentication
   // failed, 1 in case of success and 2 in case remote does not support
   // Kerberos5.

   int retval;
   int version, auth, kind;

   char answer[100];
   int type;

   // first check if protocol version supports kerberos, krb5 support
   // was introduced in rootd version 6
   if (socket->Send(kROOTD_PROTOCOL) == -1) {
      Error("Krb5Authenticate", "unable to read remote protocol version");
      return 2;
   }
   if (socket->Recv(version, kind) <= 0 || kind != kROOTD_PROTOCOL) {
      Error("Krb5Authenticate", "unable to read remote protocol version");
      return 2;
   }

   if (version < 6)
      return 2;

   // now we can send a kROOTD_KRB5 message to ask whether the server
   // supports krb5 auth

   socket->Send(kROOTD_KRB5);
   socket->Recv(retval, kind);

   if (retval == 0 || kind != kROOTD_KRB5)
      return 2;

   // ok, krb5 is supported

   // get a context
   krb5_context context;
   retval = krb5_init_context(&context);
   if (retval) {
      com_err("<Krb5Authenticate>", retval, "while initializing krb5");
      return 0;  // this is a memory leak, but it's small
   }

   // ignore broken connection signal handling
   gSystem->IgnoreSignal(kSigPipe, kTRUE);

   // test for CRC-32
   if (!valid_cksumtype(CKSUMTYPE_CRC32)) {
      com_err("<Krb5Authenticate>", KRB5_PROG_SUMTYPE_NOSUPP, "while using CRC-32");
      gSystem->IgnoreSignal(kSigPipe, kFALSE);
      return 0;
   }

   // get service principal from service and host names
   const char *service = socket->GetService();
   const char *serv_host = socket->GetInetAddress().GetHostName();
   krb5_principal server;
   if ((retval = krb5_sname_to_principal(context, serv_host, service,
                                         KRB5_NT_SRV_HST, &server))) {
      com_err("<Krb5Authenticate>", retval, "while generating service principal "
              "%s/%s", serv_host, service);
      gSystem->IgnoreSignal(kSigPipe, kFALSE);
      return 0;
   }

   // get our credentials cache
   krb5_ccache ccdef;
   if ((retval = krb5_cc_default(context, &ccdef))) {
      com_err("<Krb5Authenticate>", retval, "while getting default cache");
      gSystem->IgnoreSignal(kSigPipe, kFALSE);
      return 0;
   }

   // get our principal from the cache
   krb5_principal client;
   if ((retval = krb5_cc_get_principal(context, ccdef, &client))) {
      com_err("<Krb5Authenticate>", retval, "while getting client principal name");
      gSystem->IgnoreSignal(kSigPipe, kFALSE);
      return 0;
   }

   // authenticate
   krb5_auth_context auth_context = 0;
   int sock = socket->GetDescriptor();
   char proto_version[100] = "krootd_v_1";
   krb5_data cksum_data;
   cksum_data.data = (char *)serv_host; // eeew yuck
   cksum_data.length = strlen(serv_host);
   krb5_error *err_ret;
   krb5_ap_rep_enc_part *rep_ret;
   retval = krb5_sendauth(context, &auth_context, (krb5_pointer)&sock,
                          proto_version, client, server,
                          AP_OPTS_MUTUAL_REQUIRED,
                          &cksum_data,
                          0, // not rolling our own creds, using ccache
                          ccdef, &err_ret, &rep_ret, 0); // ugh!

   krb5_free_principal(context, server);
   krb5_free_principal(context, client);
   krb5_cc_close(context, ccdef);
   if (auth_context)
      krb5_auth_con_free(context, auth_context);

   // handle the reply (this is a verbatim copy from the kerberos
   // sample client source)
   if (retval && retval != KRB5_SENDAUTH_REJECTED) {
      com_err("<Krb5Authenticate>", retval, "while using sendauth");
      gSystem->IgnoreSignal(kSigPipe, kFALSE);
      return 0;
   }
   if (retval == KRB5_SENDAUTH_REJECTED) {
      // got an error
      Error("Krb5Authenticate", "sendauth rejected, error reply is:\n\t\"%*s\"\n",
            err_ret->text.length, err_ret->text.data);
      gSystem->IgnoreSignal(kSigPipe, kFALSE);
      return 0;
   } else if (!rep_ret) {
      // no reply
      gSystem->IgnoreSignal(kSigPipe, kFALSE);
      return 0;
   }

   // got a reply
   krb5_free_ap_rep_enc_part(context, rep_ret);
   krb5_free_context(context);

   gSystem->IgnoreSignal(kSigPipe, kFALSE);

   type = kMESS_STRING;
   socket->Recv(answer, 100, type);  // returns user@realm

   socket->Recv(answer, 100, type);  // returns user
   user = answer;

   socket->Recv(auth, kind);

   if (auth && kind == kROOTD_AUTH)
      return 1;
   return 0;
}
