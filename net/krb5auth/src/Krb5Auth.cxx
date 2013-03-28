// @(#)root/krb5auth:$Id$
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
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <time.h>

#include "Krb5Auth.h"
#include "TSocket.h"
#include "TAuthenticate.h"
#include "TDatime.h"
#include "TROOT.h"
#include "THostAuth.h"
#include "TError.h"
#include "TSystem.h"
#include "TEnv.h"
#include "NetErrors.h"
#include "Getline.h"

#if defined(R__KRB5_NEED_VCST_DEFINE)
#define krb5_c_valid_cksumtype valid_cksumtype
#endif
#if defined(R__KRB5_NEED_VCST_PROTO)
krb5_boolean krb5_c_valid_cksumtype(krb5_cksumtype ctype);
#endif

Int_t Krb5Authenticate(TAuthenticate *, TString &, TString &, Int_t);

static Int_t Krb5InitCred(const char *clientPrincipal, Bool_t promptPrinc = kFALSE);
static Int_t Krb5CheckCred(krb5_context, krb5_ccache, TString, TDatime &);
static Int_t Krb5CheckSecCtx(const char *, TRootSecContext *);

class TKrb5AuthInit {
public:
   TKrb5AuthInit() { TAuthenticate::SetKrb5AuthHook(&Krb5Authenticate); }
};
static TKrb5AuthInit gKrb5authInit;

class TKrb5CleanUp {
public:
   Bool_t                fSignal;
   krb5_context          fContext;
   krb5_ccache           fCcdef;
   krb5_principal        fClient;
   krb5_principal        fServer;
   krb5_auth_context     fAuthContext;
   krb5_ap_rep_enc_part *fRepRet;
   char                 *fData;

   TKrb5CleanUp() : fSignal(false), fContext(0), fCcdef(0), fClient(0),
      fServer(0), fAuthContext(0), fRepRet(0), fData(0) {
   }

   ~TKrb5CleanUp() {
      if (fSignal) gSystem->IgnoreSignal(kSigPipe, kFALSE);

      if (fData) free(fData);
      if (fRepRet) krb5_free_ap_rep_enc_part(fContext, fRepRet);

      if (fAuthContext) krb5_auth_con_free(fContext, fAuthContext);

      if (fServer) krb5_free_principal(fContext, fServer);
      if (fClient) krb5_free_principal(fContext, fClient);


      if (fCcdef)   krb5_cc_close(fContext, fCcdef);
      if (fContext) krb5_free_context(fContext);
   }
};


//______________________________________________________________________________
Int_t Krb5Authenticate(TAuthenticate *auth, TString &user, TString &det,
                       Int_t version)
{
   // Kerberos v5 authentication code. Returns 0 in case authentication
   // failed, 1 in case of success and 2 in case remote does not support
   // Kerberos5.
   // Protocol 'version':  3    supports alternate username
   //                      2    supports negotiation and auth reuse
   //                      1    first kerberos implementation
   //                      0    no kerberos support (function should not be called)
   // user is used to input the target username and return the name in the
   // principal used


   TKrb5CleanUp cleanup;

   int retval;
   int kind;
   TSocket *sock = auth->GetSocket();

   char answer[256];
   int type;
   Int_t nsen = 0, nrec = 0;

   TString targetUser(user);
   TString localUser;
   // The default will be the one related to the logged user
   UserGroup_t *u = gSystem->GetUserInfo();
   if (u) {
      localUser = u->fUser;
      delete u;
   } else
      localUser = TAuthenticate::GetDefaultUser();
   Bool_t promptPrinc = (targetUser != localUser);

   // first check if protocol version supports kerberos, krb5 support
   // was introduced in rootd version 6
   // This checked in the calling rootines and the result is contained in
   // the argument 'version'
   if (version <= 0) return 2; // ... but you shouldn't have got here ...

   // get a context
   krb5_context context;
   retval = krb5_init_context(&context);
   cleanup.fContext = context;

   if (retval) {
      Error("Krb5Authenticate","failed <krb5_init_context>: %s\n",
            error_message(retval));
      return -1;
   }

   // ignore broken connection signal handling
   gSystem->IgnoreSignal(kSigPipe, kTRUE);
   cleanup.fSignal = kTRUE;

   // get our credentials cache
   krb5_ccache ccdef;
   if (gDebug > 2) {
      if (gSystem->Getenv("KRB5CCNAME"))
         Info("Krb5Authenticate",
              "Use credential file from $KRB5CCNAME: %s\n",
              gSystem->Getenv("KRB5CCNAME"));
      else
         Info("Krb5Authenticate",
              "Use default credential file ($KRB5CCNAME undefined)");
   }
   if ((retval = krb5_cc_default(context, &ccdef))) {
      Error("Krb5Authenticate","failed <krb5_cc_default>: %s\n",
            error_message(retval));
      return -1;
   }
   cleanup.fCcdef = ccdef;

   // get our principal from the cache
   krb5_principal client;
   TString principal = TString(TAuthenticate::GetKrb5Principal());
   Bool_t gotPrincipal = (principal.Length() > 0) ? kTRUE : kFALSE;
   //
   // if not defined or incomplete, complete with defaults;
   // but only if interactive
   if (!principal.Length() || !principal.Contains("@")) {
      if (gDebug > 3)
         Info("Krb5Authenticate",
              "incomplete principal: complete using defaults");
      krb5_principal default_princ;

      if (!principal.Length()) {
         // Try the default user
         if ((retval = krb5_parse_name(context, localUser.Data(),
                                       &default_princ))) {
            Error("Krb5Authenticate","failed <krb5_parse_name>: %s\n",
                                        error_message(retval));
         }
      } else {
         // Try first the name specified
         if ((retval = krb5_parse_name(context, principal.Data(),
                                       &default_princ))) {
            TString errmsg = TString(Form("First: %s",error_message(retval)));
            // Try the default user in case of failure
            if ((retval = krb5_parse_name(context, localUser.Data(),
                                           &default_princ))) {
               errmsg.Append(Form("- Second: %s",error_message(retval)));
               Error("Krb5Authenticate","failed <krb5_parse_name>: %s\n",
                                        errmsg.Data());
            }
         }
      }
      //
      // If successful, we get the string principal
      if (!retval) {
         char *default_name;
         if ((retval = krb5_unparse_name(context, default_princ, &default_name))) {
               Error("Krb5Authenticate","failed <krb5_unparse_name>: %s\n",
                     error_message(retval));
         } else {
            principal = TString(default_name);
            free(default_name);
         }
         krb5_free_principal(context, default_princ);
      }
   }
   // Notify
   if (gDebug > 3) {
      if (gotPrincipal)
         Info("Krb5Authenticate",
              "user requested principal: %s", principal.Data());
      else
         Info("Krb5Authenticate",
              "default principal: %s", principal.Data());
   }

   if ((retval = krb5_cc_get_principal(context, ccdef, &client))) {

      if (isatty(0) && isatty(1)) {

         if (gDebug > 1)
            Info("Krb5Authenticate",
                 "valid credentials not found: try initializing (Principal: %s)",
                 principal.Data());
         if (Krb5InitCred(principal, promptPrinc)) {
            Error("Krb5Authenticate","error executing kinit");
            return -1;
         }
         if ((retval = krb5_cc_get_principal(context, ccdef, &client))) {
            Error("Krb5Authenticate","failed <krb5_cc_get_principal>: %s\n",
                  error_message(retval));
            return -1;
         }
      } else {
         Warning("Krb5Authenticate",
                 "not a tty: cannot prompt for credentials, returning failure");
         return -1;
      }
   }

   // If a principal was specified by the user, we must check that the
   // principal for which we have a cached ticket is the one that we want
   TString ticketPrincipal =
      TString(Form("%.*s@%.*s",client->data->length, client->data->data,
                               client->realm.length, client->realm.data));
   if (gotPrincipal) {

      // If interactive require the ticket principal to be the same as the
      // required or default one
      if (isatty(0) && isatty(1) && principal != ticketPrincipal) {
         if (gDebug > 3)
            Info("Krb5Authenticate",
                 "got credentials for different principal %s - try"
                 " initialization credentials for principal: %s",
                 ticketPrincipal.Data(), principal.Data());
         if (Krb5InitCred(principal)) {
            Error("Krb5Authenticate","error executing kinit");
            return -1;
         }
         if ((retval = krb5_cc_get_principal(context, ccdef, &client))) {
            Error("Krb5Authenticate","failed <krb5_cc_get_principal>: %s\n",
                  error_message(retval));
            return -1;
         }
         // This may have changed
         ticketPrincipal =
             TString(Form("%.*s@%.*s",client->data->length, client->data->data,
                                      client->realm.length, client->realm.data));
      }
   }

   cleanup.fClient = client;

   TDatime expDate;
   if (Krb5CheckCred(context, ccdef, ticketPrincipal, expDate) != 1) {

      // If the ticket expired we tray to re-initialize it for the same
      // principal, which may be different from the default

      if (isatty(0) && isatty(1)) {

         if (gDebug >2)
            Info("Krb5Authenticate",
                 "credentials found have expired: try initializing"
                 " (Principal: %s)", ticketPrincipal.Data());
         if (Krb5InitCred(ticketPrincipal)) {
            Error("Krb5Authenticate","error executing kinit");
            return -1;
         }
         if ((retval = krb5_cc_get_principal(context, ccdef, &client))) {
            Error("Krb5Authenticate","failed <krb5_cc_get_principal>: %s\n",
                  error_message(retval));
            return -1;
         }
         // Check credentials and get expiration time
         if (Krb5CheckCred(context, ccdef, ticketPrincipal, expDate) != 1) {
            Info("Krb5Authenticate",
                 "ticket re-initialization failed for principal %s",
                 ticketPrincipal.Data());
            return -1;
         }
      } else {
         Warning("Krb5Authenticate",
                 "not a tty: cannot prompt for credentials, returning failure");
         return -1;
      }
   }
   cleanup.fClient = client;

   // At this point we know which is the principal we will be using
   if (gDebug > 3)
      Info("Krb5Authenticate",
           "using valid ticket for principal: %s", ticketPrincipal.Data());

   // Get a normal string for user
   TString normUser(client->data->data, client->data->length);

   if (gDebug > 3) {
      Info("Krb5Authenticate", "cc_get_principal: client: %.*s %.*s",
           client->data->length, client->data->data,
           client->realm.length, client->realm.data);
   }

   Int_t reuse = 1, prompt = 0;
   TString details;

   if (version > 1) {

      // Check ReUse
      reuse  = TAuthenticate::GetAuthReUse();
      prompt = TAuthenticate::GetPromptUser();

      // Build auth details
      details = Form("pt:%d ru:%d us:%s", prompt, reuse, ticketPrincipal.Data());

      // Create Options string
      int opt = reuse * kAUTH_REUSE_MSK +
                auth->GetRSAKeyType() * kAUTH_RSATY_MSK;
      TString options(Form("%d %d %s", opt, normUser.Length(), normUser.Data()));

      // Now we are ready to send a request to the rootd/proofd daemons
      // to check if we have already a valid security context and
      // eventually to start a negotiation to get one ...
      kind = kROOTD_KRB5;
      retval = reuse;
      int rc = 0;
      if ((rc = auth->AuthExists(ticketPrincipal, TAuthenticate::kKrb5,
                options, &kind, &retval, &Krb5CheckSecCtx)) == 1) {
         // A valid authentication exists: we are done ...
         return 1;
      }
      if (rc == -2) {
         return rc;
      }

      if (kind == kROOTD_ERR) {
         TString serv = "sockd";
         if (strstr(auth->GetProtocol(),"root"))
            serv = "rootd";
         if (strstr(auth->GetProtocol(),"proof"))
            serv = "proofd";
         if (retval == kErrConnectionRefused) {
            if (gDebug > 0)
               Error("Krb5Authenticate",
                  "%s@%s does not accept connections from %s%s",
                  serv.Data(), auth->GetRemoteHost(),
                  auth->GetUser(), gSystem->HostName());
            return -2;
         } else if (retval == kErrNotAllowed) {
            if (gDebug > 0)
               Error("Krb5Authenticate",
                  "%s@%s does not accept %s authentication from %s@%s",
                  serv.Data(), auth->GetRemoteHost(),
                  TAuthenticate::GetAuthMethod(2),
                  auth->GetUser(), gSystem->HostName());
         } else
            TAuthenticate::AuthError("Krb5Authenticate", retval);
         return 0;
      }

   } else {

      nsen = sock->Send(kROOTD_KRB5);
      if (nsen <= 0) {
         Error("Krb5Authenticate","Sending kROOTD_KRB5");
         return 0;
      }
      nrec = sock->Recv(retval, kind);
      if (nrec <= 0) {
         Error("Krb5Authenticate","Receiving kROOTD_KRB5");
         return 0;
      }

      // retval == 0 when no Krb5 support compiled in remote rootd
      if (retval == 0 || kind != kROOTD_KRB5)
         return 2;
   }

   // ok, krb5 is supported
   // ignore broken connection signal handling
   gSystem->IgnoreSignal(kSigPipe, kTRUE);
   cleanup.fSignal = kFALSE;

   // test for CRC-32
   if (!krb5_c_valid_cksumtype(CKSUMTYPE_CRC32)) {
      Error("Krb5Authenticate","failed <krb5_c_valid_cksumtype>: %s\n",
                  error_message(KRB5_PROG_SUMTYPE_NOSUPP));
      return 0;
   }

   // get service principal from service and host names --
   // hard coding of service names avoids having the have these
   // services in the local /etc/services file
   TString service = TString("host");

   TString serv_host(sock->GetInetAddress().GetHostName());
   krb5_principal server;

   if (gDebug > 3)
      Info("Krb5Authenticate","serv_host: %s service: %s",
           serv_host.Data(),service.Data());

   if ((retval = krb5_sname_to_principal(context, serv_host.Data(),
                 service.Data(), KRB5_NT_SRV_HST, &server))) {

      Error("Krb5Authenticate","failed <krb5_sname_to_principal>: %s\n",
             error_message(retval));
      return 0;
   }
   cleanup.fServer = server;

   if (gDebug > 3) {
      Info("Krb5Authenticate","sname_to_principal: server: %.*s %.*s",
           server->data->length, server->data->data,
           server->realm.length, server->realm.data);
   }

   // authenticate
   krb5_auth_context auth_context = 0;
   int sockd = sock->GetDescriptor();
   char proto_version[100] = "krootd_v_1";
   krb5_data cksum_data;
   cksum_data.data = (char *)serv_host.Data(); // eeew yuck
   cksum_data.length = serv_host.Length();
   krb5_error *err_ret;
   krb5_ap_rep_enc_part *rep_ret;

   retval = krb5_auth_con_init(context, &auth_context);
   if (retval)
      Error("Krb5Authenticate","failed auth_con_init: %s\n",
            error_message(retval));
   cleanup.fAuthContext = auth_context;

   retval = krb5_auth_con_setflags(context, auth_context,
                                   KRB5_AUTH_CONTEXT_RET_TIME);
   if (retval)  Error("Krb5Authenticate","failed auth_con_setflags: %s\n",
                      error_message(retval));

   if (gDebug > 1)
     Info("Krb5Authenticate",
          "Sending kerberos authentication to %s",
          serv_host.Data());

   retval = krb5_sendauth(context, &auth_context, (krb5_pointer)&sockd,
                          proto_version, client, server,
                          AP_OPTS_MUTUAL_REQUIRED,
                          &cksum_data,
                          0, // not rolling our own creds, using ccache
                          ccdef, &err_ret, &rep_ret, 0); // ugh!

   // handle the reply (this is a verbatim copy from the kerberos
   // sample client source)
   if (retval && retval != KRB5_SENDAUTH_REJECTED) {
      Error("Krb5Authenticate","failed <krb5_sendauth>: %s\n",
             error_message(retval));
      return 0;
   }
   if (retval == KRB5_SENDAUTH_REJECTED) {
      // got an error
      Error("Krb5Authenticate", "sendauth rejected, error reply "
            "is:\n\t\"%.*s\"",
            err_ret->text.length, err_ret->text.data);
      return 0;
   } else if (!rep_ret) {
      // no reply
      return 0;
   }
   cleanup.fRepRet = rep_ret;

   if (version > 2) {

      // Send the targetUser name

     if (gDebug > 0)
        Info("Krb5Authenticate","client is %s target is %s",
                                normUser.Data(),targetUser.Data());
      nsen = sock->Send(targetUser.Data(), kROOTD_KRB5);
      if (nsen <= 0) {
         Error("Krb5Authenticate","Sending <targetUser>");
         return 0;
      }

      // If PROOF, send credentials
      if (sock->GetServType() == TSocket::kPROOFD || version < 4) {

         krb5_data outdata;
         outdata.data = 0;

         retval = krb5_auth_con_genaddrs(context, auth_context,
                       sockd, KRB5_AUTH_CONTEXT_GENERATE_LOCAL_FULL_ADDR);

         if (retval) {
            Error("Krb5Authenticate","failed auth_con_genaddrs is: %s\n",
                  error_message(retval));
         }

         retval = krb5_fwd_tgt_creds(context, auth_context, 0 /*host*/,
                                     client, server, ccdef, true,
                                     &outdata);
         if (retval) {
            Error("Krb5Authenticate","fwd_tgt_creds failed: %s\n",
                  error_message(retval));
            return 0;
         }

         cleanup.fData = outdata.data;

         if (gDebug > 3)
            Info("Krb5Authenticate",
                 "Sending kerberos forward ticket to %s %p %d [%d,%d,%d,...]",
                 serv_host.Data(), outdata.data, outdata.length,
                 outdata.data[0], outdata.data[1], outdata.data[2]);

         // Send length first
         char buflen[20];
         snprintf(buflen, 20, "%d", outdata.length);
         nsen = sock->Send(buflen, kROOTD_KRB5);
         if (nsen <= 0) {
            Error("Krb5Authenticate","Sending <buflen>");
            return 0;
         }

         // Send Key. second ...
         nsen = sock->SendRaw(outdata.data, outdata.length);
         if (nsen <= 0) {
            Error("Krb5Authenticate","Sending <Key>");
            return 0;
         }

         if (gDebug>3)
            Info("Krb5Authenticate",
                 "For kerberos forward ticket sent %d bytes (expected %d)",
                 nsen, outdata.length);
      }
   }

   // restore attention to broken connection signal handling
   gSystem->IgnoreSignal(kSigPipe, kFALSE);

   // returns user@realm
   type = kMESS_STRING;
   nrec = sock->Recv(answer, 100, type);

   if (type == kROOTD_ERR) {
      TAuthenticate::AuthError("Krb5Authenticate", kErrNoHome);
      return 0;
   }

   if (nrec <= 0) {
      Error("Krb5Authenticate","Receiving <user@realm>");
      return 0;
   }
   if (gDebug > 3)
      Info("Krb5Auth","%s",answer);

   if (version > 1) {

      // Receive key request
      nrec = sock->Recv(retval, type);
      if (nrec <= 0) {
         Error("Krb5Authenticate","Receiving <key request>");
         return 0;
      }

      Int_t rsaKey = 0;
      if (reuse == 1) {

         if (type != kROOTD_RSAKEY  || retval < 1 || retval > 2 ) {
            Error("Krb5Auth",
                  "problems recvn RSA key flag: got message %d, flag: %d",
                  type, rsaKey);
            return 0;
         }
         rsaKey = retval - 1;

         // Send the key securely
         TAuthenticate::SendRSAPublicKey(sock, rsaKey);

         // get length of user + offset string
         nrec = sock->Recv(retval, type);
         if (nrec <= 0) {
            Error("Krb5Authenticate","Receiving <length of user+offset string>");
            return 0;
         }
      }

      if (type != kROOTD_KRB5 || retval < 1) {
         Warning("Krb5Auth",
                 "problems recvn (user,offset) length (%d:%d bytes:%d)",
                  type, retval, nrec);
         return 0;
      }
      char *rfrm = new char[retval+1];
      nrec = sock->Recv(rfrm,retval+1, type);  // receive user,offset) info
      if (nrec <= 0) {
         Error("Krb5Authenticate","Receiving <user+offset string>");
         delete[] rfrm;
         return 0;
      }

      // Parse answer
      char lUser[128];
      int offset = -1;
      sscanf(rfrm,"%127s %d", lUser, &offset);
      // Save username
      user = lUser;

      // Receive token
      char *token = 0;
      if (reuse == 1 && offset > -1) {
         if (TAuthenticate::SecureRecv(sock, 1, rsaKey, &token) == -1) {
            Warning("Krb5Auth",
                    "problems secure-receiving token - may result in corrupted token");
         }
         if (gDebug > 3)
            Info("Krb5Auth","received from server: token: '%s' ", token);
      } else {
         token = StrDup("");
      }

      // Create SecContext object
      TRootSecContext *ctx =
         auth->GetHostAuth()->CreateSecContext((const char *)lUser,
             auth->GetRemoteHost(), (Int_t)TAuthenticate::kKrb5, offset,
             details, token, expDate, 0, rsaKey);
      // Transmit it to TAuthenticate
      auth->SetSecContext(ctx);

      det = details;
      if (token) delete[] token;
   } else {
      nrec = sock->Recv(answer, 100, type);  // returns user
      if (nrec <= 0) {
         Error("Krb5Authenticate","Receiving <user string>");
         return 0;
      }
      user = answer;

      // Get a SecContext for the record and avoid problems
      // with fSecContext undefined in TAuthenticate
      TRootSecContext *ctx =
         auth->GetHostAuth()->CreateSecContext((const char *)user,
             auth->GetRemoteHost(), (Int_t)TAuthenticate::kKrb5, -1,
             details, 0);
      // Transmit it to TAuthenticate
      auth->SetSecContext(ctx);
   }

   // Receive auth from remote login function
   int authok = 0;
   nrec = sock->Recv(authok, kind);
   if (nrec <= 0) {
      Error("Krb5Authenticate", "Receiving <Auth flag>");
      return 0;
   }

   if (authok && kind == kROOTD_AUTH)
      return 1;
   return 0;
}

//______________________________________________________________________________
Int_t Krb5InitCred(const char *clientPrincipal, Bool_t promptPrinc)
{
   // Checks if there are valid credentials in the cache.
   // If not, tries to initialise them.

   if (gDebug > 2)
       Info("Krb5InitCred","enter: %s", clientPrincipal);

   // Check if the user wants to be prompt about principal
   TString principal = TString(clientPrincipal);
   if (TAuthenticate::GetPromptUser() || promptPrinc) {
      const char *usr = Getline(Form("Principal (%s): ", principal.Data()));
      if (usr[0]) {
         TString usrs(usr);
         usrs.Remove(usrs.Length() - 1); // get rid of \n
         if (!usrs.IsNull())
            principal = usrs;
      }
   }

   // Prepare command
   TString cmd;

   if (strlen(R__KRB5INIT) <= 0)
      cmd = Form("/usr/kerberos/bin/kinit -f %s", principal.Data());
   else
      cmd = Form("%s -f %s",R__KRB5INIT, principal.Data());

   if (gDebug > 2)
      Info("Krb5InitCred","executing: %s", cmd.Data());
   Int_t rc = gSystem->Exec(cmd);
   if (rc)
      if (gDebug > 0)
         Info("Krb5InitCred", "error: return code: %d", rc);
   return rc;
}

//______________________________________________________________________________
Int_t Krb5CheckCred(krb5_context kCont, krb5_ccache Cc,
                    TString principal, TDatime &expDate)
{
   // Checks if there are valid credentials.

   Int_t retval;
   Int_t now = time(0);
   Int_t valid = -1;

   TString pdata = principal;
   TString prealm = principal;
   pdata.Resize(pdata.Index("@"));
   prealm.Remove(0,prealm.Index("@")+1);
   if (gDebug > 2)
      Info("Krb5CheckCred","enter: principal '%s'", principal.Data());

   // Init to now
   expDate = TDatime();

   krb5_cc_cursor cur;
   if ((retval = krb5_cc_start_seq_get(kCont, Cc, &cur))) {
      if (gDebug > 2)
         Error("Krb5Authenticate","failed <krb5_cc_start_seq_get>: %s\n",
                error_message(retval));
      return 0;
   }

   krb5_creds creds;
   while (!(retval = krb5_cc_next_cred(kCont, Cc, &cur, &creds)) && valid == -1) {

      if (gDebug > 3) {
         Info("Krb5CheckCred","creds.server->length: %d",
               creds.server->length);
         Info("Krb5CheckCred","Realms data: '%.*s' '%s'",
               creds.server->realm.length, creds.server->realm.data,
               prealm.Data());
         Info("Krb5CheckCred","Srv data[0]: '%.*s' ",
               creds.server->data[0].length, creds.server->data[0].data);
         Info("Krb5CheckCred","Data data: '%.*s' '%s'",
               creds.server->data[1].length, creds.server->data[1].data,
               prealm.Data());
         Info("Krb5CheckCred","Endtime: %d ", creds.times.endtime);
      }

      if (creds.server->length == 2 &&
         !strncmp(creds.server->realm.data,
                  prealm.Data(),creds.server->realm.length) &&
         !strncmp((char *)creds.server->data[0].data,
                  "krbtgt",creds.server->data[0].length) &&
         !strncmp((char *)creds.server->data[1].data,
                  prealm.Data(),creds.server->data[1].length)) {
         // Check expiration time
         valid = (creds.times.endtime >= now) ? 1 : 0;
         // Return expiration time
         expDate.Set(creds.times.endtime);
      }
      krb5_free_cred_contents(kCont, &creds);
   }
   return valid;
}

//______________________________________________________________________________
Int_t Krb5CheckSecCtx(const char *principal, TRootSecContext *ctx)
{
   // Krb5 version of CheckSecCtx to be passed to TAuthenticate::AuthExists
   // Check if principal is matches the one used to instantiate Ctx
   // Returns: 1 if ok, 0 if not
   // Deactivates Ctx is not valid

   Int_t rc = 0;

   if (ctx->IsActive()) {
      if (strstr(ctx->GetID(), principal))
         rc = 1;
   }
   return rc;
}
