// @(#)root/krb5auth:$Name:  $:$Id: Krb5Auth.cxx,v 1.21 2004/05/08 13:40:18 rdm Exp $
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

#include "config.h"

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
#include "TSecContext.h"
#include "TDatime.h"
#include "TROOT.h"
#include "THostAuth.h"
#include "TError.h"
#include "TSystem.h"
#include "TEnv.h"
#include "NetErrors.h"
#include "Getline.h"

Int_t Krb5Authenticate(TAuthenticate *, TString &, TString &, Int_t);

static void  Krb5InitCred(const char *ClientPrincipal, Bool_t PromptPrinc = kFALSE);
static Int_t Krb5CheckCred(krb5_context, krb5_ccache, TString, TDatime &);
static Int_t Krb5CheckSecCtx(const char *, TSecContext *);

class Krb5AuthInit {
public:
   Krb5AuthInit() { TAuthenticate::SetKrb5AuthHook(&Krb5Authenticate); }
};
static Krb5AuthInit krb5auth_init;

struct TKrb5CleanUp {

   bool                  fSignal;
   krb5_context          context;
   krb5_ccache           ccdef;
   krb5_principal        client;
   krb5_principal        server;
   krb5_auth_context     auth_context;
   krb5_ap_rep_enc_part *rep_ret;
   char*                 data;

   TKrb5CleanUp() : fSignal(false),context(0),ccdef(0),client(0),
      server(0),auth_context(0),rep_ret(0), data(0) {
   }

   ~TKrb5CleanUp() {
      if (fSignal) gSystem->IgnoreSignal(kSigPipe, kFALSE);

      if (data) free(data);
      if (rep_ret) krb5_free_ap_rep_enc_part(context, rep_ret);

      if (auth_context)  krb5_auth_con_free(context, auth_context);

      if (server) krb5_free_principal(context, server);
      if (client) krb5_free_principal(context, client);


      if (ccdef)   krb5_cc_close(context, ccdef);
      if (context) krb5_free_context(context);

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
   int Auth, kind;
   TSocket *sock = auth->GetSocket();

   char answer[256];
   int type;
   Int_t Nsen = 0, Nrec = 0;

   TString targetUser(user);
   TString localUser;
   // The default will be the one related to the logged user
   UserGroup_t *u = gSystem->GetUserInfo();
   if (u) {
      localUser = u->fUser;
      delete u;
   } else
      localUser = TAuthenticate::GetDefaultUser();
   Bool_t PromptPrinc = (targetUser != localUser);

   // first check if protocol version supports kerberos, krb5 support
   // was introduced in rootd version 6
   // This checked in the calling rootines and the result is contained in
   // the argument 'version'
   if (version <= 0) return 2; // ... but you shouldn't have got here ...

   // get a context
   krb5_context context;
   retval = krb5_init_context(&context);
   cleanup.context = context;

   if (retval) {
      Error("Krb5Authenticate","failed <krb5_init_context>: %s\n",
            error_message(retval));
      return -1;
   }

   // ignore broken connection signal handling
   gSystem->IgnoreSignal(kSigPipe, kTRUE);
   cleanup.fSignal = true;

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
   cleanup.ccdef = ccdef;

   // get our principal from the cache
   krb5_principal client;
   TString Principal = TString(TAuthenticate::GetKrb5Principal());
   Bool_t GotPrincipal = (Principal.Length() > 0) ? kTRUE : kFALSE;
   //
   // if not defined or incomplete, complete with defaults;
   // but only if interactive
   if (!Principal.Length() || !Principal.Contains("@")) {
      if (gDebug > 3)
         Info("Krb5Authenticate",
              "incomplete principal: complete using defaults");
      krb5_principal default_princ;

      if (!Principal.Length()) {
         // Try the default user
         if ((retval = krb5_parse_name(context, localUser.Data(),
                                       &default_princ))) {
            Error("Krb5Authenticate","failed <krb5_parse_name>: %s\n",
                                        error_message(retval));
         }
      } else {
         // Try first the name specified
         if ((retval = krb5_parse_name(context, Principal.Data(),
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
            Principal = TString(default_name);
            free(default_name);
         }
         krb5_free_principal(context,default_princ);
      }
   }
   // Notify
   if (gDebug > 3)
      if (GotPrincipal)
         Info("Krb5Authenticate",
              "user requested principal: %s", Principal.Data());
      else
         Info("Krb5Authenticate",
              "default principal: %s", Principal.Data());

   if ((retval = krb5_cc_get_principal(context, ccdef, &client))) {

      if (isatty(0) && isatty(1)) {

         if (gDebug > 1)
            Info("Krb5Authenticate",
                 "valid credentials not found: try initializing (Principal: %s)",
                 Principal.Data());
         Krb5InitCred(Principal,PromptPrinc);
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
   TString TicketPrincipal =
      TString(Form("%.*s@%.*s",client->data->length, client->data->data,
                               client->realm.length, client->realm.data));
   if (GotPrincipal) {

      // If interactive require the ticket principal to be the same as the
      // required or default one
      if (isatty(0) && isatty(1) && Principal != TicketPrincipal) {
         if (gDebug > 3)
            Info("Krb5Authenticate",
                 "got credentials for different principal %s - try"
                 " initialization credentials for principal: %s",
                 TicketPrincipal.Data(), Principal.Data());
         Krb5InitCred(Principal);
         if ((retval = krb5_cc_get_principal(context, ccdef, &client))) {
            Error("Krb5Authenticate","failed <krb5_cc_get_principal>: %s\n",
                  error_message(retval));
            return -1;
         }
         // This may have changed
         TString TicketPrincipal =
             TString(Form("%.*s@%.*s",client->data->length, client->data->data,
                                      client->realm.length, client->realm.data));
      }
   }

   cleanup.client = client;

   TDatime ExpDate;
   if (Krb5CheckCred(context,ccdef,TicketPrincipal,ExpDate) != 1) {

      // If the ticket expired we tray to re-initialize it for the same
      // principal, which may be different from the default

      if (isatty(0) && isatty(1)) {

         if (gDebug >2)
            Info("Krb5Authenticate",
                 "credentials found have expired: try initializing"
                 " (Principal: %s)", TicketPrincipal.Data());
         Krb5InitCred(TicketPrincipal);
         if ((retval = krb5_cc_get_principal(context, ccdef, &client))) {
            Error("Krb5Authenticate","failed <krb5_cc_get_principal>: %s\n",
                  error_message(retval));
            return -1;
         }
         // Check credentials and get expiration time
         if (Krb5CheckCred(context,ccdef,TicketPrincipal,ExpDate) != 1) {
            Info("Krb5Authenticate",
                 "ticket re-initialization failed for principal %s",
                 TicketPrincipal.Data());
            return -1;
         }
      } else {
         Warning("Krb5Authenticate",
                 "not a tty: cannot prompt for credentials, returning failure");
         return -1;
      }
   }
   cleanup.client = client;

   // At this point we know which is the principal we will be using
   if (gDebug > 3)
      Info("Krb5Authenticate",
           "using valid ticket for principal: %s", TicketPrincipal.Data());

   // Get a normal string for user
   TString User(client->data->data,client->data->length);

   if (gDebug > 3) {
      Info("Krb5Authenticate", "cc_get_principal: client: %.*s %.*s",
           client->data->length, client->data->data,
           client->realm.length, client->realm.data);
   }

   Int_t ReUse= 1, Prompt= 0;
   TString Details;

   if (version > 1) {

      // Check ReUse
      ReUse  = TAuthenticate::GetAuthReUse();
      Prompt = TAuthenticate::GetPromptUser();

      // Build auth details
      Details = Form("pt:%d ru:%d us:%s",Prompt,ReUse,TicketPrincipal.Data());

      // Create Options string
      int Opt = ReUse * kAUTH_REUSE_MSK;
      TString Options(Form("%d %ld %s", Opt, User.Length(), User.Data()));

      // Now we are ready to send a request to the rootd/proofd daemons
      // to check if we have already a valid security context and
      // eventually to start a negotiation to get one ...
      kind = kROOTD_KRB5;
      retval = ReUse;
      int rc = 0;
      if ((rc = auth->AuthExists(TicketPrincipal, TAuthenticate::kKrb5,
                Options, &kind, &retval, &Krb5CheckSecCtx)) == 1) {
         // A valid authentication exists: we are done ...
         return 1;
      }
      if (rc == -2) {
         return rc;
      }

      if (kind == kROOTD_ERR) {
         TString Server = "sockd";
         if (strstr(auth->GetProtocol(),"root"))
            Server = "rootd";
         if (strstr(auth->GetProtocol(),"proof"))
            Server = "proofd";
         if (retval == kErrConnectionRefused) {
            if (gDebug > 0)
               Error("Krb5Authenticate",
                  "%s@%s does not accept connections from %s%s",
                  Server.Data(),auth->GetRemoteHost(),
                  auth->GetUser(),gSystem->HostName());
            return -2;
         } else if (retval == kErrNotAllowed) {
            if (gDebug > 0)
               Error("Krb5Authenticate",
                  "%s@%s does not accept %s authentication from %s@%s",
                  Server.Data(),auth->GetRemoteHost(),
                  TAuthenticate::GetAuthMethod(2),
                  auth->GetUser(),gSystem->HostName());
         } else {
           if (gDebug > 0)
              TAuthenticate::AuthError("Krb5Authenticate", retval);
         }
         return 0;
      }

   } else {

      Nsen = sock->Send(kROOTD_KRB5);
      if (Nsen <= 0) {
         Error("Krb5Authenticate","Sending kROOTD_KRB5");
         return 0;
      }
      Nrec = sock->Recv(retval, kind);
      if (Nrec <= 0) {
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
   cleanup.fSignal = false;

   // test for CRC-32
   if (!valid_cksumtype(CKSUMTYPE_CRC32)) {
      Error("Krb5Authenticate","failed <valid_cksumtype>: %s\n",
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
   cleanup.server = server;

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

   retval = krb5_auth_con_init(context,&auth_context);
   if (retval)
      Error("Krb5Authenticate","failed auth_con_init: %s\n",
            error_message(retval));
   cleanup.auth_context = auth_context;

   retval = krb5_auth_con_setflags(context, auth_context,
                                   KRB5_AUTH_CONTEXT_RET_TIME);
   if (retval)  Error("Krb5Authenticate","failed auth_con_setflags: %s\n",
                      error_message(retval));

   if (gDebug > 1)
     Info("Krb5Authenticate",
          "Sending kerberos authentification to %s",
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
   cleanup.rep_ret = rep_ret;

   if (version > 2) {

      // Send the targetUser name

     if (gDebug > 0)
        Info("Krb5Authenticate","client is %s target is %s",
                                User.Data(),targetUser.Data());
      Nsen = sock->Send(targetUser.Data(),kROOTD_KRB5);
      if (Nsen <= 0) {
         Error("Krb5Authenticate","Sending <targetUser>");
         return 0;
      }

      // If PROOF, send credentials
      if (sock->GetServType() == TSocket::kPROOFD) {

         krb5_data outdata;
         outdata.data = 0;

         retval = krb5_auth_con_genaddrs(context, auth_context,
                       sockd, KRB5_AUTH_CONTEXT_GENERATE_LOCAL_FULL_ADDR);

         if (retval) {
            Error("Krb5Authenticate","failed auth_con_genaddrs is: %s\n",
                  error_message(retval));
         }

         retval = krb5_fwd_tgt_creds(context,auth_context, 0 /*host*/,
                                     client, server, ccdef, true,
                                     &outdata);
         if (retval) {
            Error("Krb5Authenticate","fwd_tgt_creds failed: %s\n",
                  error_message(retval));
            return 0;
         }

         cleanup.data = outdata.data;

         if (gDebug > 3)
            Info("Krb5Authenticate",
                 "Sending kerberos forward ticket to %s %p %d [%d,%d,%d,...]",
                 serv_host.Data(),outdata.data,outdata.length,
                 outdata.data[0],outdata.data[1],outdata.data[2]);

         // Send length first
         char BufLen[20];
         sprintf(BufLen, "%d",outdata.length);
         Nsen = sock->Send(BufLen,kROOTD_KRB5);
         if (Nsen <= 0) {
            Error("Krb5Authenticate","Sending <BufLen>");
            return 0;
         }

         // Send Key. second ...
         Nsen = sock->SendRaw(outdata.data, outdata.length);
         if (Nsen <= 0) {
            Error("Krb5Authenticate","Sending <Key>");
            return 0;
         }

         if (gDebug>3)
            Info("Krb5Authenticate",
                 "For kerberos forward ticket sent %d bytes (expected %d)",
                 Nsen,outdata.length);
      }
   }

   // restore attention to broken connection signal handling
   gSystem->IgnoreSignal(kSigPipe, kFALSE);

   // returns user@realm
   type = kMESS_STRING;
   Nrec = sock->Recv(answer, 100, type);

   if (type == kROOTD_ERR) {
      if (gDebug > 0)
         TAuthenticate::AuthError("Krb5Authenticate", kErrNoHome);
      return 0;
   }

   if (Nrec <= 0) {
      Error("Krb5Authenticate","Receiving <user@realm>");
      return 0;
   }
   if (gDebug > 3)
      Info("Krb5Auth","%s",answer);

   if (version > 1) {

      // Receive key request
      Nrec=sock->Recv(retval, type);
      if (Nrec <= 0) {
         Error("Krb5Authenticate","Receiving <key request>");
         return 0;
      }

      Int_t RSAKey = 0;
      if (ReUse == 1) {

         if (type != kROOTD_RSAKEY)
            Warning("Krb5Auth",
                    "problems recvn RSA key flag: got message %d, flag: %d",
                     type,RSAKey);
         RSAKey = 1;

         // Send the key securely
         TAuthenticate::SendRSAPublicKey(sock);

         // get length of user + OffSet string
         Nrec = sock->Recv(retval, type);
         if (Nrec <= 0) {
            Error("Krb5Authenticate","Receiving <length of user+offset string>");
            return 0;
         }
      }

      if (type != kROOTD_KRB5 || retval < 1) {
         Warning("Krb5Auth",
                 "problems recvn (user,offset) length (%d:%d bytes:%d)",
                  type,retval,Nrec);
         return 0;
      }
      char *rfrm = new char[retval+1];
      Nrec = sock->Recv(rfrm,retval+1, type);  // receive user,offset) info
      if (Nrec <= 0) {
         Error("Krb5Authenticate","Receiving <user+offset string>");
         delete[] rfrm;
         return 0;
      }

      // Parse answer
      char *lUser= new char[retval];
      int OffSet = -1;
      sscanf(rfrm,"%s %d",lUser,&OffSet);
      // Save username
      user = lUser;

      // Receive Token
      char *Token = 0;
      if (ReUse == 1 && OffSet > -1) {
         if (TAuthenticate::SecureRecv(sock,RSAKey,&Token) == -1) {
            Warning("Krb5Auth",
                    "problems secure-receiving Token - may result in corrupted Token");
         }
         if (gDebug > 3)
            Info("Krb5Auth","received from server: token: '%s' ",Token);
      } else {
         Token = StrDup("");
      }

      // Create SecContext object
      TSecContext *ctx =
         auth->GetHostAuth()->CreateSecContext((const char *)lUser,
             auth->GetRemoteHost(), (Int_t)TAuthenticate::kKrb5, OffSet,
             Details, (const char *)Token, ExpDate, 0, RSAKey);
      // Transmit it to TAuthenticate
      auth->SetSecContext(ctx);

      det = Details;
      if (Token) delete[] Token;
      if (lUser) delete[] lUser;
   } else {
      Nrec = sock->Recv(answer, 100, type);  // returns user
      if (Nrec <= 0) {
         Error("Krb5Authenticate","Receiving <user string>");
         return 0;
      }
      user = answer;
   }

   // Receive auth from remote login function
   Nrec = sock->Recv(Auth, kind);
   if (Nrec <= 0) {
      Error("Krb5Authenticate","Receiving <Auth flag>");
      return 0;
   }

   if (Auth && kind == kROOTD_AUTH)
      return 1;
   return 0;
}

//______________________________________________________________________________
void Krb5InitCred(const char *ClientPrincipal, Bool_t PromptPrinc)
{
   // Checks if there are valid credentials in the cache.
   // If not, tries to initialise them.

   if (gDebug > 2)
       Info("Krb5InitCred","enter: %s", ClientPrincipal);

   // Check if the user wants to be prompt about principal
   TString Principal = TString(ClientPrincipal);
   if (TAuthenticate::GetPromptUser() || PromptPrinc) {
      char *usr = Getline(Form("Principal (%s): ", Principal.Data()));
      if (usr[0]) {
         usr[strlen(usr) - 1] = 0; // get rid of \n
         if (strlen(usr))
            Principal = StrDup(usr);
      }
   }

   // Prepare command
   TString Cmd;

   if (strlen(R__KRB5INIT) <= 0)
      Cmd = Form("/usr/kerberos/bin/kinit -f %s",Principal.Data());
   else
      Cmd = Form("%s -f %s",R__KRB5INIT,Principal.Data());

   if (gDebug > 2)
      Info("Krb5InitCred","executing: %s",Cmd.Data());
   Int_t rc = gSystem->Exec(Cmd);
   if (rc)
      if (gDebug > 0)
         Info("Krb5InitCred", "error: return code: %d", rc);
}

//______________________________________________________________________________
Int_t Krb5CheckCred(krb5_context kCont, krb5_ccache Cc,
                    TString Principal, TDatime &ExpDate)
{
   // Checks if there are valid credentials.

   Int_t retval;
   Int_t Now = time(0);
   Int_t Valid = -1;

   TString PData = Principal;
   TString PRealm = Principal;
   PData.Resize(PData.Index("@"));
   PRealm.Remove(0,PRealm.Index("@")+1);
   if (gDebug > 2)
      Info("Krb5CheckCred","enter: principal '%s'",Principal.Data());

   // Init to now
   ExpDate = TDatime();

   krb5_cc_cursor Cur;
   if ((retval = krb5_cc_start_seq_get(kCont, Cc, &Cur))) {
      if (gDebug > 2)
         Error("Krb5Authenticate","failed <krb5_cc_start_seq_get>: %s\n",
                error_message(retval));
      return 0;
   }

   krb5_creds Creds;
   while (!(retval = krb5_cc_next_cred(kCont, Cc, &Cur, &Creds)) && Valid == -1) {

      if (gDebug > 3) {
         Info("Krb5CheckCred","Creds.server->length: %d",
               Creds.server->length);
         Info("Krb5CheckCred","Realms data: '%.*s' '%s'",
               Creds.server->realm.length, Creds.server->realm.data,
               PRealm.Data());
         Info("Krb5CheckCred","Srv data[0]: '%.*s' ",
               Creds.server->data[0].length, Creds.server->data[0].data);
         Info("Krb5CheckCred","Data data: '%.*s' '%s'",
               Creds.server->data[1].length, Creds.server->data[1].data,
               PRealm.Data());
         Info("Krb5CheckCred","Endtime: %d ",Creds.times.endtime);
      }

      if (Creds.server->length == 2 &&
         !strncmp(Creds.server->realm.data,
                  PRealm.Data(),Creds.server->realm.length) &&
         !strncmp((char *)Creds.server->data[0].data,
                  "krbtgt",Creds.server->data[0].length) &&
         !strncmp((char *)Creds.server->data[1].data,
                  PRealm.Data(),Creds.server->data[1].length)) {
         // Check expiration time
         Valid = (Creds.times.endtime >= Now) ? 1 : 0;
         // Return expiration time
         ExpDate.Set(Creds.times.endtime);
      }
      krb5_free_cred_contents(kCont, &Creds);
   }
   return Valid;
}

//______________________________________________________________________________
Int_t Krb5CheckSecCtx(const char *Principal, TSecContext *Ctx)
{
   // Krb5 version of CheckSecCtx to be passed to TAuthenticate::AuthExists
   // Check if Principal is matches the one used to instantiate Ctx
   // Returns: 1 if ok, 0 if not
   // Deactivates Ctx is not valid

   Int_t rc = 0;

   if (Ctx->IsActive()) {
      if (strstr(Ctx->GetDetails(),Principal))
         rc = 1;
   }
   return rc;
}
