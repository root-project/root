// @(#)root/krb5auth:$Name:  $:$Id: Krb5Auth.cxx,v 1.14 2003/11/20 23:00:46 rdm Exp $
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
#include "TROOT.h"
#include "THostAuth.h"
#include "TError.h"
#include "TSystem.h"
#include "TEnv.h"
#include "rpderr.h"

Int_t Krb5Authenticate(TAuthenticate *, TString &, TString &, Int_t);

void  Krb5InitCred(char *ClientPrincipal);
Int_t Krb5CheckCred(krb5_context kCont, krb5_ccache Cc, krb5_principal Principal);

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
Int_t Krb5Authenticate(TAuthenticate *auth, TString &user, TString &det, Int_t version)
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

   char answer[100];
   int type;

   TString targetUser(user);

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
      com_err("<Krb5Authenticate>", retval, "while initializing krb5");
      return -1;
   }

   // ignore broken connection signal handling
   gSystem->IgnoreSignal(kSigPipe, kTRUE);
   cleanup.fSignal = true;

   // get our credentials cache
   krb5_ccache ccdef;
   if ((retval = krb5_cc_default(context, &ccdef))) {
      com_err("<Krb5Authenticate>", retval, "while getting default cache");
      return -1;
   }
   cleanup.ccdef = ccdef;

   // get our principal from the cache
   krb5_principal client;
   char *ClientPrincipal = StrDup(TAuthenticate::GetDefaultUser());

   if ((retval = krb5_cc_get_principal(context, ccdef, &client))) {

      if (isatty(0) && isatty(1)) {

         if (gDebug > 1)
            Info("Krb5Authenticate",
                 "valid credentials not found: try initializing (Principal: %s)",
                 ClientPrincipal);
         Krb5InitCred(ClientPrincipal);
         if ((retval = krb5_cc_get_principal(context, ccdef, &client))) {
            com_err("<Krb5Authenticate>", retval, "while getting client principal name");
            return -1;
         }
      } else {
         Warning("Krb5Authenticate",
                 "not a tty: cannot prompt for credentials, returning failure");
         return -1;
      }
   }
   cleanup.client = client;

   if (Krb5CheckCred(context,ccdef,client) != 1) {

      if (isatty(0) && isatty(1)) {

         if (gDebug >2)
            Info("Krb5Authenticate",
                 "credentials found have expired: try initializing (Principal: %s)",
                  ClientPrincipal);
         Krb5InitCred(ClientPrincipal);
         if ((retval = krb5_cc_get_principal(context, ccdef, &client))) {
            com_err("<Krb5Authenticate>", retval, "while getting client principal name");
            return -1;
         }
      } else {
         Warning("Krb5Authenticate",
                 "not a tty: cannot prompt for credentials, returning failure");
         return -1;
      }
   }
   if (ClientPrincipal) delete[] ClientPrincipal;
   cleanup.client = client;

   // Get a normal string for user
   char   User[64];
   strcpy(User,client->data->data);

   if (gDebug > 3) {
      char *Realm = new char[client->realm.length+1];
      strncpy(Realm,client->realm.data,client->realm.length);
      Realm[client->realm.length]= '\0';
      Info("Krb5Authenticate", "cc_get_principal: client: %s %s (%d %d)",
           client->data->data,Realm, client->data->length, client->realm.length);
      delete [] Realm;
   }

   Int_t ReUse= 1, Prompt= 0;
   TString Details;

   if (version > 1) {

      // Check ReUse
      ReUse  = TAuthenticate::GetAuthReUse();
      Prompt = TAuthenticate::GetPromptUser();

      // Build auth details
      Details = Form("pt:%d ru:%d us:%s@%s",
                     Prompt,ReUse,client->data->data,client->realm.data);

      // Create Options string
      char *Options= new char[strlen(User)+20];
      int Opt = ReUse * kAUTH_REUSE_MSK;
      sprintf(Options,"%d %ld %s", Opt, (Long_t)strlen(User), User);

      // Now we are ready to send a request to the rootd/proofd daemons
      // to check if we have already a valid security context and
      // eventually to start a negotiation to get one ...
      kind = kROOTD_KRB5;
      retval = ReUse;
      int rc = 0;
      if ((rc = TAuthenticate::AuthExists(auth,(Int_t)TAuthenticate::kKrb5,
                Details,Options,&kind,&retval)) == 1) {
         // A valid authentication exists: we are done ...
         if (Options) delete[] Options;
         return 1;
      }
      if (Options) delete[] Options;
      if (rc == -2) {
         return rc;
      }
      if (retval == kErrNotAllowed && kind == kROOTD_ERR) {
         return 0;
      }

   } else {

      sock->Send(kROOTD_KRB5);
      sock->Recv(retval, kind);

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
      com_err("<Krb5Authenticate>", KRB5_PROG_SUMTYPE_NOSUPP,
              "while using CRC-32");
      return 0;
   }

   // get service principal from service and host names --
   // hard coding of service names avoids having the have these
   // services in the local /etc/services file
   const char *service;
   if (sock->GetPort() == 1093)
     service = "proofd";
   else if (sock->GetPort() == 1094)
     service = "rootd";
   else {
      service = sock->GetService();

      // if port is not 1093 or 1094 AND the
      // service is not described in /etc/service
      // let's default to "host"

      int port = atoi(service);
      if (port != 0) service = "host";
   }

   const char *serv_host = sock->GetInetAddress().GetHostName();

   krb5_principal server;

   if (gDebug > 3)
      Info("Krb5Authenticate","serv_host: %s service: %s",serv_host,service);

   if ((retval = krb5_sname_to_principal(context, serv_host, service,
                                         KRB5_NT_SRV_HST, &server))) {
      com_err("<Krb5Authenticate>", retval, "while generating service "
              "principal %s/%s", serv_host, service);
      return 0;
   }
   cleanup.server = server;

   if (gDebug > 3) {
      char *Realm = new char[server->realm.length+1];
      strncpy(Realm,server->realm.data,server->realm.length);
      Realm[server->realm.length]= '\0';
      Info("Krb5Authenticate","sname_to_principal: server: %s %s (%d %d)",
           server->data->data, Realm, server->data->length, server->realm.length);
      delete [] Realm;
   }

   // authenticate
   krb5_auth_context auth_context = 0;
   int sockd = sock->GetDescriptor();
   char proto_version[100] = "krootd_v_1";
   krb5_data cksum_data;
   cksum_data.data = (char *)serv_host; // eeew yuck
   cksum_data.length = strlen(serv_host);
   krb5_error *err_ret;
   krb5_ap_rep_enc_part *rep_ret;

   retval = krb5_auth_con_init(context,&auth_context);
   if (retval)  Error("Krb5Authenticate","failed auth_con_init: %s\n",
                      error_message(retval));
   cleanup.auth_context = auth_context;

   retval = krb5_auth_con_setflags(context, auth_context,
                                   KRB5_AUTH_CONTEXT_RET_TIME);
   if (retval)  Error("Krb5Authenticate","failed auth_con_setflags: %s\n",
                      error_message(retval));

   if (gDebug > 1)
     Info("Krb5Authenticate",
          "Sending kerberos authentification to %s",
          serv_host);

   retval = krb5_sendauth(context, &auth_context, (krb5_pointer)&sockd,
                          proto_version, client, server,
                          AP_OPTS_MUTUAL_REQUIRED,
                          &cksum_data,
                          0, // not rolling our own creds, using ccache
                          ccdef, &err_ret, &rep_ret, 0); // ugh!

   // handle the reply (this is a verbatim copy from the kerberos
   // sample client source)
   if (retval && retval != KRB5_SENDAUTH_REJECTED) {
      com_err("<Krb5Authenticate>", retval, "while using sendauth");
      return 0;
   }
   if (retval == KRB5_SENDAUTH_REJECTED) {
      // got an error
      Error("Krb5Authenticate", "sendauth rejected, error reply "
            "is:\n\t\"%*s\"",
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
                                User,targetUser.Data());
      sock->Send(targetUser.Data(),kROOTD_KRB5);

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
              serv_host,outdata.data,outdata.length,
              outdata.data[0],outdata.data[1],outdata.data[2]);

      // Send length first
      char BufLen[20];
      sprintf(BufLen, "%d",outdata.length);
      sock->Send(BufLen,kROOTD_KRB5);

      // Send Key. second ...
      Int_t Nsen = sock->SendRaw(outdata.data, outdata.length);

      if (gDebug>3)
         Info("Krb5Authenticate",
              "For kerberos forward ticket sent %d bytes (expected %d)",
              Nsen,outdata.length);
   }

   // restore attention to broken connection signal handling
   gSystem->IgnoreSignal(kSigPipe, kFALSE);

   // returns user@realm
   type = kMESS_STRING;
   sock->Recv(answer, 100, type);

   if (version > 1) {

      // Receive key request
      int nrec=sock->Recv(retval, type);

      Int_t RSAKey = 0;
      if (ReUse == 1) {

         if (type != kROOTD_RSAKEY)
            Warning("Krb5Auth",
                    "problems recvn RSA key flag: got message %d, flag: %d",
                     type,RSAKey);
         RSAKey = 1;

         // Send the key securely
         TAuthenticate::SendRSAPublicKey(sock);

         // returns user + OffSet
         nrec = sock->Recv(retval, type);

      }

      if (type != kROOTD_KRB5 || retval < 1) {
         Warning("Krb5Auth",
                 "problems recvn (user,offset) length (%d:%d bytes:%d)",
                  type,retval,nrec);
         return 0;
      }
      char *rfrm = new char[retval+1];
      nrec = sock->Recv(rfrm,retval+1, type);  // receive user,offset) info

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

      // Create and save AuthDetails object
      TAuthenticate::SaveAuthDetails(auth,(Int_t)TAuthenticate::kKrb5,
                                     OffSet,ReUse,Details,lUser,RSAKey,Token);
      det = Details;
      if (Token) delete[] Token;
      if (lUser) delete[] lUser;
   } else {
      sock->Recv(answer, 100, type);  // returns user
      user = answer;
   }

   // Receive auth from remote login function
   sock->Recv(Auth, kind);

   if (Auth && kind == kROOTD_AUTH)
      return 1;
   return 0;
}

//______________________________________________________________________________
void Krb5InitCred(char *ClientPrincipal)
{
   // Checks if there are valid credentials in the cache.
   // If not, tries to initialise them.

   if (gDebug > 2)
       Info("Krb5InitCred","enter: %s", ClientPrincipal);

   // Get klist output ...
   char cmd[kMAXPATHLEN]= { 0 };

   // Prepare command
   char *Krb5Init = R__KRB5INIT;
   if (gDebug > 2) Info("Krb5InitCred","krb5init is %s",Krb5Init);

   if (Krb5Init==0 || strlen(Krb5Init)<=0) {
      if (Krb5Init!=0) delete[] Krb5Init;
      Krb5Init = "/usr/kerberos/bin/kinit";
  }
  sprintf(cmd, "%s -f %s",Krb5Init,ClientPrincipal);

  if (gDebug > 2) Info("Krb5InitCred","executing: %s",cmd);
  gSystem->Exec(cmd);
}

//______________________________________________________________________________
Int_t Krb5CheckCred(krb5_context kCont, krb5_ccache Cc, krb5_principal Principal)
{
   // Checks if there are valid credentials.

   Int_t retval;
   Int_t Now = time(0);
   Int_t Valid = -1;

   if (gDebug > 2)
      Info("Krb5CheckCred","enter: principal '%s@%s'",
            Principal->data->data,Principal->realm.data);

   krb5_cc_cursor Cur;
   if ((retval = krb5_cc_start_seq_get(kCont, Cc, &Cur))) {
      if (gDebug > 2) com_err("<Krb5CheckCred>", retval, "while starting seq get");
      return 0;
   }

   krb5_creds Creds;
   while (!(retval = krb5_cc_next_cred(kCont, Cc, &Cur, &Creds)) && Valid == -1) {

      if (gDebug > 3) {
         Info("Krb5CheckCred","Creds.server->length: %d",
               Creds.server->length);
         Info("Krb5CheckCred","Realms data: '%s' '%s'",
               Creds.server->realm.data,Principal->realm.data);
         Info("Krb5CheckCred","Srv data[0]: '%s' ",
               Creds.server->data[0].data);
         Info("Krb5CheckCred","Data data: '%s' '%s'",
               Creds.server->data[1].data,Principal->realm.data);
         Info("Krb5CheckCred","Endtime: %d ",Creds.times.endtime);
      }

      if (Creds.server->length == 2 &&
         strcmp(Creds.server->realm.data, Principal->realm.data) == 0 &&
         strcmp((char *)Creds.server->data[0].data, "krbtgt") == 0 &&
         strcmp((char *)Creds.server->data[1].data,Principal->realm.data) == 0) {
         // Check expiration time
         Valid = (Creds.times.endtime >= Now) ? 1 : 0;
      }
      krb5_free_cred_contents(kCont, &Creds);
   }
   return Valid;
}

