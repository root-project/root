// @(#)root/krb5auth:$Name:  $:$Id: Krb5Auth.cxx,v 1.9 2003/10/07 14:03:02 rdm Exp $
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

TSocket *sock = 0;
THostAuth *HostAuth = 0;
Int_t  gRSAKey = 0;

//______________________________________________________________________________
Int_t Krb5Authenticate(TAuthenticate *auth, TString &user, TString &det, Int_t version)
{
   // Kerberos v5 authentication code. Returns 0 in case authentication
   // failed, 1 in case of success and 2 in case remote does not support
   // Kerberos5.
   // Protocol 'version':  2    supports negotiation and auth reuse
   //                      1    first kerberos implementation
   //                      0    no kerberos support (function should not be called)

   int retval;
   int Auth, kind;

   char answer[100];
   int type;

   // From the calling TAuthenticate
   sock = auth->GetSocket();
   HostAuth = auth->GetHostAuth();

   // first check if protocol version supports kerberos, krb5 support
   // was introduced in rootd version 6
   // This checked in the calling rootines and the result is contained in
   // the argument 'version'
   if (version <= 0) return 2; // ... but you shouldn't have got here ...

   // get a context
   krb5_context context;
   retval = krb5_init_context(&context);
   if (retval) {
      com_err("<Krb5Authenticate>", retval, "while initializing krb5");
      return -1;  // this is a memory leak, but it's small
   }

   // ignore broken connection signal handling
   gSystem->IgnoreSignal(kSigPipe, kTRUE);

   // get our credentials cache
   krb5_ccache ccdef;
   if ((retval = krb5_cc_default(context, &ccdef))) {
      com_err("<Krb5Authenticate>", retval, "while getting default cache");
      gSystem->IgnoreSignal(kSigPipe, kFALSE);
      return -1;
   }

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
            gSystem->IgnoreSignal(kSigPipe, kFALSE);
            return -1;
	 }
      } else {
         Warning("Krb5Authenticate",
                 "not a tty: cannot prompt for credentials, returning failure");
         gSystem->IgnoreSignal(kSigPipe, kFALSE);
         return -1;
      }
   }

   if (Krb5CheckCred(context,ccdef,client) != 1) {

      if (isatty(0) && isatty(1)) {

         if (gDebug >2)
            Info("Krb5Authenticate",
                 "credentials found have expired: try initializing (Principal: %s)",
                  ClientPrincipal);
         Krb5InitCred(ClientPrincipal);
         if ((retval = krb5_cc_get_principal(context, ccdef, &client))) {
            com_err("<Krb5Authenticate>", retval, "while getting client principal name");
            gSystem->IgnoreSignal(kSigPipe, kFALSE);
            return -1;
	 }
      } else {
         Warning("Krb5Authenticate",
                 "not a tty: cannot prompt for credentials, returning failure");
         gSystem->IgnoreSignal(kSigPipe, kFALSE);
         return -1;
      }
   }
   if (ClientPrincipal) delete[] ClientPrincipal;

   // Get a normal string for user
   char   User[64];
   strcpy(User,client->data->data);

   if (gDebug > 3) {
      char *Realm = StrDup(client->realm.data);
      int k = 0, len = strlen(client->realm.data);
      for (k = 0; k < len; k++ ) { if ((int)Realm[k] == 32) Realm[k] = '\0'; }
      Info("Krb5Authenticate", "cc_get_principal: client: %s %s (%d %d)",
           client->data->data,Realm, strlen(client->data->data), strlen(Realm));
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

      // Now we are ready to send a request to the rootd/proofd daemons to check if we have already
      // a valid security context and eventually to start a negotiation to get one ...
      kind = kROOTD_KRB5;
      retval = ReUse;
      int rc = 0;
      if ((rc = TAuthenticate::AuthExists(auth,(Int_t)TAuthenticate::kKrb5,Details,Options,&kind,&retval)) == 1) {
         // A valid authentication exists: we are done ...
         if (Options) delete[] Options;
         return 1;
      }
      if (rc == -2) {
         if (Options) delete[] Options;
         return rc;
      }

   } else {

      sock->Send(kROOTD_KRB5);
      sock->Recv(retval, kind);

      if (kind == kROOTD_ERR) {
         TAuthenticate::AuthError("Krb5Authenticate", retval);
         if (retval == kErrConnectionRefused) return -2;
         return 0;
      }
      // retval == 0 when no Krb5 support compiled in remote rootd
      if (retval == 0 || kind != kROOTD_KRB5)
         return 2;
   }

   // ok, krb5 is supported
   // ignore broken connection signal handling
   gSystem->IgnoreSignal(kSigPipe, kTRUE);

   // test for CRC-32
   if (!valid_cksumtype(CKSUMTYPE_CRC32)) {
      com_err("<Krb5Authenticate>", KRB5_PROG_SUMTYPE_NOSUPP,
              "while using CRC-32");
      gSystem->IgnoreSignal(kSigPipe, kFALSE);
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
   else
      service = sock->GetService();
   const char *serv_host = sock->GetInetAddress().GetHostName();
   krb5_principal server;

   if (gDebug > 3)
      Info("Krb5Authenticate","serv_host: %s service: %s",serv_host,service);

   if ((retval = krb5_sname_to_principal(context, serv_host, service,KRB5_NT_SRV_HST, &server))) {
      com_err("<Krb5Authenticate>", retval, "while generating service "
              "principal %s/%s", serv_host, service);
      gSystem->IgnoreSignal(kSigPipe, kFALSE);
      return 0;
   }

   if (gDebug > 3) {
      char *Realm = StrDup(server->realm.data);
      int k = 0, len = strlen(server->realm.data);
      for (k = 0; k < len; k++) { if ((int)Realm[k] == 32) Realm[k] = '\0'; }
      Info("Krb5Authenticate","sname_to_principal: server: %s %s (%d %d)",
           server->data->data, Realm, strlen(server->data->data), strlen(Realm));
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
   retval = krb5_sendauth(context, &auth_context, (krb5_pointer)&sockd,
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
      Error("Krb5Authenticate", "sendauth rejected, error reply "
            "is:\n\t\"%*s\"\n",
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

   // restore attention to broken connection signal handling
   gSystem->IgnoreSignal(kSigPipe, kFALSE);

   // returns user@realm
   type = kMESS_STRING;
   sock->Recv(answer, 100, type);

   if (version > 1) {

      // Receive key request
      int nrec=sock->Recv(retval, type);

      if (ReUse == 1) {

         if (type != kROOTD_RSAKEY)
            Warning("Krb5Auth", "problems recvn RSA key flag: got message %d, flag: %d",type,gRSAKey);
         gRSAKey = 1;

         // Send the key securely
         TAuthenticate::SendRSAPublicKey(sock);

         // returns user + OffSet
         nrec = sock->Recv(retval, type);
      }

      if (type != kROOTD_KRB5 || retval < 1)
         Warning("Krb5Auth", "problems recvn (user,offset) length (%d:%d bytes:%d)",type,retval,nrec);
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
         if (TAuthenticate::SecureRecv(sock,gRSAKey,&Token) == -1) {
            Warning("Krb5Auth","problems secure-receiving Token - may result in corrupted Token");
         }
         if (gDebug > 3)
            Info("Krb5Auth","received from server: token: '%s' ",Token);
      } else {
         Token = StrDup("");
      }

      // Create and save AuthDetails object
      TAuthenticate::SaveAuthDetails(auth,(Int_t)TAuthenticate::kKrb5,
                                     OffSet,ReUse,Details,lUser,gRSAKey,Token);
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
  sprintf(cmd, "%s %s",Krb5Init,ClientPrincipal);

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
