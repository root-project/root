// @(#)root/auth:$Id$
// Author: Gerardo Ganis   08/07/05

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRootAuth                                                            //
//                                                                      //
// TVirtualAuth implementation based on the old client authentication   //
// code.                                                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TAuthenticate.h"
#include "TEnv.h"
#include "TError.h"
#include "THostAuth.h"
#include "TRootAuth.h"
#include "TRootSecContext.h"
#include "TSocket.h"
#include "TSystem.h"
#include "TUrl.h"

//______________________________________________________________________________
TSecContext *TRootAuth::Authenticate(TSocket *s, const char *host,
                                     const char *user, Option_t *opts)
{
   // Runs authentication on socket s.
   // Invoked when dynamic loading is needed.
   // Returns 1 on success, 0 on failure.
   TSecContext *ctx = 0;
   Int_t rc = 0;

   Int_t rproto =  s->GetRemoteProtocol() % 1000;
   if (s->GetServType() == (Int_t)TSocket::kROOTD) {
      if (rproto > 6 && rproto < 10) {
         // Middle aged versions expect client protocol now
         s->Send(Form("%d", TSocket::GetClientProtocol()), kROOTD_PROTOCOL2);
         Int_t kind = 0;
         if (s->Recv(rproto, kind) < 0) {
            Error("Authenticate", "receiving remote protocol");
            return ctx;
         }
         s->SetRemoteProtocol(rproto);
      }
   }

   // Find out if we are a PROOF master
   Bool_t isPROOF = (s->GetServType() == (Int_t)TSocket::kPROOFD);
   Bool_t isMASTER = kFALSE;
   if (isPROOF) {
      // Master by default
      isMASTER = kTRUE;
      // Parse option
      TString opt(TUrl(s->GetUrl()).GetOptions());
      if (!strncasecmp(opt.Data()+1, "C", 1)) {
         isMASTER = kFALSE;
      }
   }

   // Find out whether we are a proof serv
   Bool_t isPROOFserv = (opts[0] == 'P') ? kTRUE : kFALSE;

   // Build the protocol string for TAuthenticate
   TString proto = TUrl(s->GetUrl()).GetProtocol();
   if (proto == "") {
      proto = "root";
   } else if (proto.Contains("sockd") || proto.Contains("rootd") ||
              proto.Contains("proofd")) {
      proto.ReplaceAll("d",1,"",0);
   }
   proto += Form(":%d",rproto);

   // Init authentication
   TAuthenticate *auth =
      new TAuthenticate(s, host, proto, user);

   // If PROOF client and trasmission of the SRP password is
   // requested make sure that ReUse is switched on to get and
   // send also the Public Key
   // Masters do this automatically upon reception of valid info
   // (see TSlave.cxx)
   if (isMASTER && !isPROOFserv) {
      if (gEnv->GetValue("Proofd.SendSRPPwd",0)) {
         Int_t kSRP = TAuthenticate::kSRP;
         TString detsSRP(auth->GetHostAuth()->GetDetails(kSRP));
         Int_t pos = detsSRP.Index("ru:0");
         if (pos > -1) {
            detsSRP.ReplaceAll("ru:0",4,"ru:1",4);
            auth->GetHostAuth()->SetDetails(kSRP,detsSRP);
         } else {
            TSubString ss = detsSRP.SubString("ru:no",TString::kIgnoreCase);
            if (!ss.IsNull()) {
               detsSRP.ReplaceAll(ss.Data(),5,"ru:1",4);
               auth->GetHostAuth()->SetDetails(kSRP,detsSRP);
            }
         }
      }
   }

   // No control on credential forwarding in case of SSH authentication;
   // switched it off on PROOF servers, unless the user knows what (s)he
   // is doing
   if (isPROOFserv) {
      if (!(gEnv->GetValue("ProofServ.UseSSH",0)))
         auth->GetHostAuth()->RemoveMethod(TAuthenticate::kSSH);
   }

   // Attempt authentication
   if (!auth->Authenticate()) {
      // Close the socket if unsuccessful
      if (auth->HasTimedOut() > 0)
         Error("Authenticate",
               "timeout expired for %s@%s", auth->GetUser(), host);
      else
         Error("Authenticate",
               "authentication failed for %s@%s", auth->GetUser(), host);
      // This is to terminate properly remote proofd in case of failure
      if (isPROOF)
         s->Send(Form("%d %s", gSystem->GetPid(), host), kROOTD_CLEANUP);
   } else {
      // Set return flag;
      rc = 1;
      // Search pointer to relevant TSecContext
      ctx = auth->GetSecContext();
      s->SetSecContext(ctx);
   }
   // Cleanup
   delete auth;

   // If we are talking to a recent proofd send over a buffer with the
   // remaining authentication related stuff
   if (rc && isPROOF && rproto > 11) {
      Bool_t client = !isPROOFserv;
      if (TAuthenticate::ProofAuthSetup(s, client) !=0 ) {
         Error("Authenticate", "PROOF: failed to finalize setup");
      }
   }

   // We are done
   return ctx;
}

//______________________________________________________________________________
Int_t TRootAuth::ClientVersion()
{
   // Return client version;

   return TSocket::GetClientProtocol();
}

//______________________________________________________________________________
void TRootAuth::ErrorMsg(const char *where, Int_t ecode)
{
   // Print error string corresponding to ecode, prepending location

   TAuthenticate::AuthError(where, ecode);
}
