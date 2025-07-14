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
#include "TError.h"
#include "THostAuth.h"
#include "TRootAuth.h"
#include "TRootSecContext.h"
#include "TSocket.h"
#include "TSystem.h"
#include "TUrl.h"

////////////////////////////////////////////////////////////////////////////////
/// Runs authentication on socket s.
/// Invoked when dynamic loading is needed.
/// Returns 1 on success, 0 on failure.

TSecContext *TRootAuth::Authenticate(TSocket *s, const char *host,
                                     const char *user, Option_t */*opts*/)
{
   TSecContext *ctx = 0;

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

   // Build the protocol string for TAuthenticate
   TString proto = TUrl(s->GetUrl()).GetProtocol();
   if (proto == "") {
      proto = "root";
   } else if (proto.Contains("sockd") || proto.Contains("rootd")) {
      proto.ReplaceAll("d",1,"",0);
   }
   proto += Form(":%d",rproto);

   // Init authentication
   TAuthenticate *auth =
      new TAuthenticate(s, host, proto, user);

   // Attempt authentication
   if (!auth->Authenticate()) {
      // Close the socket if unsuccessful
      if (auth->HasTimedOut() > 0)
         Error("Authenticate",
               "timeout expired for %s@%s", auth->GetUser(), host);
      else
         Error("Authenticate",
               "authentication failed for %s@%s", auth->GetUser(), host);
   } else {
      // Search pointer to relevant TSecContext
      ctx = auth->GetSecContext();
      s->SetSecContext(ctx);
   }
   // Cleanup
   delete auth;

   // We are done
   return ctx;
}

////////////////////////////////////////////////////////////////////////////////
/// Return client version;

Int_t TRootAuth::ClientVersion()
{
   return TSocket::GetClientProtocol();
}

////////////////////////////////////////////////////////////////////////////////
/// Print error string corresponding to ecode, prepending location

void TRootAuth::ErrorMsg(const char *where, Int_t ecode)
{
   TAuthenticate::AuthError(where, ecode);
}
