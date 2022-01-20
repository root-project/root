// @(#)root/auth:$Id$
// Author: G. Ganis   08/07/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRootSecContext                                                      //
//                                                                      //
// Special implementation of TSecContext                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "RConfigure.h"

#include <stdlib.h>

#include "TError.h"
#include "TRootSecContext.h"
#include "TROOT.h"
#include "TSocket.h"
#include "TUrl.h"
#include "TVirtualMutex.h"

ClassImp(TRootSecContext);

////////////////////////////////////////////////////////////////////////////////
/// Ctor for SecContext object.

   TRootSecContext::TRootSecContext(const char *user, const char *host, Int_t meth,
                                    Int_t offset, const char *id,
                                    const char *token, TDatime expdate,
                                    void *ctx, Int_t key)
      : TSecContext(user, host, meth, offset, id, token, expdate, ctx)
{
   R__ASSERT(gROOT);

   fRSAKey  = key;
   fMethodName = TAuthenticate::GetAuthMethod(fMethod);
}

////////////////////////////////////////////////////////////////////////////////
/// Ctor for SecContext object.
/// User and host from url = `user@host` .

TRootSecContext::TRootSecContext(const char *url, Int_t meth, Int_t offset,
                                 const char *id, const char *token,
                                 TDatime expdate, void *ctx, Int_t key)
   : TSecContext(url, meth, offset, id, token, expdate, ctx)
{
   R__ASSERT(gROOT);

   fRSAKey  = key;
   fMethodName = TAuthenticate::GetAuthMethod(fMethod);
}

////////////////////////////////////////////////////////////////////////////////
/// Dtor: delete (deActivate, local/remote cleanup, list removal)
/// all what is still active

TRootSecContext::~TRootSecContext()
{
   TSecContext::Cleanup();
}

////////////////////////////////////////////////////////////////////////////////
/// Set OffSet to -1 and expiring Date to default
/// Remove from the list
/// If Opt contains "C" or "c", ask for remote cleanup
/// If Opt contains "R" or "r", remove from the list
/// Default Opt="CR"

void TRootSecContext::DeActivate(Option_t *Opt)
{
   // Ask remote cleanup of this context
   Bool_t clean = (strstr(Opt,"C") || strstr(Opt,"c"));
   if (clean && fOffSet > -1)
      CleanupSecContext(kFALSE);

   // Cleanup TPwdCtx object fro UsrPwd
   if (fMethod == TAuthenticate::kClear)
      if (fContext) {
         delete (TPwdCtx *)fContext;
         fContext = 0;
      }

   Bool_t remove = (strstr(Opt,"R") || strstr(Opt,"r"));
   if (remove && fOffSet > -1){
      R__LOCKGUARD(gROOTMutex);
      // Remove from the global list
      gROOT->GetListOfSecContexts()->Remove(this);
      // Remove also from local lists in THostAuth objects
      TAuthenticate::RemoveSecContext(this);
   }

   // Set inactive
   fOffSet  = -1;
   fExpDate = kROOTTZERO;

}

////////////////////////////////////////////////////////////////////////////////
/// Ask remote client to cleanup security context 'ctx'
/// If 'all', all sec context with the same host as ctx
/// are cleaned.

Bool_t TRootSecContext::CleanupSecContext(Bool_t all)
{
   Bool_t cleaned = kFALSE;

   // Nothing to do if inactive ...
   if (!IsActive())
      return kTRUE;

   // Contact remote services that used this context,
   // starting from the last ...
   TIter last(fCleanup,kIterBackward);
   TSecContextCleanup *nscc = 0;
   while ((nscc = (TSecContextCleanup *)last()) && !cleaned) {

      // First check if remote daemon supports cleaning
      Int_t srvtyp = nscc->GetType();
      Int_t rproto = nscc->GetProtocol();
      Int_t level = 2;
      if ((srvtyp == TSocket::kROOTD && rproto < 10) ||
          (srvtyp == TSocket::kPROOFD && rproto < 9))
         level = 1;
      if ((srvtyp == TSocket::kROOTD && rproto < 8) ||
          (srvtyp == TSocket::kPROOFD && rproto < 7))
         level = 0;
      if (level) {
         Int_t port = nscc->GetPort();

         TSocket *news = new TSocket(fHost.Data(),port,-1);

         if (news && news->IsValid()) {
            if (srvtyp == TSocket::kPROOFD) {
               news->SetOption(kNoDelay, 1);
               news->Send("cleaning request");
            } else
               news->SetOption(kNoDelay, 0);

            // Backward compatibility: send socket size
            if (srvtyp == TSocket::kROOTD && level == 1)
               news->Send((Int_t)0, (Int_t)0);

            if (all || level == 1) {
               news->Send(Form("%d",TAuthenticate::fgProcessID), kROOTD_CLEANUP);
               cleaned = kTRUE;
            } else {
               news->Send(Form("%d %d %d %s", TAuthenticate::fgProcessID, fMethod,
                               fOffSet, fUser.Data()), kROOTD_CLEANUP);
               if (TAuthenticate::SecureSend(news, 1, fRSAKey,
                                             (char *)(fToken.Data())) == -1) {
                  Info("CleanupSecContext", "problems secure-sending token");
               } else {
                  cleaned = kTRUE;
               }
            }
            if (cleaned && gDebug > 2) {
               char srvname[3][10] = {"sockd", "rootd", "proofd"};
               Info("CleanupSecContext",
                    "remote %s notified for cleanup (%s,%d)",
                    srvname[srvtyp],fHost.Data(),port);
            }
         }
         SafeDelete(news);
      }
   }

   if (!cleaned)
      if (gDebug > 2)
         Info("CleanupSecContext",
              "unable to open valid socket for cleanup for %s", fHost.Data());

   return cleaned;
}

////////////////////////////////////////////////////////////////////////////////
/// If opt is "F" (default) print object content.
/// If opt is "<number>" print in special form for calls within THostAuth
/// with cardinality "<number>"
/// If opt is "S" prints short in-line form for calls within TFTP,
/// TSlave, TProof ...

void TRootSecContext::Print(Option_t *opt) const
{
   // Check if option is numeric
   Int_t ord = -1, i = 0;
   for (; i < (Int_t)strlen(opt); i++) {
      if (opt[i] < 48 || opt[i] > 57) {
         ord = -2;
         break;
      }
   }
   // If numeric get the cardinality and prepare the strings
   if (ord == -1)
      ord = atoi(opt);

   if (!strncasecmp(opt,"F",1)) {
      Info("Print",
           "+------------------------------------------------------+");
      Info("Print",
           "+ Host:%s Method:%d (%s) User:'%s'",
           GetHost(), fMethod, GetMethodName(),
           fUser.Data());
      Info("Print",
           "+         OffSet:%d Id: '%s'", fOffSet, fID.Data());
      if (fOffSet > -1)
         Info("Print",
              "+         Expiration time: %s",fExpDate.AsString());
      Info("Print",
           "+------------------------------------------------------+");
   } else if (!strncasecmp(opt,"S",1)) {
      if (fOffSet > -1) {
         if (fID.BeginsWith("AFS"))
            Printf("Security context:     Method: AFS, not reusable");
         else
            Printf("Security context:     Method: %d (%s) expiring on %s",
                   fMethod, GetMethodName(),
                   fExpDate.AsString());
      } else {
         Printf("Security context:     Method: %d (%s) not reusable",
                fMethod, GetMethodName());
      }
   } else {
      // special printing form for THostAuth
      Info("PrintEstblshed","+ %d \t h:%s met:%d (%s) us:'%s'",
                               ord, GetHost(), fMethod, GetMethodName(),
                               fUser.Data());
      Info("PrintEstblshed","+ \t offset:%d id: '%s'", fOffSet, fID.Data());
      if (fOffSet > -1)
         Info("PrintEstblshed","+ \t expiring: %s",fExpDate.AsString());
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Returns short string with relevant information about this
/// security context

const char *TRootSecContext::AsString(TString &out)
{
   if (fOffSet > -1) {
      if (fID.BeginsWith("AFS"))
         out = Form("Method: AFS, not reusable");
      else {
         char expdate[32];
         out = Form("Method: %d (%s) expiring on %s",
                    fMethod, GetMethodName(), fExpDate.AsString(expdate));
      }
   } else {
      if (fOffSet == -1)
         out = Form("Method: %d (%s) not reusable", fMethod, GetMethodName());
      else if (fOffSet == -3)
         out = Form("Method: %d (%s) authorized by /etc/hosts.equiv or $HOME/.rhosts",
                    fMethod, GetMethodName());
      else if (fOffSet == -4)
         out = Form("No authentication required remotely");
   }
   return out.Data();
}
