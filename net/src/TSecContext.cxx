// @(#)root/net:$Name:  $:$Id: TSecContext.cxx,v 1.3 2004/05/18 11:56:38 rdm Exp $
// Author: G. Ganis   19/03/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSecContext                                                          //
//                                                                      //
// Contains details about an established security context               //
// Used by THostAuth                                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifdef HAVE_CONFIG
#include "config.h"
#endif

#include <stdlib.h>

#include "TSecContext.h"
#include "TUrl.h"
#include "TROOT.h"
#include "TError.h"

ClassImp(TSecContext)
ClassImp(TSecContextCleanup)

//______________________________________________________________________________
TSecContext::TSecContext(const char *user, const char *host, Int_t meth,
                         Int_t offset, const char *details,
                         const char *token, TDatime expdate, void *ctx, Int_t key)
            : TObject()
{
   // Ctor for SecContext object.
   Assert(gROOT);

   fContext = ctx;
   fCleanup = new TList;
   fDetails = details;
   fExpDate = expdate;
   if (offset > -1) {
      if (fExpDate < TDatime()) {
         // This means expdate was not initialized
         // We set it to default, ie 1 day from now
         fExpDate.Set(TDatime().GetDate() + 1, TDatime().GetTime());
      }
   }
   fHost    = host;
   fMethod  = meth;
   fOffSet  = offset;
   fRSAKey  = key;
   fToken   = token;
   fUser    = user;

   // Keep official list updated with active TSecContexts
   if (fOffSet > -1)
      gROOT->GetListOfSecContexts()->Add(this);

}

//______________________________________________________________________________
TSecContext::TSecContext(const char *url, Int_t meth, Int_t offset,
                         const char *details, const char *token,
                         TDatime expdate, void *ctx, Int_t key)
            : TObject()
{
   // Ctor for SecContext object.
   // User and host from url = user@host .
   Assert(gROOT);

   fContext = ctx;
   fCleanup = new TList;
   fDetails = details;
   fExpDate = expdate;
   if (offset > -1) {
      if (fExpDate < TDatime()) {
         // This means expdate was not initialized
         // We set it to default, ie 1 day from now
         fExpDate.Set(TDatime().GetDate() + 1, TDatime().GetTime());
      }
   }
   fHost    = TUrl(url).GetHost();
   fMethod  = meth;
   fOffSet  = offset;
   fRSAKey  = key;
   fToken   = token;
   fUser    = TUrl(url).GetUser();

   // Keep official list updated with active TSecContexts
   if (fOffSet > -1)
      gROOT->GetListOfSecContexts()->Add(this);
}

//______________________________________________________________________________
TSecContext::~TSecContext()
{
   // Dtor: DeActivate (local/remote cleanup, list removal),
   // if still Active

   if (IsActive())
      DeActivate();

   // Delete the cleanup list
   if (fCleanup) {
      fCleanup->Delete();
      delete fCleanup;
   }
}

//______________________________________________________________________________
void TSecContext::AddForCleanup(Int_t port, Int_t proto, Int_t type)
{
   // Create a new TSecContextCleanup
   // Internally is added to the list

   TSecContextCleanup *tscc = new TSecContextCleanup(port, proto, type);
   fCleanup->Add(tscc);

}

//______________________________________________________________________________
void TSecContext::Cleanup()
{
   // Ask remote cleanup of this context

   TAuthenticate::CleanupSecContext(this,kFALSE);
}

//______________________________________________________________________________
void TSecContext::DeActivate(Option_t *Opt)
{
   // Set OffSet to -1 and expiring Date to default
   // Remove from the list
   // If globus, cleanup local stuff
   // If Opt contains "C" or "c", ask for remote cleanup
   // If Opt contains "R" or "r", remove from the list
   // Default Opt="CR"

   // Ask remote cleanup of this context
   Bool_t clean = (strstr(Opt,"C") || strstr(Opt,"c"));
   if (clean && fOffSet > -1)
      Cleanup();

   // Cleanup TPwdCtx object fro UsrPwd and SRP
   if (fMethod == TAuthenticate::kClear ||
       fMethod == TAuthenticate::kSRP)
      if (fContext) {
         delete (TPwdCtx *)fContext;
         fContext = 0;
      }

   // Cleanup globus security context if needed
   if (fMethod == TAuthenticate::kGlobus && fContext) {
      GlobusAuth_t GlobusAuthHook = TAuthenticate::GetGlobusAuthHook();
      if (GlobusAuthHook != 0) {
         TString det("context");
         TString us("-1");
         (*GlobusAuthHook)((TAuthenticate *)fContext,us,det);
         fContext = 0;
      }
   }

   Bool_t remove = (strstr(Opt,"R") || strstr(Opt,"r"));
   if (remove && fOffSet > -1){
      // Remove from the global list
      gROOT->GetListOfSecContexts()->Remove(this);
      // Remove also from local lists in THostAuth objects
      TAuthenticate::RemoveSecContext(this);
   }

   // Set inactive
   fOffSet  = -1;
   fExpDate = kROOTTZERO;

}

//______________________________________________________________________________
Bool_t TSecContext::IsA(const char *methname) const
{
   // Checks if this security context is for method named 'methname'
   // Valid names: UsrPwd, SRP, Krb5, Globus, SSH, UidGid
   // (Case sensitive)
   // (see TAuthenticate.cxx for updated list)

   TString ThisMethod(TAuthenticate::GetAuthMethod(fMethod));
   return (ThisMethod == methname);
}

//______________________________________________________________________________
Bool_t TSecContext::IsActive() const
{
   // Check remote OffSet and expiring Date

   if (fOffSet > -1 && fExpDate > TDatime())
      return kTRUE;
   // Invalid
   return kFALSE;
}

//______________________________________________________________________________
void TSecContext::Print(Option_t *opt) const
{
   // If opt is "F" (default) print object content.
   // If opt is "<number>" print in special form for calls within THostAuth
   // with cardinality <number>
   // If opt is "S" prints short in-line form for calls within TFTP,
   // TSlave, TProof ...

   char Ord[10] = {0};
   char Spc[10] = {0};

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

   // If asked to print ordinal number, preapre the string
   if (ord > -1) {
      sprintf(Ord,"%d)",ord);
      // and take care of alignment
      Int_t len=strlen(Ord);
      while (len--)
         strcat(Spc," ");
   }

   if (!strncasecmp(opt,"F",1)) {
      Info("Print",
           "+------------------------------------------------------+");
      Info("Print",
           "+ Host:%s Method:%d (%s) User:'%s'",
            GetHost(),fMethod,TAuthenticate::GetAuthMethod(fMethod),
            fUser.Data());
      Info("Print",
           "+         OffSet:%d Details: '%s'",
                      fOffSet,fDetails.Data());
      if (fOffSet > -1)
         Info("Print",
           "+         Expiration time: %s",fExpDate.AsString());
      Info("Print",
           "+------------------------------------------------------+");
   } else if (!strncasecmp(opt,"S",1)) {
      if (fOffSet > -1)
         Printf("Security context:     Method: %d (%s) expiring on %s",
                fMethod,TAuthenticate::GetAuthMethod(fMethod),fExpDate.AsString());
      else
         Printf("Security context:     Method: %d (%s) not reusable",
                fMethod,TAuthenticate::GetAuthMethod(fMethod));
   } else {
      // special printing form for THostAuth
      Info("PrintEstblshed","+ %s h:%s met:%d (%s) us:'%s'",
            Ord, GetHost(), fMethod, TAuthenticate::GetAuthMethod(fMethod),
            fUser.Data());
      Info("PrintEstblshed","+ %s offset:%d det: '%s'",
            Spc,fOffSet,fDetails.Data());
      if (fOffSet > -1)
         Info("PrintEstblshed","+ %s expiring: %s",Spc,fExpDate.AsString());
   }
}

//______________________________________________________________________________
const char *TSecContext::AsString() const
{
   // Returns short string with relevant information about this
   // security context

   static TString thestring(256);

   if (fOffSet > -1)
      thestring =
         Form("Method: %d (%s) expiring on %s",
              fMethod,TAuthenticate::GetAuthMethod(fMethod),fExpDate.AsString());
   else {
      if (fOffSet == -1)
         thestring =
            Form("Method: %d (%s) not reusable",
                 fMethod,TAuthenticate::GetAuthMethod(fMethod));
      else if (fOffSet == -3)
         thestring =
            Form("Method: %d (%s) authorized by /etc/hosts.equiv or $HOME/.rhosts",
                 fMethod,TAuthenticate::GetAuthMethod(fMethod));
      else if (fOffSet == -4)
         thestring =
            Form("No authentication required remotely");
   }

   return thestring;
}

