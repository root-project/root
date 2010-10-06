// @(#)root/auth:$Id$
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
// THostAuth                                                            //
//                                                                      //
// Contains details about host-specific authentication methods and the  //
// result of their application.                                         //
// Used by TAuthenticate.                                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "RConfigure.h"
#include "TSystem.h"
#include "THostAuth.h"
#include "TRootSecContext.h"
#include "TAuthenticate.h"
#include "TSocket.h"
#include "TUrl.h"
#include <stdlib.h>


ClassImp(THostAuth)

//______________________________________________________________________________
   THostAuth::THostAuth() : TObject()
{
   // Deafult constructor.

   Create(0, 0);
}

//______________________________________________________________________________
THostAuth::THostAuth(const char *host, const char *user, Int_t nmeth,
                     Int_t *authmeth, char **details) : TObject()
{
   // Create hostauth object.
   // 'host' may contain also the server for whicb these directives
   // are valid in the form 'host:server' or 'server://host'
   // with server either "sock[d]", "root[d]", "proof[d]" or
   // 0, 1, 2, respectively.

   Create(host, user, nmeth, authmeth, details);
}

//______________________________________________________________________________
THostAuth::THostAuth(const char *host, Int_t server, const char *user,
                     Int_t nmeth, Int_t *authmeth, char **details) : TObject()
{
   // Create hostauth object.
   // 'host' may contain also the server for whicb these directives
   // are valid in the form 'host:server' or 'server://host'
   // with server either "sock[d]", "root[d]", "proof[d]" or
   // 0, 1, 2, respectively.

   Create(host, user, nmeth, authmeth, details);

   fServer = server;
}

//______________________________________________________________________________
THostAuth::THostAuth(const char *host, const char *user, Int_t authmeth,
                     const char *details) : TObject()
{
   // Create hostauth object with one method only.
   // 'host' may contain also the server for whicb these directives
   // are valid in the form 'host:server' or 'server://host'

   Create(host, user, 1, &authmeth, (char **)&details);
}

//______________________________________________________________________________
THostAuth::THostAuth(const char *host, Int_t server, const char *user,
                     Int_t authmeth, const char *details) : TObject()
{
   // Create hostauth object with one method only.
   // 'host' may contain also the server for whicb these directives
   // are valid in the form 'host:server' or 'server://host'

   Create(host, user, 1, &authmeth, (char **)&details);
   fServer = server;
}

//______________________________________________________________________________
void THostAuth::Create(const char *host, const char *user, Int_t nmeth,
                       Int_t *authmeth, char **details)
{
   // Create hostauth object.
   // 'host' may contain also the server for whicb these directives
   // are valid in the form 'host:server' or 'server://host'
   // with server either "sock[d]", "root[d]", "proof[d]" or
   // 0, 1, 2, respectively.

   int i;

   // Host
   fHost = host;

   fServer = -1;
   // Extract server, if given
   TString srv("");
   if (fHost.Contains(":")) {
      // .rootauthrc form: host:server
      srv = fHost;
      fHost.Remove(fHost.Index(":"));
      srv.Remove(0,srv.Index(":")+1);
   } else if (fHost.Contains("://")) {
      // Url form: server://host
      srv = TUrl(fHost).GetProtocol();
      fHost.Remove(0,fHost.Index("://")+3);
   }
   if (srv.Length()) {
      if (srv == "0" || srv.BeginsWith("sock"))
         fServer = TSocket::kSOCKD;
      else if (srv == "1" || srv.BeginsWith("root"))
         fServer = TSocket::kROOTD;
      else if (srv == "2" || srv.BeginsWith("proof"))
         fServer = TSocket::kPROOFD;
   }

   // Check and save the host FQDN ...
   if (fHost != "default" && !fHost.Contains("*")) {
      TInetAddress addr = gSystem->GetHostByName(fHost);
      if (addr.IsValid())
         fHost = addr.GetHostName();
   }

   // User
   fUser = user;
   if (fUser == "")
      fUser = gSystem->Getenv("USER");
   if (fUser == "") {
      UserGroup_t *u = gSystem->GetUserInfo();
      if (u)
         fUser = u->fUser;
      delete u;
   }

   // Methods indexes
   fNumMethods = nmeth;
   if (fNumMethods > 0) {
      if (!authmeth)
         fNumMethods = 0;
      for (i = 0; i < kMAXSEC; i++) {
         if (i < fNumMethods) {
            fMethods[i] = authmeth[i];
            fSuccess[i] = 0;
            fFailure[i] = 0;
         } else {
            fMethods[i] = -1;
            fSuccess[i] = -1;
            fFailure[i] = -1;
         }
      }
   }

   // Method details
   if (fNumMethods > 0) {
      for (i = 0; i < fNumMethods; i++) {
         if (details && details[i] && strlen(details[i]) > 0) {
            fDetails[i] = details[i];
         } else {
            // Use default instead
            char *tmp = TAuthenticate::GetDefaultDetails(fMethods[i],0,fUser);
            fDetails[i] = (const char *)tmp;
            delete[] tmp;
         }
      }
   }

   // List of TSecContext
   fSecContexts = new TList;

   // Active when created
   fActive = kTRUE;
}


//______________________________________________________________________________
THostAuth::THostAuth(const char *asstring) : TObject()
{
   // Create hostauth object from directives given as a compact string
   // See THostAuth::AsString().
   // Used in proof context only; fServer not set; to be set by hand
   // with SetServer() method if really needed

   fServer = -1;

   TString strtmp(asstring);
   char *tmp = new char[strlen(asstring)+1];
   strncpy(tmp,asstring,strlen(asstring));
   tmp[strlen(asstring)] = 0;

   fHost = TString((const char *)strtok(tmp," "));
   strtmp.ReplaceAll(fHost,"");
   fHost.Remove(0,fHost.Index(":")+1);

   fUser = TString((const char *)strtok(0," "));
   strtmp.ReplaceAll(fUser,"");
   fUser.Remove(0,fUser.Index(":")+1);

   TString fNmet;
   fNmet = TString((const char *)strtok(0," "));
   strtmp.ReplaceAll(fNmet,"");
   fNmet.Remove(0,fNmet.Index(":")+1);

   delete[] tmp;

   fNumMethods = atoi(fNmet.Data());
   Int_t i = 0;
   for (; i < fNumMethods; i++) {
      TString det = strtmp;
      det.Remove(0,det.Index("'")+1);
      det.Resize(det.Index("'"));
      // Remove leading spaces, if
      char cmet[20];
      sscanf(det.Data(),"%10s",cmet);
      Int_t met = atoi(cmet);
      if (met > -1 && met < kMAXSEC) {
         det.ReplaceAll(cmet,"");
         while (det.First(' ') == 0)
            det.Remove(0,1);
         while (det.Last(' ') == (det.Length() - 1))
            det.Resize(det.Length() - 1);
         fMethods[i] = met;
         fSuccess[i] = 0;
         fFailure[i] = 0;
         fDetails[i] = det;
      }
      strtmp.Remove(0,strtmp.Index("'",strtmp.Index("'")+1)+1);
   }
   for (i = fNumMethods; i < kMAXSEC ; i++) {
      fMethods[i] = -1;
      fSuccess[i] = -1;
      fFailure[i] = -1;
   }

   // List of TSecContext
   fSecContexts = new TList;

   // Active when created
   fActive = kTRUE;
}


//______________________________________________________________________________
THostAuth::THostAuth(THostAuth &ha) : TObject()
{
   // Copy ctor ...

   fHost = ha.fHost;
   fServer = ha.fServer;
   fUser = ha.fUser;
   fNumMethods  = ha.fNumMethods;
   Int_t i = 0;
   for (; i < kMAXSEC; i++) {
      fMethods[i] = ha.fMethods[i];
      fSuccess[i] = ha.fSuccess[i];
      fFailure[i] = ha.fFailure[i];
      fDetails[i] = ha.fDetails[i];
   }
   fSecContexts = ha.Established();
   fActive = ha.fActive;
}

//______________________________________________________________________________
void  THostAuth::AddMethod(Int_t meth, const char *details)
{
   // Add method to the list. If already there, change its
   // details to 'details'

   // Check 'meth'
   if (meth < 0 || meth >= kMAXSEC) return;

   // If already there, set details and return
   if (HasMethod(meth)) {
      SetDetails(meth,details);
      return;
   }

   // This is a new method
   fMethods[fNumMethods] = meth;
   fSuccess[fNumMethods] = 0;
   fFailure[fNumMethods] = 0;
   if (details && strlen(details) > 0) {
      fDetails[fNumMethods] = details;
   } else {
      // Use default instead
      char *tmp = TAuthenticate::GetDefaultDetails(meth,0,fUser);
      fDetails[fNumMethods] = (const char *)tmp;
      delete[] tmp;
   }

   // Increment total number
   fNumMethods++;

   if (gDebug > 3) Print();
}

//______________________________________________________________________________
void  THostAuth::RemoveMethod(Int_t meth)
{
   // Remove method 'meth' from the list, if there ...

   // If we don't have it, nothing to do
   Int_t pos = -1;
   if (!HasMethod(meth,&pos)) return;

   // Now rescale info
   Int_t i = 0, k = 0;
   for (; i < fNumMethods; i++) {
      if (i != pos) {
         fMethods[k] = fMethods[i];
         fSuccess[k] = fSuccess[i];
         fFailure[k] = fFailure[i];
         fDetails[k] = fDetails[i];
         k++;
      }
   }

   // Decrement total number
   fNumMethods--;

   // Free last position
   fMethods[fNumMethods] = -1;
   fSuccess[fNumMethods] = -1;
   fFailure[fNumMethods] = -1;
   fDetails[fNumMethods].Resize(0);

   if (gDebug > 3) Print();
}

//______________________________________________________________________________
void  THostAuth::Reset()
{
   // Remove all methods, leaving Active status and
   // list of associted TSceContexts unchanged

   // Free all filled positions
   Int_t i = 0;
   for (; i < fNumMethods; i++) {
      fMethods[i] = -1;
      fSuccess[i] = -1;
      fFailure[i] = -1;
      fDetails[i].Resize(0);
   }

   // Set total number to 0
   fNumMethods = 0;
}

//______________________________________________________________________________
THostAuth::~THostAuth()
{
   // The dtor.

   delete    fSecContexts;
}

//______________________________________________________________________________
const char *THostAuth::GetDetails(Int_t level)
{
   // Return authentication details for specified level
   // or "" if the specified level does not exist for this host.

   Int_t i = -1;
   if (HasMethod(level,&i)) {
      if (gDebug > 3)
         Info("GetDetails"," %d: returning fDetails[%d]: %s",
              level,i,fDetails[i].Data());
      return fDetails[i];
   }
   static const char *empty = " ";
   return empty;
}

//______________________________________________________________________________
Bool_t THostAuth::HasMethod(Int_t level, Int_t *pos)
{
   // Return kTRUE if method 'level' is in the list

   int i;
   for (i = 0; i < fNumMethods; i++) {
      if (fMethods[i] == level) {
         if (pos) *pos = i;
         return kTRUE;
      }
   }
   if (pos) *pos = -1;
   return kFALSE;
}

//______________________________________________________________________________
void THostAuth::SetDetails(Int_t level, const char *details)
{
   // Set authentication details for specified level.

   Int_t i = -1;
   if (HasMethod(level,&i)) {
      if (details && strlen(details) > 0) {
         fDetails[i] = details;
      } else {
         // Use default instead
         char *tmp = TAuthenticate::GetDefaultDetails(level,0,fUser);
         fDetails[i] = (const char *)tmp;
         delete[] tmp;
      }
   } else {
      // Add new method ...
      AddMethod(level, details);
   }
}

//______________________________________________________________________________
void THostAuth::Print(Option_t *proc) const
{
   // Print object content.

   char srvnam[5][8] = { "any", "sockd", "rootd", "proofd", "???" };

   Int_t isrv = (fServer >= -1 && fServer <= TSocket::kPROOFD) ?
      fServer+1 : TSocket::kPROOFD+2;

   Info("Print",
        "%s +------------------------------------------------------------------+",proc);
   Info("Print","%s + Host:%s - srv:%s - User:%s - # of available methods:%d",
        proc, fHost.Data(), srvnam[isrv], fUser.Data(), fNumMethods);
   int i = 0;
   for (i = 0; i < fNumMethods; i++){
      Info("Print","%s + Method: %d (%s) Ok:%d Ko:%d Dets:%s", proc,
           fMethods[i],TAuthenticate::GetAuthMethod(fMethods[i]),
           fSuccess[i], fFailure[i], fDetails[i].Data());
   }
   Info("Print",
        "%s +------------------------------------------------------------------+",proc);
}

//______________________________________________________________________________
void THostAuth::PrintEstablished() const
{
   // Print info about established authentication vis-a-vis of this Host.

   Info("PrintEstablished",
        "+------------------------------------------------------------------------------+");
   Info("PrintEstablished","+ Host:%s - Number of active sec contexts: %d",
        fHost.Data(), fSecContexts->GetSize());

   // Check list
   if (fSecContexts->GetSize()>0) {
      TIter next(fSecContexts);
      TSecContext *ctx = 0;
      Int_t k = 1;
      while ((ctx = (TSecContext *) next())) {
         TString opt;
         opt += k++;
         ctx->Print(opt);
      }
   }
   Info("PrintEstablished",
        "+------------------------------------------------------------------------------+");
}

//______________________________________________________________________________
void  THostAuth::ReOrder(Int_t nmet, Int_t *fmet)
{
   // Reorder nmet methods according fmet[nmet]

   // Temporary arrays
   Int_t   tMethods[kMAXSEC] = {0};
   Int_t   tSuccess[kMAXSEC] = {0};
   Int_t   tFailure[kMAXSEC] = {0};
   TString tDetails[kMAXSEC];
   Int_t   flag[kMAXSEC] = {0};

   // Copy info in the new order
   Int_t j = 0;
   for ( ; j < nmet; j++) {
      Int_t i = -1;
      if (HasMethod(fmet[j],&i)) {
         tMethods[j] = fMethods[i];
         tSuccess[j] = fSuccess[i];
         tFailure[j] = fFailure[i];
         tDetails[j] = fDetails[i];
         flag[i]++;
      } else if (fmet[j] >= 0 && fmet[j] < kMAXSEC) {
         tMethods[j] = fmet[j];
         tSuccess[j] = 0;
         tFailure[j] = 0;
         char *tmp = TAuthenticate::GetDefaultDetails(fmet[j],0,fUser);
         tDetails[j] = (const char *)tmp;
         delete[] tmp;
      } else {
         Warning("ReOrder","Method id out of range (%d) - skipping",fmet[j]);
      }
   }

   // Add existing methods not listed ... if any
   Int_t k = nmet, i = 0;
   for(; i < fNumMethods; i++){
      if (flag[i] == 0) {
         tMethods[k] = fMethods[i];
         tSuccess[k] = fSuccess[i];
         tFailure[k] = fFailure[i];
         tDetails[k] = fDetails[i];
         k++;
         flag[i] = 1;
      }
   }

   // Restore from temporary
   fNumMethods = k;
   for (i = 0; i < fNumMethods; i++) {
      fMethods[i] = tMethods[i];
      fSuccess[i] = tSuccess[i];
      fFailure[i] = tFailure[i];
      fDetails[i] = tDetails[i];
   }

   if (gDebug > 3) Print();
}

//______________________________________________________________________________
void  THostAuth::Update(THostAuth *ha)
{
   // Update info with the one in ha
   // Remaining methods, if any, get lower priority

   // Temporary arrays
   Int_t   tNumMethods = fNumMethods;
   Int_t   tMethods[kMAXSEC];
   Int_t   tSuccess[kMAXSEC];
   Int_t   tFailure[kMAXSEC];
   TString tDetails[kMAXSEC];

   // Save existing info in temporary arrays
   Int_t i = 0;
   for ( ; i < fNumMethods; i++) {
      tMethods[i] = fMethods[i];
      tSuccess[i] = fSuccess[i];
      tFailure[i] = fFailure[i];
      tDetails[i] = fDetails[i];
   }

   // Reset
   Reset();

   // Get ha content in
   for(i = 0; i < ha->NumMethods(); i++){
      fMethods[i] = ha->GetMethod(i);
      fSuccess[i] = ha->GetSuccess(i);
      fFailure[i] = ha->GetFailure(i);
      fDetails[i] = ha->GetDetailsByIdx(i);
   }

   // Set new tmp size
   fNumMethods = ha->NumMethods();

   // Add remaining methods with low priority
   if (fNumMethods < kMAXSEC) {
      for (i = 0; i < tNumMethods; i++) {
         if (!HasMethod(tMethods[i]) && fNumMethods < kMAXSEC) {
            fMethods[fNumMethods] = tMethods[i];
            fSuccess[fNumMethods] = tSuccess[i];
            fFailure[fNumMethods] = tFailure[i];
            fDetails[fNumMethods] = tDetails[i];
            fNumMethods++;
         }
      }
   }
   if (gDebug > 3) Print();
}

//______________________________________________________________________________
void  THostAuth::SetFirst(Int_t method)
{
   // Set 'method' to be the first used (if in the list ...).

   Int_t i = -1;
   if (HasMethod(method,&i)) {

      Int_t tMe = fMethods[i];
      Int_t tSu = fSuccess[i];
      Int_t tFa = fFailure[i];
      TString tDe = fDetails[i];

      // Rescale methods
      Int_t j = i;
      for (; j > 0; j--) {
         fMethods[j] = fMethods[j-1];
         fSuccess[j] = fSuccess[j-1];
         fFailure[j] = fFailure[j-1];
         fDetails[j] = fDetails[j-1];
      }

      // The saved method first
      fMethods[0] = tMe;
      fSuccess[0] = tSu;
      fFailure[0] = tFa;
      fDetails[0] = tDe;
   }

   if (gDebug > 3) Print();
}

//______________________________________________________________________________
void THostAuth::SetLast(Int_t method)
{
   // Set 'method' to be the last used (if in the list ...).

   Int_t i = -1;
   if (HasMethod(method,&i)) {

      Int_t tMe = fMethods[i];
      Int_t tSu = fSuccess[i];
      Int_t tFa = fFailure[i];
      TString tDe = fDetails[i];

      // Rescale methods
      Int_t j = i;
      for (; j < (fNumMethods - 1); j++) {
         fMethods[j] = fMethods[j+1];
         fSuccess[j] = fSuccess[j+1];
         fFailure[j] = fFailure[j+1];
         fDetails[j] = fDetails[j+1];
      }

      // The saved method first
      Int_t lp = fNumMethods - 1;
      fMethods[lp] = tMe;
      fSuccess[lp] = tSu;
      fFailure[lp] = tFa;
      fDetails[lp] = tDe;
   }

   if (gDebug > 3) Print();
}

//______________________________________________________________________________
void THostAuth::AddFirst(Int_t level, const char *details)
{
   // Add new method in first position
   // If already in the list, set as first method 'level' with
   // authentication 'details'.
   // Faster then AddMethod(method,details)+SetFirst(method).

   Int_t i = -1;
   if (HasMethod(level,&i)) {
      if (i > 0) {
         SetDetails(level, details);
         SetFirst(level);
      }
      if (gDebug > 3) Print();
      return;
   }

   // Rescale methods
   for (i = fNumMethods; i > 0; i--) {
      fMethods[i] = fMethods[i-1];
      fSuccess[i] = fSuccess[i-1];
      fFailure[i] = fFailure[i-1];
      fDetails[i] = fDetails[i-1];
   }

   // This method first
   fMethods[0] = level;
   fSuccess[0] = 0;
   fFailure[0] = 0;
   if (details && strlen(details) > 0) {
      fDetails[0] = details;
   } else {
      // Use default instead
      char *tmp = TAuthenticate::GetDefaultDetails(level,0,fUser);
      fDetails[0] = (const char *)tmp;
      delete[] tmp;
   }

   // Increment total number
   fNumMethods++;

   if (gDebug > 3) Print();
}


//______________________________________________________________________________
void THostAuth::CountSuccess(Int_t method)
{
   // Count successes for 'method'

   int i;
   for (i = 0; i < fNumMethods; i++) {
      if (fMethods[i] == method) {
         fSuccess[i]++;
         break;
      }
   }
}

//______________________________________________________________________________
void THostAuth::CountFailure(Int_t method)
{
   // Count failures for 'method'

   int i;
   for (i = 0; i < fNumMethods; i++) {
      if (fMethods[i] == method) {
         fFailure[i]++;
         break;
      }
   }
}

//______________________________________________________________________________
TRootSecContext *THostAuth::CreateSecContext(const char *user, const char *host,
                                             Int_t meth, Int_t offset,
                                             const char *details, const char *token,
                                             TDatime expdate, void *sctx, Int_t key)
{
   // Create a Security context and add it to local list
   // Return pointer to it to be stored in TAuthenticate

   TRootSecContext *ctx = new TRootSecContext(user, host, meth, offset, details,
                                              token, expdate, sctx, key);
   // Add it also to the local list if active
   if (ctx->IsActive())
      fSecContexts->Add(ctx);

   return ctx;

}

//______________________________________________________________________________
void THostAuth::AsString(TString &Out) const
{
   // Return a static string with all info in a serialized form

   Out = Form("h:%s u:%s n:%d",GetHost(),GetUser(),fNumMethods);

   Int_t i = 0;
   for (; i < fNumMethods; i++) {
      Out += TString(Form(" '%d %s'",fMethods[i],fDetails[i].Data()));
   }

}
