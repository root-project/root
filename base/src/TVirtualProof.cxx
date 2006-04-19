// @(#)root/base:$Name:  $:$Id: TVirtualProof.cxx,v 1.4 2006/03/03 15:42:37 rdm Exp $
// Author: Fons Rademakers   16/09/02

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualProof                                                        //
//                                                                      //
// Abstract interface to the Parallel ROOT Facility, PROOF.             //
// For more information on PROOF see the TProof class.                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TError.h"
#include "TList.h"
#include "TPluginManager.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TVirtualProof.h"

TVirtualProof *gProof = 0;

TProof_t TVirtualProof::fgProofHook = 0; // Hook to the TProof constructor

ClassImp(TVirtualProof)

//_____________________________________________________________________________
TVirtualProof *TVirtualProof::Open(const char *cluster, const char *conffile,
                                   const char *confdir, Int_t loglevel)
{
   // Start a PROOF session on a specific cluster. If cluster is 0
   // (the default) then the PROOF Session Viewer GUI pops up; 0 is returned.
   // If cluster is "" (empty string) then we connect to a PROOF session
   // on the localhost ("proof://localhost"). The appropriate instance of
   // TVirtualProofMgr is created, if not yet existing. The instantiated
   // TProof object is returned.
   // Use TProof::cd() to switch between PROOF sessions.
   // For more info on PROOF see the TProof ctor.

   const char *pn = "TVirtualProof::Open";

   // Make sure libProof and dependents are loaded and TProof can be created,
   // dependents are loaded via the information in the [system].rootmap file
   if (!cluster) {

      TPluginManager *pm = gROOT->GetPluginManager();
      if (!pm) {
         ::Error(pn, "plugin manager not found");
         return 0;
      }

      if (gROOT->IsBatch()) {
         ::Error(pn, "we are in batch mode, cannot show PROOF Session Viewer");
         return 0;
      }
      // start PROOF Session Viewer
      TPluginHandler *sv = pm->FindHandler("TSessionViewer", "");
      if (!sv) {
         ::Error(pn, "no plugin found for TSessionViewer");
         return 0;
      }
      if (sv->LoadPlugin() == -1) {
         ::Error(pn, "plugin for TSessionViewer could not be loaded");
         return 0;
      }
      sv->ExecPlugin(0);
      return 0;

   } else {

      TVirtualProof *proof = 0;

      // If the master was specified as "", try to get the localhost FQDN
      if (!strlen(cluster))
         cluster = gSystem->GetHostByName(gSystem->HostName()).GetHostName();

      TUrl u(cluster);
      // in case user gave as url: "machine.dom.ain", replace
      // "http" by "proof" and "80" by "1093"
      if (!strcmp(u.GetProtocol(), TUrl("a").GetProtocol()))
         u.SetProtocol("proof");
      if (u.GetPort() == TUrl("a").GetPort())
         u.SetPort(1093);

      // Find out if we are required to attach to a specific session
      TString o(u.GetOptions());
      Int_t locid = -1;
      Bool_t create = kFALSE;
      if (o.Length() > 0) {
         if (o.BeginsWith("N",TString::kIgnoreCase)) {
            create = kTRUE;
         } else if (o.IsDigit()) {
            locid = o.Atoi();
         }
         u.SetOptions("");
      }

      // Attach-to or create the appropriate manager
      TVirtualProofMgr *mgr = TVirtualProofMgr::Create(u.GetUrl());

      if (mgr && mgr->IsValid()) {

         // If XProofd we always attempt an attach first (unless
         // explicitely not requested).
         Bool_t attach = (create || mgr->IsProofd()) ? kFALSE : kTRUE;
         if (attach) {
            TVirtualProofDesc *d = 0;
            if (locid < 0)
               // Get the list of sessions
               d = (TVirtualProofDesc *) mgr->QuerySessions("")->First();
            else
               d = (TVirtualProofDesc *) mgr->GetProofDesc(locid);
            if (d) {
               proof = (TVirtualProof*) mgr->AttachSession(d->GetLocalId());
               if (!proof || !proof->IsValid()) {
                  if (locid)
                     ::Error(pn, "new session could not be attached");
                  SafeDelete(proof);
               }
            }
         }

         // start the PROOF session
         if (!proof) {
            proof = (TVirtualProof*) mgr->CreateSession(conffile, confdir, loglevel);
            if (!proof || !proof->IsValid()) {
               ::Error(pn, "new session could not be created");
               SafeDelete(proof);
            }
         }
      }
      return proof;
   }
}

//_____________________________________________________________________________
void TVirtualProof::SetTProofHook(TProof_t proofhook)
{
   // Set hook to TProof ctor.

   fgProofHook = proofhook;
}

//_____________________________________________________________________________
TProof_t TVirtualProof::GetTProofHook()
{
   // Get the TProof hook.

   return fgProofHook;
}
