// @(#)root/proof:$Name:  $:$Id: TProofMgr.cxx,v 1.11 2007/03/19 15:14:09 rdm Exp $
// Author: G. Ganis, Nov 2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofMgr                                                            //
//                                                                      //
// The PROOF manager interacts with the PROOF server coordinator to     //
// create or destroy a PROOF session, attach to or detach from          //
// existing one, and to monitor any client activity on the cluster.     //
// At most one manager instance per server is allowed.                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TError.h"
#include "TList.h"
#include "TProof.h"
#include "TProofMgr.h"
#include "TROOT.h"

ClassImp(TProofMgr)

// Sub-list of TROOT::fProofs with managers
TList TProofMgr::fgListOfManagers;
TProofMgr_t TProofMgr::fgTXProofMgrHook = 0;

//______________________________________________________________________________
TProofMgr::TProofMgr(const char *url, Int_t, const char *alias)
          : TNamed("",""), fRemoteProtocol(-1), fServType(kXProofd), fSessions(0)
{
   // Create a PROOF manager for the standard (old) environment.

   fServType = kProofd;

   // AVoid problems with empty URLs
   fUrl = (!url || strlen(url) <= 0) ? TUrl("proof://localhost") : TUrl(url);

   // Correct URL protocol
   if (!strcmp(fUrl.GetProtocol(), TUrl("a").GetProtocol()))
      fUrl.SetProtocol("proof");

   // Check and save the host FQDN ...
   if (strcmp(fUrl.GetHost(), fUrl.GetHostFQDN()))
      fUrl.SetHost(fUrl.GetHostFQDN());

   SetName(fUrl.GetUrl());
   if (alias)
      SetAlias(alias);
   else
      SetAlias(fUrl.GetHost());
}

//______________________________________________________________________________
TProofMgr::~TProofMgr()
{
   // Destroy a TProofMgr instance

   SafeDelete(fSessions);

   fgListOfManagers.Remove(this);
   gROOT->GetListOfProofs()->Remove(this);
}

//______________________________________________________________________________
TProof *TProofMgr::AttachSession(Int_t id, Bool_t)
{
   // Dummy version provided for completeness. Just returns a pointer to
   // existing session 'id' (as shown by TProof::QuerySessions) or 0 if 'id' is
   // not valid.

   TProofDesc *d = GetProofDesc(id);
   if (d) {
      if (d->GetProof())
         // Nothing to do if already in contact with proofserv
         return d->GetProof();
   }

   Info("AttachSession","invalid proofserv id (%d)", id);
   return 0;
}

//______________________________________________________________________________
void TProofMgr::DetachSession(Int_t id, Option_t *opt)
{
   // Detach session with 'id' from its proofserv. The 'id' is the number
   // shown by QuerySessions.

   TProofDesc *d = GetProofDesc(id);
   if (d) {
      if (d->GetProof())
         d->GetProof()->Detach(opt);
      fSessions->Remove(d);
      delete d;
   }

   return;
}

//______________________________________________________________________________
TList *TProofMgr::QuerySessions(Option_t *opt)
{
   // Get list of sessions accessible to this manager.

   if (opt && !strncasecmp(opt,"L",1))
      // Just return the existing list
      return fSessions;

   // Create list if not existing
   if (!fSessions) {
      fSessions = new TList();
      fSessions->SetOwner();
   }

   // Fill-in entries from the official list
   if (gROOT->GetListOfProofs()) {
      // Loop over
      TIter nxp(gROOT->GetListOfProofs());
      TProof *p = 0;
      Int_t ns = 0;
      while ((p = (TProof *)nxp())) {
         // Only those belonginh to this server
         if (MatchUrl(p->GetUrl())) {
            if (!(fSessions->FindObject(p->GetSessionTag()))) {
               Int_t st = (p->IsIdle()) ? TProofDesc::kIdle
                                        : TProofDesc::kRunning;
               TProofDesc *d =
                   new TProofDesc(p->GetName(), p->GetTitle(), p->GetUrl(),
                                         ++ns, p->GetSessionID(), st, p);
               fSessions->Add(d);
            }
         }
      }
   }

   // Drop entries not existing any longer
   if (fSessions->GetSize() > 0) {
      TIter nxd(fSessions);
      TProofDesc *d = 0;
      while ((d = (TProofDesc *)nxd())) {
         if (d->GetProof()) {
            if (!(gROOT->GetListOfProofs()->FindObject(d->GetProof()))) {
               fSessions->Remove(d);
               SafeDelete(d);
            } else {
               if (opt && !strncasecmp(opt,"S",1))
                  d->Print("");
            }
         }
      }
   }

   // We are done
   return fSessions;
}

//______________________________________________________________________________
Int_t TProofMgr::Reset(const char *)
{
   // Send a cleanup request for the sessions associated with the current
   // user.
   // Not supported.

   Warning("Reset","functionality not supported");

   return -1;
}

//______________________________________________________________________________
void TProofMgr::ShowWorkers()
{
   // Show available workers

   AbstractMethod("ShowWorkers");
}

//______________________________________________________________________________
TProofDesc *TProofMgr::GetProofDesc(Int_t id)
{
   // Get TProof instance corresponding to 'id'.

   if (id > 0) {
      TProofDesc *d = 0;
      // Retrieve an updated list
      QuerySessions("");
      if (fSessions) {
         TIter nxd(fSessions);
         while ((d = (TProofDesc *)nxd())) {
            if (d->MatchId(id))
               return d;
         }
      }
   }

   return 0;
}

//______________________________________________________________________________
void TProofMgr::ShutdownSession(TProof *p)
{
   // Discard PROOF session 'p'

   if (p) {
      TProofDesc *d = 0;
      if (fSessions) {
         TIter nxd(fSessions);
         while ((d = (TProofDesc *)nxd())) {
            if (p == d->GetProof()) {
               fSessions->Remove(d);
               delete d;
               break;
            }
         }
      }
   }
}

//______________________________________________________________________________
TProof *TProofMgr::CreateSession(const char *cfg,
                                 const char *cfgdir, Int_t loglevel)
{
   // Create a new remote session (master and associated workers).

   // Create
   if (IsProofd())
      fUrl.SetOptions("std");

   // Create the instance
   TProof *p = new TProof(fUrl.GetUrl(), cfg, cfgdir, loglevel, 0, this);

   if (p && p->IsValid()) {

      // Save record about this session
      Int_t ns = 1;
      if (fSessions) {
         // To avoid ambiguities in case of removal of some elements
         if (fSessions->Last())
            ns = ((TProofDesc *)(fSessions->Last()))->GetLocalId() + 1;
      } else {
         // Create the list
         fSessions = new TList;
      }

      // Create the description class
      Int_t st = (p->IsIdle()) ? TProofDesc::kIdle : TProofDesc::kRunning ;
      TProofDesc *d =
         new TProofDesc(p->GetName(), p->GetTitle(), p->GetUrl(),
                               ns, p->GetSessionID(), st, p);
      fSessions->Add(d);

   } else {
      // Session creation failed
      Error("CreateSession", "creating PROOF session");
      SafeDelete(p);
   }

   // We are done
   return p;
}

//______________________________________________________________________________
Bool_t TProofMgr::MatchUrl(const char *url)
{
   // Checks if 'url' refers to the same 'user@host:port' entity as the URL
   // in memory

   TUrl u(url);

   // Correct URL protocol
   if (!strcmp(u.GetProtocol(), TUrl("a").GetProtocol()))
      u.SetProtocol("proof");

   // Correct port
   if (u.GetPort() == TUrl("a").GetPort()) {
      Int_t port = gSystem->GetServiceByName("rootd");
      if (port < 0)
         port = 1094;
      u.SetPort(port);
   }

   // Now we can check
   if (!strcmp(u.GetHostFQDN(), fUrl.GetHostFQDN()))
      if (u.GetPort() == fUrl.GetPort())
         if (strlen(u.GetUser()) <= 0 || !strcmp(u.GetUser(),fUrl.GetUser()))
            return kTRUE;

   // Match failed
   return kFALSE;
}

//________________________________________________________________________
TList *TProofMgr::GetListOfManagers()
{
   // Extract pointers to PROOF managers from TROOT::fProofs.

   // Update the list with new entries
   if (gROOT->GetListOfProofs()) {
      TIter nxp(gROOT->GetListOfProofs());
      TProofMgr *p = 0;
      while ((p = dynamic_cast<TProofMgr *> (nxp())))
         if (!fgListOfManagers.FindObject(p))
            fgListOfManagers.Add(p);
   }

   // Get rid of invalid entries and notify
   if (fgListOfManagers.GetSize() > 0) {
      TIter nxp(&fgListOfManagers);
      TObject *o = 0;
      Int_t nm = 0;
      while ((o = nxp())) {
         if (!(gROOT->GetListOfProofs()->FindObject(o))) {
            fgListOfManagers.Remove(o);
         } else {
            TProofMgr *p = (TProofMgr *)o;
            if (gDebug > 0)
               Printf("// #%d: \"%s\" (%s)", ++nm, p->GetName(), p->GetTitle());
         }
      }
   } else {
      if (gDebug > 0)
         Printf("No managers found");
   }

   // We are done
   return &fgListOfManagers;
}

//______________________________________________________________________________
TProofMgr *TProofMgr::Create(const char *uin, Int_t loglevel,
                             const char *alias, Bool_t xpd)
{
   // Static method returning the appropriate TProofMgr object using
   // the plugin manager.
   TProofMgr *m= 0;

   // Resolve url; if empty input try to get the localhost FQDN
   TUrl u(uin);
   if (!uin || !strlen(uin))
      u.SetUrl("localhost");

   // in case user gave as url: "machine.dom.ain", replace
   // "http" by "proof" and "80" by "1093"
   if (!strcmp(u.GetProtocol(), TUrl("a").GetProtocol()))
      u.SetProtocol("proof");
   if (u.GetPort() == TUrl("a").GetPort())
      u.SetPort(1093);

   // Avoid multiple calls to GetUrl
   const char *url = u.GetUrl();

   // Make sure we do not have already a manager for this URL
   TList *lm = TProofMgr::GetListOfManagers();
   if (lm) {
      TIter nxm(lm);
      while ((m = (TProofMgr *)nxm()))
         if (m->MatchUrl(url))
            if (m->IsValid()) {
               return m;
            } else {
               fgListOfManagers.Remove(m);
               SafeDelete(m);
               break;
            }
   }

   m = 0;
   Bool_t trystd = kTRUE;

   // If required, we assume first that the remote server is based on XrdProofd
   if (xpd) {
      TProofMgr_t cm = TProofMgr::GetXProofMgrHook();
      if (cm) {
         m = (TProofMgr *) (*cm)(url, loglevel, alias);
         // Update trystd flag
         trystd = (m && !(m->IsValid()) && m->IsProofd()) ? kTRUE : kFALSE;
      }
   }

   // If the first attempt failed, we instantiate an old interface
   if (trystd) {
      SafeDelete(m);
      m = new TProofMgr(url, loglevel, alias);
   }

   // Record the new manager, if any
   if (m) {
      fgListOfManagers.Add(m);
      if (m->IsValid() && !(m->IsProofd())) {
         R__LOCKGUARD2(gROOTMutex);
         gROOT->GetListOfProofs()->Add(m);
         gROOT->GetListOfSockets()->Add(m);
      }
   }

   // We are done
   return m;
}

//________________________________________________________________________
TProofMgr_t TProofMgr::GetXProofMgrHook()
{
   // Get the constructor hook fro TXProofMgr.
   // We do this without the plugin manager because it blocks the
   // CINT mutex breaking the parallel startup.

   if (!fgTXProofMgrHook) {
      // Load the appropriate library ...
      TString prooflib = "libProofx";
      char *p = 0;
      if ((p = gSystem->DynamicPathName(prooflib, kTRUE))) {
         delete[] p;
         if (gSystem->Load(prooflib) == -1)
            ::Error("TProofMgr::GetXProofMgrCtor",
                    "can't load %s", prooflib.Data());
      } else
         ::Error("TProofMgr::GetXProofMgrCtor",
                 "can't locate %s", prooflib.Data());
   }

   // Done
   return fgTXProofMgrHook;
}

//_____________________________________________________________________________
void TProofMgr::SetTXProofMgrHook(TProofMgr_t pmh)
{
   // Set hook to TXProofMgr ctor

   fgTXProofMgrHook = pmh;
}


//
//  TProofDesc
//

ClassImp(TProofDesc)

//________________________________________________________________________
void TProofDesc::Print(Option_t *) const
{
   // Dump the content to the screen.
   const char *st[] = { "unknown", "idle", "processsing", "shutting down"};

   Printf("// # %d", fLocalId);
   Printf("// alias: %s, url: \"%s\"", GetTitle(), GetUrl());
   Printf("// tag: %s", GetName());
   Printf("// status: %s, attached: %s (remote ID: %d)",st[fStatus+1], (fProof ? "YES" : "NO"), fRemoteId);
}
