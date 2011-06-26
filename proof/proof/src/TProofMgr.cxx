// @(#)root/proof:$Id$
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
#include "TEnv.h"
#include "TList.h"
#include "TProof.h"
#include "TProofMgr.h"
#include "TProofMgrLite.h"
#include "TROOT.h"

ClassImp(TProofMgr)

// Sub-list of TROOT::fProofs with managers
TList TProofMgr::fgListOfManagers;
TProofMgr_t TProofMgr::fgTXProofMgrHook = 0;

//______________________________________________________________________________
TProofMgr::TProofMgr(const char *url, Int_t, const char *alias)
          : TNamed("",""), fRemoteProtocol(-1), fServType(kXProofd),
            fSessions(0), fIntHandler(0)
{
   // Create a PROOF manager for the standard (old) environment.

   fServType = kProofd;

   // AVoid problems with empty URLs
   fUrl = (!url || strlen(url) <= 0) ? TUrl("proof://localhost") : TUrl(url);

   // Correct URL protocol
   if (!strcmp(fUrl.GetProtocol(), TUrl("a").GetProtocol()))
      fUrl.SetProtocol("proof");

   // Check port
   if (fUrl.GetPort() == TUrl("a").GetPort()) {
      // For the time being we use 'rootd' service as default.
      // This will be changed to 'proofd' as soon as XRD will be able to
      // accept on multiple ports
      Int_t port = gSystem->GetServiceByName("proofd");
      if (port < 0) {
         if (gDebug > 0)
            Info("TProofMgr","service 'proofd' not found by GetServiceByName"
                              ": using default IANA assigned tcp port 1093");
         port = 1093;
      } else {
         if (gDebug > 1)
            Info("TProofMgr","port from GetServiceByName: %d", port);
      }
      fUrl.SetPort(port);
   }

   // Make sure that the user is defined
   if (strlen(fUrl.GetUser()) <= 0) {
      // Fill in the default user
      UserGroup_t *pw = gSystem->GetUserInfo();
      if (pw) {
         fUrl.SetUser(pw->fUser);
         delete pw;
      }
   }

   // Check and save the host FQDN ...
   if (strcmp(fUrl.GetHost(), "__lite__")) {
      if (strcmp(fUrl.GetHost(), fUrl.GetHostFQDN()))
         fUrl.SetHost(fUrl.GetHostFQDN());
   }

   SetName(fUrl.GetUrl(kTRUE));
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
   SafeDelete(fIntHandler);

   fgListOfManagers.Remove(this);
   gROOT->GetListOfProofs()->Remove(this);
}

//______________________________________________________________________________
TProof *TProofMgr::AttachSession(Int_t id, Bool_t gui)
{
   // Dummy version provided for completeness. Just returns a pointer to
   // existing session 'id' (as shown by TProof::QuerySessions) or 0 if 'id' is
   // not valid. The boolena 'gui' should be kTRUE when invoked from the GUI.

   TProofDesc *d = GetProofDesc(id);
   if (d)
      return AttachSession(d, gui);

   Info("AttachSession","invalid proofserv id (%d)", id);
   return 0;
}

//______________________________________________________________________________
TProof *TProofMgr::AttachSession(TProofDesc *d, Bool_t)
{
   // Dummy version provided for completeness. Just returns a pointer to
   // existing session 'id' (as shown by TProof::QuerySessions) or 0 if 'id' is
   // not valid.

   if (!d) {
      Warning("AttachSession","invalid description object - do nothing");
      return 0;
   }

   if (d->GetProof())
      // Nothing to do if already in contact with proofserv
      return d->GetProof();

   Warning("AttachSession","session not available - do nothing");
   return 0;
}

//______________________________________________________________________________
void TProofMgr::DetachSession(Int_t id, Option_t *opt)
{
   // Detach session with 'id' from its proofserv. The 'id' is the number
   // shown by QuerySessions. The correspondent TProof object is deleted.
   // If id == 0 all the known sessions are detached.
   // Option opt="S" or "s" forces session shutdown.

   if (!IsValid()) {
      Warning("DetachSession","invalid TProofMgr - do nothing");
      return;
   }

   if (id > 0) {

      TProofDesc *d = GetProofDesc(id);
      if (d) {
         if (d->GetProof())
            d->GetProof()->Detach(opt);
         TProof *p = d->GetProof();
         fSessions->Remove(d);
         SafeDelete(p);
         delete d;
      }

   } else if (id == 0) {

      // Requesto to destroy all sessions
      if (fSessions) {
         // Delete PROOF sessions
         TIter nxd(fSessions);
         TProofDesc *d = 0;
         while ((d = (TProofDesc *)nxd())) {
            if (d->GetProof())
               d->GetProof()->Detach(opt);
            TProof *p = d->GetProof();
            fSessions->Remove(d);
            SafeDelete(p);
         }
         fSessions->Delete();
      }
   }

   return;
}

//______________________________________________________________________________
void TProofMgr::DetachSession(TProof *p, Option_t *opt)
{
   // Detach session 'p' from its proofserv. The instance 'p' is invalidated
   // and should be deleted by the caller

   if (!IsValid()) {
      Warning("DetachSession","invalid TProofMgr - do nothing");
      return;
   }

   if (p) {
      // Single session request
      TProofDesc *d = GetProofDesc(p);
      if (d) {
         if (d->GetProof())
            // The session is closed here
            d->GetProof()->Detach(opt);
         fSessions->Remove(d);
         delete d;
      }
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
      TObject *o = 0;
      TProof *p = 0;
      Int_t ns = 0;
      while ((o = nxp())) {
         if (o->InheritsFrom(TProof::Class())) {
            p = (TProof *)o;
            // Only those belonging to this server
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
Int_t TProofMgr::SendMsgToUsers(const char *, const char *)
{
   // Send a message to connected users. Only superusers can do this.
   // The first argument specifies the message or the file from where to take
   // the message.
   // The second argument specifies the user to which to send the message: if
   // empty or null the message is send to all the connected users.
   // return 0 in case of success, -1 in case of error

   Warning("SendMsgToUsers","functionality not supported");

   return -1;
}

//______________________________________________________________________________
Int_t TProofMgr::Reset(Bool_t, const char *)
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
   // Get TProofDesc instance corresponding to 'id'.

   TProofDesc *d = 0;
   if (id > 0) {
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

   return d;
}

//______________________________________________________________________________
TProofDesc *TProofMgr::GetProofDesc(TProof *p)
{
   // Get TProofDesc instance corresponding to TProof object 'p'.

   TProofDesc *d = 0;
   if (p) {
      // Retrieve an updated list
      QuerySessions("");
      if (fSessions) {
         TIter nxd(fSessions);
         while ((d = (TProofDesc *)nxd())) {
            if (p == d->GetProof())
               return d;
         }
      }
   }

   return d;
}

//______________________________________________________________________________
void TProofMgr::DiscardSession(TProof *p)
{
   // Discard TProofDesc of session 'p' from the internal list

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
      if (gDebug > 0) Error("CreateSession", "PROOF session creation failed");
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
      Int_t port = gSystem->GetServiceByName("proofd");
      if (port < 0)
         port = 1093;
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
      TObject *o = 0;
      while ((o = nxp())) {
         if (o->InheritsFrom(TProofMgr::Class()) && !fgListOfManagers.FindObject(o))
            fgListOfManagers.Add(o);
      }
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

   Bool_t isLite = kFALSE;

   // Resolve url; if empty the actions depend of the default
   TUrl u(uin);
   TString proto = u.GetProtocol();
   if (proto.IsNull()) {
      u.SetUrl(gEnv->GetValue("Proof.LocalDefault", "lite://"));
      proto = u.GetProtocol();
   }
   if (proto == "lite") {
#ifndef WIN32
      isLite = kTRUE;
      u.SetHost("__lite__");
      u.SetProtocol("proof");
      u.SetPort(1093);
#else
      ::Info("TProofMgr::Create","'lite' not yet supported on Windows");
      return m;
#endif
   }

   if (!isLite) {
      // in case user gave as url: "machine.dom.ain", replace
      // "http" by "proof" and "80" by "1093"
      if (!strcmp(u.GetProtocol(), TUrl("a").GetProtocol()))
         u.SetProtocol("proof");
      if (u.GetPort() == TUrl("a").GetPort())
         u.SetPort(1093);
   }

   // Avoid multiple calls to GetUrl
   const char *url = u.GetUrl();

   // Make sure we do not have already a manager for this URL
   TList *lm = TProofMgr::GetListOfManagers();
   if (lm) {
      TIter nxm(lm);
      while ((m = (TProofMgr *)nxm())) {
         if (m->IsValid()) {
            if (m->MatchUrl(url)) return m;
         } else {
            fgListOfManagers.Remove(m);
            SafeDelete(m);
            break;
         }
      }
   }

   if (isLite) {
      // Init the lite version
      return new TProofMgrLite(url, loglevel, alias);
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
