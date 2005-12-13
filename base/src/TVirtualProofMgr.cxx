// @(#)root/base:$Name:  $:$Id: TVirtualProofMgr.cxx,v 1.2 2005/12/12 17:59:17 rdm Exp $
// Author: G. Ganis, Nov 2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualProofMgr                                                     //
//                                                                      //
// Abstract interface to the manager for PROOF sessions.                //
// The PROOF manager interacts with the PROOF server coordinator to     //
// create or destroy a PROOF session, attach to or detach from          //
// existing one, and to monitor any client activity on the cluster.     //
// At most one manager instance per server is allowed.                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TInetAddress.h"
#include "TList.h"
#include "TPluginManager.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TVirtualMutex.h"
#include "TVirtualProof.h"
#include "TVirtualProofMgr.h"

ClassImp(TVirtualProofMgr)


// Sub-list of TROOT::fProofs with managers
TList TVirtualProofMgr::fgListOfManagers;

//______________________________________________________________________________
TVirtualProofMgr::~TVirtualProofMgr()
{
   // Destroy a TVirtualProofMgr instance

   SafeDelete(fSessions);

   fgListOfManagers.Remove(this);
   gROOT->GetListOfProofs()->Remove(this);
}

//______________________________________________________________________________
TVirtualProofDesc *TVirtualProofMgr::GetProofDesc(Int_t id)
{
   // Get TVirtualProof instance corresponding to 'id'.

   if (id > 0) {
      TVirtualProofDesc *d = 0;
      // Retrieve an updated list
      QuerySessions("");
      if (fSessions) {
         TIter nxd(fSessions);
         while ((d = (TVirtualProofDesc *)nxd())) {
            if (d->MatchId(id))
               return d;
         }
      }
   }

   return 0;
}

//______________________________________________________________________________
TVirtualProof *TVirtualProofMgr::CreateSession(const char *cfg,
                                               const char *cfgdir, Int_t loglevel)
{
   // Create a new remote session (master and associated workers).

   // Create
   if (IsProofd())
      fUrl.SetOptions("std");

   TPluginManager *pm = gROOT->GetPluginManager();
   if (!pm) {
      Error("CreateSession", "plugin manager not found");
      return 0;
   }

   // Load regular TProof for client
   TPluginHandler *h = pm->FindHandler("TVirtualProof", "");
   if (!h) {
      Error("CreateSession", "no plugin found for TVirtualProof");
      return 0;
   }
   if (h->LoadPlugin() == -1) {
      Error("CreateSession", "plugin for TVirtualProof could not be loaded");
      return 0;
   }

   // Create the instance
   TVirtualProof *p =
     (TVirtualProof*) h->ExecPlugin(4, fUrl.GetUrl(), cfg, cfgdir, loglevel);
   fUrl.SetOptions("");

   if (p && p->IsValid()) {

      // Set reference manager
      p->SetManager(this);

      // Save record about this session
      Int_t ns = 1;
      if (fSessions) {
         // To avoid ambiguities in case of removal of some elements
         if (fSessions->Last())
            ns = ((TVirtualProofDesc *)(fSessions->Last()))->GetLocalId() + 1;
      } else {
         // Create the list
         fSessions = new TList;
      }

      // Create the description class
      TVirtualProofDesc *d =
         new TVirtualProofDesc(p->GetName(), p->GetTitle(), p->GetUrl(),
                               ns, p->GetSessionID(), p->IsIdle(), p);
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
Bool_t TVirtualProofMgr::MatchUrl(const char *url)
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

   // Get the url host FQDN ...
   TInetAddress addr = gSystem->GetHostByName(u.GetHost());
   if (addr.IsValid()) {
      u.SetHost(addr.GetHostName());
      if (!strcmp(u.GetProtocol(),"UnNamedHost"))
         u.SetHost(addr.GetHostAddress());
   }

   // Now we can check
   if (!strcmp(u.GetHost(), fUrl.GetHost()))
      if (u.GetPort() == fUrl.GetPort())
         if (strlen(u.GetUser()) <= 0 || !strcmp(u.GetUser(),fUrl.GetUser()))
            return kTRUE;

   // Match failed
   return kFALSE;
}

//________________________________________________________________________
TList *TVirtualProofMgr::GetListOfManagers()
{
   // Extract pointers to PROOF managers from TROOT::fProofs.

   // Update the list with new entries
   if (gROOT->GetListOfProofs()) {
      TIter nxp(gROOT->GetListOfProofs());
      TVirtualProofMgr *p = 0;
      while ((p = dynamic_cast<TVirtualProofMgr *> (nxp())))
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
            TVirtualProofMgr *p = (TVirtualProofMgr *)o;
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
TVirtualProofMgr *TVirtualProofMgr::Create(const char *url, Int_t loglevel,
                                           const char *alias, Bool_t xpd)
{
   // Static method returning the appropriate TProofMgr object using
   // the plugin manager.
   TVirtualProofMgr *m= 0;

   // Make sure we do not have already a manager for this URL
   TList *lm = TVirtualProofMgr::GetListOfManagers();
   if (lm) {
      TIter nxm(lm);
      while ((m = (TVirtualProofMgr *)nxm()))
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
   TPluginHandler *h = 0;
   Bool_t trystd = kTRUE;

   // If required, we assume first that the remote server is based on XrdProofd
   if (xpd) {
      if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualProofMgr", "xpd")) &&
           h->LoadPlugin() == 0) {
         m = (TVirtualProofMgr *) h->ExecPlugin(3, url, loglevel, alias);
         // Update trystd flag
         trystd = (m && !(m->IsValid()) && m->IsProofd()) ? kTRUE : kFALSE;
      }
   }

   // If the first attempt failed, we instantiate an old interface
   if (trystd) {
      SafeDelete(m);
      if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualProofMgr", "std")) &&
           h->LoadPlugin() == 0) {
         m = (TVirtualProofMgr *) h->ExecPlugin(3, url, loglevel, alias);
      }
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

ClassImp(TVirtualProofDesc)

//________________________________________________________________________
void TVirtualProofDesc::Print(Option_t *) const
{
   // Dump the content to the screen.

   Printf("// # %d", fLocalId);
   Printf("// alias: %s, url: \"%s\"", GetTitle(), GetUrl());
   Printf("// tag: %s", GetName());
   Printf("// status: %s, attached: %s (remote ID: %d)",
         (fIdle ? "idle" : "processing"), (fProof ? "YES" : "NO"), fRemoteId);
}
