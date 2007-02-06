// @(#)root/proofx:$Name:  $:$Id: TXProofMgr.cxx,v 1.16 2006/12/03 23:34:04 rdm Exp $
// Author: Gerardo Ganis  12/12/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TXProofMgr                                                           //
//                                                                      //
// The PROOF manager interacts with the PROOF server coordinator to     //
// create or destroy a PROOF session, attach to or detach from          //
// existing one, and to monitor any client activity on the cluster.     //
// At most one manager instance per server is allowed.                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TList.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TProof.h"
#include "TProofLog.h"
#include "TXProofMgr.h"
#include "TXSocket.h"
#include "TROOT.h"

ClassImp(TXProofMgr)

// Autoloading hooks.
// These are needed to avoid using the plugin manager which may create
// problems in multi-threaded environments.
extern "C" {
   TProofMgr *GetTXProofMgr(const char *url, Int_t l, const char *al)
   { return ((TProofMgr *) new TXProofMgr(url, l, al)); }
}
class TXProofMgrInit {
public:
   TXProofMgrInit() {
      TProofMgr::SetTXProofMgrHook(&GetTXProofMgr);
}};
static TXProofMgrInit gxproofmgr_init;

//______________________________________________________________________________
TXProofMgr::TXProofMgr(const char *url, Int_t dbg, const char *alias)
          : TProofMgr(url)
{
   // Create a PROOF manager for the standard (old) environment.

   // Set the correct servert type
   fServType = kXProofd;

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
            Info("TXProofMgr","service 'proofd' not found by GetServiceByName"
                              ": using default IANA assigned tcp port 1093");
         port = 1093;
      } else {
         if (gDebug > 1)
            Info("TXProofMgr","port from GetServiceByName: %d", port);
      }
      fUrl.SetPort(port);
   }

   // Check and save the host FQDN ...
   if (strcmp(fUrl.GetHost(), fUrl.GetHostFQDN()))
      fUrl.SetHost(fUrl.GetHostFQDN());

   SetName(fUrl.GetUrl(kTRUE));
   if (alias)
      SetAlias(alias);
   else
      SetAlias(fUrl.GetHost());

   // Initialize
   if (Init(dbg) != 0) {
      // Failure: make sure the socket is deleted so that its lack of
      // validity is correctly transmitted
      SafeDelete(fSocket);
   }
}

//______________________________________________________________________________
Int_t TXProofMgr::Init(Int_t)
{
   // Do real initialization: open the connection and set the relevant
   // variables.
   // Login and authentication are dealt with at this level, if required.
   // Return 0 in case of success, 1 if the remote server is a 'proofd',
   // -1 in case of error.

   // Here we make sure that the port is explicitly specified in the URL,
   // even when it matches the default value
   TString u = fUrl.GetUrl(kTRUE);

   if (!(fSocket = new TXSocket(u, 'C', kPROOF_Protocol,
                                kXPROOF_Protocol, 0, -1, this)) ||
       !(fSocket->IsValid())) {
      if (!fSocket || !(fSocket->IsServProofd()))
         Error("Init", "while opening the connection to %s - exit", u.Data());
      if (fSocket && fSocket->IsServProofd())
         fServType = TProofMgr::kProofd;
      return -1;
   }

   // Protocol run by remote PROOF server
   fRemoteProtocol = fSocket->GetRemoteProtocol();

   // We add the manager itself for correct destruction
   {  R__LOCKGUARD2(gROOTMutex);
      gROOT->GetListOfSockets()->Remove(fSocket);
   }

   // We are done
   return 0;
}

//______________________________________________________________________________
TXProofMgr::~TXProofMgr()
{
   // Destructor: close the connection

   if (fSocket)
      fSocket->Close("P");
   SafeDelete(fSocket);
}

//______________________________________________________________________________
TProof *TXProofMgr::AttachSession(Int_t id, Bool_t gui)
{
   // Dummy version provided for completeness. Just returns a pointer to
   // existing session 'id' (as shown by TProof::QuerySessions) or 0 if 'id' is
   // not valid. The boolena 'gui' should be kTRUE when invoked from the GUI.

   if (!IsValid()) {
      Warning("AttachSession","invalid TXProofMgr - do nothing");
      return 0;
   }

   TProofDesc *d = GetProofDesc(id);
   if (d) {
      if (d->GetProof())
         // Nothing to do if already in contact with proofserv
         return d->GetProof();

      // Re-compose url
      TString u(Form("%s/?%d", fUrl.GetUrl(kTRUE), d->GetRemoteId()));

      // We need this to set correctly the kUsingSessionGui bit before the first
      // feedback messages arrive
      if (gui)
         u += "GUI";

      // Attach
      TProof *p = new TProof(u);
      if (p && p->IsValid()) {

         // Set reference manager
         p->SetManager(this);

         // Save record about this session
         Int_t st = (p->IsIdle()) ? TProofDesc::kIdle
                                  : TProofDesc::kRunning;
         d->SetStatus(st);
         d->SetProof(p);

         // Set session tag
         p->SetName(d->GetName());

      } else {
         // Session creation failed
         Error("AttachSession", "attaching to PROOF session");
      }
      return p;
   }

   Info("AttachSession","invalid proofserv id (%d)", id);
   return 0;
}

//______________________________________________________________________________
void TXProofMgr::DetachSession(Int_t id, Option_t *opt)
{
   // Detach session with 'id' from its proofserv. The 'id' is the number
   // shown by QuerySessions.

   if (!IsValid()) {
      Warning("DetachSession","invalid TXProofMgr - do nothing");
      return;
   }

   if (id > 0) {
      // Single session request
      TProofDesc *d = GetProofDesc(id);
      if (d) {
         if (fSocket)
            fSocket->DisconnectSession(d->GetRemoteId(), opt);
         TProof *p = d->GetProof();
         SafeDelete(p);
         fSessions->Remove(d);
         delete d;
      }
   } else if (id == 0) {

      // Requesto to destroy all sessions
      if (fSocket) {
         TString o = Form("%sA",opt);
         fSocket->DisconnectSession(-1, o);
      }
      if (fSessions) {
         // Delete PROOF sessions
         TIter nxd(fSessions);
         TProofDesc *d = 0;
         while ((d = (TProofDesc *)nxd())) {
            TProof *p = d->GetProof();
            SafeDelete(p);
         }
         fSessions->Delete();
      }
   }

   return;
}

//______________________________________________________________________________
Bool_t TXProofMgr::MatchUrl(const char *url)
{
   // Checks if 'url' refers to the same 'user@host:port' entity as the URL
   // in memory. TProofMgr::MatchUrl cannot be used here because of the
   // 'double' default port, implying an additional check on the port effectively
   // open.

   if (!IsValid()) {
      Warning("MatchUrl","invalid TXProofMgr - do nothing");
      return 0;
   }

   TUrl u(url);

   // Correct URL protocol
   if (!strcmp(u.GetProtocol(), TUrl("a").GetProtocol()))
      u.SetProtocol("proof");

   if (u.GetPort() == TUrl("a").GetPort()) {
      // Set default port
      Int_t port = gSystem->GetServiceByName("proofd");
      if (port < 0)
         port = 1093;
      u.SetPort(port);
   }

   // Now we can check
   if (!strcmp(u.GetHostFQDN(), fUrl.GetHost()))
      if (u.GetPort() == fUrl.GetPort() ||
          u.GetPort() == fSocket->GetPort())
         if (strlen(u.GetUser()) <= 0 || !strcmp(u.GetUser(),fUrl.GetUser()))
            return kTRUE;

   // Match failed
   return kFALSE;
}

//______________________________________________________________________________
void TXProofMgr::ShowWorkers()
{
   // Show available workers

   if (!IsValid()) {
      Warning("ShowWorkers","invalid TXProofMgr - do nothing");
      return;
   }

   // Send the request
   TObjString *os = fSocket->SendCoordinator(TXSocket::kQueryWorkers);
   if (os) {
      TObjArray *oa = TString(os->GetName()).Tokenize(TString("&"));
      if (oa) {
         TIter nxos(oa);
         TObjString *to = 0;
         while ((to = (TObjString *) nxos()))
            // Now parse them ...
            Printf("+  %s", to->GetName());
      }
   }
}

//______________________________________________________________________________
TList *TXProofMgr::QuerySessions(Option_t *opt)
{
   // Get list of sessions accessible to this manager

   if (opt && !strncasecmp(opt,"L",1))
      // Just return the existing list
      return fSessions;

   // Nothing to do if not in contact with proofserv
   if (!IsValid()) {
      Warning("QuerySessions","invalid TXProofMgr - do nothing");
      return 0;
   }

   // Create list if not existing
   if (!fSessions) {
      fSessions = new TList();
      fSessions->SetOwner();
   }

   // Send the request
   TList *ocl = new TList;
   TObjString *os = fSocket->SendCoordinator(TXSocket::kQuerySessions);
   if (os) {
      TObjArray *oa = TString(os->GetName()).Tokenize(TString("|"));
      if (oa) {
         TProofDesc *d = 0;
         TIter nxos(oa);
         TObjString *to = (TObjString *) nxos();
         while ((to = (TObjString *) nxos())) {
            // Now parse them ...
            char al[256];
            char tg[256];
            Int_t id = -1, st = -1, nc = 0;
            sscanf(to->GetName(),"%d %s %s %d %d", &id, tg, al, &st, &nc);
            // Add to the list, if not already there
            if (!(d = (TProofDesc *) fSessions->FindObject(tg))) {
               Int_t locid = fSessions->GetSize() + 1;
               d = new TProofDesc(tg, al, GetUrl(), locid, id, st, 0);
               fSessions->Add(d);
            } else {
               // Set missing / update info
               d->SetStatus(st);
               d->SetRemoteId(id);
               d->SetTitle(al);
            }
            // Add to the list for final garbage collection
            ocl->Add(new TObjString(tg));
         }
         SafeDelete(oa);
      }
      SafeDelete(os);
   }

   // Printout and Garbage collection
   if (fSessions->GetSize() > 0) {
      TIter nxd(fSessions);
      TProofDesc *d = 0;
      while ((d = (TProofDesc *)nxd())) {
         if (ocl->FindObject(d->GetName())) {
            if (opt && !strncasecmp(opt,"S",1))
               d->Print("");
         } else {
            fSessions->Remove(d);
            SafeDelete(d);
         }
      }
   }

   // We are done
   return fSessions;
}

//_____________________________________________________________________________
Bool_t TXProofMgr::HandleError(const void *)
{
   // Handle error on the input socket

   Printf("TXProofMgr::HandleError: %p: got called ...", this);

   // Interrupt any PROOF session in Collect
   if (fSessions && fSessions->GetSize() > 0) {
      TIter nxd(fSessions);
      TProofDesc *d = 0;
      while ((d = (TProofDesc *)nxd())) {
         TProof *p = (TProof *) d->GetProof();
         if (p)
            p->InterruptCurrentMonitor();
      }
   }
   if (gDebug > 0)
      Printf("TXProofMgr::HandleError: %p: DONE ... ", this);

   // We are done
   return kTRUE;
}

//______________________________________________________________________________
Int_t TXProofMgr::Reset(const char *usr)
{
   // Send a cleanup request for the sessions associated with the current
   // user.
   // A user with superuser privileges can also asks cleaning for an different
   // user, specified by 'usr', or for all users (usr = *)
   // Return 0 on success, -1 in case of error.

   // Nothing to do if not in contact with proofserv
   if (!IsValid()) {
      Warning("Reset","invalid TXProofMgr - do nothing");
      return -1;
   }

   fSocket->SendCoordinator(TXSocket::kCleanupSessions, usr);

   return 0;
}

//_____________________________________________________________________________
TProofLog *TXProofMgr::GetSessionLogs(Int_t isess, const char *stag)
{
   // Get logs or log tails from last session associated with this manager
   // instance.
   // The arguments allow to specify a session different from the last one:
   //      isess   specifies a position relative to the last one, i.e. 1
   //              for the next to last session; the absolute value is taken
   //              so -1 and 1 are equivalent.
   //      stag    specifies the unique tag of the wanted session
   // If 'stag' is specified 'isess' is ignored.
   // Returns a TProofLog object (to be deleted by the caller) on success,
   // 0 if something wrong happened.

   // Nothing to do if not in contact with proofserv
   if (!IsValid()) {
      Warning("GetSessionLogs","invalid TXProofMgr - do nothing");
      return 0;
   }

   TProofLog *pl = 0;

   // The absolute value of isess counts
   isess = (isess > 0) ? -isess : isess;

   // Get the list of paths
   TObjString *os = fSocket->SendCoordinator(TXSocket::kQueryLogPaths, stag, isess);

   // Analyse it now
   Int_t ii = 0;
   if (os) {
      TString rs(os->GetName());
      Ssiz_t from = 0;
      // The session tag
      TString tag;
      if (!rs.Tokenize(tag, from, "|")) {
         Warning("GetSessionLogs", "Session tag undefined: corruption?\n"
                                   " (received string: %s)", os->GetName());
         return (TProofLog *)0;
      }
      // The pool url
      TString purl;
      if (!rs.Tokenize(purl, from, "|")) {
         Warning("GetSessionLogs", "Pool URL undefined: corruption?\n"
                                   " (received string: %s)", os->GetName());
         return (TProofLog *)0;
      }
      // Create the instance now
      if (!pl)
         pl = new TProofLog(tag, purl, this);

      // Per-node info
      TString to;
      while (rs.Tokenize(to, from, "|")) {
         if (!to.IsNull()) {
            TString ord(to);
            ord.Strip(TString::kLeading, ' ');
            TString url(ord);
            if ((ii = ord.Index(" ")) != kNPOS)
               ord.Remove(ii);
            if ((ii = url.Index(" ")) != kNPOS)
               url.Remove(0, ii + 1);
            // Add to the list
            pl->Add(ord, url);
            // Notify
            if (gDebug > 1)
               Info("GetSessionLogs", "ord: %s, url: %s", ord.Data(), url.Data());
         }
      }
      // Cleanup
      SafeDelete(os);
      // Retrieve the default part
      if (pl)
         pl->Retrieve();
   }

   // Done
   return pl;
}

//______________________________________________________________________________
TObjString *TXProofMgr::ReadBuffer(const char *fin, Long64_t ofs, Int_t len)
{
   // Read, via the coordinator, 'len' bytes from offset 'ofs' of 'file'.
   // Returns a TObjString with the content or 0, in case of failure

   // Nothing to do if not in contact with proofserv
   if (!IsValid()) {
      Warning("ReadBuffer","invalid TXProofMgr - do nothing");
      return (TObjString *)0;
   }

   // Send the request
   return fSocket->SendCoordinator(TXSocket::kReadBuffer, fin, len, ofs);
}
