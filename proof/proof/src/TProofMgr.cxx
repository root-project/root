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

#include "Bytes.h"
#include "TError.h"
#include "TEnv.h"
#include "TFile.h"
#include "TFileCollection.h"
#include "TFileInfo.h"
#include "TList.h"
#include "TParameter.h"
#include "TProof.h"
#include "TProofMgr.h"
#include "TProofMgrLite.h"
#include "TSocket.h"
#include "TROOT.h"

ClassImp(TProofMgr)

// Sub-list of TROOT::fProofs with managers
TList TProofMgr::fgListOfManagers;
TProofMgr_t TProofMgr::fgTXProofMgrHook = 0;

// Auxilliary structures for pinging
// The client request
typedef struct {
   int first;
   int second;
   int third;
   int fourth;
   int fifth;
} clnt_HS_t;
// The body received after the first handshake's header
typedef struct {
   int msglen;
   int protover;
   int msgval;
} srv_HS_t;

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
   TString host = u.GetHost();
   if (proto == "lite" || host == "__lite__" ) {
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

//______________________________________________________________________________
Int_t TProofMgr::Ping(const char *url, Bool_t checkxrd)
{
   // Non-blocking check for a PROOF (or Xrootd, if checkxrd) service at 'url'
   // Return
   //        0 if a XProofd (or Xrootd, if checkxrd) daemon is listening at 'url'
   //       -1 if nothing is listening on the port (connection cannot be open)
   //        1 if something is listening but not XProofd (or not Xrootd, if checkxrd)

   if (!url || (url && strlen(url) <= 0)) {
      ::Error("TProofMgr::Ping", "empty url - fail");
      return -1;
   }

   TUrl u(url);
   // Check the port and set the defaults
   if (!strcmp(u.GetProtocol(), "http") && u.GetPort() == 80) {
      if (!checkxrd) {
         u.SetPort(1093);
      } else {
         u.SetPort(1094);
      }
   }
   
   // Open the connection, disabling warnings ...
   Int_t oldLevel = gErrorIgnoreLevel;
   gErrorIgnoreLevel = kSysError+1;
   TSocket s(u.GetHost(), u.GetPort());
   if (!(s.IsValid())) {
      if (gDebug > 0)
         ::Info("TProofMgr::Ping", "could not open connection to %s:%d", u.GetHost(), u.GetPort());
      gErrorIgnoreLevel = oldLevel;
      return -1;
   }
   // Send the first bytes
   int writeCount = -1;
   clnt_HS_t initHS;
   memset(&initHS, 0, sizeof(initHS));
   int len = sizeof(initHS);
   if (checkxrd) {
      initHS.fourth = (int)host2net((int)4);
      initHS.fifth = (int)host2net((int)2012);
      if ((writeCount = s.SendRaw(&initHS, len)) != len) {
         if (gDebug > 0)
            ::Info("TProofMgr::Ping", "1st: wrong number of bytes sent: %d (expected: %d)",
                                    writeCount, len);
         gErrorIgnoreLevel = oldLevel;
         return 1;
      }
   } else {
      initHS.third  = (int)host2net((int)1);
      if ((writeCount = s.SendRaw(&initHS, len)) != len) {
         if (gDebug > 0)
            ::Info("TProofMgr::Ping", "1st: wrong number of bytes sent: %d (expected: %d)",
                                    writeCount, len);
         gErrorIgnoreLevel = oldLevel;
         return 1;
      }
      // These 8 bytes are need by 'proofd' and discarded by XPD
      int dum[2];
      dum[0] = (int)host2net((int)4);
      dum[1] = (int)host2net((int)2012);
      if ((writeCount = s.SendRaw(&dum[0], sizeof(dum))) !=  sizeof(dum)) {
         if (gDebug > 0)
            ::Info("TProofMgr::Ping", "2nd: wrong number of bytes sent: %d (expected: %d)",
                                    writeCount, (int) sizeof(dum));
         gErrorIgnoreLevel = oldLevel;
         return 1;
      }
   }
   // Read first server response
   int type;
   len = sizeof(type);
   int readCount = s.RecvRaw(&type, len); // 4(2+2) bytes
   if (readCount != len) {
      if (gDebug > 0)
         ::Info("TProofMgr::Ping", "1st: wrong number of bytes read: %d (expected: %d)",
                        readCount, len);
      gErrorIgnoreLevel = oldLevel;
      return 1;
   }
   // to host byte order
   type = net2host(type);
   // Check if the server is the eXtended proofd
   if (type == 0) {
      srv_HS_t xbody;
      len = sizeof(xbody);
      readCount = s.RecvRaw(&xbody, len); // 12(4+4+4) bytes
      if (readCount != len) {
         if (gDebug > 0)
            ::Info("TProofMgr::Ping", "2nd: wrong number of bytes read: %d (expected: %d)",
                           readCount, len);
         gErrorIgnoreLevel = oldLevel;
         return 1;
      }
      xbody.protover = net2host(xbody.protover);
      xbody.msgval = net2host(xbody.msglen);
      xbody.msglen = net2host(xbody.msgval);

   } else if (type == 8) {
      // Standard proofd
      if (gDebug > 0) ::Info("TProofMgr::Ping", "server is old %s", (checkxrd ? "ROOTD" : "PROOFD"));
      gErrorIgnoreLevel = oldLevel;
      return 1;
   } else {
      // We don't know the server type
      if (gDebug > 0) ::Info("TProofMgr::Ping", "unknown server type: %d", type);
      gErrorIgnoreLevel = oldLevel;
      return 1;
   }
   
   // Restore ignore level
   gErrorIgnoreLevel = oldLevel;
   // Done
   return 0;
}

//________________________________________________________________________
void TProofMgr::ReplaceSubdirs(const char *fn, TString &fdst, TList &dirph)
{
   // Parse file name extracting the directory subcomponents in dirs, stored
   // as TObjStrings.

   if (!fn || (fn && strlen(fn) <= 0)) return;
   if (dirph.GetSize() <= 0) return;

   // Parse fn
   TList dirs;
   TString dd(fn), d;
   Ssiz_t from = 0;
   while (dd.Tokenize(d, from, "/")) {
      if (!d.IsNull()) dirs.Add(new TObjString(d));
   }
   if (dirs.GetSize() <= 0) return;
   dirs.SetOwner(kTRUE);
   
   TIter nxph(&dirph);
   TParameter<Int_t> *pi = 0;
   while ((pi = (TParameter<Int_t> *) nxph())) {
      if (pi->GetVal() < dirs.GetSize()) {
         TObjString *os = (TObjString *) dirs.At(pi->GetVal());
         if (os) fdst.ReplaceAll(pi->GetName(), os->GetName());
      } else {
         ::Warning("TProofMgr::ReplaceSubdirs",
                   "requested directory level '%s' is not available in the file path",
                   pi->GetName());
      }
   }
}

//________________________________________________________________________
TFileCollection *TProofMgr::UploadFiles(TList *src,
                                        const char *mss, const char *dest)
{
   // Upload files provided via the list 'src' (as TFileInfo or TObjString)
   // to 'mss'. The path under 'mss' is determined by 'dest'; the following
   // place-holders can be used in 'dest':
   //      <d0>, <d1>, <d2>, ...         referring to the n-th sub-component
   //                                    of the src path
   //      <bn>                          basename in the source path
   //      <sn>                          serial number of file in the list
   //      <fn>                          the full file path
   //      <us>, <gr>                    the local user and group names.
   // So, for example, if the source filename for the 99-th file is
   //               protosrc://host//d0/d1/d2/d3/d4/d5/myfile
   // then with dest = '/pool/user/<d3>/<d4>/<d5>/<s>/<bn>' and 
   //           mss = 'protodst://hostdst//nm/
   // the corresponding destination path is
   //           protodst://hostdst//nm/pool/user/d3/d4/d5/99/myfile
   //
   // If 'dest' is empty, <fn> is used.
   //
   // Returns a TFileCollection with the destination files created; this
   // TFileCollection is, for example, ready to be registered as dataset.

   TFileCollection *ds = 0;

   // The inputs must be make sense
   if (!src || (src && src->GetSize() <= 0)) {
      ::Warning("TProofMgr::UploadFiles", "list is empty!");
      return ds;
   }
   if (!mss || (mss && strlen(mss) <= 0)) {
      ::Warning("TProofMgr::UploadFiles", "MSS is undefined!");
      return ds;
   }
   
   TList dirph;

   // If the destination is defined we need to understand if we have place-holders
   if (dest && strlen(dest) > 0) {
      TString dst(dest), dt;
      Ssiz_t from = 0;
      TRegexp re("<d+[0-9]>");
      while (dst.Tokenize(dt, from, "/")) {
         if (dt.Contains(re)) {
            TParameter<Int_t> *pi = new TParameter<Int_t>(dt, -1);
            dt.ReplaceAll("<d", "");
            dt.ReplaceAll(">", "");
            if (dt.IsDigit()) {
               pi->SetVal(dt.Atoi());
               dirph.Add(pi);
            } else {
               SafeDelete(pi);
            }
         }
      }
      dirph.SetOwner(kTRUE);
   }

   // Now we will actually copy files and create the TList object
   ds = new TFileCollection();
   TIter nxf(src);
   TObject *o = 0;
   TObjString *os = 0;
   TFileInfo *fi = 0;
   Int_t kn = 0;
   while ((o = nxf())) {
      TUrl *furl = 0;
      if (!strcmp(o->ClassName(), "TFileInfo")) {
         if (!(fi = dynamic_cast<TFileInfo *>(o))) {
            ::Warning("TProofMgr::UploadFiles",
                      "object of class name '%s' does not cast to %s - ignore",
                      o->ClassName(), o->ClassName());
            continue;
         }
         furl = fi->GetFirstUrl();
      } else if (!strcmp(o->ClassName(), "TObjString")) {
         if (!(os = dynamic_cast<TObjString *>(o))) {
            ::Warning("TProofMgr::UploadFiles",
                      "object of class name '%s' does not cast to %s - ignore",
                      o->ClassName(), o->ClassName());
            continue;
         }
         furl = new TUrl(os->GetName());
      } else {
         ::Warning("TProofMgr::UploadFiles",
                   "object of unsupported class '%s' found in list - ignore", o->ClassName());
         continue;
      }

      // The file must be accessible
      if (gSystem->AccessPathName(furl->GetUrl()) == kFALSE) {
         
         // Create the destination path
         TString fdst(mss);
         if (dest && strlen(dest) > 0) {
            fdst += dest;
         } else {
            fdst += TString::Format("/%s", furl->GetFile());
         }
         
         // Replace filename and basename
         if (fdst.Contains("<bn>")) fdst.ReplaceAll("<bn>", gSystem->BaseName(furl->GetFile()));
         if (fdst.Contains("<fn>")) fdst.ReplaceAll("<fn>", furl->GetFile());

         // Replace serial number
         TString skn = TString::Format("%d", kn++);
         if (fdst.Contains("<sn>")) fdst.ReplaceAll("<sn>", skn);
         
         // Replace user and group name
         UserGroup_t *pw = gSystem->GetUserInfo();
         if (pw) {
            if (fdst.Contains("<us>")) fdst.ReplaceAll("<us>", pw->fUser);
            if (fdst.Contains("<gr>")) fdst.ReplaceAll("<gr>", pw->fGroup);
            delete pw;
         }

         // Now replace the subdirs, if required
         if (dirph.GetSize() > 0)
            TProofMgr::ReplaceSubdirs(gSystem->DirName(furl->GetFile()), fdst, dirph);

         // Check double slashes in the file field (Turl sets things correctly inside)
         TUrl u(fdst);
         fdst = u.GetUrl();

         // Copy the file now
         ::Info("TProofMgr::UploadFiles", "uploading '%s' to '%s'", furl->GetUrl(), fdst.Data());
         if (TFile::Cp(furl->GetUrl(), fdst.Data())) {
            // Build TFileCollection
            ds->Add(new TFileInfo(fdst.Data()));
         } else {
            ::Error("TProofMgr::UploadFiles", "file %s was not copied", furl->GetUrl());
         }
      }
   }
   
   // Return the TFileCollection
   return ds;
}

//______________________________________________________________________________
TFileCollection *TProofMgr::UploadFiles(const char *srcfiles,
                                        const char *mss, const char *dest)
{
   // Upload to 'mss' the files listed in the text file 'srcfiles' or contained
   // in the directory 'srcfiles'.
   // In the case 'srcfiles' is a text file, the files must be specified one per
   // line, with line beginning by '#' ignored (i.e. considered comments).
   // The path under 'mss' is defined by 'dest'; the following
   // place-holders can be used in 'dest':
   //      <d0>, <d1>, <d2>, ...         referring to the n-th sub-component
   //                                    of the src path
   //      <bn>                          basename in the source path
   //      <sn>                          serial number of file in the list
   //      <fn>                          the full file path
   //      <us>, <gr>                    the local user and group names.
   // So, for example, if the source filename for the 99-th file is
   //               protosrc://host//d0/d1/d2/d3/d4/d5/myfile
   // then with dest = '/pool/user/<d3>/<d4>/<d5>/<s>/<bn>' and 
   //           mss = 'protodst://hostdst//nm/
   // the corresponding destination path is
   //           protodst://hostdst//nm/pool/user/d3/d4/d5/99/myfile
   //
   // If 'dest' is empty, <fn> is used.
   //
   // Returns a TFileCollection with the destination files created; this
   // TFileCollection is, for example, ready to be registered as dataset.

   TFileCollection *ds = 0;

   // The inputs must be make sense
   if (!srcfiles || (srcfiles && strlen(srcfiles) <= 0)) {
      ::Error("TProofMgr::UploadFiles", "input text file or directory undefined!");
      return ds;
   }
   if (!mss || (mss && strlen(mss) <= 0)) {
      ::Error("TProofMgr::UploadFiles", "MSS is undefined!");
      return ds;
   }

   TString inpath(gSystem->ExpandPathName(srcfiles));
 
   FileStat_t fst;
   if (gSystem->GetPathInfo(inpath.Data(), fst)) {
      ::Error("TProofMgr::UploadFiles",
              "could not get information about the input path '%s':"
              " make sure that it exists and is readable", srcfiles);
      return ds;
   }

   // Create the list to feed UploadFile(TList *, ...)
   TList files;
   files.SetOwner();

   TString line;
   if (R_ISREG(fst.fMode)) {
      // Text file
      ifstream f;
      f.open(inpath.Data(), ifstream::out);
      if (f.is_open()) {
         while (f.good()) {
            line.ReadToDelim(f);
            line.Strip(TString::kTrailing, '\n');
            // Skip comments
            if (line.BeginsWith("#")) continue;
            if (gSystem->AccessPathName(line, kReadPermission) == kFALSE)
               files.Add(new TFileInfo(line));
         }
         f.close();
      } else {
         ::Error("TProofMgr::UploadFiles", "unable to open file '%s'", srcfiles);
      }
   } else if (R_ISDIR(fst.fMode)) {
      // Directory
      void *dirp = gSystem->OpenDirectory(inpath.Data());
      if (dirp) {
         const char *ent = 0;
         while ((ent = gSystem->GetDirEntry(dirp))) {
            if (!strcmp(ent, ".") || !strcmp(ent, "..")) continue;
            line.Form("%s/%s", inpath.Data(), ent);
            if (gSystem->AccessPathName(line, kReadPermission) == kFALSE)
               files.Add(new TFileInfo(line));
         }
         gSystem->FreeDirectory(dirp);
      } else {
         ::Error("TProofMgr::UploadFiles", "unable to open directory '%s'", inpath.Data());
      }
   } else {
      ::Error("TProofMgr::UploadFiles",
              "input path '%s' is neither a regular file nor a directory!", inpath.Data());
      return ds;
   }
   if (files.GetSize() <= 0) {
      ::Warning("TProofMgr::UploadFiles", "no files found in file or directory '%s'", inpath.Data());
   } else {
      ds = TProofMgr::UploadFiles(&files, mss, dest);
   }
   // Done
   return ds;
}

//______________________________________________________________________________
Int_t TProofMgr::Rm(const char *what, const char *, const char *)
{
   // Run 'rm' on 'what'. Locally it is just a call to TSystem::Unlink .

   Int_t rc = -1;
   // Nothing to do if not in contact with proofserv
   if (!IsValid()) {
      Error("Rm", "invalid TProofMgr - do nothing");
      return rc;
   }
   // Nothing to do if not in contact with proofserv
   if (!what || (what && strlen(what) <= 0)) {
      Error("Rm", "path undefined!");
      return rc;
   }
   
   TUrl u(what);
   if (!strcmp(u.GetProtocol(), "file")) {
      rc = gSystem->Unlink(u.GetFile());
   } else {
      rc = gSystem->Unlink(what);
   }
   // Done
   return (rc == 0) ? 0 : -1;
}

//
//  TProofDesc
//

ClassImp(TProofDesc)

//________________________________________________________________________
void TProofDesc::Print(Option_t *) const
{
   // Dump the content to the screen.
   const char *st[] = { "unknown", "idle", "processing", "shutting down"};

   Printf("// # %d", fLocalId);
   Printf("// alias: %s, url: \"%s\"", GetTitle(), GetUrl());
   Printf("// tag: %s", GetName());
   Printf("// status: %s, attached: %s (remote ID: %d)",st[fStatus+1], (fProof ? "YES" : "NO"), fRemoteId);
}
