// @(#)root/proofx:$Id$
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

#include <errno.h>
#ifdef WIN32
#include <io.h>
#endif

#include "TList.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TProof.h"
#include "TProofLog.h"
#include "TXProofMgr.h"
#include "TXSocket.h"
#include "TROOT.h"
#include "XProofProtocol.h"

ClassImp(TXProofMgr)

// Autoloading hooks.
// These are needed to avoid using the plugin manager which may create
// problems in multi-threaded environments.
TProofMgr *GetTXProofMgr(const char *url, Int_t l, const char *al)
{ return ((TProofMgr *) new TXProofMgr(url, l, al)); }

class TXProofMgrInit {
public:
   TXProofMgrInit() {
      TProofMgr::SetTXProofMgrHook(&GetTXProofMgr);
}};
static TXProofMgrInit gxproofmgr_init;

//______________________________________________________________________________
TXProofMgr::TXProofMgr(const char *url, Int_t dbg, const char *alias)
          : TProofMgr(url, dbg, alias)
{
   // Create a PROOF manager for the standard (old) environment.

   // Set the correct servert type
   fServType = kXProofd;

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

   fSocket = 0;
   if (!(fSocket = new TXSocket(u, 'C', kPROOF_Protocol,
                                kXPROOF_Protocol, 0, -1, this)) ||
       !(fSocket->IsValid())) {
      if (!fSocket || !(fSocket->IsServProofd()))
         if (gDebug > 0)
            Error("Init", "while opening the connection to %s - exit (error: %d)",
                          u.Data(), (fSocket ? fSocket->GetOpenError() : -1));
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

   SetInvalid();
}

//______________________________________________________________________________
void TXProofMgr::SetInvalid()
{
   // Invalidate this manager by closing the connection

   if (fSocket)
      fSocket->Close("P");
   SafeDelete(fSocket);

   // Avoid destroying twice
   {  R__LOCKGUARD2(gROOTMutex);
      gROOT->GetListOfSockets()->Remove(this);
   }
}

//______________________________________________________________________________
TProof *TXProofMgr::AttachSession(TProofDesc *d, Bool_t gui)
{
   // Dummy version provided for completeness. Just returns a pointer to
   // existing session 'id' (as shown by TProof::QuerySessions) or 0 if 'id' is
   // not valid. The boolena 'gui' should be kTRUE when invoked from the GUI.

   if (!IsValid()) {
      Warning("AttachSession","invalid TXProofMgr - do nothing");
      return 0;
   }
   if (!d) {
      Warning("AttachSession","invalid description object - do nothing");
      return 0;
   }

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
   TProof *p = new TProof(u, 0, 0, gDebug, 0, this);
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

//______________________________________________________________________________
void TXProofMgr::DetachSession(Int_t id, Option_t *opt)
{
   // Detach session with 'id' from its proofserv. The 'id' is the number
   // shown by QuerySessions. The correspondent TProof object is deleted.
   // If id == 0 all the known sessions are detached.
   // Option opt="S" or "s" forces session shutdown.

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
         fSessions->Remove(d);
         SafeDelete(p);
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
void TXProofMgr::DetachSession(TProof *p, Option_t *opt)
{
   // Detach session 'p' from its proofserv. The instance 'p' is invalidated
   // and should be deleted by the caller

   if (!IsValid()) {
      Warning("DetachSession","invalid TXProofMgr - do nothing");
      return;
   }

   if (p) {
      // Single session request
      TProofDesc *d = GetProofDesc(p);
      if (d) {
         if (fSocket)
            fSocket->DisconnectSession(d->GetRemoteId(), opt);
         fSessions->Remove(d);
         p->Close(opt);
         delete d;
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
   TObjString *os = fSocket->SendCoordinator(kQueryWorkers);
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
   TObjString *os = fSocket->SendCoordinator(kQuerySessions);
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
Bool_t TXProofMgr::HandleInput(const void *)
{
   // Handle asynchronous input on the socket

   if (fSocket && fSocket->IsValid()) {
      TMessage *mess;
      if (fSocket->Recv(mess) >= 0) {
         Int_t what = mess->What();
         if (gDebug > 0)
            Info("HandleInput", "%p: got message type: %d", this, what);
         switch (what) {
            case kPROOF_TOUCH:
               fSocket->RemoteTouch();
               break;
            default:
               Warning("HandleInput", "%p: got unknown message type: %d", what);
               break;
         }
      }
   } else {
      Warning("HandleInput", "%p: got message but socket is invalid!");
   }

   // We are done
   return kTRUE;
}

//_____________________________________________________________________________
Bool_t TXProofMgr::HandleError(const void *in)
{
   // Handle error on the input socket

   XHandleErr_t *herr = in ? (XHandleErr_t *)in : 0;

   // Try reconnection
   if (fSocket && herr && (herr->fOpt == 1)) {
      fSocket->Reconnect();
      if (fSocket && fSocket->IsValid()) {
         if (gDebug > 0)
            Printf("ProofMgr: connection to coordinator at %s re-established",
                   fUrl.GetUrl());
         return kFALSE;
      }
   }
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
Int_t TXProofMgr::Reset(Bool_t hard, const char *usr)
{
   // Send a cleanup request for the sessions associated with the current user.
   // If 'hard' is true sessions are signalled for termination and moved to
   // terminate at all stages (top master, sub-master, workers). Otherwise
   // (default) only top-master sessions are asked to terminate, triggering
   // a gentle session termination. In all cases all sessions should be gone
   // after a few (2 or 3) session checking cycles.
   // A user with superuser privileges can also asks cleaning for an different
   // user, specified by 'usr', or for all users (usr = *)
   // Return 0 on success, -1 in case of error.

   // Nothing to do if not in contact with proofserv
   if (!IsValid()) {
      Warning("Reset","invalid TXProofMgr - do nothing");
      return -1;
   }

   Int_t h = (hard) ? 1 : 0;
   fSocket->SendCoordinator(kCleanupSessions, usr, h);

   return 0;
}

//_____________________________________________________________________________
TProofLog *TXProofMgr::GetSessionLogs(Int_t isess,
                                      const char *stag, const char *pattern)
{
   // Get logs or log tails from last session associated with this manager
   // instance.
   // The arguments allow to specify a session different from the last one:
   //      isess   specifies a position relative to the last one, i.e. 1
   //              for the next to last session; the absolute value is taken
   //              so -1 and 1 are equivalent.
   //      stag    specifies the unique tag of the wanted session
   // The special value stag = "NR" allows to just initialize the TProofLog
   // object w/o retrieving the files; this may be useful when the number
   // of workers is large and only a subset of logs is required.
   // If 'stag' is specified 'isess' is ignored (unless stag = "NR").
   // If 'pattern' is specified only the lines containing it are retrieved
   // (remote grep functionality); to filter out a pattern 'pat' use
   // pattern = "-v pat".
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

   // Special option in stag
   bool retrieve = 1;
   TString sesstag(stag);
   if (sesstag == "NR") {
      retrieve = 0;
      sesstag = "";
   }

   // Get the list of paths
   TObjString *os = fSocket->SendCoordinator(kQueryLogPaths, sesstag.Data(), isess);

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
         pl = new TProofLog(tag, GetUrl(), this);

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
            // Add to the list (special tag for valgrind outputs)
            if (url.Contains(".valgrind")) ord += "-valgrind";
            pl->Add(ord, url);
            // Notify
            if (gDebug > 1)
               Info("GetSessionLogs", "ord: %s, url: %s", ord.Data(), url.Data());
         }
      }
      // Cleanup
      SafeDelete(os);
      // Retrieve the default part if required
      if (pl && retrieve) {
         if (pattern && strlen(pattern) > 0)
            pl->Retrieve("*", TProofLog::kGrep, 0, pattern);
         else
            pl->Retrieve();
      }
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
   return fSocket->SendCoordinator(kReadBuffer, fin, len, ofs, 0);
}

//______________________________________________________________________________
TObjString *TXProofMgr::ReadBuffer(const char *fin, const char *pattern)
{
   // Read, via the coordinator, lines containing 'pattern' in 'file'.
   // Returns a TObjString with the content or 0, in case of failure

   // Nothing to do if not in contact with proofserv
   if (!IsValid()) {
      Warning("ReadBuffer","invalid TXProofMgr - do nothing");
      return (TObjString *)0;
   }

   // Prepare the buffer
   Int_t plen = strlen(pattern);
   Int_t lfi = strlen(fin);
   char *buf = new char[lfi + plen + 1];
   memcpy(buf, fin, lfi);
   memcpy(buf+lfi, pattern, plen);
   buf[lfi+plen] = 0;

   // Send the request
   return fSocket->SendCoordinator(kReadBuffer, buf, plen, 0, 1);
}

//______________________________________________________________________________
void TXProofMgr::ShowROOTVersions()
{
   // Display what ROOT versions are available on the cluster

   // Nothing to do if not in contact with proofserv
   if (!IsValid()) {
      Warning("ShowROOTVersions","invalid TXProofMgr - do nothing");
      return;
   }

   // Send the request
   TObjString *os = fSocket->SendCoordinator(kQueryROOTVersions);
   if (os) {
      // Display it
      Printf("----------------------------------------------------------\n");
      Printf("Available versions (tag ROOT-vers remote-path PROOF-version):\n");
      Printf("%s", os->GetName());
      Printf("----------------------------------------------------------");
      SafeDelete(os);
   }

   // We are done
   return;
}

//______________________________________________________________________________
void TXProofMgr::SetROOTVersion(const char *tag)
{
   // Set the default ROOT version to be used

   // Nothing to do if not in contact with proofserv
   if (!IsValid()) {
      Warning("SetROOTVersion","invalid TXProofMgr - do nothing");
      return;
   }

   // Send the request
   fSocket->SendCoordinator(kROOTVersion, tag);

   // We are done
   return;
}

//______________________________________________________________________________
Int_t TXProofMgr::SendMsgToUsers(const char *msg, const char *usr)
{
   // Send a message to connected users. Only superusers can do this.
   // The first argument specifies the message or the file from where to take
   // the message.
   // The second argument specifies the user to which to send the message: if
   // empty or null the message is send to all the connected users.
   // return 0 in case of success, -1 in case of error

   Int_t rc = 0;

   // Check input
   if (!msg || strlen(msg) <= 0) {
      Error("SendMsgToUsers","no message to send - do nothing");
      return -1;
   }

   // Buffer (max 32K)
   const Int_t kMAXBUF = 32768;
   char buf[kMAXBUF] = {0};
   char *p = &buf[0];
   Int_t space = kMAXBUF - 1;
   Int_t len = 0;
   Int_t lusr = 0;

   // A specific user?
   if (usr && strlen(usr) > 0 && (strlen(usr) != 1 || usr[0] != '*')) {
      lusr = (strlen(usr) + 3);
      sprintf(buf, "u:%s ", usr);
      p += lusr;
      space -= lusr;
   }

   // Is it from file ?
   if (!gSystem->AccessPathName(msg, kFileExists)) {
      // From file: can we read it ?
      if (gSystem->AccessPathName(msg, kReadPermission)) {
         Error("SendMsgToUsers","request to read message from unreadable file '%s'", msg);
         return -1;
      }
      // Open the file
      FILE *f = 0;
      if (!(f = fopen(msg, "r"))) {
         Error("SendMsgToUsers", "file '%s' cannot be open", msg);
         return -1;
      }
      // Determine the number of bytes to be read from the file.
      Int_t left = (Int_t) lseek(fileno(f), (off_t) 0, SEEK_END);
      lseek(fileno(f), (off_t) 0, SEEK_SET);
      // Now readout from file
      Int_t wanted = left;
      if (wanted > space) {
         wanted = space;
         Warning("SendMsgToUsers",
                 "requested to send %d bytes: max size is %d bytes: truncating", left, space);
      }
      do {
         while ((len = read(fileno(f), p, wanted)) < 0 &&
                  TSystem::GetErrno() == EINTR)
            TSystem::ResetErrno();
         if (len < 0) {
            SysError("SendMsgToUsers", "error reading file");
            break;
         }

         // Update counters
         left -= len;
         p += len;
         wanted = (left > kMAXBUF-1) ? kMAXBUF-1 : left;

      } while (len > 0 && left > 0);
   } else {
      // Add the message to the buffer
      len = strlen(msg);
      if (len > space) {
         Warning("SendMsgToUsers",
                 "requested to send %d bytes: max size is %d bytes: truncating", len, space);
         len = space;
      }
      memcpy(p, msg, len);
   }

   // Null-terminate
   buf[len + lusr] = 0;
//   fprintf(stderr,"%s\n", buf);

   // Send the request
   fSocket->SendCoordinator(kSendMsgToUser, buf);

   return rc;
}
