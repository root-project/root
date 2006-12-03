// @(#)root/proofx:$Name:  $:$Id: TXSlave.cxx,v 1.15 2006/11/28 12:10:52 rdm Exp $
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
// TXSlave                                                              //
//                                                                      //
// This is the version of TSlave for slave servers based on XRD.        //
// See TSlave for details.                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TXSlave.h"
#include "TProof.h"
#include "TProofServ.h"
#include "TSystem.h"
#include "TEnv.h"
#include "TROOT.h"
#include "TUrl.h"
#include "TMessage.h"
#include "TMonitor.h"
#include "TError.h"
#include "TSysEvtHandler.h"
#include "TVirtualMutex.h"
#include "TThread.h"
#include "TXSocket.h"
#include "TXSocketHandler.h"

ClassImp(TXSlave)

//______________________________________________________________________________

//---- Hook to the constructor -------------------------------------------------
//---- This is needed to avoid using the plugin manager which may create -------
//---- problems in multi-threaded environments. --------------------------------

extern "C" {
   TSlave *GetTXSlave(const char *url, const char *ord, Int_t perf,
                      const char *image, TProof *proof, Int_t stype,
                      const char *workdir, const char *msd)
   {
      return ((TSlave *)(new TXSlave(url, ord, perf, image,
                                     proof, stype, workdir, msd)));
   }
}
class XSlaveInit {
 public:
   XSlaveInit() {
      TSlave::SetTXSlaveHook(&GetTXSlave);
}};
static XSlaveInit xslave_init;

//______________________________________________________________________________

//---- error handling ----------------------------------------------------------
//---- Needed to avoid blocking on the CINT mutex in printouts -----------------

//______________________________________________________________________________
void TXSlave::DoError(int level, const char *location, const char *fmt, va_list va) const
{
   // Interface to ErrorHandler (protected).

   ::ErrorHandler(level, Form("TXSlave::%s", location), fmt, va);
}

//
// Specific Interrupt signal handler
//
class TXSlaveInterruptHandler : public TSignalHandler {
private:
   TXSocket *fSocket;
public:
   TXSlaveInterruptHandler(TXSocket *s = 0)
      : TSignalHandler(kSigInterrupt, kFALSE), fSocket(s) { }
   Bool_t Notify();
};

//______________________________________________________________________________
Bool_t TXSlaveInterruptHandler::Notify()
{
   // TXSlave interrupt handler.

   Info("Notify","Processing interrupt signal ...");

   // Handle also interrupt condition on socket(s)
   if (fSocket)
      fSocket->SetInterrupt();

   return kTRUE;
}

//______________________________________________________________________________
TXSlave::TXSlave(const char *url, const char *ord, Int_t perf,
               const char *image, TProof *proof, Int_t stype,
               const char *workdir, const char *msd) : TSlave()
{
   // Create a PROOF slave object. Called via the TProof ctor.
   fImage = image;
   fProofWorkDir = workdir;
   fWorkDir = workdir;
   fOrdinal = ord;
   fPerfIdx = perf;
   fProof = proof;
   fSlaveType = (ESlaveType)stype;
   fMsd = msd;
   fIntHandler = 0;
   fValid = kFALSE;

   // Instance of the socket input handler to monitor all the XPD sockets
   TXSocketHandler *sh = TXSocketHandler::GetSocketHandler();
   gSystem->AddFileHandler(sh);

   TXSocket::fgLoc = (fProof->IsMaster()) ? "master" : "client" ;

   Init(url, stype);
}

//______________________________________________________________________________
void TXSlave::Init(const char *host, Int_t stype)
{
   // Init a PROOF slave object. Called via the TXSlave ctor.
   // The Init method is technology specific and is overwritten by derived
   // classes.

   // Url string with host, port information; 'host' may contain 'user' information
   // in the usual form 'user@host'

   // Auxilliary url
   TUrl url(host);
   url.SetProtocol(fProof->fUrl.GetProtocol());
   // Check port
   if (url.GetPort() == TUrl("a").GetPort()) {
      // For the time being we use 'rootd' service as default.
      // This will be changed to 'proofd' as soon as XRD will be able to
      // accept on multiple ports
      Int_t port = gSystem->GetServiceByName("proofd");
      if (port < 0) {
         if (gDebug > 0)
            Info("Init","service 'proofd' not found by GetServiceByName"
                        ": using default IANA assigned tcp port 1093");
         port = 1094;
      } else {
         if (gDebug > 1)
            Info("Init","port from GetServiceByName: %d", port);
      }
      url.SetPort(port);
   }

   // Fill members
   fName = url.GetHost();
   fPort = url.GetPort(); // We get the right default if the port is not specified


   // The field 'psid' is interpreted as session ID when we are attaching
   // to an existing session (ID passed in the options field of the url) or
   // to our PROOF protocl version when we are creating a new session
   TString opts(url.GetOptions());
   Bool_t attach = (opts.Length() > 0 && opts.IsDigit()) ? kTRUE : kFALSE;
   Int_t psid = (attach) ? opts.Atoi() : kPROOF_Protocol;

   // Add information about our status (Client or Master)
   TString iam;
   Char_t mode = 's';
   TString alias = fProof->GetTitle();
   if (fProof->IsMaster() && stype == kSlave) {
      iam = "Master";
      mode = 's';
      // Send session tag of the closest master to the slaves
      alias = Form("session-%s|ord:%s", fProof->GetName(), fOrdinal.Data());
   } else if (fProof->IsMaster() && stype == kMaster) {
      iam = "Master";
      mode = 'm';
      // Send session tag of the closest master to the slaves
      alias = Form("session-%s|ord:%s", fProof->GetName(), fOrdinal.Data());
   } else if (!fProof->IsMaster() && stype == kMaster) {
      iam = "Local Client";
      mode = (attach) ? 'A' : 'M';
   } else {
      Error("Init","Impossible PROOF <-> SlaveType Configuration Requested");
      R__ASSERT(0);
   }

   // Add conf file, if required
   if (fProof->fConfFile.Length() > 0)
      alias += Form("|cf:%s",fProof->fConfFile.Data());

   // Send over env variables (may not be supported remotely)
   TString envlist;
   if (!fProof->GetManager() ||
        fProof->GetManager()->GetRemoteProtocol() > 1001) {
         const TList *envs = TProof::GetEnvVars();
         if (envs != 0 ) {
            TIter next(envs);
            for (TObject *o = next(); o != 0; o = next()) {
               TNamed *env = dynamic_cast<TNamed*>(o);
               if (env != 0) {
                  if (!envlist.IsNull())
                     envlist += ",";
                  envlist += Form("%s=%s", env->GetName(), env->GetTitle());
               }
            }
         }
   } else {
      if (fProof->GetManager() && TProof::GetEnvVars())
         Info("Init", "** NOT ** sending user envs - RemoteProtocol : %d",
                      fProof->GetManager()->GetRemoteProtocol());
   }

   // Add to the buffer
   if (!envlist.IsNull())
      alias += Form("|envs:%s", envlist.Data());

   // Open connection to a remote XrdPROOF slave server.
   // Login and authentication are dealt with at this level, if required.
   if (!(fSocket = new TXSocket(url.GetUrl(kTRUE), mode, psid,
                                -1, alias, fProof->GetLogLevel(), this))) {
      Error("Init", "while opening the connection to %s - exit", url.GetUrl(kTRUE));
      return;
   }

   // The socket may not be valid
   if (!(fSocket->IsValid())) {
      Error("Init", "some severe error occurred while opening "
                    "the connection at %s - exit", url.GetUrl(kTRUE));
      SafeDelete(fSocket);
      return;
   }

   // Check if the remote server supports user envs setting
   if (!fProof->GetManager() && !envlist.IsNull() &&
      ((TXSocket *)fSocket)->GetXrdProofdVersion() <= 1001) {
      Info("Init","user envs setting sent but unsupported remotely - RemoteProtocol : %d",
                     ((TXSocket *)fSocket)->GetXrdProofdVersion()); 
   }

   // Set the reference to TProof
   ((TXSocket *)fSocket)->fReference = fProof;

   // Protocol run by remote PROOF server
   fProtocol = fSocket->GetRemoteProtocol();

   // Set server type
   fProof->fServType = TProofMgr::kXProofd;

   // Set remote session ID
   fProof->fSessionID = ((TXSocket *)fSocket)->GetSessionID();

   // Set URL entry point for the default data pool
   TString dpu(((TXSocket *)fSocket)->fBuffer);
   if (dpu.Length() > 0)
      fProof->SetDataPoolUrl(dpu);

   // Remove socket from global TROOT socket list. Only the TProof object,
   // representing all slave sockets, will be added to this list. This will
   // ensure the correct termination of all proof servers in case the
   // root session terminates.
   {
      R__LOCKGUARD2(gROOTMutex);
      gROOT->GetListOfSockets()->Remove(fSocket);
   }

   R__LOCKGUARD2(gProofMutex);

   // Fill some useful info
   fUser = ((TXSocket *)fSocket)->fUser;
   PDB(kGlobal,3) {
      Info("Init","%s: fUser is .... %s", iam.Data(), fUser.Data());
   }

   // Set valid
   fValid = kTRUE;
}

//______________________________________________________________________________
Int_t TXSlave::SetupServ(Int_t, const char *)
{
   // Init a PROOF slave object. Called via the TXSlave ctor.
   // The Init method is technology specific and is overwritten by derived
   // classes.

   // get back startup message of proofserv (we are now talking with
   // the real proofserver and not anymore with the proofd front-end)
   Int_t what;
   char buf[512];
   if (fSocket->Recv(buf, sizeof(buf), what) <= 0) {
      Error("SetupServ", "failed to receive slave startup message");
      Close("S");
      SafeDelete(fSocket);
      fValid = kFALSE;
      return -1;
   }

   if (what == kMESS_NOTOK) {
      SafeDelete(fSocket);
      fValid = kFALSE;
      return -1;
   }

   // protocols less than 4 are incompatible
   if (fProtocol < 4) {
      Error("SetupServ", "incompatible PROOF versions (remote version "
                         "must be >= 4, is %d)", fProtocol);
      SafeDelete(fSocket);
      fValid = kFALSE;
      return -1;
   }

   fProof->fProtocol   = fProtocol;   // protocol of last slave on master

   // set some socket options
   fSocket->SetOption(kNoDelay, 1);

   // We are done
   return 0;
}

//______________________________________________________________________________
TXSlave::~TXSlave()
{
   // Destroy slave.

   Close();
}

//______________________________________________________________________________
void TXSlave::Close(Option_t *opt)
{
   // Close slave socket.

   if (fSocket)
      // Closing socket ...
      fSocket->Close(opt);

   SafeDelete(fInput);
   SafeDelete(fSocket);
}

//______________________________________________________________________________
Int_t TXSlave::Ping()
{
   // Ping the remote master or slave servers.
   // Returns 0 if ok, -1 in case of error

   if (!IsValid()) return -1;

   return ((TXSocket *)fSocket)->Ping();
}

//______________________________________________________________________________
void TXSlave::Interrupt(Int_t type)
{
   // Send interrupt to master or slave servers.
   // Returns 0 if ok, -1 in case of error

   if (!IsValid()) return;

   if (type == TProof::kLocalInterrupt) {
      // Equivalent to an error condition on the socket
      HandleError();
      return;
   }

   ((TXSocket *)fSocket)->SendInterrupt(type);
   Info("Interrupt","Interrupt of type %d sent", type);
}

//______________________________________________________________________________
void TXSlave::StopProcess(Bool_t abort, Int_t timeout)
{
   // Sent stop/abort request to PROOF server. It will be
   // processed asynchronously by a separate thread.
   if (!IsValid()) return;

   ((TXSocket *)fSocket)->SendUrgent(TXSocket::kStopProcess, (Int_t)abort, timeout);
   if (gDebug > 0)
      Info("StopProcess", "Request of type %d sent over", abort);
}

//_____________________________________________________________________________
Int_t TXSlave::GetProofdProtocol(TSocket *s)
{
   // Find out the remote proofd protocol version.
   // Returns -1 in case of error.

   Int_t rproto = -1;

   UInt_t cproto = 0;
   Int_t len = sizeof(cproto);
   memcpy((char *)&cproto,
      Form(" %d", TSocket::GetClientProtocol()),len);
   Int_t ns = s->SendRaw(&cproto, len);
   if (ns != len) {
      ::Error("TXSlave::GetProofdProtocol",
              "sending %d bytes to proofd server [%s:%d]",
              len, (s->GetInetAddress()).GetHostName(), s->GetPort());
      return -1;
   }

   // Get the remote protocol
   Int_t ibuf[2] = {0};
   len = sizeof(ibuf);
   Int_t nr = s->RecvRaw(ibuf, len);
   if (nr != len) {
      ::Error("TXSlave::GetProofdProtocol",
              "reading %d bytes from proofd server [%s:%d]",
              len, (s->GetInetAddress()).GetHostName(), s->GetPort());
      return -1;
   }
   Int_t kind = net2host(ibuf[0]);
   if (kind == kROOTD_PROTOCOL) {
      rproto = net2host(ibuf[1]);
   } else {
      kind = net2host(ibuf[1]);
      if (kind == kROOTD_PROTOCOL) {
         len = sizeof(rproto);
         nr = s->RecvRaw(&rproto, len);
         if (nr != len) {
            ::Error("TXSlave::GetProofdProtocol",
                    "reading %d bytes from proofd server [%s:%d]",
                    len, (s->GetInetAddress()).GetHostName(), s->GetPort());
            return -1;
         }
         rproto = net2host(rproto);
      }
   }
   if (gDebug > 2)
      ::Info("TXSlave::GetProofdProtocol",
             "remote proofd: buf1: %d, buf2: %d rproto: %d",
             net2host(ibuf[0]),net2host(ibuf[1]),rproto);

   // We are done
   return rproto;
}

//______________________________________________________________________________
TObjString *TXSlave::SendCoordinator(Int_t kind, const char *msg, Int_t int2)
{
   // Send message to intermediate coordinator.
   // If any output is due, this is returned as a generic message

   return ((TXSocket *)fSocket)->SendCoordinator(kind, msg, int2);
}

//______________________________________________________________________________
void TXSlave::SetAlias(const char *alias)
{
   // Set an alias for this session. If reconnection is supported, the alias
   // will be communicated to the remote coordinator so that it can be recovered
   // when reconnecting

   // Nothing to do if not in contact with coordinator
   if (!IsValid()) return;

   ((TXSocket *)fSocket)->SendCoordinator(TXSocket::kSessionAlias, alias);

   return;
}

//_____________________________________________________________________________
Bool_t TXSlave::HandleError(const void *)
{
   // Handle error on the input socket

   Info("HandleError", "%p: got called ... fProof: %p", this, fProof);

   // Interrupt underlying socket operations
   ((TXSocket *)fSocket)->SetInterrupt();

   // Remove signal handler
   SetInterruptHandler(kFALSE);

   if (fProof) {

      // Remove PROOF signal handler
      if (fProof->fIntHandler)
         fProof->fIntHandler->Remove();

      // Attach to the monitor instance, if any
      TMonitor *mon = fProof->fCurrentMonitor;

      if (gDebug > 2)
         Info("HandleError", "%p: proof: %p, mon: %p", this, fProof, mon);

      if (mon && mon->GetListOfActives()->FindObject(fSocket)) {
         // Synchronous collection in TProof
         if (gDebug > 2)
            Info("HandleError", "%p: deactivating from monitor %p", this, mon);
         mon->DeActivate(fSocket);
      }
      // Update lists:
      if (fProof->IsMaster()) {
         // On masters we have to update the lists
         TString msg(Form("Worker '%s-%s' has been removed from the active list",
                          fName.Data(), fOrdinal.Data()));
         TMessage m(kPROOF_MESSAGE);
         m << msg;
         if (gProofServ)
            gProofServ->GetSocket()->Send(m);
         else
            Warning("HandleError", "%p: global reference to TProofServ missing");
         // The session is gone
         ((TXSocket *)fSocket)->SetSessionID(-1);
         fProof->MarkBad(this);
      } else {
         // On clients the proof session should be removed from the lists
         // and deleted, since it is not valid anymore
         fProof->GetListOfSlaves()->Remove(this);
         TProofMgr *mgr= fProof->GetManager();
         if (mgr)
            mgr->ShutdownSession(fProof);
      }
   } else {
      Warning("HandleError", "%p: reference to PROOF missing", this);
   }

   // Post semaphore to wake up anybody waiting; send as many posts as needed
   if (fSocket) {
      R__LOCKGUARD(((TXSocket *)fSocket)->fAMtx);
      TSemaphore *sem = &(((TXSocket *)fSocket)->fASem);
      while (sem->TryWait() != 1)
         sem->Post();
   }

   if (gDebug > 0)
      Info("HandleError", "%p: DONE ... ", this);

   // We are done
   return kTRUE;
}

//_____________________________________________________________________________
Bool_t TXSlave::HandleInput(const void *)
{
   // Handle asynchronous input on the socket

   if (fProof) {

      // Attach to the monitor instance, if any
      TMonitor *mon = fProof->fCurrentMonitor;

      if (gDebug > 2)
         Info("HandleInput", "%p: %s: proof: %p, mon: %p",
                             this, GetOrdinal(), fProof, mon);

      if (mon) {
         if (mon->IsActive(fSocket)) {
            // Synchronous collection in TProof
            if (gDebug > 2)
               Info("HandleInput","%p: %s: posting monitor %p", this, GetOrdinal(), mon);
            mon->SetReady(fSocket);
         }
      } else {
         // Asynchronous collection in TProof
         if (gDebug > 2)
            Info("HandleInput","%p: %s: calling TProof::CollectInputFrom", this, GetOrdinal());
         fProof->CollectInputFrom(fSocket);
      }
   } else {
      Warning("HandleInput", "%p: %s: reference to PROOF missing", this, GetOrdinal());
      return kFALSE;
   }

   // We are done
   return kTRUE;
}

//_____________________________________________________________________________
void TXSlave::SetInterruptHandler(Bool_t on)
{
   // Set/Unset the interrupt handler

   if (gDebug > 1)
      Info("SetInterruptHandler", "enter: %d", on);

   if (on) {
      if (!fIntHandler)
         fIntHandler = new TXSlaveInterruptHandler((TXSocket *)fSocket);
      fIntHandler->Add();
   } else {
      if (fIntHandler)
         fIntHandler->Remove();
   }
}

//_____________________________________________________________________________
void TXSlave::FlushSocket()
{
   // Clean any input on the socket

   if (gDebug > 1)
      Info("FlushSocket", "enter: %p", fSocket);

   if (fSocket)
      TXSocket::FlushPipe(fSocket);
}
