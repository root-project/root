// @(#)root/proofx:$Name:  $:$Id: TXSlave.cxx,v 1.4 2006/03/01 10:55:21 rdm Exp $
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
#include "TSystem.h"
#include "TEnv.h"
#include "TROOT.h"
#include "TUrl.h"
#include "TMessage.h"
#include "TError.h"
#include "TVirtualMutex.h"
#include "TThread.h"
#include "TXSocket.h"
#include "TXSocketHandler.h"

ClassImp(TXSlave)

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
      Int_t port = gSystem->GetServiceByName("rootd");
      if (port < 0) {
         if (gDebug > 0)
            Info("Init","service 'rootd' not found by GetServiceByName"
                        ": using default IANA assigned tcp port 1094");
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

   // If we are attaching to an existing process, the ID is passed in the
   // options field of the url
   Int_t psid = (strlen(url.GetOptions()) > 0) ? atoi(url.GetOptions()) : -1;

   // Add information about our status (Client or Master)
   TString iam;
   Char_t mode = 's';
   if (fProof->IsMaster() && stype == kSlave) {
      iam = "Master";
      mode = 's';
   } else if (fProof->IsMaster() && stype == kMaster) {
      iam = "Master";
      mode = 'm';
   } else if (!fProof->IsMaster() && stype == kMaster) {
      iam = "Local Client";
      mode = 'M';
   } else {
      Error("Init","Impossible PROOF <-> SlaveType Configuration Requested");
      R__ASSERT(0);
   }

   // Open connection to a remote XrdPROOF slave server.
   // Login and authentication are dealt with at this level, if required.
   if (!(fSocket = new TXSocket(url.GetUrl(kTRUE),
                                mode, psid, -1, fProof->GetTitle()))) {
      Error("Init", "while opening the connection to %s - exit", url.GetUrl(kTRUE));
      return;
   }

   // The socket may not be valid
   if (!(fSocket->IsValid())) {
      Error("Init", "some severe error occurred while opening"
                        " the connection at %s - exit", url.GetUrl(kTRUE));
      return;
   }

   // Set the this as reference of this socket
   ((TXSocket *)fSocket)->fReference = this;

   // Set server type
   fProof->fServType = TVirtualProofMgr::kXProofd;

   // Set remote session ID
   fProof->fSessionID = ((TXSocket *)fSocket)->GetSessionID();

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
}

//______________________________________________________________________________
void TXSlave::SetupServ(Int_t stype, const char *conffile)
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
      SafeDelete(fSocket);
      return;
   }

   if (what == kMESS_NOTOK) {
      SafeDelete(fSocket);
      return;
   }

   // exchange protocol level between client and master and between
   // master and slave
   if (fSocket->Send(kPROOF_Protocol, kROOTD_PROTOCOL) != 2*sizeof(Int_t)) {
      Error("SetupServ", "failed to send local PROOF protocol");
      SafeDelete(fSocket);
      return;
   }

   if (fSocket->Recv(fProtocol, what) != 2*sizeof(Int_t)) {
      Error("SetupServ", "failed to receive remote PROOF protocol");
      SafeDelete(fSocket);
      return;
   }

   // protocols less than 4 are incompatible
   if (fProtocol < 4) {
      Error("SetupServ", "incompatible PROOF versions (remote version"
                      " must be >= 4, is %d)", fProtocol);
      SafeDelete(fSocket);
      return;
   }

   fProof->fProtocol   = fProtocol;   // protocol of last slave on master

   if (fProtocol < 5) {
      //
      // Setup authentication related stuff for ald versions
      Bool_t isMaster = (stype == kMaster);
      TString wconf = isMaster ? TString(conffile) : fProofWorkDir;
      if (OldAuthSetup(isMaster, wconf) != 0) {
         Error("SetupServ", "OldAuthSetup: failed to setup authentication");
         SafeDelete(fSocket);
         return;
      }
   } else {
      //
      // Send ordinal (and config) info to slave (or master)
      TMessage mess;
      if (stype == kMaster)
         mess << fUser << fOrdinal << TString(conffile);
      else
         mess << fUser << fOrdinal << fProofWorkDir;

      if (fSocket->Send(mess) < 0) {
         Error("SetupServ", "failed to send ordinal and config info");
         SafeDelete(fSocket);
         return;
      }
   }

   // set some socket options
   fSocket->SetOption(kNoDelay, 1);
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
   // Send interrupt OOB byte to master or slave servers.
   // Returns 0 if ok, -1 in case of error

   if (!IsValid()) return;

   ((TXSocket *)fSocket)->SendInterrupt(type);
   Info("Interrupt","Interrupt of type %d sent", type);
}

//_____________________________________________________________________________
Int_t TXSlave::GetProofdProtocol(TSocket *s)
{
   // Find out the remote proofd protocol version.
   // Returns -1 in case of error
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
TObjString *TXSlave::SendCoordinator(Int_t kind, const char *msg)
{
   // Send message to intermediate coordinator.
   // If any output is due, this is returned as a generic message

   return ((TXSocket *)fSocket)->SendCoordinator(kind, msg);
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
