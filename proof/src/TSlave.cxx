// @(#)root/proof:$Name:  $:$Id: TSlave.cxx,v 1.39 2005/06/23 00:29:38 rdm Exp $
// Author: Fons Rademakers   14/02/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSlave                                                               //
//                                                                      //
// This class describes a PROOF slave server.                           //
// It contains information like the slaves host name, ordinal number,   //
// performance index, socket, etc. Objects of this class can only be    //
// created via TProof member functions.                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TSlave.h"
#include "TProof.h"
#include "TSystem.h"
#include "TEnv.h"
#include "TROOT.h"
#include "TUrl.h"
#include "TMessage.h"
#include "TError.h"
#include "TVirtualMutex.h"
#include "TThread.h"
#include "TSocket.h"

ClassImp(TSlave)

//______________________________________________________________________________
TSlave::TSlave(const char *host, Int_t port, const char *ord, Int_t perf,
               const char *image, TProof *proof, ESlaveType stype,
               const char *workdir, const char *conffile, const char *msd)
  : fName(host), fImage(image), fProofWorkDir(workdir),
    fWorkDir(workdir), fUser(), fPort(port),
    fOrdinal(ord), fPerfIdx(perf), fSecContext(0),
    fProtocol(0), fSocket(0), fProof(proof),
    fInput(0), fBytesRead(0), fRealTime(0),
    fCpuTime(0), fSlaveType(stype), fStatus(0),
    fParallel(0), fMsd(msd)
{
   // Create a PROOF slave object. Called via the TProof ctor.

   // The url contains information about the server type: make sure
   // it is 'proofd' or alike
   TString hurl(proof->GetUrlProtocol());
   hurl.Insert(5, 'd');
   // Add host, port (and user) information
   if (proof->GetUser() && strlen(proof->GetUser())) {
      hurl += TString(Form("://%s@%s:%d", proof->GetUser(), host, port));
   } else {
      hurl += TString(Form("://%s:%d", host, port));
   }

   // Add information about our status (Client or Master)
   TString iam;
   if (proof->IsMaster() && stype == kSlave) {
      iam = "Master";
      hurl += TString("/?SM");
   } else if (proof->IsMaster() && stype == kMaster) {
      iam = "Master";
      hurl += TString("/?MM");
   } else if (!proof->IsMaster() && stype == kMaster) {
      iam = "Local Client";
      hurl += TString("/?MC");
   } else {
      Error("TSlave","Impossible PROOF <-> SlaveType Configuration Requested");
      Assert(0);
   }

   // Open authenticated connection to remote PROOF slave server.
   Int_t wsize = 65536;
   fSocket = TSocket::CreateAuthSocket(hurl, 0, wsize);

   if (!fSocket || !fSocket->IsAuthenticated()) {
      SafeDelete(fSocket);
      return;
   }

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
   fSecContext        = fSocket->GetSecContext();
   fUser              = fSecContext->GetUser();
   Int_t ProofdProto  = fSocket->GetRemoteProtocol();
   Int_t RemoteOffSet = fSocket->GetSecContext()->GetOffSet();

   PDB(kGlobal,3) {
     fSocket->GetSecContext()->Print("e");
     Info("TSlave",
         "%s: fUser is .... %s", iam.Data(), proof->fUser.Data());
   }

   char buf[512];
   fSocket->Recv(buf, sizeof(buf));
   if (strcmp(buf, "Okay")) {
      Printf("%s", buf);
      SafeDelete(fSocket);
      return;
   }

   // get back startup message of proofserv (we are now talking with
   // the real proofserver and not anymore with the proofd front-end)
   Int_t what;
   if (fSocket->Recv(buf, sizeof(buf), what) <= 0) {
      Error("TSlave", "failed to receive slave startup message");
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
      Error("TSlave", "failed to send local PROOF protocol");
      SafeDelete(fSocket);
      return;
   }

   if (fSocket->Recv(fProtocol, what) != 2*sizeof(Int_t)) {
      Error("TSlave", "failed to receive remote PROOF protocol");
      SafeDelete(fSocket);
      return;
   }

   // protocols less than 4 are incompatible
   if (fProtocol < 4) {
      Error("TSlave", "incompatible PROOF versions (remote version must be >= 4, is %d)", fProtocol);
      SafeDelete(fSocket);
      return;
   }

   proof->fProtocol   = fProtocol;   // on master this is the protocol
   proof->fSecContext = fSecContext;
   proof->fUser       = fUser;
                                        // of the last slave
   // send user name to remote host
   // for UsrPwd and SRP methods send also passwd, rsa encoded
   TMessage pubkey;
   TString passwd = "";
   Bool_t  pwhash = kFALSE;
   Bool_t  srppwd = kFALSE;
   Bool_t  sndsrp = kFALSE;

   Bool_t upwd = fSecContext->IsA("UsrPwd");
   Bool_t srp = fSecContext->IsA("SRP");

   TPwdCtx *pwdctx = 0;
   if (RemoteOffSet > -1 && (upwd || srp))
      pwdctx = (TPwdCtx *)(fSecContext->GetContext());

   if (!proof->IsMaster()) {
      if ((gEnv->GetValue("Proofd.SendSRPPwd",0)) && (RemoteOffSet > -1))
         sndsrp = kTRUE;
   } else {
      if (srp && pwdctx) {
         if (pwdctx->GetPasswd() != "" && RemoteOffSet > -1)
            sndsrp = kTRUE;
      }
   }

   if ((upwd && pwdctx) || (srp  && sndsrp)) {

      // Send offset to identify remotely the public part of RSA key
      if (fSocket->Send(RemoteOffSet,kROOTD_RSAKEY) != 2*sizeof(Int_t)) {
         Error("TSlave", "failed to send offset in RSA key");
         SafeDelete(fSocket);
         return;
      }

      if (pwdctx) {
         passwd = pwdctx->GetPasswd();
         pwhash = pwdctx->IsPwHash();
      }

      if (fSocket->SecureSend(passwd,1,fSecContext->GetRSAKey()) == -1) {
         if (RemoteOffSet > -1)
            Warning("TSlave","problems secure-sending pass hash %s",
                    "- may result in failures");
         // If non RSA encoding available try passwd inversion
         if (upwd) {
            for (int i = 0; i < passwd.Length(); i++) {
               char inv = ~passwd(i);
               passwd.Replace(i, 1, inv);
            }
            TMessage mess;
            mess << passwd;
            if (fSocket->Send(mess) < 0) {
               Error("TSlave", "failed to send inverted password");
               SafeDelete(fSocket);
               return;
            }
         }
      }

   } else {

      // Send notification of no offset to be sent ...
      if (fSocket->Send(-2, kROOTD_RSAKEY) != 2*sizeof(Int_t)) {
         Error("TSlave", "failed to send no offset notification in RSA key");
         SafeDelete(fSocket);
         return;
      }
   }

   // Send ordinal (and config) info to slave (or master)
   TMessage mess;
   if (stype == kMaster)
      mess << fUser << pwhash << srppwd << fOrdinal << TString(conffile);
   else
      mess << fUser << pwhash << srppwd << fOrdinal << fProofWorkDir;

   if (fSocket->Send(mess) < 0) {
      Error("TSlave", "failed to send ordinal and config info");
      SafeDelete(fSocket);
      return;
   }

   if (ProofdProto > 6) {
      // Now we send authentication details to access, e.g., data servers
      // not in the proof cluster and to be propagated to slaves.
      // This is triggered by the 'proofserv <dserv1> <dserv2> ...'
      // line in .rootauthrc
      if (fSocket->SendHostAuth() < 0) {
         Error("TSlave", "failed to send HostAuth info");
         SafeDelete(fSocket);
         return;
      }
   }

   // set some socket options
   fSocket->SetOption(kNoDelay, 1);
}

//______________________________________________________________________________
TSlave::~TSlave()
{
   // Destroy slave.

   Close();
}

//______________________________________________________________________________
void TSlave::Close(Option_t *)
{
   // Close slave socket.

   SafeDelete(fInput);
   SafeDelete(fSocket);
}

//______________________________________________________________________________
Int_t TSlave::Compare(const TObject *obj) const
{
   // Used to sort slaves by performance index.

   const TSlave *sl = dynamic_cast<const TSlave*>(obj);

   if (fPerfIdx > sl->GetPerfIdx()) return 1;
   if (fPerfIdx < sl->GetPerfIdx()) return -1;
   const char *myord = GetOrdinal();
   const char *otherord = sl->GetOrdinal();
   while (myord && otherord) {
      Int_t myval = atoi(myord);
      Int_t otherval = atoi(otherord);
      if (myval < otherval) return 1;
      if (myval > otherval) return -1;
      myord = strchr(myord, '.');
      if (myord) myord++;
      otherord = strchr(otherord, '.');
      if (otherord) otherord++;
   }
   if (myord) return -1;
   if (otherord) return 1;
   return 0;
}

//______________________________________________________________________________
void TSlave::Print(Option_t *) const
{
   // Printf info about slave.

   Printf("*** Slave %s  (%s)", fOrdinal.Data(), fSocket ? "valid" : "invalid");
   Printf("    Host name:               %s", GetName());
   Printf("    Port number:             %d", GetPort());
   if (fSocket) {
      Printf("    User:                    %s", GetUser());
      Printf("    Security context:        %s", fSecContext->AsString());
      Printf("    Proofd protocol version: %d", fSocket->GetRemoteProtocol());
      Printf("    Image name:              %s", GetImage());
      Printf("    Working directory:       %s", GetWorkDir());
      Printf("    Performance index:       %d", GetPerfIdx());
      Printf("    MB's processed:          %.2f", float(GetBytesRead())/(1024*1024));
      Printf("    MB's sent:               %.2f", float(fSocket->GetBytesRecv())/(1024*1024));
      Printf("    MB's received:           %.2f", float(fSocket->GetBytesSent())/(1024*1024));
      Printf("    Real time used (s):      %.3f", GetRealTime());
      Printf("    CPU time used (s):       %.3f", GetCpuTime());
   }
}

//______________________________________________________________________________
void TSlave::SetInputHandler(TFileHandler *ih)
{
   // Adopt and register input handler for this slave. Handler will be deleted
   // by the slave.

   fInput = ih;
   fInput->Add();
}
