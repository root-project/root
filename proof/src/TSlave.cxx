// @(#)root/proof:$Name:  $:$Id: TSlave.cxx,v 1.30 2004/06/25 17:27:09 rdm Exp $
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

ClassImp(TSlave)

//______________________________________________________________________________
TSlave::TSlave(const char *host, Int_t port, Int_t ord, Int_t perf,
               const char *image, TProof *proof)
{
   // Create a PROOF slave object. Called via the TProof ctor.

   fName     = host;
   fPort     = port;
   fImage    = image;
   fWorkDir  = kPROOF_WorkDir;
   fOrdinal  = ord;
   fPerfIdx  = perf;
   fSecContext = 0;
   fProof    = proof;
   fSocket   = 0;
   fInput    = 0;

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
   if (proof->IsMaster()) {
      iam = "Master";
      hurl += TString("/?M");
   } else {
      iam = "Local Client";
      hurl += TString("/?C");
   }

   // Open authenticated connection to remote PROOF slave server.
   Int_t wsize = 65536;
   fSocket = TSocket::CreateAuthSocket(hurl, 0, wsize);
   if (fSocket && fSocket->IsAuthenticated()) {

      // Remove socket from global TROOT socket list. Only the TProof object,
      // representing all slave sockets, will be added to this list. This will
      // ensure the correct termination of all proof servers in case the
      // root session terminates.
      gROOT->GetListOfSockets()->Remove(fSocket);

      // Fill some useful info
      fSecContext        = fSocket->GetSecContext();
      fUser              = fSecContext->GetUser();
      proof->fSecContext = fSecContext;
      proof->fUser       = fUser;
      Int_t ProofdProto  = fSocket->GetRemoteProtocol();

      PDB(kGlobal,3) {
         fSocket->GetSecContext()->Print("e");
         Info("TSlave",
              "%s: fUser is .... %s", iam.Data(), proof->fUser.Data());
      }

      TString Details = fSocket->GetSecContext()->GetDetails();
      Int_t RemoteOffSet = fSocket->GetSecContext()->GetOffSet();

      char buf[512];
      fSocket->Recv(buf, sizeof(buf));

      if (strcmp(buf, "Okay")) {
         Printf("%s", buf);
         SafeDelete(fSocket);
      } else {
         // get back startup message of proofserv (we are now talking with
         // the real proofserver and not anymore with the proofd front-end)

         Int_t what;
         fSocket->Recv(buf, sizeof(buf), what);

         if (what == kMESS_NOTOK) {
            SafeDelete(fSocket);
            return;
         }

         // exchange protocol level between client and master and between
         // master and slave
         fSocket->Send(kPROOF_Protocol, kROOTD_PROTOCOL);
         fSocket->Recv(fProtocol, what);
         fProof->fProtocol = fProtocol;   // on master this is the protocol
                                          // of the last slave
         // send user name to remote host
         // for UsrPwd and SRP methods send also passwd, rsa encoded
         TMessage pubkey;
         TString passwd = "";
         Bool_t  pwhash = kFALSE;
         Bool_t  srppwd = kFALSE;
         Bool_t  sndsrp = kFALSE;

         TPwdCtx *pwdctx = 0;
         if (RemoteOffSet > -1 &&
            (fSecContext->IsA("UsrPwd") || fSecContext->IsA("SRP")))
            pwdctx = (TPwdCtx *)(fSecContext->GetContext());

         if (!fProof->IsMaster()) {
            if (gEnv->GetValue("Proofd.SendSRPPwd",0))
               if (RemoteOffSet > -1)
                  sndsrp = kTRUE;
         } else {
            if (fSecContext->IsA("SRP") && pwdctx) {
               if (pwdctx->GetPasswd() != "" && RemoteOffSet > -1)
                  sndsrp = kTRUE;
            }
         }

         if ((fSecContext->IsA("UsrPwd") && pwdctx) ||
             (fSecContext->IsA("SRP")    && sndsrp)) {

            // Send offset to identify remotely the public part of RSA key
            fSocket->Send(RemoteOffSet,kROOTD_RSAKEY);

            if (pwdctx) {
               passwd = pwdctx->GetPasswd();
               pwhash = pwdctx->IsPwHash();
            }
            srppwd = fSecContext->IsA("SRP");

            if (fSocket->SecureSend(passwd,1,fSecContext->GetRSAKey()) == -1) {
               if (RemoteOffSet > -1)
                  Warning("TSlave","problems secure-sending pass hash %s",
                          "- may result in failures");
               // If non RSA encoding available try passwd inversion
               if (fSecContext->IsA("UsrPwd")) {
                  for (int i = 0; i < passwd.Length(); i++) {
                     char inv = ~passwd(i);
                     passwd.Replace(i, 1, inv);
                  }
                  TMessage mess;
                  mess << passwd;
                  fSocket->Send(mess);
               }
            }

         } else {

            // Send notification of no offset to be sent ...
            fSocket->Send(-2,kROOTD_RSAKEY);

         }

         TMessage mess;
         if (!fProof->IsMaster())
            mess << fUser << pwhash << srppwd << fProof->fConfFile;
         else
            mess << fUser << pwhash << srppwd << fOrdinal;

         fSocket->Send(mess);

         if (ProofdProto > 6) {
            // Now we send authentication details to access, eg, data servers
            // not in the proof cluster and to be propagated to slaves.
            // This is triggered by the 'proofserv <dserv1> <dserv2> ...'
            // card in .rootauthrc
            fSocket->SendHostAuth();
         }

         // set some socket options
         fSocket->SetOption(kNoDelay, 1);
      }
   } else
      SafeDelete(fSocket);
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

   TSlave *sl = (TSlave *) obj;

   if (fPerfIdx > sl->GetPerfIdx()) return 1;
   if (fPerfIdx < sl->GetPerfIdx()) return -1;
   return 0;
}

//______________________________________________________________________________
void TSlave::Print(Option_t *) const
{
   // Printf info about slave.

   Printf("*** Slave %d  (%s)", fOrdinal, fSocket ? "valid" : "invalid");
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
      Printf("    MB's sent:               %.2f", fSocket ? float(fSocket->GetBytesRecv())/(1024*1024) : 0.0);
      Printf("    MB's received:           %.2f", fSocket ? float(fSocket->GetBytesSent())/(1024*1024) : 0.0);
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
