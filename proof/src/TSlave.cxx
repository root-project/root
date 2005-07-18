// @(#)root/proof:$Name:  $:$Id: TSlave.cxx,v 1.41 2005/06/23 10:51:56 rdm Exp $
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
    fOrdinal(ord), fPerfIdx(perf),
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
   fUser              = fSocket->GetSecContext()->GetUser();
   proof->fUser       = fUser;
   PDB(kGlobal,3) {
      Info("TSlave","%s: fUser is .... %s", iam.Data(), fUser.Data());
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
      Error("TSlave", "incompatible PROOF versions (remote version"
                      " must be >= 4, is %d)", fProtocol);
      SafeDelete(fSocket);
      return;
   }

   proof->fProtocol   = fProtocol;   // protocol of last slave on master
   proof->fUser       = fUser;

   if (fProtocol < 5) {
      //
      // Setup authentication related stuff for ald versions
      Bool_t isMaster = (stype == kMaster);
      TString wconf = isMaster ? TString(conffile) : fProofWorkDir;
      if (OldAuthSetup(isMaster, wconf) != 0) {
         Error("TSlave", "OldAuthSetup: failed to setup authentication");
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
         Error("TSlave", "failed to send ordinal and config info");
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

   // deactivate used sec context if talking to proofd daemon running
   // an old protocol (sec context disactivated remotely)
   TSecContext *sc = fSocket->GetSecContext();
   if (sc && sc->IsActive()) {
      TIter last(sc->GetSecContextCleanup(), kIterBackward);
      TSecContextCleanup *nscc = 0;
      while ((nscc = (TSecContextCleanup *)last())) {
         if (nscc->GetType() == TSocket::kPROOFD &&
             nscc->GetProtocol() < 9) {
            sc->DeActivate("");
            break;
         }
      }
   }

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

   TString sc;

   Printf("*** Slave %s  (%s)", fOrdinal.Data(), fSocket ? "valid" : "invalid");
   Printf("    Host name:               %s", GetName());
   Printf("    Port number:             %d", GetPort());
   if (fSocket) {
      Printf("    User:                    %s", GetUser());
      Printf("    Security context:        %s", fSocket->GetSecContext()->AsString(sc));
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

//______________________________________________________________________________
Int_t TSlave::OldAuthSetup(Bool_t master, TString wconf)
{
   // Setup authentication related stuff for old versions.
   // Provided for backward compatibility.
   static OldSlaveAuthSetup_t oldAuthSetupHook = 0;

   if (!oldAuthSetupHook) {
      // Load libraries needed for (server) authentication ...
#ifdef ROOTLIBDIR
      TString authlib = TString(ROOTLIBDIR) + "/libRootAuth";
#else
      TString authlib = TString(gRootDir) + "/lib/libRootAuth";
#endif
      char *p = 0;
      // The generic one
      if ((p = gSystem->DynamicPathName(authlib, kTRUE))) {
         delete[] p;
         if (gSystem->Load(authlib) == -1) {
            Error("OldAuthSetup", "can't load %s",authlib.Data());
            return kFALSE;
         }
      } else {
         Error("OldAuthSetup", "can't locate %s",authlib.Data());
         return -1;
      }
      //
      // Locate OldSlaveAuthSetup
      Func_t f = gSystem->DynFindSymbol(authlib,"OldSlaveAuthSetup");
      if (f)
         oldAuthSetupHook = (OldSlaveAuthSetup_t)(f);
      else {
         Error("OldAuthSetup", "can't find OldSlaveAuthSetup");
         return -1;
      }
   }
   //
   // Setup
   if (oldAuthSetupHook) {
      return (*oldAuthSetupHook)(fSocket, master, fOrdinal, wconf);
   } else {
      Error("OldAuthSetup", "hook to method OldSlaveAuthSetup is undefined");
      return -1;
   }
}
