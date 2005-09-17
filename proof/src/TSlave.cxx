// @(#)root/proof:$Name:  $:$Id: TSlave.cxx,v 1.44 2005/08/15 15:57:18 rdm Exp $
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
#include "TPluginManager.h"

ClassImp(TSlave)

//______________________________________________________________________________
TSlave::TSlave(const char *host, Int_t port, const char *ord, Int_t perf,
               const char *image, TProof *proof, ESlaveType stype,
               const char *workdir, const char *msd)
  : fName(host), fImage(image), fProofWorkDir(workdir),
    fWorkDir(workdir), fPort(port),
    fOrdinal(ord), fPerfIdx(perf),
    fProtocol(0), fSocket(0), fProof(proof),
    fInput(0), fBytesRead(0), fRealTime(0),
    fCpuTime(0), fSlaveType(stype), fStatus(0),
    fParallel(0), fMsd(msd)
{
   // Create a PROOF slave object. Called via the TProof ctor.

   Init(host, port, stype);
}

//______________________________________________________________________________
TSlave::TSlave()
{
   // Default constructor used by derived classes

   fPort      = -1;
   fOrdinal   = "-1";
   fPerfIdx   = -1;
   fProof     = 0;
   fSlaveType = kMaster;
   fProtocol  = 0;
   fSocket    = 0;
   fInput     = 0;
   fBytesRead = 0;
   fRealTime  = 0;
   fCpuTime   = 0;
   fStatus    = 0;
   fParallel  = 0;
}

//______________________________________________________________________________
void TSlave::Init(const char *host, Int_t port, ESlaveType stype)
{
   // Init a PROOF slave object. Called via the TSlave ctor.
   // The Init method is technology specific and is overwritten by derived
   // classes.

   // The url contains information about the server type: make sure
   // it is 'proofd' or alike
   TString hurl(fProof->GetUrlProtocol());
   hurl.Insert(5, 'd');
   // Add host, port (and user) information
   if (fProof->GetUser() && strlen(fProof->GetUser())) {
      hurl += TString(Form("://%s@%s:%d", fProof->GetUser(), host, port));
   } else {
      hurl += TString(Form("://%s:%d", host, port));
   }

   // Add information about our status (Client or Master)
   TString iam;
   if (fProof->IsMaster() && stype == kSlave) {
      iam = "Master";
      hurl += TString("/?SM");
   } else if (fProof->IsMaster() && stype == kMaster) {
      iam = "Master";
      hurl += TString("/?MM");
   } else if (!fProof->IsMaster() && stype == kMaster) {
      iam = "Local Client";
      hurl += TString("/?MC");
   } else {
      Error("Init","Impossible PROOF <-> SlaveType Configuration Requested");
      Assert(0);
   }

   // Open authenticated connection to remote PROOF slave server.
   // If a connection was already open (fSocket != 0), re-use it
   // to perform authentication (optimization needed to avoid a double
   // opening in case this is called by TXSlave).
   Int_t wsize = 65536;
   fSocket = TSocket::CreateAuthSocket(hurl, 0, wsize, fSocket);

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
   fProof->fUser      = fUser;
   PDB(kGlobal,3) {
      Info("Init","%s: fUser is .... %s", iam.Data(), fUser.Data());
   }

   char buf[512];
   fSocket->Recv(buf, sizeof(buf));
   if (strcmp(buf, "Okay")) {
      Printf("%s", buf);
      SafeDelete(fSocket);
      return;
   }

}

//______________________________________________________________________________
void TSlave::SetupServ(ESlaveType stype, const char *conffile)
{
   // Init a PROOF slave object. Called via the TSlave ctor.
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
   fProof->fUser       = fUser;

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
void TSlave::Init(TSocket *s, ESlaveType stype)
{
   // Init a PROOF slave object using the connection opened via s. Used to
   // avoid double opening when an attempt via TXSlave found a remote proofd.

   Init(s->GetInetAddress().GetHostName(), s->GetPort(), stype);
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

   if (fSocket) {
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

//______________________________________________________________________________
TSlave *TSlave::Create(const char *host, Int_t port, const char *ord, Int_t perf,
                       const char *image, TProof *proof, ESlaveType stype,
                       const char *workdir, const char *msd)
{
   // Static method returning the appropriate TSlave object for the remote
   // server.

   TSlave *s = 0;

   // If we have TXSlave the plugin manager will find it
   TPluginHandler *h = 0;
   if ((h = gROOT->GetPluginManager()->FindHandler("TSlave")) &&
       h->LoadPlugin() == 0) {
      s = (TSlave *) h->ExecPlugin(10, host, port, ord, perf, image, proof, stype,
                                       workdir, msd);
   } else {
      s = new TSlave(host, port, ord, perf, image, proof, stype, workdir, msd);
   }

   return s;
}

//______________________________________________________________________________
Int_t TSlave::Ping()
{
   // Ping the remote master or slave servers.
   // Returns 0 if ok, -1 in case of error

   if (!IsValid()) return -1;

   TMessage mess(kPROOF_PING | kMESS_ACK);
   fSocket->Send(mess);
   if (fSocket->Send(mess) == -1) {
      Warning("Ping","%s: acknowledgement not received", GetOrdinal());
      return -1;
   }
   return 0;
}

//______________________________________________________________________________
void TSlave::Interrupt(Int_t type)
{
   // Send interrupt OOB byte to master or slave servers.
   // Returns 0 if ok, -1 in case of error

   if (!IsValid()) return;

   char oobc = (char) type;
   const int kBufSize = 1024;
   char waste[kBufSize];

   // Send one byte out-of-band message to server
   if (fSocket->SendRaw(&oobc, 1, kOob) <= 0) {
      Error("Interrupt", "error sending oobc to slave %s", GetOrdinal());
      return;
   }

   if (type == TProof::kHardInterrupt) {
      char  oob_byte;
      int   n, nch, nbytes = 0, nloop = 0;

      // Receive the OOB byte
      while ((n = fSocket->RecvRaw(&oob_byte, 1, kOob)) < 0) {
         if (n == -2) {   // EWOULDBLOCK
            //
            // The OOB data has not yet arrived: flush the input stream
            //
            // In some systems (Solaris) regular recv() does not return upon
            // receipt of the oob byte, which makes the below call to recv()
            // block indefinitely if there are no other data in the queue.
            // FIONREAD ioctl can be used to check if there are actually any
            // data to be flushed.  If not, wait for a while for the oob byte
            // to arrive and try to read it again.
            //
            fSocket->GetOption(kBytesToRead, nch);
            if (nch == 0) {
               gSystem->Sleep(1000);
               continue;
            }

            if (nch > kBufSize) nch = kBufSize;
            n = fSocket->RecvRaw(waste, nch);
            if (n <= 0) {
               Error("Interrupt", "error receiving waste from slave %s",
                     GetOrdinal());
               break;
            }
            nbytes += n;
         } else if (n == -3) {   // EINVAL
            //
            // The OOB data has not arrived yet
            //
            gSystem->Sleep(100);
            if (++nloop > 100) {  // 10 seconds time-out
               Error("Interrupt", "server %s does not respond", GetOrdinal());
               break;
            }
         } else {
            Error("Interrupt", "error receiving OOB from server %s",
                  GetOrdinal());
            break;
         }
      }

      //
      // Continue flushing the input socket stream until the OOB
      // mark is reached
      //
      while (1) {
         int atmark;

         fSocket->GetOption(kAtMark, atmark);

         if (atmark)
            break;

         // find out number of bytes to read before atmark
         fSocket->GetOption(kBytesToRead, nch);
         if (nch == 0) {
            gSystem->Sleep(1000);
            continue;
         }

         if (nch > kBufSize) nch = kBufSize;
         n = fSocket->RecvRaw(waste, nch);
         if (n <= 0) {
            Error("Interrupt", "error receiving waste (2) from slave %s",
                  GetOrdinal());
            break;
         }
         nbytes += n;
      }
      if (nbytes > 0) {
         if (fProof->IsMaster())
            Info("Interrupt", "slave %s:%s synchronized: %d bytes discarded",
                 GetName(), GetOrdinal(), nbytes);
         else
            Info("Interrupt", "PROOF synchronized: %d bytes discarded", nbytes);
      }

      // Get log file from master or slave after a hard interrupt
      fProof->Collect(this);

   } else if (type == TProof::kSoftInterrupt) {

      // Get log file from master or slave after a soft interrupt
      fProof->Collect(this);

   } else if (type == TProof::kShutdownInterrupt) {

      ; // nothing expected to be returned

   } else {

      // Unexpected message, just receive log file
      fProof->Collect(this);
   }
}
