// @(#)root/proof:$Name:  $:$Id: TProofServ.cxx,v 1.1.1.1 2000/05/16 17:00:46 rdm Exp $
// Author: Fons Rademakers   16/02/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofServ                                                           //
//                                                                      //
// TProofServ is the PROOF server. It can act either as the master      //
// server or as a slave server, depending on its startup arguments. It  //
// receives and handles message coming from the client or from the      //
// master server.                                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifdef WIN32
#include <io.h>
typedef long off_t;
#endif

#include "TProofServ.h"
#include "TProof.h"
#include "TROOT.h"
#include "TFile.h"
#include "TSysEvtHandler.h"
#include "TSystem.h"
#include "TInterpreter.h"
#include "TException.h"
#include "TSocket.h"
#include "TStopwatch.h"
#include "TMessage.h"
#include "TEnv.h"
#include "TError.h"
#include "TTree.h"

TProofServ *gProofServ;


//______________________________________________________________________________
void ProofErrorHandler(int level, Bool_t abort, const char *location, const char *msg)
{
   // The PROOF error handler function. It prints the message on stderr and
   // if abort is set it aborts the application.

   if (level < gErrorIgnoreLevel)
      return;

   const char *type   = 0;
   ELogLevel loglevel = kLogWarning;

   if (level >= kWarning) {
      loglevel = kLogWarning;
      type = "Warning";
   }
   if (level >= kError) {
      loglevel = kLogErr;
      type = "Error";
   }
   if (level >= kSysError) {
      loglevel = kLogErr;
      type = "SysError";
   }
   if (level >= kFatal) {
      //loglevel = kLogEmerg;
      loglevel = kLogErr;
      type = "Fatal";
   }

   char *bp;
   if (!location || strlen(location) == 0) {
      fprintf(stderr, "%s: %s\n", type, msg);
      bp = Form("%s:%s:%s", gProofServ->GetUser(), type, msg);
   } else {
      fprintf(stderr, "%s in <%s>: %s\n", type, location, msg);
      bp = Form("%s:%s:%s:%s", gProofServ->GetUser(), type, location, msg);
   }
   fflush(stderr);
   gSystem->Syslog(loglevel, bp);

   if (abort) {
      gProofServ->GetSocket()->Send(kPROOF_FATAL);

      fprintf(stderr, "aborting\n");
      fflush(stderr);
      gSystem->StackTrace();
      gSystem->Abort();
   }
}

//----- Interrupt signal handler -----------------------------------------------
//______________________________________________________________________________
class TProofInterruptHandler : public TSignalHandler {
public:
   TProofInterruptHandler() : TSignalHandler(kSigUrgent, kFALSE) { }
   Bool_t  Notify();
};

//______________________________________________________________________________
Bool_t TProofInterruptHandler::Notify()
{
   gProofServ->HandleUrgentData();
   if (TROOT::Initialized()) {
      Throw(GetSignal());
   }
   return kTRUE;
}

//----- Socket Input handler --------------------------------------------
//______________________________________________________________________________
class TSocketInputHandler : public TFileHandler {
public:
   TSocketInputHandler(int fd) : TFileHandler(fd, 1) { }
   Bool_t Notify();
   Bool_t ReadNotify() { return Notify(); }
};

//______________________________________________________________________________
Bool_t TSocketInputHandler::Notify()
{
   gProofServ->HandleSocketInput();
   return kTRUE;
}


ClassImp(TProofServ)

//______________________________________________________________________________
TProofServ::TProofServ(int *argc, char **argv)
       : TApplication("proofserv", argc, argv)
{
   // Create an application environment. The TProofServ environment provides
   // an eventloop via inheritance of TApplication.

   // Make sure all registered dictionaries have been initialized
   gInterpreter->InitializeDictionaries();

   // abort on kSysError's or higher and set error handler
   gErrorAbortLevel = kSysError;
   SetErrorHandler(ProofErrorHandler);

   fNcmd        = 0;
   fInterrupt   = kFALSE;
   fLogLevel    = 1;
   fRealTime    = 0.0;
   fCpuTime     = 0.0;
   fSocket      = new TSocket(0);

   Setup();
   RedirectOutput();

   // Load user functions
   const char *logon;
   logon = gEnv->GetValue("Proof.Load", (char*)0);
   if (logon && !gSystem->AccessPathName(logon, kReadPermission))
      ProcessLine(Form(".L %s",logon));

   // Execute logon macro
   logon = gEnv->GetValue("Proof.Logon", (char*)0);
   if (logon && !NoLogOpt() && !gSystem->AccessPathName(logon, kReadPermission))
      ProcessFile(logon);

   gInterpreter->SaveContext();
   gInterpreter->SaveGlobalsContext();

   // Install interrupt and terminal input handlers
   TProofInterruptHandler *ih = new TProofInterruptHandler;
   gSystem->AddSignalHandler(ih);

   TSocketInputHandler *th = new TSocketInputHandler(0);
   gSystem->AddFileHandler(th);

   gProofServ = this;
}

//______________________________________________________________________________
TProofServ::~TProofServ()
{
   // Cleanup. Not really necessary since after this dtor there is no
   // live anyway.

   delete fSocket;
}

//______________________________________________________________________________
TObject *TProofServ::Get(const char *namecycle)
{
   // Get object with name "name;cycle" (e.g. "aap;2") from master or client.
   // This method is called by TDirectory::Get() in case the object can not
   // be found locally.

   fSocket->Send(namecycle, kPROOF_GETOBJECT);

   TMessage *mess;
   if (fSocket->Recv(mess) < 0)
      return 0;

   TObject *idcur = 0;
   if (mess->What() == kMESS_OBJECT)
      idcur = mess->ReadObject(mess->GetClass());
   delete mess;

   return idcur;
}

//______________________________________________________________________________
void TProofServ::GetLimits(Int_t dim, Int_t nentries, Int_t *nbins, Float_t *vmin, Float_t *vmax)
{
   // Get limits of histogram from master. This method is called by
   // TTree::TakeEstimate().

   TMessage mess(kPROOF_LIMITS);

   mess << dim << nentries << nbins[0] << vmin[0] << vmax[0];

   if (dim == 2)
      mess << nbins[1] << vmin[1] << vmax[1];

   if (dim == 3)
      mess << nbins[2] << vmin[2] << vmax[2];

   fSocket->Send(mess);

   TMessage *answ;
   if (fSocket->Recv(answ) != -1) {
      (*answ) >> nbins[0] >> vmin[0] >> vmax[0];

      if (dim == 2)
         (*answ) >> nbins[1] >> vmin[1] >> vmax[1];

      if (dim == 3)
         (*answ) >> nbins[2] >> vmin[2] >> vmax[2];

      delete answ;
   }
}

//______________________________________________________________________________
Bool_t TProofServ::GetNextPacket(Int_t &nentries, Stat_t &firstentry)
{
   // Get next range of entries to be processed on this server.

   fSocket->Send(kPROOF_GETPACKET);

   TMessage *mess;
   if (fSocket->Recv(mess) < 0)
      return kFALSE;

   (*mess) >> nentries >> firstentry >> fEntriesProcessed;

   if (nentries == -1)
      return kFALSE;
   return kTRUE;
}

//______________________________________________________________________________
void TProofServ::GetOptions(int *argc, char **argv)
{
   // Get and handle command line options.

   for (int i = 1; i < *argc; i++) {
      if (!strcmp(argv[i], "proofserv") || !strcmp(argv[i], "proofslave")) {
         fService = argv[i];
         fMasterServ = kTRUE;
         if (!strcmp(argv[i], "proofslave")) fMasterServ = kFALSE;
      } else {
         fConfDir = argv[i];
      }
   }
}

//______________________________________________________________________________
void TProofServ::HandleSocketInput()
{
   // Handle input coming from the client or from the master server.

   static TStopwatch timer;

   TMessage *mess;
   char      str[2048];
   Int_t     what;

   if (fSocket->Recv(mess) < 0)
      return;                    // do something more intelligent here

   what = mess->What();

   timer.Start();
   fNcmd++;

   switch (what) {

      case kMESS_CINT:
         mess->ReadString(str, sizeof(str));
         if (fLogLevel > 1) printf("Processing: %s...\n", str);
         //gSystem->Syslog(kLogInfo, "%s", str);
         ProcessLine(str);
         SendLogFile();
         break;

      case kMESS_STRING:
         mess->ReadString(str, sizeof(str));
         break;

      case kMESS_OBJECT:
         mess->ReadObject(mess->GetClass());
         break;

      case kPROOF_GROUPVIEW:
         mess->ReadString(str, sizeof(str));
         sscanf(str, "%d %d", &fGroupId, &fGroupSize);
         break;

      case kPROOF_LOGLEVEL:
         mess->ReadString(str, sizeof(str));
         sscanf(str, "%d", &fLogLevel);
         break;

      case kPROOF_PING:
         // do nothing
         break;

      case kPROOF_RESET:
         mess->ReadString(str, sizeof(str));
         Reset(str);
         break;

      case kPROOF_STATUS:
         SendStatus();
         break;

      case kPROOF_STOP:
         Terminate(0);
         break;

      case kPROOF_TREEDRAW:
         mess->ReadString(str, sizeof(str));
         {
            Int_t maxv, est;
            char name[64];
            sscanf(str, "%s %d %d", name, &maxv, &est);
            TTree *t = (TTree*)gDirectory->Get(name);
            if (t) {
               t->SetMaxVirtualSize(maxv);
               t->SetEstimate(est);
            }
         }
         break;

      default:
         Error("HandleSocketInput", "unknown command");
         break;
   }

   fRealTime += (Float_t)timer.RealTime();
   fCpuTime  += (Float_t)timer.CpuTime();

   delete mess;
}

//______________________________________________________________________________
void TProofServ::HandleUrgentData()
{
   // Handle Out-Of-Band data sent by the master or client.

   char  oob_byte;
   int   n, nch, wasted = 0;

   const int kBufSize = 1024;
   char waste[kBufSize];

   if (fLogLevel > 5)
      printf("HandleUrgentData()...");

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
            Error("HandleUrgentData", "error receiving waste");
            break;
         }
         wasted = 1;
      } else {
         Error("HandleUrgentData", "error receiving OOB");
         return;
      }
   }

   if (fLogLevel > 5)
      printf("got OOB byte: %d\n", oob_byte);

   switch (oob_byte) {

      case TProof::kHardInterrupt:
         if (IsMaster())
            gSystem->Syslog(kLogInfo, "*** Master: Hard Interrupt");
         else
            gSystem->Syslog(kLogInfo, Form("*** Slave %d: Hard Interrupt", fOrdinal));

         // If master server, propagate interrupt to slaves
         if (IsMaster() && gProof)
            gProof->Interrupt(TProof::kHardInterrupt);

         // Flush input socket
         while (1) {
            int atmark;

            fSocket->GetOption(kAtMark, atmark);

            if (atmark) {
               // Send the OOB byte back so that the client knows where
               // to stop flushing its input stream of obsolete messages
               n = fSocket->SendRaw(&oob_byte, 1, kOob);
               if (n <= 0)
                  Error("HandleUrgentData", "error sending OOB");
               break;
            }

            // find out number of bytes to read before atmark
            fSocket->GetOption(kBytesToRead, nch);
            if (nch == 0) {
               gSystem->Sleep(1000);
               continue;
            }

            if (nch > kBufSize) nch = kBufSize;
            n = fSocket->RecvRaw(waste, nch);
            if (n <= 0) {
               Error("HandleUrgentData", "error receiving waste (2)");
               break;
            }
         }

         break;

      case TProof::kSoftInterrupt:
         if (IsMaster())
            gSystem->Syslog(kLogInfo, "Master: Soft Interrupt");
         else
            gSystem->Syslog(kLogInfo, Form("Slave %d: Soft Interrupt", fOrdinal));

         // If master server, propagate interrupt to slaves
         if (IsMaster() && gProof)
            gProof->Interrupt(TProof::kSoftInterrupt);

         if (wasted) {
            Error("HandleUrgentData", "soft interrupt flushed stream");
            break;
         }

         Interrupt();

         break;

      case TProof::kShutdownInterrupt:
         if (IsMaster())
            gSystem->Syslog(kLogInfo, "Master: Shutdown Interrupt");
         else
            gSystem->Syslog(kLogInfo, Form("Slave %d: Shutdown Interrupt", fOrdinal));

         // If master server, propagate interrupt to slaves
         if (IsMaster() && gProof)
            gProof->Interrupt(TProof::kShutdownInterrupt);

         Terminate(0);  // will not return from here....

         break;

      default:
         Error("HandleUrgentData", "unexpected OOB byte");
         break;
   }

   SendLogFile();
}

//______________________________________________________________________________
void TProofServ::Print(Option_t *)
{
   // Print status of slave server.

   Printf("This is slave %s", gSystem->HostName());
}

//______________________________________________________________________________
void TProofServ::RedirectOutput()
{
   // Redirect stdout to a log file. This log file will be flushed to the
   // client or master after each command.

   // Duplicate the initial (0) socket, this will yield a socket with a
   // descriptor >0, which will free id=0 for stdout.
   int isock;
   if ((isock = dup(fSocket->GetDescriptor())) < 0)
      SysError("RedirectOutput", "could not duplicate output socket");
   fSocket->SetDescriptor(isock);

   // Remove all previous log files and create new log files.
   char logfile[512];

   if (IsMaster()) {
      gSystem->Exec(Form("/bin/rm -f %s/proof_*.log", fLogDir.Data()));
      sprintf(logfile, "%s/proof_%d.log", fLogDir.Data(), gSystem->GetPid());
   } else {
      gSystem->Exec(Form("/bin/rm -f %s/proofs%d_*.log", fLogDir.Data(),
                    fOrdinal));
      sprintf(logfile, "%s/proofs%d_%d.log", fLogDir.Data(), fOrdinal,
              gSystem->GetPid());
   }

   if ((freopen(logfile, "w", stdout)) == 0)
      SysError("RedirectOutput", "could not freopen stdout");

   if ((dup2(fileno(stdout), fileno(stderr))) < 0)
      SysError("RedirectOutput", "could not redirect stderr");

   if ((fLogFile = fopen(logfile, "r")) == 0)
      SysError("RedirectOutput", "could not open logfile");

#if 0
   // Send message of the day to the client.
   if (IsMaster()) CatMotd();
#endif
}

//______________________________________________________________________________
void TProofServ::Reset(const char *dir)
{
   // Reset PROOF environment to be ready for execution of next command.

   // First go to new directory.
   gDirectory->cd(dir);

   // Clear interpreter environment.
   gROOT->Reset();

   // Make sure current directory is empty (don't delete anything when
   // we happen to be in the ROOT memory only directory!?)
   if (gDirectory != gROOT) {
      TObject *obj;
      TIter next(gDirectory->GetList());
      while ((obj = next()))
         if (!obj->InheritsFrom(TTree::Class())) {
            gDirectory->GetList()->Remove(obj);
            delete obj;
         }
   }
}

//______________________________________________________________________________
void TProofServ::Run(Bool_t retrn)
{
   // Main server eventloop.

   TApplication::Run(retrn);
}

//______________________________________________________________________________
void TProofServ::SendLogFile()
{
   // Send log file to master.

   // Determine the number of bytes left to be read from the log file.
   fflush(stdout);

   off_t ltot, lnow;
   Int_t left;

   ltot = lseek(fileno(stdout),   (off_t) 0, SEEK_END);
   lnow = lseek(fileno(fLogFile), (off_t) 0, SEEK_CUR);
   left = Int_t(ltot - lnow);

   if (left <= 0)
      fSocket->Send(kPROOF_LOGDONE);
   else {
      fSocket->Send(kPROOF_LOGFILE);

      while (left > 0) {
         char line[256];

         if (fgets(line, sizeof(line), fLogFile) == 0) {
            left = 0;
         } else {
            left -= strlen(line);
            fSocket->Send(line);
         }
         if (!left)
            fSocket->Send(kPROOF_LOGDONE);
      }
   }
}

//______________________________________________________________________________
void TProofServ::SendStatus()
{
   // Send status of slave server to master or client.

   char str[64];

   sprintf(str, "%g %.3f %.3f", TFile::GetFileBytesRead(), fRealTime, fCpuTime);
   fSocket->Send(str, kPROOF_STATUS);
}

//______________________________________________________________________________
void TProofServ::Setup()
{
   // Print the ProofServ logo on standard output.

   char str[512];

   if (IsMaster()) {
      sprintf(str, "**** Welcome to the Proof server @ %s ****", gSystem->HostName());
   } else {
      sprintf(str, "*** Proof slave server @ %s started ****", gSystem->HostName());
   }
   fSocket->Send(str);

   fSocket->Recv(str, sizeof(str));

   char user[16], vers[16], userpass[64], curdir[256];
   if (IsMaster()) {
      sscanf(str, "%s %s %s %d", user, vers, userpass, &fProtocol);
      fUserPass = userpass;
   } else {
      sscanf(str, "%s %s %s %d %d %d", user, vers, curdir, &fProtocol,
             &fMasterPid, &fOrdinal);
   }
   fUser    = user;
   fVersion = vers;

   // deny write access for group and world
   gSystem->Umask(022);

   if (IsMaster())
      gSystem->Openlog("proofserv", kLogPid | kLogCons, kLogLocal6);
   else
      gSystem->Openlog("proofslave", kLogPid | kLogCons, kLogLocal7);

   // Set $HOME and $PATH. The HOME directory was already set to the
   // user's home directory by proofd.
   gSystem->Setenv("HOME", gSystem->HomeDirectory());
#ifdef R__UNIX
   gSystem->Setenv("PATH", "/bin:/usr/bin:/usr/contrib/bin:/usr/local/bin");
#endif

   // set the working directory to $HOME/proof
   char workdir[256];
   sprintf(workdir, "%s/proof", gSystem->HomeDirectory());

   if (gSystem->AccessPathName(workdir)) {
      gSystem->MakeDirectory(workdir);
      if (!gSystem->ChangeDirectory(workdir)) {
         SysError("Setup", "can not change working directory");
      }
   } else {
      if (!gSystem->ChangeDirectory(workdir)) {
         gSystem->Unlink(workdir);
         gSystem->MakeDirectory(workdir);
         if (!gSystem->ChangeDirectory(workdir)) {
            SysError("Setup", "can not change working directory");
         }
      }
   }

   // for master server the work and log directory are the same
   fLogDir = workdir;

   // Slave servers set their work directory to the work directory of the
   // master server.
   if (!IsMaster()) {
      if (!gSystem->ChangeDirectory(curdir))
         SysError("Setup", "can not change to the current directory");
   }

   // Incoming OOB should generate a SIGURG
   fSocket->SetOption(kProcessGroup, gSystem->GetPid());

   // Send packages of immediately to reduce latency
   fSocket->SetOption(kNoDelay, 1);
}

//______________________________________________________________________________
void TProofServ::Terminate(int status)
{
   // Terminate the proof server.

   gSystem->Exit(status);
}

//______________________________________________________________________________
Bool_t TProofServ::IsActive()
{
   // Static function that returns kTRUE in case we are a PROOF server.

   return gProofServ ? kTRUE : kFALSE;
}

//______________________________________________________________________________
TProofServ *TProofServ::This()
{
   // Static function returning pointer to global object gProofServ.
   // Mainly for use via CINT, where the gProofServ symbol might be
   // deleted from the symbol table.

   return gProofServ;
}

