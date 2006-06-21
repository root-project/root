// @(#)root/proofx:$Name:  $:$Id: TXProofServ.cxx,v 1.8 2006/06/05 22:51:14 rdm Exp $
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
// TXProofServ                                                          //
//                                                                      //
// TXProofServ is the XRD version of the PROOF server. It differs from  //
// TXProofServ only for the underlying connection technology             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifdef HAVE_CONFIG
#include "config.h"
#endif

#include "RConfig.h"
#include "Riostream.h"

#ifdef WIN32
   #include <io.h>
   typedef long off_t;
#endif
#include <errno.h>
#include <time.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <netinet/in.h>

#if (defined(__FreeBSD__) && (__FreeBSD__ < 4)) || \
    (defined(__APPLE__) && (!defined(MAC_OS_X_VERSION_10_3) || \
     (MAC_OS_X_VERSION_MAX_ALLOWED < MAC_OS_X_VERSION_10_3)))
#include <sys/file.h>
#define lockf(fd, op, sz)   flock((fd), (op))
#ifndef F_LOCK
#define F_LOCK             (LOCK_EX | LOCK_NB)
#endif
#ifndef F_ULOCK
#define F_ULOCK             LOCK_UN
#endif
#endif

#include "TXProofServ.h"
#include "TDSetProxy.h"
#include "TEnv.h"
#include "TError.h"
#include "TException.h"
#include "TFile.h"
#include "TInterpreter.h"
#include "TKey.h"
#include "TMessage.h"
#include "TPerfStats.h"
#include "TProofDebug.h"
#include "TProof.h"
#include "TProofLimitsFinder.h"
#include "TProofPlayer.h"
#include "TProofQueryResult.h"
#include "TRegexp.h"
#include "TROOT.h"
#include "TSelector.h"
#include "TSlave.h"
#include "TSocket.h"
#include "TStopwatch.h"
#include "TSysEvtHandler.h"
#include "TSystem.h"
#include "TTimeStamp.h"
#include "TUrl.h"
#include "TTree.h"
#include "TPluginManager.h"
#include "TObjString.h"
#include "TXSocketHandler.h"
#include "TXUnixSocket.h"
#include "compiledata.h"
#include "TProofResourcesStatic.h"
#include "TProofNodeInfo.h"
#include "TTimer.h"
#include "TMutex.h"

#include "XProofProtocol.h"

#include <XrdClient/XrdClientConst.hh>
#include <XrdClient/XrdClientEnv.hh>

#ifndef R__WIN32
const char* const kCP     = "/bin/cp -f";
const char* const kRM     = "/bin/rm -rf";
const char* const kLS     = "/bin/ls -l";
const char* const kUNTAR  = "%s -c %s/%s | (cd %s; tar xf -)";
const char* const kGUNZIP = "gunzip";
#else
const char* const kCP     = "copy";
const char* const kRM     = "delete";
const char* const kLS     = "dir";
const char* const kUNTAR  = "...";
const char* const kGUNZIP = "gunzip";
#endif

// debug hook
static volatile Int_t gProofServDebug = 1;

//----- Interrupt signal handler -----------------------------------------------
//______________________________________________________________________________
class TXProofServInterruptHandler : public TSignalHandler {
   TXProofServ  *fServ;
public:
   TXProofServInterruptHandler(TXProofServ *s)
      : TSignalHandler(kSigUrgent, kFALSE) { fServ = s; }
   Bool_t  Notify();
};

//______________________________________________________________________________
Bool_t TXProofServInterruptHandler::Notify()
{
   fServ->HandleUrgentData();
   if (TROOT::Initialized()) {
      Throw(GetSignal());
   }
   return kTRUE;
}

//----- SigPipe signal handler -------------------------------------------------
//______________________________________________________________________________
class TXProofServSigPipeHandler : public TSignalHandler {
   TXProofServ  *fServ;
public:
   TXProofServSigPipeHandler(TXProofServ *s) : TSignalHandler(kSigPipe, kFALSE)
      { fServ = s; }
   Bool_t  Notify();
};

//______________________________________________________________________________
Bool_t TXProofServSigPipeHandler::Notify()
{
   fServ->HandleSigPipe();
   return kTRUE;
}

//----- Termination signal handler ---------------------------------------------
//______________________________________________________________________________
class TXProofServTerminationHandler : public TSignalHandler {
   TXProofServ  *fServ;
public:
   TXProofServTerminationHandler(TXProofServ *s)
      : TSignalHandler(kSigTermination, kFALSE) { fServ = s; }
   Bool_t  Notify();
};

//______________________________________________________________________________
Bool_t TXProofServTerminationHandler::Notify()
{
   Printf("TXProofServTerminationHandler::Notify: wake up!");

   fServ->HandleTermination();
   return kTRUE;
}

//----- Seg violation signal handler ---------------------------------------------
//______________________________________________________________________________
class TXProofServSegViolationHandler : public TSignalHandler {
   TXProofServ  *fServ;
public:
   TXProofServSegViolationHandler(TXProofServ *s)
      : TSignalHandler(kSigSegmentationViolation, kFALSE) { fServ = s; }
   Bool_t  Notify();
};

//______________________________________________________________________________
Bool_t TXProofServSegViolationHandler::Notify()
{
   Printf("**** ");
   Printf("**** Segmentation violation: terminating ****");
   Printf("**** ");
   fServ->HandleTermination();
   return kTRUE;
}

//----- Input handler for messages from parent or master -----------------------
//______________________________________________________________________________
class TXProofServInputHandler : public TFileHandler {
   TXProofServ  *fServ;
public:
   TXProofServInputHandler(TXProofServ *s, Int_t fd) : TFileHandler(fd, 1)
      { fServ = s; }
   Bool_t Notify();
   Bool_t ReadNotify() { return Notify(); }
};

//______________________________________________________________________________
Bool_t TXProofServInputHandler::Notify()
{
   fServ->HandleSocketInput();
   // This request has been completed: remove the client ID from the pipe
   ((TXUnixSocket *) fServ->GetSocket())->RemoveClientID();
   return kTRUE;
}

ClassImp(TXProofServ)

// Hook to the constructor. This is needed to avoid using the plugin manager
// which may create problems in multi-threaded environments.
extern "C" {
   TApplication *GetTXProofServ(Int_t *argc, char **argv, FILE *flog)
   { return new TXProofServ(argc, argv, flog); }
}

//______________________________________________________________________________
void TXProofServ::CreateServer()
{
   // Finalize the server setup. If master, create the TProof instance to talk
   // the worker or submaster nodes.

   if (gProofDebugLevel > 0)
      Info("CreateServer", "starting server creation");

   // Get file descriptor for log file
   if (fLogFile) {
      // Use the file already open by pmain
      if ((fLogFileDes = fileno(fLogFile)) < 0) {
         Error("CreateServer", "resolving the log file description number");
         exit(1);
      }
   }

   // Global location string in TXSocket
   TXSocket::fgLoc = (IsMaster()) ? "master" : "slave" ;

   // Set debug level in XrdClient
   EnvPutInt(NAME_DEBUG, gEnv->GetValue("XNet.Debug", 0));

   // Get socket to be used to call back our xpd
   const char *sockpath = 0;
   if (!(sockpath = gSystem->Getenv("ROOTOPENSOCK"))) {
     Error("CreateServer", "Socket setup by xpd undefined");
     exit(1);
   }

   // If test session, just send the protocol version and exit
   if (Argc() > 3 && !strcmp(Argv(3), "test")) {
      Int_t fpw = (Int_t) strtol(sockpath, 0, 10);
      int proto = htonl(kPROOF_Protocol);
      if (write(fpw, &proto, sizeof(proto)) != sizeof(proto)) {
         Error("CreateServer", "test: sending protocol number");
         exit(1);
      }
      exit(0);
   }

   // Get the sessions ID
   const char *sessID = 0;
   if (!(sessID = gSystem->Getenv("ROOTSESSIONID"))) {
     Error("CreateServer", "Session ID undefined");
     exit(1);
   }
   Int_t psid = (Int_t) strtol(sessID, 0, 10);

   // Call back the server
   fSocket = new TXUnixSocket(sockpath, psid);
   if (!fSocket || !(fSocket->IsValid())) {
      Error("CreateServer", "Failed to open connection to XrdProofd coordinator");
      exit(1);
   }

   // Get socket descriptor
   Int_t sock = fSocket->GetDescriptor();

   // Get the client ID
   const char *clntID = 0;
   if (!(clntID = gSystem->Getenv("ROOTCLIENTID"))) {
     Error("CreateServer", "Client ID undefined");
     SendLogFile();
     exit(1);
   }
   Int_t cid = (Int_t) strtol(clntID, 0, 10);
   ((TXSocket *)fSocket)->SetClientID(cid);

   // debug hooks
   if (IsMaster()) {
      // wait (loop) in master to allow debugger to connect
      if (gEnv->GetValue("Proof.GdbHook",0) == 1) {
         while (gProofServDebug)
            ;
      }
   } else {
      // wait (loop) in slave to allow debugger to connect
      if (gEnv->GetValue("Proof.GdbHook",0) == 2) {
         while (gProofServDebug)
            ;
      }
   }

   if (gProofDebugLevel > 0)
      Info("CreateServer", "Service: %s, ConfDir: %s, IsMaster: %d",
           fService.Data(), fConfDir.Data(), (Int_t)fMasterServ);

   Setup();

   if (!fLogFile) {
      RedirectOutput();
      // If for some reason we failed setting a redirection fole for the logs
      // we cannot continue
      if (!fLogFile || (fLogFileDes = fileno(fLogFile)) < 0) {
         SendLogFile(-98);
         Terminate(0);
      }
   }

   // Send message of the day to the client
   if (IsMaster()) {
      if (CatMotd() == -1) {
         SendLogFile(-99);
         Terminate(0);
      }
   } else {
      THLimitsFinder::SetLimitsFinder(new TProofLimitsFinder);
   }

   // Everybody expects iostream to be available, so load it...
   ProcessLine("#include <iostream>", kTRUE);
   ProcessLine("#include <_string>",kTRUE); // for std::string iostream.

   // Allow the usage of ClassDef and ClassImp in interpreted macros
   ProcessLine("#include <RtypesCint.h>", kTRUE);

   // Disallow the interpretation of Rtypes.h, TError.h and TGenericClassInfo.h
   ProcessLine("#define ROOT_Rtypes 0", kTRUE);
   ProcessLine("#define ROOT_TError 0", kTRUE);
   ProcessLine("#define ROOT_TGenericClassInfo 0", kTRUE);

   // Load user functions
   const char *logon;
   logon = gEnv->GetValue("Proof.Load", (char *)0);
   if (logon) {
      char *mac = gSystem->Which(TROOT::GetMacroPath(), logon, kReadPermission);
      if (mac)
         ProcessLine(Form(".L %s", logon), kTRUE);
      delete [] mac;
   }

   // Execute logon macro
   logon = gEnv->GetValue("Proof.Logon", (char *)0);
   if (logon && !NoLogOpt()) {
      char *mac = gSystem->Which(TROOT::GetMacroPath(), logon, kReadPermission);
      if (mac)
         ProcessFile(logon);
      delete [] mac;
   }

   // Save current interpreter context
   gInterpreter->SaveContext();
   gInterpreter->SaveGlobalsContext();

   // Install interrupt and message input handlers
   fInterruptHandler = new TXProofServInterruptHandler(this);
   gSystem->AddSignalHandler(fInterruptHandler);
   fInputHandler =
      TXSocketHandler::GetSocketHandler(new TXProofServInputHandler(this, sock), fSocket);
   gSystem->AddFileHandler(fInputHandler);

   // Set the this as reference of this socket
   ((TXSocket *)fSocket)->fReference = this;
   // Set this has handler
   ((TXSocket *)fSocket)->fHandler = this;

   // if master, start slave servers
   if (IsMaster()) {
      TString master = "proof://__master__";

      // Add port, if defined
      const char *port = 0;
      if ((port = (char *) gSystem->Getenv("ROOTXPDPORT"))) {
         master += ":";
         master += port;
      }

      // Make sure that parallel startup via threads is not active
      // (it is broken for xpd because of the locks on gCINTMutex)
      gEnv->SetValue("Proof.ParallelStartup", 0);

      // Get plugin manager to load appropriate TVirtualProof from
      TPluginManager *pm = gROOT->GetPluginManager();
      if (!pm) {
         Error("CreateServer", "no plugin manager found");
         SendLogFile(-99);
         Terminate(0);
      }

      // Find the appropriate handler
      TPluginHandler *h = pm->FindHandler("TVirtualProof", fConfFile);
      if (!h) {
         Error("CreateServer", "no plugin found for TVirtualProof with a"
                             " config file of '%s'", fConfFile.Data());
         SendLogFile(-99);
         Terminate(0);
      }

      // load the plugin
      if (h->LoadPlugin() == -1) {
         Error("CreateServer", "plugin for TVirtualProof could not be loaded");
         SendLogFile(-99);
         Terminate(0);
      }

      // make instance of TProof
      fProof = reinterpret_cast<TProof*>(h->ExecPlugin(5, master.Data(),
                                                          fConfFile.Data(),
                                                          fConfDir.Data(),
                                                          fLogLevel,
                                                          fSessionTag.Data()));
      if (!fProof || !fProof->IsValid()) {
         Error("CreateServer", "plugin for TVirtualProof could not be executed");
         delete fProof;
         fProof = 0;
         SendLogFile(-99);
         Terminate(0);
      }
      // Find out if we are a master in direct contact only with workers
      fEndMaster = fProof->IsEndMaster();

   }
   SendLogFile();
}

//______________________________________________________________________________
TXProofServ::~TXProofServ()
{
   // Cleanup. Not really necessary since after this dtor there is no
   // live anyway.

   delete fSocket;
}

//______________________________________________________________________________
void TXProofServ::HandleUrgentData()
{
   // Handle high priority data sent by the master or client.

   // Get interrupt
   Int_t iLev = ((TXSocket *)fSocket)->GetInterrupt();
   if (iLev < 0) {
      Error("HandleUrgentData", "error receiving interrupt");
      return;
   }

   PDB(kGlobal, 5)
      Info("HandleUrgentData", "got interrupt: %d\n", iLev);

   if (fProof)
      fProof->SetActive();

   switch (iLev) {

      case TProof::kPing:
         PDB(kGlobal, 5)
            Info("HandleUrgentData", "*** Ping");

         // If master server, propagate interrupt to slaves
         if (IsMaster()) {
            Int_t nbad = fProof->fActiveSlaves->GetSize()-fProof->Ping();
            if (nbad > 0) {
               Info("HandleUrgentData","%d slaves did not reply to ping",nbad);
            }
         }

         // Reply to ping
         ((TXSocket *)fSocket)->Ping();

         // Send log with result of ping
         if (IsMaster())
            SendLogFile();

         break;

      case TProof::kHardInterrupt:
         Info("HandleUrgentData", "*** Hard Interrupt");

         // If master server, propagate interrupt to slaves
         if (IsMaster())
            fProof->Interrupt(TProof::kHardInterrupt);

         // Flush input socket
         ((TXSocket *)fSocket)->Flush();

         break;

      case TProof::kSoftInterrupt:
         Info("HandleUrgentData", "Soft Interrupt");

         // If master server, propagate interrupt to slaves
         if (IsMaster())
            fProof->Interrupt(TProof::kSoftInterrupt);

         Interrupt();

         break;


      case TProof::kShutdownInterrupt:
         Info("HandleUrgentData", "Shutdown Interrupt");

         HandleTermination();

         break;

      default:
         Error("HandleUrgentData", "unexpected type");
         break;
   }

   SendLogFile();

   if (fProof) fProof->SetActive(kFALSE);
}

//______________________________________________________________________________
void TXProofServ::HandleSigPipe()
{
   // Called when the client is not alive anymore; terminate the session.

   // If master server, propagate interrupt to slaves
   // (shutdown interrupt send internally).
   if (IsMaster())
      fProof->Close("S");

   Terminate(0);  // will not return from here....
}

//______________________________________________________________________________
void TXProofServ::HandleTermination()
{
   // Called when the client is not alive anymore; terminate the session.

   // If master server, propagate interrupt to slaves
   // (shutdown interrupt send internally).
   if (IsMaster()) {
      // If not idle, try first to stop processing
      if (!fIdle) {
         // Remove pending requests
         fWaitingQueries->Delete();
         // Do not wait for ever, but al least 20 seconds
         Long_t timeout = gEnv->GetValue("Proof.ShutdownTimeout", 60);
         timeout = (timeout > 20) ? timeout : 20;
         // Processing will be aborted
         fProof->StopProcess(kTRUE, (Long_t) (timeout / 2));
         // Receive end-of-processing messages, but do not wait for ever
         fProof->Collect(TProof::kActive, timeout);
         // Still not idle
         if (!fIdle)
            Warning("HandleTermination","processing could not be stopped");
      }
      // Close the session
      fProof->Close("S");
   }

   // Close link with coordinator
   ((TXSocket *)fSocket)->SetSessionID(-1);
   fSocket->Close();

   Terminate(0);  // will not return from here....
}

//______________________________________________________________________________
void TXProofServ::Setup()
{
   // Print the ProofServ logo on standard output.

   char str[512];

   if (IsMaster()) {
      sprintf(str, "**** Welcome to the PROOF server @ %s ****", gSystem->HostName());
   } else {
      sprintf(str, "**** PROOF worker server @ %s started ****", gSystem->HostName());
   }

   if (fSocket->Send(str) != 1+static_cast<Int_t>(strlen(str))) {
      Error("Setup", "failed to send proof server startup message");
      SendLogFile();
      gSystem->Exit(1);
   }

   // Get client protocol
   if (!gSystem->Getenv("ROOTPROOFCLNTVERS")) {
      Error("Setup", "remote proof protocol missing");
      SendLogFile();
      gSystem->Exit(1);
   }
   fProtocol = atoi(gSystem->Getenv("ROOTPROOFCLNTVERS"));

   // The local user
   UserGroup_t *pw = gSystem->GetUserInfo();
   if (pw) {
      fUser = pw->fUser;
      delete pw;
   }

   // Work dir and ...
   if (IsMaster())
      if (gSystem->Getenv("ROOTPROOFCFGFILE"))
         fConfFile = gSystem->Getenv("ROOTPROOFCFGFILE");
   if (gSystem->Getenv("ROOTPROOFWORKDIR"))
      fWorkDir = gSystem->Getenv("ROOTPROOFWORKDIR");
   else
      fWorkDir = kPROOF_WorkDir;

   // goto to the main PROOF working directory
   char *workdir = gSystem->ExpandPathName(fWorkDir.Data());
   fWorkDir = workdir;
   delete [] workdir;
   if (gProofDebugLevel > 0)
      Info("Setup", "working directory set to %s", fWorkDir.Data());

   // deny write access for group and world
   gSystem->Umask(022);

   if (IsMaster())
      gSystem->Openlog("proofserv", kLogPid | kLogCons, kLogLocal5);
   else
      gSystem->Openlog("proofslave", kLogPid | kLogCons, kLogLocal6);

   // Set $HOME and $PATH. The HOME directory was already set to the
   // user's home directory by proofd.
   gSystem->Setenv("HOME", gSystem->HomeDirectory());

#ifdef R__UNIX
   TString bindir;
# ifdef ROOTBINDIR
   bindir = ROOTBINDIR;
# else
   bindir = gSystem->Getenv("ROOTSYS");
   if (!bindir.IsNull()) bindir += "/bin";
# endif
# ifdef COMPILER
   TString compiler = COMPILER;
   compiler.Remove(0, compiler.Index("is ") + 3);
   compiler = gSystem->DirName(compiler);
   if (!bindir.IsNull()) bindir += ":";
   bindir += compiler;
#endif
   if (!bindir.IsNull()) bindir += ":";
   bindir += "/bin:/usr/bin:/usr/local/bin";
   gSystem->Setenv("PATH", bindir);
#endif

   if (gSystem->AccessPathName(fWorkDir)) {
      gSystem->mkdir(fWorkDir, kTRUE);
      if (!gSystem->ChangeDirectory(fWorkDir)) {
         Error("Setup", "can not change to PROOF directory %s",
               fWorkDir.Data());
         SendLogFile();
         gSystem->Exit(1);
      }
   } else {
      if (!gSystem->ChangeDirectory(fWorkDir)) {
         gSystem->Unlink(fWorkDir);
         gSystem->mkdir(fWorkDir, kTRUE);
         if (!gSystem->ChangeDirectory(fWorkDir)) {
            Error("Setup", "can not change to PROOF directory %s",
                     fWorkDir.Data());
            SendLogFile();
            gSystem->Exit(1);
         }
      }
   }

   // check and make sure "cache" directory exists
   fCacheDir = fWorkDir;
   fCacheDir += TString("/") + kPROOF_CacheDir;
   if (gSystem->AccessPathName(fCacheDir))
      gSystem->MakeDirectory(fCacheDir);
   if (gProofDebugLevel > 0)
      Info("Setup", "cache directory set to %s", fCacheDir.Data());
   fCacheLock =
      new TProofLockPath(Form("%s%s",kPROOF_CacheLockFile,fUser.Data()));

   // check and make sure "packages" directory exists
   fPackageDir = fWorkDir;
   fPackageDir += TString("/") + kPROOF_PackDir;
   if (gSystem->AccessPathName(fPackageDir))
      gSystem->MakeDirectory(fPackageDir);
   if (gProofDebugLevel > 0)
      Info("Setup", "package directory set to %s", fPackageDir.Data());
   fPackageLock =
      new TProofLockPath(Form("%s%s",kPROOF_PackageLockFile,fUser.Data()));

   // Get Session tag
   if (!gSystem->Getenv("ROOTPROOFSESSIONTAG")) {
     Error("Setup", "Session tag missing");
     SendLogFile();
     gSystem->Exit(1);
   }
   fSessionTag = gSystem->Getenv("ROOTPROOFSESSIONTAG");
   if (gProofDebugLevel > 0)
      Info("Setup", "session tag is %s", fSessionTag.Data());

   // Get Session dir (sandbox)
   if (!gSystem->Getenv("ROOTPROOFSESSDIR")) {
     Error("Setup", "Session dir missing");
     SendLogFile();
     gSystem->Exit(1);
   }
   fSessionDir = gSystem->Getenv("ROOTPROOFSESSDIR");

   if (gSystem->AccessPathName(fSessionDir)) {
      gSystem->MakeDirectory(fSessionDir);
      if (!gSystem->ChangeDirectory(fSessionDir)) {
         Error("Setup", "can not change to working directory %s",
               fSessionDir.Data());
         SendLogFile();
         gSystem->Exit(1);
      } else {
         gSystem->Setenv("PROOF_SANDBOX", fSessionDir);
      }
   }
   if (gProofDebugLevel > 0)
      Info("Setup", "session dir is %s", fSessionDir.Data());

   // On masters, check and make sure that "queries" and "datasets"
   // directories exist
   if (IsMaster()) {

      // Create 'queries' locker instance and lock it
      fQueryLock = new TProofLockPath(Form("%s%s-%s",
                       kPROOF_QueryLockFile,fSessionTag.Data(),fUser.Data()));

      // Make sure that the 'queries' dir exist
      fQueryDir = fWorkDir;
      fQueryDir += TString("/") + kPROOF_QueryDir;
      if (gSystem->AccessPathName(fQueryDir))
         gSystem->MakeDirectory(fQueryDir);
      else
         ScanPreviousQueries(fQueryDir);
      fQueryDir += TString("/session-") + fSessionTag;
      if (gSystem->AccessPathName(fQueryDir))
         gSystem->MakeDirectory(fQueryDir);
      if (gProofDebugLevel > 0)
         Info("Setup", "queries dir is %s", fQueryDir.Data());
      fQueryLock->Lock();

      // 'datasets'
      fDataSetDir = fWorkDir;
      fDataSetDir += TString("/") + kPROOF_DataSetDir;
      if (gSystem->AccessPathName(fDataSetDir))
         gSystem->MakeDirectory(fDataSetDir);
      if (gProofDebugLevel > 0)
         Info("Setup", "dataset dir is %s", fDataSetDir.Data());

      fDataSetLock =
         new TProofLockPath(Form("%s%s", kPROOF_DataSetLockFile,fUser.Data()));

      // Send session tag to client
      TMessage m(kPROOF_SESSIONTAG);
      m << fSessionTag;
      fSocket->Send(m);

      // ... and to the coordinator to record in the session proxy
      ((TXSocket *)fSocket)->SendCoordinator(TXSocket::kSessionTag, fSessionTag);
   }

   // Send packages off immediately to reduce latency
   fSocket->SetOption(kNoDelay, 1);

   // Check every two hours if client is still alive
   fSocket->SetOption(kKeepAlive, 1);

   // Install SigPipe handler to handle kKeepAlive failure
   gSystem->AddSignalHandler(new TXProofServSigPipeHandler(this));

   // Install Termination handler
   gSystem->AddSignalHandler(new TXProofServTerminationHandler(this));

   // Install seg violation handler
   gSystem->AddSignalHandler(new TXProofServSegViolationHandler(this));

   if (gProofDebugLevel > 0)
      Info("Setup", "successfully completed");
}

//______________________________________________________________________________
void TXProofServ::SendLogFile(Int_t status, Int_t start, Int_t end)
{
   // Send log file to master.
   // If start > -1 send only bytes in the range from start to end,
   // if end <= start send everything from start.

   // Determine the number of bytes left to be read from the log file.
   fflush(stdout);

   off_t ltot, lnow;
   Int_t left;

   ltot = lseek(fileno(stdout),   (off_t) 0, SEEK_END);
   lnow = lseek(fLogFileDes, (off_t) 0, SEEK_CUR);

   Bool_t adhoc = kFALSE;
   if (start > -1) {
      lseek(fLogFileDes, (off_t) start, SEEK_SET);
      if (end <= start || end > ltot)
         end = ltot;
      left = (Int_t)(end - start);
      if (end < ltot)
         left++;
      adhoc = kTRUE;
   } else {
      left = (Int_t)(ltot - lnow);
   }

   if (left > 0) {
      fSocket->Send(left, kPROOF_LOGFILE);

      const Int_t kMAXBUF = 32768;  //16384  //65536;
      char buf[kMAXBUF];
      Int_t wanted = (left > kMAXBUF) ? kMAXBUF : left;
      Int_t len;
      do {
         while ((len = read(fLogFileDes, buf, wanted)) < 0 &&
                TSystem::GetErrno() == EINTR)
            TSystem::ResetErrno();

         if (len < 0) {
            SysError("SendLogFile", "error reading log file");
            break;
         }

         if (end == ltot && len == wanted)
            buf[len-1] = '\n';

         if (fSocket->SendRaw(buf, len, kDontBlock) < 0) {
            SysError("SendLogFile", "error sending log file");
            break;
         }

         // Update counters
         left -= len;
         wanted = (left > kMAXBUF) ? kMAXBUF : left;

      } while (len > 0 && left > 0);
   }

   // Restore initial position if partial send
   if (adhoc)
      lseek(fLogFileDes, lnow, SEEK_SET);

   TMessage mess(kPROOF_LOGDONE);
   if (IsMaster())
      mess << status << (fProof ? fProof->GetParallel() : 0);
   else
      mess << status << (Int_t) 1;

   fSocket->Send(mess);
}

//______________________________________________________________________________
TProofServ::EQueryAction TXProofServ::GetWorkers(TList *workers,
                                                 Int_t & /* prioritychange */)
{
   // Get list of workers to be used from now on.
   // The list must be provide by the caller.

   // Needs a list where to store the info
   if (!workers) {
      Error("GetWorkers", "output list undefined");
      return kQueryStop;
   }

   // If user config files are enabled, check them first
   if (gSystem->Getenv("ROOTUSEUSERCFG")) {
      Int_t pc = 1;
      TProofServ::EQueryAction rc = TProofServ::GetWorkers(workers, pc);
      if (rc == kQueryOK)
         return rc;
   }

   // Send request to the coordinator
   TObjString *os = ((TXSocket *)fSocket)->SendCoordinator(TXSocket::kGetWorkers);

   // The reply contains some information about the master (image, workdir)
   // followed by the information about the workers; the tokens for each node
   // are separated by '&'
   if (os) {
      TObjArray *oa = TString(os->GetName()).Tokenize(TString("&"));
      if (oa) {
         TIter nxos(oa);
         // The master, first
         TObjString *to = (TObjString *) nxos();
         TProofNodeInfo *master = new TProofNodeInfo(to->GetName());
         // Image
         fImage = master->GetImage();
         if (fImage.Length() <= 0) {
            Error("GetWorkers", "no appropriate master line got from coordinator");
            SafeDelete(oa);
            SafeDelete(os);
            SafeDelete(master);
            return kQueryStop;
         }

         // Now the workers
         while ((to = (TObjString *) nxos()))
            workers->Add(new TProofNodeInfo(to->GetName()));

         // Cleanup
         SafeDelete(oa);
         SafeDelete(master);
      }
      // Cleanup
      SafeDelete(os);
   }

   // We are done
   return kQueryOK;
}

//_____________________________________________________________________________
Bool_t TXProofServ::HandleError(const void *)
{
   // Handle error on the input socket

   Printf("TXProofServ::HandleError: %p: got called ...", this);

   // If master server, propagate interrupt to slaves
   // (shutdown interrupt send internally).
   if (IsMaster())
      fProof->Close("S");

   // Close link with coordinator
   ((TXSocket *)fSocket)->SetSessionID(-1);
   fSocket->Close();

   Terminate(0);  // will not return from here....

   Printf("TXProofServ::HandleError: %p: DONE ... ", this);

   // We are done
   return kTRUE;
}

//_____________________________________________________________________________
Bool_t TXProofServ::HandleInput(const void *in)
{
   // Handle asynchronous input on the input socket

   if (gDebug > 2)
      Printf("TXProofServ::HandleInput %p, in: %p", this, in);

   XHandleIn_t *hin = (XHandleIn_t *) in;
   Int_t acod = (hin) ? hin->fInt1 : kXPD_msg;

   // Act accordingly
   if (acod == kXPD_ping || acod == kXPD_interrupt) {
      // Interrupt or Ping
      HandleUrgentData();

   } else if (acod == kXPD_timer) {
      // Shutdown option
      fShutdownWhenIdle = (hin->fInt2 == 2) ? kFALSE : kTRUE;
      if (hin->fInt2 > 0)
         // Setup Shutdown timer 
         SetShutdownTimer(kTRUE, hin->fInt3);
      else
         // Stop Shutdown timer, if any
         SetShutdownTimer(kFALSE);

   } else if (acod == kXPD_urgent) {
      // Get type
      Int_t type = hin->fInt2;
      switch (type) {
      case TXSocket::kStopProcess:
         {
            // Abort or Stop ?
            Bool_t abort = (hin->fInt3 != 0) ? kTRUE : kFALSE;
            // Timeout
            Int_t timeout = hin->fInt4;
            // Act now
            if (fProof)
               fProof->StopProcess(abort, timeout);
            else
               if(fPlayer)
                  fPlayer->StopProcess(abort, timeout);
         }
         break;
      default:
         Info("HandleInput","kXPD_urgent: unknown type: %d", type);
      }

   } else {
      // Standard socket input
      HandleSocketInput();
      // This request has been completed: remove the client ID from the pipe
      ((TXSocket *)fSocket)->RemoveClientID();
   }

   // We are done
   return kTRUE;
}

//______________________________________________________________________________
void TXProofServ::DisableTimeout()
{
   // Disable read timeout on the underlying socket

   if (fSocket)
     ((TXSocket *)fSocket)->DisableTimeout();
}

//______________________________________________________________________________
void TXProofServ::EnableTimeout()
{
   // Enable read timeout on the underlying socket

   if (fSocket)
     ((TXSocket *)fSocket)->EnableTimeout();
}

//______________________________________________________________________________
void TXProofServ::Terminate(Int_t status)
{
   // Terminate the proof server.
   if (fTerminated)
      // Avoid doubling the exit operations
      exit(1);
   fTerminated = kTRUE;

   // Cleanup session directory
   if (status == 0) {
      // make sure we remain in a "connected" directory
      gSystem->ChangeDirectory("/");
      // needed in case fSessionDir is on NFS ?!
      gSystem->MakeDirectory(fSessionDir+"/.delete");
      gSystem->Exec(Form("%s %s", kRM, fSessionDir.Data()));
   }

   // Cleanup queries directory if empty
   if (IsMaster()) {
      if (!(fQueries->GetSize())) {
         // make sure we remain in a "connected" directory
         gSystem->ChangeDirectory("/");
         // needed in case fQueryDir is on NFS ?!
         gSystem->MakeDirectory(fQueryDir+"/.delete");
         gSystem->Exec(Form("%s %s", kRM, fQueryDir.Data()));
         // Remove lock file
         if (fQueryLock)
            gSystem->Unlink(fQueryLock->GetName());
       }

      // Unlock the query dir owned by this session
      if (fQueryLock)
         fQueryLock->Unlock();
   }

   // Remove input handler to avoid spurious signals in socket
   // selection for closing activities executed upon exit()
   gSystem->RemoveFileHandler(fInputHandler);
   gSystem->RemoveSignalHandler(fInterruptHandler);

   gSystem->Exit(status);
}

//______________________________________________________________________________
Int_t TXProofServ::LockSession(const char *sessiontag, TProofLockPath **lck)
{
   // Try locking query area of session tagged sessiontag.
   // The id of the locking file is returned in fid and must be
   // unlocked via UnlockQueryFile(fid).

   // We do not need to lock our own session
   if (strstr(sessiontag, fSessionTag))
      return 0;

   if (!lck) {
      Info("LockSession","locker space undefined");
      return -1;
   }
   *lck = 0;

   // Check the format
   TString stag = sessiontag;
   TRegexp re("session-.*-.*-.*");
   Int_t i1 = stag.Index(re);
   if (i1 == kNPOS) {
      Info("LockSession","bad format: %s", sessiontag);
      return -1;
   }
   stag.ReplaceAll("session-","");

   // Drop query number, if any
   Int_t i2 = stag.Index(":q");
   if (i2 != kNPOS)
      stag.Remove(i2);

   // Make sure that parent process does not exist anylonger
   TString parlog = fSessionDir;
   parlog = parlog.Remove(parlog.Index("master-")+strlen("master-"));
   parlog += stag;
   if (!gSystem->AccessPathName(parlog)) {
      Info("LockSession","parent still running: do nothing");
      return -1;
   }

   // Lock the query lock file
   TString qlock = fQueryLock->GetName();
   qlock.ReplaceAll(fSessionTag, stag);

   if (!gSystem->AccessPathName(qlock)) {
      *lck = new TProofLockPath(qlock);
      if (((*lck)->Lock()) < 0) {
         Info("LockSession","problems locking query lock file");
         SafeDelete(*lck);
         return -1;
      }
   }

   // We are done
   return 0;
}

//______________________________________________________________________________
void TXProofServ::SetShutdownTimer(Bool_t on, Int_t delay)
{
   // Enable/disable the timer for delayed shutdown; the delay will be 'delay'
   // seconds; depending on fShutdownWhenIdle, the countdown will start
   // immediately or when the session is idle.

   fShutdownTimerMtx = (fShutdownTimerMtx) ? fShutdownTimerMtx : new TMutex(kTRUE);
   R__LOCKGUARD(fShutdownTimerMtx);

   if (delay < 0 && !fShutdownTimer)
      // No shutdown request, nothing to do
      return;

   // Make sure that 'delay' make sense, i.e. not larger than 10 days
   if (delay > 864000) {
      Warning("SetShutdownTimer",
              "Abnormous delay value (%d): corruption? setting to 0", delay);
      delay = 1;
   }
   // Set a minimum value (0 does not seem to start the timer ...)
   delay = (delay <= 0) ? 1 : delay;

   if (on) {
      if (!fShutdownTimer) {
         // First setup call: create timer
         fShutdownTimer = new TTimer((delay * 1000), kFALSE);
         // Connect it to the HandleTermination
         fShutdownTimer->Connect("Timeout()", "TXProofServ", this, "HandleTermination()");
         // Start the countdown if requested
         if (!fShutdownWhenIdle || fIdle)
            fShutdownTimer->Start(-1, kTRUE);
      } else {
         // Start the countdown
         fShutdownTimer->Start(-1, kTRUE);
      }
   } else {
      if (fShutdownTimer) {
         // Stop timer (client has reattached)
         fShutdownTimer->Stop();
         // Disconnect from HandleTermination
         fShutdownTimer->Disconnect("Timeout()", this, "HandleTermination()");
         // Clean-up the timer
         SafeDelete(fShutdownTimer);
      }
   }
}
