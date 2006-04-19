// @(#)root/proofx:$Name:  $:$Id: TXProofServ.cxx,v 1.4 2006/02/26 16:09:57 rdm Exp $
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

#include <XrdClient/XrdClientConst.hh>
#include <XrdClient/XrdClientEnv.hh>

// debug hook
static volatile Int_t gProofServDebug = 1;

//______________________________________________________________________________
static void ProofServErrorHandler(Int_t level, Bool_t abort, const char *location,
                                  const char *msg)
{
   // The PROOF error handler function. It prints the message on stderr and
   // if abort is set it aborts the application.

   if (!gProofServ)
      return;

   if (level < gErrorIgnoreLevel)
      return;

   const char *type   = 0;
   ELogLevel loglevel = kLogInfo;

   if (level >= kInfo) {
      loglevel = kLogInfo;
      type = "Info";
   }
   if (level >= kWarning) {
      loglevel = kLogWarning;
      type = "Warning";
   }
   if (level >= kError) {
      loglevel = kLogErr;
      type = "Error";
   }
   if (level >= kBreak) {
      loglevel = kLogErr;
      type = "*** Break ***";
   }
   if (level >= kSysError) {
      loglevel = kLogErr;
      type = "SysError";
   }
   if (level >= kFatal) {
      loglevel = kLogErr;
      type = "Fatal";
   }

   TString node = gProofServ->IsMaster() ? "master" : "slave ";
   node += gProofServ->GetOrdinal();
   char *bp;

   if (!location || strlen(location) == 0 ||
       (level >= kBreak && level < kSysError)) {
      fprintf(stderr, "%s on %s: %s\n", type, node.Data(), msg);
      bp = Form("%s:%s:%s:%s", gProofServ->GetUser(), node.Data(), type, msg);
   } else {
      fprintf(stderr, "%s in <%s> on %s: %s\n", type, location, node.Data(), msg);
      bp = Form("%s:%s:%s:<%s>:%s", gProofServ->GetUser(), node.Data(), type, location, msg);
   }
   fflush(stderr);
   gSystem->Syslog(loglevel, bp);

   if (abort) {

      static Bool_t recursive = kFALSE;

      if (!recursive) {
         recursive = kTRUE;
         gProofServ->GetSocket()->Send(kPROOF_FATAL);
         recursive = kFALSE;
      }

      fprintf(stderr, "aborting\n");
      fflush(stderr);
      gSystem->StackTrace();
      gSystem->Abort();
   }
}

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
   TApplication *GetTXProofServ(Int_t *argc, char **argv)
   { return ((TApplication *)(new TXProofServ(argc, argv))); }
}

//______________________________________________________________________________
TXProofServ::TXProofServ(Int_t *argc, char **argv) : TProofServ(argc, argv)
{
   // Main constructor. Create an application environment. The TProofServ
   // environment provides an eventloop via inheritance of TApplication.
   // Actual server creation work is done in CreateServer() to allow
   // overloading.

   // crude check on number of arguments
   if (*argc < 2) {
     Fatal("TXProofServ", "Must have at least 1 arguments (see  proofd).");
     exit(1);
   }
}

//______________________________________________________________________________
void TXProofServ::CreateServer()
{
   // Finalize the server setup. If master, create the TProof instance to talk
   // the worker or submaster nodes.


   // wait (loop) to allow debugger to connect
   if ((gEnv->GetValue("Proof.GdbHook",0) == 3 && fService != "prooftest") ||
       (gEnv->GetValue("Proof.GdbHook",0) == 4 && fService == "prooftest")) {
      while (gProofServDebug)
         ;
   }

   // abort on higher than kSysError's and set error handler
   gErrorAbortLevel = kSysError + 1;
   SetErrorHandler(ProofServErrorHandler);

   fNcmd            = 0;
   fInterrupt       = kFALSE;
   fProtocol        = 0;
   fOrdinal         = "-1";
   fGroupId         = -1;
   fGroupSize       = 0;
   fLogLevel        = gEnv->GetValue("Proof.DebugLevel",0);
   fRealTime        = 0.0;
   fCpuTime         = 0.0;
   fProof           = 0;
   fPlayer          = 0;
   fEnabledPackages = new TList;
   fEnabledPackages->SetOwner();

   // Global location string in TXSocket
   TXSocket::fgLoc = (IsMaster()) ? "master" : "slave" ;

   // Set debug level in XrdClient
   EnvPutInt(NAME_DEBUG, gEnv->GetValue("XNet.Debug", 0));

   // Get socket to be used to call back our xpd
   const char *sockpath = 0;
   if (!(sockpath = gSystem->Getenv("ROOTOPENSOCK"))) {
     Fatal("CreateServer", "Socket setup by xpd undefined");
     exit(1);
   }
   // If test session, just send the protcol version and exit
   if (Argc() > 3) {
      Int_t fpw = (Int_t) strtol(sockpath, 0, 10);
      int proto = htonl(kPROOF_Protocol);
      if (write(fpw, &proto, sizeof(proto)) != sizeof(proto)) {
         Error("CreateServer", "test: sending protocol number");
      }
      exit(0);
   }

   // Get the sessions ID
   const char *sessID = 0;
   if (!(sessID = gSystem->Getenv("ROOTSESSIONID"))) {
     Fatal("CreateServer", "Session ID undefined");
     exit(1);
   }
   Int_t psid = (Int_t) strtol(sessID, 0, 10);

   // Call back the server
   fSocket = new TXUnixSocket(sockpath, psid);
   if (!fSocket || !(fSocket->IsValid())) {
      Fatal("CreateServer", "Failed to open connection to XrdProofd coordinator");
      exit(1);
   }

   // Get socket descriptor
   Int_t sock = fSocket->GetDescriptor();

   // Get the client ID
   const char *clntID = 0;
   if (!(clntID = gSystem->Getenv("ROOTCLIENTID"))) {
     Fatal("CreateServer", "Client ID undefined");
     exit(1);
   }
   Int_t cid = (Int_t) strtol(clntID, 0, 10);
   ((TXSocket *)fSocket)->SetClientID(cid);

   fArchivePath     = "";

   fSeqNum          = 0;
   fDrawQueries     = 0;
   fKeptQueries     = 0;
   fQueries         = new TList;
   fWaitingQueries  = new TList;
   fPreviousQueries = 0;
   fIdle            = kTRUE;

   if (gErrorIgnoreLevel == kUnset) {
      gErrorIgnoreLevel = 0;
      if (gEnv) {
         TString level = gEnv->GetValue("Root.ErrorIgnoreLevel", "Info");
         if (!level.CompareTo("Info",TString::kIgnoreCase))
            gErrorIgnoreLevel = kInfo;
         else if (!level.CompareTo("Warning",TString::kIgnoreCase))
            gErrorIgnoreLevel = kWarning;
         else if (!level.CompareTo("Error",TString::kIgnoreCase))
            gErrorIgnoreLevel = kError;
         else if (!level.CompareTo("Break",TString::kIgnoreCase))
            gErrorIgnoreLevel = kBreak;
         else if (!level.CompareTo("SysError",TString::kIgnoreCase))
            gErrorIgnoreLevel = kSysError;
         else if (!level.CompareTo("Fatal",TString::kIgnoreCase))
            gErrorIgnoreLevel = kFatal;
      }
   }

   gProofDebugLevel = gEnv->GetValue("Proof.DebugLevel",0);
   gProofDebugMask = (TProofDebug::EProofDebugMask) gEnv->GetValue("Proof.DebugMask",~0);
   if (gProofDebugLevel > 0)
      Info("CreateServer", "DebugLevel %d Mask %u", gProofDebugLevel, gProofDebugMask);

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
      Info("CreateServer", "Service %s ConfDir %s IsMaster %d\n",
           fService.Data(), fConfDir.Data(), (Int_t)fMasterServ);

   Setup();
   RedirectOutput();

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
   gSystem->AddSignalHandler(new TXProofServInterruptHandler(this));
   TXSocketHandler *sh =
      TXSocketHandler::GetSocketHandler(new TXProofServInputHandler(this, sock), fSocket);
   gSystem->AddFileHandler(sh);

   // Set the this as reference of this socket
   ((TXSocket *)fSocket)->fReference = this;
   // Set this has handler
   ((TXSocket *)fSocket)->fHandler = this;

   gProofServ = this;

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
      fProof = reinterpret_cast<TProof*>(h->ExecPlugin(4, master.Data(),
                                                          fConfFile.Data(),
                                                          fConfDir.Data(),
                                                          fLogLevel));
      if (!fProof || !fProof->IsValid()) {
         Error("CreateServer", "plugin for TVirtualProof could not be executed");
         delete fProof;
         fProof = 0;
         SendLogFile(-99);
         Terminate(0);
      }

      SendLogFile();
   }
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

         // If master server, propagate interrupt to slaves
         // (shutdown interrupt send internally).
         if (IsMaster())
            fProof->Close("S");

         // Close link with coordinator
         ((TXSocket *)fSocket)->SetSessionID(-1);
         fSocket->Close();

         Terminate(0);  // will not return from here....

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
void TXProofServ::Setup()
{
   // Print the ProofServ logo on standard output.

   char str[512];

   if (IsMaster()) {
      sprintf(str, "**** Welcome to the PROOF server @ %s ****", gSystem->HostName());
   } else {
      sprintf(str, "**** PROOF slave server @ %s started ****", gSystem->HostName());
   }

   if (fSocket->Send(str) != 1+static_cast<Int_t>(strlen(str))) {
      Error("Setup", "failed to send proof server startup message");
      gSystem->Exit(1);
   }

   // exchange protocol level between client and master and between
   // master and slave
   Int_t what;
   if (fSocket->Recv(fProtocol, what) != 2*sizeof(Int_t)) {
      Error("Setup", "failed to receive remote proof protocol");
      gSystem->Exit(1);
   }
   if (fSocket->Send(kPROOF_Protocol, kROOTD_PROTOCOL) != 2*sizeof(Int_t)) {
      Error("Setup", "failed to send local proof protocol");
      gSystem->Exit(1);
   }

   // Receive some useful information
   TMessage *mess;
   if ((fSocket->Recv(mess) <= 0) || !mess) {
      Error("Setup", "failed to receive ordinal and config info");
      gSystem->Exit(1);
   }
   if (IsMaster()) {
      (*mess) >> fUser >> fOrdinal >> fConfFile;
      fWorkDir = kPROOF_WorkDir;
   } else {
      (*mess) >> fUser >> fOrdinal >> fWorkDir;
      if (fWorkDir.IsNull()) fWorkDir = kPROOF_WorkDir;
   }
   delete mess;

   if (IsMaster()) {

      // strip off any prooftype directives
      TString conffile = fConfFile;
      conffile.Remove(0, 1 + conffile.Index(":"));
      // parse config file to find working directory
      TProofResourcesStatic resources(fConfDir, conffile);
      if (resources.IsValid()) {
         if (resources.GetMaster()) {
            TString tmpWorkDir = resources.GetMaster()->GetWorkDir();
            if (tmpWorkDir != "")
               fWorkDir = tmpWorkDir;
         }
      } else {
         Error("Setup", "reading config file %s",
                        resources.GetFileName().Data());
         gSystem->Exit(1);
      }
   }

   // goto to the main PROOF working directory
   char *workdir = gSystem->ExpandPathName(fWorkDir.Data());
   fWorkDir = workdir;
   delete [] workdir;

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
         SysError("Setup", "can not change to PROOF directory %s",
                  fWorkDir.Data());
      }
   } else {
      if (!gSystem->ChangeDirectory(fWorkDir)) {
         gSystem->Unlink(fWorkDir);
         gSystem->mkdir(fWorkDir, kTRUE);
         if (!gSystem->ChangeDirectory(fWorkDir)) {
            SysError("Setup", "can not change to PROOF directory %s",
                     fWorkDir.Data());
         }
      }
   }

   // check and make sure "cache" directory exists
   fCacheDir = fWorkDir;
   fCacheDir += TString("/") + kPROOF_CacheDir;
   if (gSystem->AccessPathName(fCacheDir))
      gSystem->MakeDirectory(fCacheDir);

   fCacheLock = kPROOF_CacheLockFile;
   fCacheLock += fUser;

   // check and make sure "packages" directory exists
   fPackageDir = fWorkDir;
   fPackageDir += TString("/") + kPROOF_PackDir;
   if (gSystem->AccessPathName(fPackageDir))
      gSystem->MakeDirectory(fPackageDir);

   fPackageLock = kPROOF_PackageLockFile;
   fPackageLock += fUser;

   // host first name
   TString host = gSystem->HostName();
   if (host.Index(".") != kNPOS)
      host.Remove(host.Index("."));

   // Session tag
   fSessionTag = Form("%s-%s-%d-%d", fOrdinal.Data(), host.Data(),
                          TTimeStamp().GetSec(),gSystem->GetPid());

   // create session directory and make it the working directory
   fSessionDir = fWorkDir;
   if (IsMaster())
      fSessionDir += "/master-";
   else
      fSessionDir += "/slave-";
   fSessionDir += fSessionTag;

   if (gSystem->AccessPathName(fSessionDir)) {
      gSystem->MakeDirectory(fSessionDir);
      if (!gSystem->ChangeDirectory(fSessionDir)) {
         SysError("Setup", "can not change to working directory %s",
                  fSessionDir.Data());
      } else {
         gSystem->Setenv("PROOF_SANDBOX", fSessionDir);
      }
   }

   // On masters, check and make sure "queries" directory exists
   if (IsMaster()) {
      fQueryDir = fWorkDir;
      fQueryDir += TString("/") + kPROOF_QueryDir;
      if (gSystem->AccessPathName(fQueryDir))
         gSystem->MakeDirectory(fQueryDir);
      else
         ScanPreviousQueries(fQueryDir);
      fQueryDir += TString("/session-") + fSessionTag;
      if (gSystem->AccessPathName(fQueryDir))
         gSystem->MakeDirectory(fQueryDir);

      fQueryLock = kPROOF_QueryLockFile;
      fQueryLock += fSessionTag;
      fQueryLock += fUser;

      // Lock the query dir owned by this session
      fQueryLockId = LockQueryFile(fQueryLock);

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
   lnow = lseek(fileno(fLogFile), (off_t) 0, SEEK_CUR);

   Bool_t adhoc = kFALSE;
   if (start > -1) {
      lseek(fileno(fLogFile), (off_t) start, SEEK_SET);
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
         while ((len = read(fileno(fLogFile), buf, wanted)) < 0 &&
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
      lseek(fileno(fLogFile), lnow, SEEK_SET);

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
         // Work dir, if defined
         TString tmpwrk = master->GetWorkDir(); 
         if (tmpwrk != "")
            fWorkDir = tmpwrk;

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
Bool_t TXProofServ::HandleError()
{
   // Handle error on the input socket

   Printf("HandleError: %p: got called ...", this);

   // If master server, propagate interrupt to slaves
   // (shutdown interrupt send internally).
   if (IsMaster())
      fProof->Close("S");

   // Close link with coordinator
   ((TXSocket *)fSocket)->SetSessionID(-1);
   fSocket->Close();

   Terminate(0);  // will not return from here....

   Printf("HandleError: %p: DONE ... ", this);

   // We are done
   return kTRUE;
}

//_____________________________________________________________________________
Bool_t TXProofServ::HandleInput()
{
   // Handle asynchronous input on the input socket

   if (gDebug > 2)
      Info("HandleInput","%p", this);
   HandleSocketInput();
   // This request has been completed: remove the client ID from the pipe
   ((TXSocket *)fSocket)->RemoveClientID();
   // We are done
   return kTRUE;
}
