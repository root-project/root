// @(#)root/proof:$Id$
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

#include "RConfigure.h"
#include "RConfig.h"
#include "Riostream.h"

#ifdef WIN32
   #include <process.h>
   #include <io.h>
   #include "snprintf.h"
   typedef long off_t;
#endif
#include <errno.h>
#include <time.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#ifndef WIN32
#include <sys/wait.h>
#endif
#include <cstdlib>

// To handle exceptions
#include <exception>
#include <new>

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

#include "TProofServ.h"
#include "TDSetProxy.h"
#include "TEnv.h"
#include "TError.h"
#include "TEventList.h"
#include "TEntryList.h"
#include "TException.h"
#include "TFile.h"
#include "THashList.h"
#include "TInterpreter.h"
#include "TKey.h"
#include "TMessage.h"
#include "TVirtualPerfStats.h"
#include "TProofDebug.h"
#include "TProof.h"
#include "TVirtualProofPlayer.h"
#include "TProofQueryResult.h"
#include "TQueryResultManager.h"
#include "TRegexp.h"
#include "TROOT.h"
#include "TSocket.h"
#include "TStopwatch.h"
#include "TSystem.h"
#include "TTimeStamp.h"
#include "TUrl.h"
#include "TPluginManager.h"
#include "TObjString.h"
#include "compiledata.h"
#include "TProofResourcesStatic.h"
#include "TProofNodeInfo.h"
#include "TFileInfo.h"
#include "TMutex.h"
#include "TClass.h"
#include "TSQLServer.h"
#include "TSQLResult.h"
#include "TSQLRow.h"
#include "TPRegexp.h"
#include "TParameter.h"
#include "TMap.h"
#include "TSortedList.h"
#include "TParameter.h"
#include "TFileCollection.h"
#include "TLockFile.h"
#include "TDataSetManagerFile.h"
#include "TProofProgressStatus.h"
#include "TServerSocket.h"
#include "TMonitor.h"
#include "TFunction.h"
#include "TMethodArg.h"
#include "TMethodCall.h"

// global proofserv handle
TProofServ *gProofServ = 0;

// debug hook
static volatile Int_t gProofServDebug = 1;

// Syslog control
Int_t TProofServ::fgLogToSysLog = 0;
TString TProofServ::fgSysLogService("proof");
TString TProofServ::fgSysLogEntity("undef:default");

// File where to log: default stderr
FILE *TProofServ::fgErrorHandlerFile = 0;

// To control allowed actions while processing
Int_t TProofServ::fgRecursive = 0;

// Last message before exceptions
TString TProofServ::fgLastMsg("<undef>");

// Memory controllers
Long_t TProofServ::fgVirtMemMax = -1;
Long_t TProofServ::fgResMemMax = -1;
Float_t TProofServ::fgMemHWM = 0.80;
Float_t TProofServ::fgMemStop = 0.95;

//----- Termination signal handler ---------------------------------------------
//______________________________________________________________________________
class TProofServTerminationHandler : public TSignalHandler {
   TProofServ  *fServ;
public:
   TProofServTerminationHandler(TProofServ *s)
      : TSignalHandler(kSigTermination, kFALSE) { fServ = s; }
   Bool_t  Notify();
};

//______________________________________________________________________________
Bool_t TProofServTerminationHandler::Notify()
{
   // Handle this interrupt

   Printf("Received SIGTERM: terminating");
   fServ->HandleTermination();
   return kTRUE;
}

//----- Interrupt signal handler -----------------------------------------------
//______________________________________________________________________________
class TProofServInterruptHandler : public TSignalHandler {
   TProofServ  *fServ;
public:
   TProofServInterruptHandler(TProofServ *s)
      : TSignalHandler(kSigUrgent, kFALSE) { fServ = s; }
   Bool_t  Notify();
};

//______________________________________________________________________________
Bool_t TProofServInterruptHandler::Notify()
{
   // Handle this interrupt

   fServ->HandleUrgentData();
   if (TROOT::Initialized()) {
      Throw(GetSignal());
   }
   return kTRUE;
}

//----- SigPipe signal handler -------------------------------------------------
//______________________________________________________________________________
class TProofServSigPipeHandler : public TSignalHandler {
   TProofServ  *fServ;
public:
   TProofServSigPipeHandler(TProofServ *s) : TSignalHandler(kSigPipe, kFALSE)
      { fServ = s; }
   Bool_t  Notify();
};

//______________________________________________________________________________
Bool_t TProofServSigPipeHandler::Notify()
{
   // Handle this signal

   fServ->HandleSigPipe();
   return kTRUE;
}

//----- Input handler for messages from parent or master -----------------------
//______________________________________________________________________________
class TProofServInputHandler : public TFileHandler {
   TProofServ  *fServ;
public:
   TProofServInputHandler(TProofServ *s, Int_t fd) : TFileHandler(fd, 1)
      { fServ = s; }
   Bool_t Notify();
   Bool_t ReadNotify() { return Notify(); }
};

//______________________________________________________________________________
Bool_t TProofServInputHandler::Notify()
{
   // Handle this input

   fServ->HandleSocketInput();
   return kTRUE;
}

TString TProofServLogHandler::fgPfx = ""; // Default prefix to be prepended to messages
Int_t TProofServLogHandler::fgCmdRtn = 0; // Return code of the command execution (available only
                                          // after closing the pipe)
//______________________________________________________________________________
TProofServLogHandler::TProofServLogHandler(const char *cmd,
                                             TSocket *s, const char *pfx)
                     : TFileHandler(-1, 1), fSocket(s), fPfx(pfx)
{
   // Execute 'cmd' in a pipe and handle output messages from the related file

   ResetBit(kFileIsPipe);
   fgCmdRtn = 0;
   fFile = 0;
   if (s && cmd) {
      fFile = gSystem->OpenPipe(cmd, "r");
      if (fFile) {
         SetFd(fileno(fFile));
         // Notify what already in the file
         Notify();
         // Used in the destructor
         SetBit(kFileIsPipe);
      } else {
         fSocket = 0;
         Error("TProofServLogHandler", "executing command in pipe");
         fgCmdRtn = -1;
      }
   } else {
      Error("TProofServLogHandler",
            "undefined command (%p) or socket (%p)", (int *)cmd, s);
   }
}
//______________________________________________________________________________
TProofServLogHandler::TProofServLogHandler(FILE *f, TSocket *s, const char *pfx)
                     : TFileHandler(-1, 1), fSocket(s), fPfx(pfx)
{
   // Handle available message from the open file 'f'

   ResetBit(kFileIsPipe);
   fgCmdRtn = 0;
   fFile = 0;
   if (s && f) {
      fFile = f;
      SetFd(fileno(fFile));
      // Notify what already in the file
      Notify();
   } else {
      Error("TProofServLogHandler", "undefined file (%p) or socket (%p)", f, s);
   }
}
//______________________________________________________________________________
TProofServLogHandler::~TProofServLogHandler()
{
   // Handle available message in the open file

   if (TestBit(kFileIsPipe) && fFile) {
      Int_t rc = gSystem->ClosePipe(fFile);
#ifdef WIN32
      fgCmdRtn = rc;
#else
      fgCmdRtn = WIFEXITED(rc) ? WEXITSTATUS(rc) : -1;
#endif
   }
   fFile = 0;
   fSocket = 0;
   ResetBit(kFileIsPipe);
}
//______________________________________________________________________________
Bool_t TProofServLogHandler::Notify()
{
   // Handle available message in the open file

   if (IsValid()) {
      TMessage m(kPROOF_MESSAGE);
      // Read buffer
      char line[4096];
      char *plf = 0;
      while (fgets(line, sizeof(line), fFile)) {
         if ((plf = strchr(line, '\n')))
            *plf = 0;
         // Create log string
         TString log;
         if (fPfx.Length() > 0) {
            // Prepend prefix specific to this instance
            log.Form("%s: %s", fPfx.Data(), line);
         } else if (fgPfx.Length() > 0) {
            // Prepend default prefix
            log.Form("%s: %s", fgPfx.Data(), line);
         } else {
            // Nothing to prepend
            log = line;
         }
         // Send the message one level up
         m.Reset(kPROOF_MESSAGE);
         m << log;
         fSocket->Send(m);
      }
   }
   return kTRUE;
}
//______________________________________________________________________________
void TProofServLogHandler::SetDefaultPrefix(const char *pfx)
{
   // Static method to set the default prefix

   fgPfx = pfx;
}
//______________________________________________________________________________
Int_t TProofServLogHandler::GetCmdRtn()
{
   // Static method to get the return code from the execution of a command via
   // the pipe. This is always 0 when the log handler is not used with a pipe

   return fgCmdRtn;
}

//______________________________________________________________________________
TProofServLogHandlerGuard::TProofServLogHandlerGuard(const char *cmd, TSocket *s,
                                                     const char *pfx, Bool_t on)
{
   // Init a guard for executing a command in a pipe

   fExecHandler = 0;
   if (cmd && on) {
      fExecHandler = new TProofServLogHandler(cmd, s, pfx);
      if (fExecHandler->IsValid()) {
         gSystem->AddFileHandler(fExecHandler);
      } else {
         Error("TProofServLogHandlerGuard","invalid handler");
      }
   } else {
      if (on)
         Error("TProofServLogHandlerGuard","undefined command");
   }
}

//______________________________________________________________________________
TProofServLogHandlerGuard::TProofServLogHandlerGuard(FILE *f, TSocket *s,
                                                     const char *pfx, Bool_t on)
{
   // Init a guard for executing a command in a pipe

   fExecHandler = 0;
   if (f && on) {
      fExecHandler = new TProofServLogHandler(f, s, pfx);
      if (fExecHandler->IsValid()) {
         gSystem->AddFileHandler(fExecHandler);
      } else {
         Error("TProofServLogHandlerGuard","invalid handler");
      }
   } else {
      if (on)
         Error("TProofServLogHandlerGuard","undefined file");
   }
}

//______________________________________________________________________________
TProofServLogHandlerGuard::~TProofServLogHandlerGuard()
{
   // Close a guard for executing a command in a pipe

   if (fExecHandler && fExecHandler->IsValid()) {
      gSystem->RemoveFileHandler(fExecHandler);
      SafeDelete(fExecHandler);
   }
}

//--- Special timer to control delayed shutdowns ----------------------------//
//______________________________________________________________________________
Bool_t TShutdownTimer::Notify()
{
   // Handle expiration of the shutdown timer. In the case of low activity the
   // process will be aborted.

   if (gDebug > 0)
      Info ("Notify","checking activity on the input socket");

   // Check activity on the socket
   TSocket *xs = 0;
   if (fProofServ && (xs = fProofServ->GetSocket())) {
      TTimeStamp now;
      TTimeStamp ts = xs->GetLastUsage();
      Long_t dt = (Long_t)(now.GetSec() - ts.GetSec()) * 1000 +
                  (Long_t)(now.GetNanoSec() - ts.GetNanoSec()) / 1000000 ;
      Int_t to = gEnv->GetValue("ProofServ.ShutdonwTimeout", 20);
      if (dt > to * 60000) {
         Printf("TShutdownTimer::Notify: input socket: %p: did not show any activity"
                         " during the last %d mins: aborting", xs, to);
         // At this point we lost our controller: we need to abort to avoid
         // hidden timeouts or loops
         gSystem->Abort();
      } else {
         if (gDebug > 0)
            Info("Notify", "input socket: %p: show activity"
                           " %ld secs ago", xs, dt / 60000);
      }
   }
   Start(-1, kFALSE);
   return kTRUE;
}

//--- Synchronous timer used to reap children processes change of state ------//
//______________________________________________________________________________
TReaperTimer::~TReaperTimer()
{
   // Destructor

   if (fChildren) {
      fChildren->SetOwner(kTRUE);
      delete fChildren;
      fChildren = 0;
   }
}

//______________________________________________________________________________
void TReaperTimer::AddPid(Int_t pid)
{
   // Add an entry for 'pid' in the internal list

   if (pid > 0) {
      if (!fChildren)
         fChildren = new TList;
      TString spid;
      spid.Form("%d", pid);
      fChildren->Add(new TParameter<Int_t>(spid.Data(), pid));
      TurnOn();
   }
}

//______________________________________________________________________________
Bool_t TReaperTimer::Notify()
{
   // Check if any of the registered children has changed its state.
   // Unregister those that are gone.

   if (fChildren) {
      TIter nxp(fChildren);
      TParameter<Int_t> *p = 0;
      while ((p = (TParameter<Int_t> *)nxp())) {
         int status;
#ifndef WIN32
         pid_t pid;
         do {
            pid = waitpid(p->GetVal(), &status, WNOHANG);
         } while (pid < 0 && errno == EINTR);
#else
         intptr_t pid;
         pid = _cwait(&status, (intptr_t)p->GetVal(), 0);
#endif
         if (pid > 0 && pid == p->GetVal()) {
            // Remove from the list
            fChildren->Remove(p);
            delete p;
         }
      }
   }

   // Stop the timer if no children
   if (!fChildren || fChildren->GetSize() <= 0) {
      Stop();
   } else {
      // Needed for the next shot
      Reset();
   }
   return kTRUE;
}

//--- Special timer to to terminate idle sessions ----------------------------//
//______________________________________________________________________________
Bool_t TIdleTOTimer::Notify()
{
   // Handle expiration of the idle timer. The session will just be terminated.

   Info ("Notify", "session idle for more then %lld secs: terminating", Long64_t(fTime)/1000);

   if (fProofServ) {
      // Set the status to timed-out
      Int_t uss_rc = -1;
      if ((uss_rc = fProofServ->UpdateSessionStatus(4)) != 0)
         Warning("Notify", "problems updating session status (errno: %d)", -uss_rc);
      // Send a terminate request
      TString msg;
      if (fProofServ->GetProtocol() < 29) {
         msg.Form("\n//\n// PROOF session at %s (%s) terminated because idle for more than %lld secs\n"
                  "// Please IGNORE any error message possibly displayed below\n//",
                  gSystem->HostName(), fProofServ->GetSessionTag(), Long64_t(fTime)/1000);
      } else {
         msg.Form("\n//\n// PROOF session at %s (%s) terminated because idle for more than %lld secs\n//",
                  gSystem->HostName(), fProofServ->GetSessionTag(), Long64_t(fTime)/1000);
      }
      fProofServ->SendAsynMessage(msg.Data());
      fProofServ->Terminate(0);
      Reset();
      Stop();
   } else {
      Warning("Notify", "fProofServ undefined!");
      Start(-1, kTRUE);
   }
   return kTRUE;
}

ClassImp(TProofServ)

// Hook to the constructor. This is needed to avoid using the plugin manager
// which may create problems in multi-threaded environments.
extern "C" {
   TApplication *GetTProofServ(Int_t *argc, char **argv, FILE *flog)
   { return new TProofServ(argc, argv, flog); }
}

//______________________________________________________________________________
TProofServ::TProofServ(Int_t *argc, char **argv, FILE *flog)
       : TApplication("proofserv", argc, argv, 0, -1)
{
   // Main constructor. Create an application environment. The TProofServ
   // environment provides an eventloop via inheritance of TApplication.
   // Actual server creation work is done in CreateServer() to allow
   // overloading.

   // Read session specific rootrc file
   TString rcfile = gSystem->Getenv("ROOTRCFILE") ? gSystem->Getenv("ROOTRCFILE")
                                                  : "session.rootrc";
   if (!gSystem->AccessPathName(rcfile, kReadPermission))
      gEnv->ReadFile(rcfile, kEnvChange);

   // Upper limit on Virtual Memory (in kB)
   fgVirtMemMax = gEnv->GetValue("Proof.VirtMemMax",-1);
   if (fgVirtMemMax < 0 && gSystem->Getenv("PROOF_VIRTMEMMAX")) {
      Long_t mmx = strtol(gSystem->Getenv("PROOF_VIRTMEMMAX"), 0, 10);
      if (mmx < kMaxLong && mmx > 0)
         fgVirtMemMax = mmx * 1024;
   }
   // Old variable for backward compatibility
   if (fgVirtMemMax < 0 && gSystem->Getenv("ROOTPROOFASHARD")) {
      Long_t mmx = strtol(gSystem->Getenv("ROOTPROOFASHARD"), 0, 10);
      if (mmx < kMaxLong && mmx > 0)
         fgVirtMemMax = mmx * 1024;
   }
   // Upper limit on Resident Memory (in kB)
   fgResMemMax = gEnv->GetValue("Proof.ResMemMax",-1);
   if (fgResMemMax < 0 && gSystem->Getenv("PROOF_RESMEMMAX")) {
      Long_t mmx = strtol(gSystem->Getenv("PROOF_RESMEMMAX"), 0, 10);
      if (mmx < kMaxLong && mmx > 0)
         fgResMemMax = mmx * 1024;
   }
   // Thresholds for warnings and stop processing
   fgMemStop = gEnv->GetValue("Proof.MemStop", 0.95);
   fgMemHWM = gEnv->GetValue("Proof.MemHWM", 0.80);
   if (fgVirtMemMax > 0 || fgResMemMax > 0) {
      if ((fgMemStop < 0.) || (fgMemStop > 1.)) {
         Warning("TProofServ", "requested memory fraction threshold to stop processing"
                               " (MemStop) out of range [0,1] - ignoring");
         fgMemStop = 0.95;
      }
      if ((fgMemHWM < 0.) || (fgMemHWM > fgMemStop)) {
         Warning("TProofServ", "requested memory fraction threshold for warning and finer monitoring"
                               " (MemHWM) out of range [0,MemStop] - ignoring");
         fgMemHWM = 0.80;
      }
   }   

   // Wait (loop) to allow debugger to connect
   Bool_t test = (*argc >= 4 && !strcmp(argv[3], "test")) ? kTRUE : kFALSE;
   if ((gEnv->GetValue("Proof.GdbHook",0) == 3 && !test) ||
       (gEnv->GetValue("Proof.GdbHook",0) == 4 && test)) {
      while (gProofServDebug)
         ;
   }

   // Test instance
   if (*argc >= 4)
      if (!strcmp(argv[3], "test"))
         fService = "prooftest";

   // crude check on number of arguments
   if (*argc < 2) {
      Error("TProofServ", "Must have at least 1 arguments (see  proofd).");
      exit(1);
   }

   // Set global to this instance
   gProofServ = this;

   // Log control flags
   fSendLogToMaster = kFALSE;

   // Abort on higher than kSysError's and set error handler
   gErrorAbortLevel = kSysError + 1;
   SetErrorHandlerFile(stderr);
   SetErrorHandler(ErrorHandler);

   fNcmd            = 0;
   fGroupPriority   = 100;
   fInterrupt       = kFALSE;
   fProtocol        = 0;
   fOrdinal         = gEnv->GetValue("ProofServ.Ordinal", "-1");
   fGroupId         = -1;
   fGroupSize       = 0;
   fRealTime        = 0.0;
   fCpuTime         = 0.0;
   fProof           = 0;
   fPlayer          = 0;
   fSocket          = 0;
   fEnabledPackages = new TList;
   fEnabledPackages->SetOwner();

   fTotSessions     = -1;
   fActSessions     = -1;
   fEffSessions     = -1.;

   fGlobalPackageDirList = 0;

   fLogFile         = flog;
   fLogFileDes      = -1;

   fArchivePath     = "";
   // Init lockers
   fPackageLock     = 0;
   fCacheLock       = 0;
   fQueryLock       = 0;

   fQMgr            = 0;
   fQMtx            = new TMutex(kTRUE);
   fWaitingQueries  = new TList;
   fIdle            = kTRUE;
   fQuerySeqNum     = -1;

   fQueuedMsg       = new TList;

   fRealTimeLog     = kFALSE;

   fShutdownTimer   = 0;
   fReaperTimer     = 0;
   fIdleTOTimer     = 0;

   fInflateFactor   = 1000;

   fDataSetManager  = 0; // Initialized in Setup()

   fInputHandler    = 0;

   // Quotas disabled by default
   fMaxQueries      = -1;
   fMaxBoxSize      = -1;
   fHWMBoxSize      = -1;

   // Submerger quantities
   fMergingSocket   = 0;
   fMergingMonitor  = 0;
   fMergedWorkers   = 0;

   // Bit to flg high-memory footprint
   ResetBit(TProofServ::kHighMemory);
   
   // Max message size
   fMsgSizeHWM = gEnv->GetValue("ProofServ.MsgSizeHWM", 1000000);

   // Message compression
   fCompressMsg     = gEnv->GetValue("ProofServ.CompressMessage", 0);

   gProofDebugLevel = gEnv->GetValue("Proof.DebugLevel",0);
   fLogLevel = gProofDebugLevel;

   gProofDebugMask = (TProofDebug::EProofDebugMask) gEnv->GetValue("Proof.DebugMask",~0);
   if (gProofDebugLevel > 0)
      Info("TProofServ", "DebugLevel %d Mask 0x%x", gProofDebugLevel, gProofDebugMask);

   // Max log file size
   fLogFileMaxSize = -1;
   TString logmx = gEnv->GetValue("ProofServ.LogFileMaxSize", "");
   if (!logmx.IsNull()) {
      Long64_t xf = 1;
      if (!logmx.IsDigit()) {
         if (logmx.EndsWith("K")) {
            xf = 1024;
            logmx.Remove(TString::kTrailing, 'K');
         } else if (logmx.EndsWith("M")) {
            xf = 1024*1024;
            logmx.Remove(TString::kTrailing, 'M');
         } if (logmx.EndsWith("G")) {
            xf = 1024*1024*1024;
            logmx.Remove(TString::kTrailing, 'G');
         }
      }
      if (logmx.IsDigit()) {
         fLogFileMaxSize = logmx.Atoi() * xf;
         if (fLogFileMaxSize > 0)
            Info("TProofServ", "keeping the log file size within %lld bytes", fLogFileMaxSize);
      } else {
         logmx = gEnv->GetValue("ProofServ.LogFileMaxSize", "");
         Warning("TProofServ", "bad formatted log file size limit ignored: '%s'", logmx.Data());
      }
   }
   
   // Parse options
   GetOptions(argc, argv);

   // Default prefix in the form '<role>-<ordinal>'
   fPrefix = (IsMaster() ? "Mst-" : "Wrk-");
   if (test) fPrefix = "Test";
   if (fOrdinal != "-1")
      fPrefix += fOrdinal;
   TProofServLogHandler::SetDefaultPrefix(fPrefix);

   // Syslog control
   TString slog = gEnv->GetValue("ProofServ.LogToSysLog", "");
   if (!(slog.IsNull())) {
      if (slog.IsDigit()) {
         fgLogToSysLog = slog.Atoi();
      } else {
         char c = (slog[0] == 'M' || slog[0] == 'm') ? 'm' : 'a';
         c = (slog[0] == 'W' || slog[0] == 'w') ? 'w' : c;
         Bool_t dosyslog = ((c == 'm' && IsMaster()) ||
                            (c == 'w' && !IsMaster()) || c == 'a') ? kTRUE : kFALSE;
         if (dosyslog) {
            slog.Remove(0,1);
            if (slog.IsDigit()) fgLogToSysLog = slog.Atoi();
            if (fgLogToSysLog <= 0)
               Warning("TProofServ", "request for syslog logging ineffective!");
         }
      }
   }
   // Initialize proper service if required
   if (fgLogToSysLog > 0) {
      fgSysLogService = (IsMaster()) ? "proofm" : "proofw";
      if (fOrdinal != "-1") fgSysLogService += TString::Format("-%s", fOrdinal.Data());
      gSystem->Openlog(fgSysLogService, kLogPid | kLogCons, kLogLocal5);
   }

   // Enable optimized sending of streamer infos to use embedded backward/forward
   // compatibility support between different ROOT versions and different versions of
   // users classes
   Bool_t enableSchemaEvolution = gEnv->GetValue("Proof.SchemaEvolution",1);
   if (enableSchemaEvolution) {
      TMessage::EnableSchemaEvolutionForAll();
   } else {
      Info("TProofServ", "automatic schema evolution in TMessage explicitely disabled");
   }
}

//______________________________________________________________________________
Int_t TProofServ::CreateServer()
{
   // Finalize the server setup. If master, create the TProof instance to talk
   // to the worker or submaster nodes.
   // Return 0 on success, -1 on error

   // Get socket to be used (setup in proofd)
   TString opensock = gSystem->Getenv("ROOTOPENSOCK");
   if (opensock.Length() <= 0)
      opensock = gEnv->GetValue("ProofServ.OpenSock", "-1");
   Int_t sock = opensock.Atoi();
   if (sock <= 0) {
      Fatal("CreateServer", "Invalid socket descriptor number (%d)", sock);
      return -1;
   }
   fSocket = new TSocket(sock);

   // Set compression level, if any
   fSocket->SetCompressionLevel(fCompressMsg);

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
           GetService(), GetConfDir(), (Int_t)fMasterServ);

   if (Setup() != 0) {
      // Setup failure
      LogToMaster();
      SendLogFile();
      Terminate(0);
      return -1;
   }

   // Set the default prefix in the form '<role>-<ordinal>' (it was already done
   // in the constructor, but for standard PROOF the ordinal number is only set in
   // Setup(), so we need to do it again here)
   TString pfx = (IsMaster() ? "Mst-" : "Wrk-");
   pfx += GetOrdinal();
   TProofServLogHandler::SetDefaultPrefix(pfx);

   if (!fLogFile) {
      RedirectOutput();
      // If for some reason we failed setting a redirection fole for the logs
      // we cannot continue
      if (!fLogFile || (fLogFileDes = fileno(fLogFile)) < 0) {
         LogToMaster();
         SendLogFile(-98);
         Terminate(0);
         return -1;
      }
   } else {
      // Use the file already open by pmain
      if ((fLogFileDes = fileno(fLogFile)) < 0) {
         LogToMaster();
         SendLogFile(-98);
         Terminate(0);
         return -1;
      }
   }

   // Send message of the day to the client
   if (IsMaster()) {
      if (CatMotd() == -1) {
         LogToMaster();
         SendLogFile(-99);
         Terminate(0);
         return -1;
      }
   }

   // Everybody expects iostream to be available, so load it...
   ProcessLine("#include <iostream>", kTRUE);
   ProcessLine("#include <string>",kTRUE); // for std::string iostream.

   // The following libs are also useful to have, make sure they are loaded...
   //gROOT->LoadClass("TMinuit",     "Minuit");
   //gROOT->LoadClass("TPostScript", "Postscript");

   // Load user functions
   const char *logon;
   logon = gEnv->GetValue("Proof.Load", (char *)0);
   if (logon) {
      char *mac = gSystem->Which(TROOT::GetMacroPath(), logon, kReadPermission);
      if (mac)
         ProcessLine(TString::Format(".L %s", logon), kTRUE);
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
   gSystem->AddSignalHandler(new TProofServTerminationHandler(this));
   gSystem->AddSignalHandler(new TProofServInterruptHandler(this));
   fInputHandler = new TProofServInputHandler(this, sock);
   gSystem->AddFileHandler(fInputHandler);

   // if master, start slave servers
   if (IsMaster()) {
      TString master = "proof://__master__";
      TInetAddress a = gSystem->GetSockName(sock);
      if (a.IsValid()) {
         master += ":";
         master += a.GetPort();
      }

      // Get plugin manager to load appropriate TProof from
      TPluginManager *pm = gROOT->GetPluginManager();
      if (!pm) {
         Error("CreateServer", "no plugin manager found");
         SendLogFile(-99);
         Terminate(0);
         return -1;
      }

      // Find the appropriate handler
      TPluginHandler *h = pm->FindHandler("TProof", fConfFile);
      if (!h) {
         Error("CreateServer", "no plugin found for TProof with a"
                             " config file of '%s'", fConfFile.Data());
         SendLogFile(-99);
         Terminate(0);
         return -1;
      }

      // load the plugin
      if (h->LoadPlugin() == -1) {
         Error("CreateServer", "plugin for TProof could not be loaded");
         SendLogFile(-99);
         Terminate(0);
         return -1;
      }

      // make instance of TProof
      fProof = reinterpret_cast<TProof*>(h->ExecPlugin(5, master.Data(),
                                                          fConfFile.Data(),
                                                          GetConfDir(),
                                                          fLogLevel, 0));
      if (!fProof || !fProof->IsValid()) {
         Error("CreateServer", "plugin for TProof could not be executed");
         SafeDelete(fProof);
         SendLogFile(-99);
         Terminate(0);
         return -1;
      }
      // Find out if we are a master in direct contact only with workers
      fEndMaster = fProof->IsEndMaster();

      SendLogFile();
   }

   // Setup the shutdown timer
   if (!fShutdownTimer) {
      // Check activity on socket every 5 mins
      fShutdownTimer = new TShutdownTimer(this, 300000);
      fShutdownTimer->Start(-1, kFALSE);
   }

   // Check if schema evolution is effective: clients running versions <=17 do not
   // support that: send a warning message
   if (fProtocol <= 17) {
      TString msg;
      msg.Form("Warning: client version is too old: automatic schema evolution is ineffective.\n"
               "         This may generate compatibility problems between streamed objects.\n"
               "         The advise is to move to ROOT >= 5.21/02 .");
      SendAsynMessage(msg.Data());
   }

   // Setup the idle timer
   if (IsMaster() && !fIdleTOTimer) {
      // Check activity on socket every 5 mins
      Int_t idle_to = gEnv->GetValue("ProofServ.IdleTimeout", -1);
      if (idle_to > 0) {
         fIdleTOTimer = new TIdleTOTimer(this, idle_to * 1000);
         fIdleTOTimer->Start(-1, kTRUE);
         if (gProofDebugLevel > 0)
            Info("CreateServer", " idle timer started (%d secs)", idle_to);
      } else if (gProofDebugLevel > 0) {
         Info("CreateServer", " idle timer not started (no idle timeout requested)");
      }
   }

   // Done
   return 0;
}

//______________________________________________________________________________
TProofServ::~TProofServ()
{
   // Cleanup. Not really necessary since after this dtor there is no
   // live anyway.

   SafeDelete(fWaitingQueries);
   SafeDelete(fQMtx);
   SafeDelete(fEnabledPackages);
   SafeDelete(fSocket);
   SafeDelete(fPackageLock);
   SafeDelete(fCacheLock);
   SafeDelete(fQueryLock);
   SafeDelete(fGlobalPackageDirList);
   close(fLogFileDes);
}

//______________________________________________________________________________
Int_t TProofServ::CatMotd()
{
   // Print message of the day (in the file pointed by the env PROOFMOTD
   // or from fConfDir/etc/proof/motd). The motd is not shown more than
   // once a day. If the file pointed by env PROOFNOPROOF exists (or the
   // file fConfDir/etc/proof/noproof exists), show its contents and close
   // the connection.

   TString lastname;
   FILE   *motd;
   Bool_t  show = kFALSE;

   // If we are disabled just print the message and close the connection
   TString motdname(GetConfDir());
   // The env variable PROOFNOPROOF allows to put the file in an alternative
   // location not overwritten by a new installation
   if (gSystem->Getenv("PROOFNOPROOF")) {
      motdname = gSystem->Getenv("PROOFNOPROOF");
   } else {
      motdname += "/etc/proof/noproof";
   }
   if ((motd = fopen(motdname, "r"))) {
      Int_t c;
      printf("\n");
      while ((c = getc(motd)) != EOF)
         putchar(c);
      fclose(motd);
      printf("\n");

      return -1;
   }

   // get last modification time of the file ~/proof/.prooflast
   lastname = TString(GetWorkDir()) + "/.prooflast";
   char *last = gSystem->ExpandPathName(lastname.Data());
   Long64_t size;
   Long_t id, flags, modtime, lasttime;
   if (gSystem->GetPathInfo(last, &id, &size, &flags, &lasttime) == 1)
      lasttime = 0;

   // show motd at least once per day
   if (time(0) - lasttime > (time_t)86400)
      show = kTRUE;

   // The env variable PROOFMOTD allows to put the file in an alternative
   // location not overwritten by a new installation
   if (gSystem->Getenv("PROOFMOTD")) {
      motdname = gSystem->Getenv("PROOFMOTD");
   } else {
      motdname = GetConfDir();
      motdname += "/etc/proof/motd";
   }
   if (gSystem->GetPathInfo(motdname, &id, &size, &flags, &modtime) == 0) {
      if (modtime > lasttime || show) {
         if ((motd = fopen(motdname, "r"))) {
            Int_t c;
            printf("\n");
            while ((c = getc(motd)) != EOF)
               putchar(c);
            fclose(motd);
            printf("\n");
         }
      }
   }

   if (lasttime)
      gSystem->Unlink(last);
   Int_t fd = creat(last, 0600);
   if (fd >= 0) close(fd);
   delete [] last;

   return 0;
}

//______________________________________________________________________________
TObject *TProofServ::Get(const char *namecycle)
{
   // Get object with name "name;cycle" (e.g. "aap;2") from master or client.
   // This method is called by TDirectory::Get() in case the object can not
   // be found locally.

   fSocket->Send(namecycle, kPROOF_GETOBJECT);

   TObject *idcur = 0;

   Bool_t notdone = kTRUE;
   while (notdone) {
      TMessage *mess = 0;
      if (fSocket->Recv(mess) < 0)
         return 0;
      Int_t what = mess->What();
      if (what == kMESS_OBJECT) {
         idcur = mess->ReadObject(mess->GetClass());
         notdone = kFALSE;
      } else {
         Int_t xrc = HandleSocketInput(mess, kFALSE);
         if (xrc == -1) {
            Error("Get", "command %d cannot be executed while processing", what);
         } else if (xrc == -2) {
            Error("Get", "unknown command %d ! Protocol error?", what);
         }
      }
      delete mess;
   }

   return idcur;
}

//______________________________________________________________________________
void TProofServ::RestartComputeTime()
{
   // Reset the compute time

   fCompute.Stop();
   if (fPlayer) {
      TProofProgressStatus *status = fPlayer->GetProgressStatus();
      if (status) status->SetLearnTime(fCompute.RealTime());
      Info("RestartComputeTime", "compute time restarted after %f secs (%d entries)",
                                 fCompute.RealTime(), fPlayer->GetLearnEntries());
   }
   fCompute.Start(kFALSE);
}

//______________________________________________________________________________
TDSetElement *TProofServ::GetNextPacket(Long64_t totalEntries)
{
   // Get next range of entries to be processed on this server.

   Long64_t bytesRead = 0;

   if (gPerfStats) bytesRead = gPerfStats->GetBytesRead();

   if (fCompute.Counter() > 0)
      fCompute.Stop();

   TMessage req(kPROOF_GETPACKET);
   Double_t cputime = fCompute.CpuTime();
   Double_t realtime = fCompute.RealTime();

   // Apply inflate factor, if needed
   PDB(kLoop, 2)
      Info("GetNextPacket", "inflate factor: %d"
                            " (realtime: %f, cputime: %f, entries: %lld)",
                            fInflateFactor, realtime, cputime, totalEntries);
   if (fInflateFactor > 1000) {
      UInt_t sleeptime = (UInt_t) (cputime * (fInflateFactor - 1000)) ;
      Int_t i = 0;
      for (i = kSigBus ; i <= kSigUser2 ; i++)
         gSystem->IgnoreSignal((ESignals)i, kTRUE);
      gSystem->Sleep(sleeptime);
      for (i = kSigBus ; i <= kSigUser2 ; i++)
         gSystem->IgnoreSignal((ESignals)i, kFALSE);
      realtime += sleeptime / 1000.;
      PDB(kLoop, 2)
         Info("GetNextPacket","slept %d millisec", sleeptime);
   }

   if (fProtocol > 18) {
      req << fLatency.RealTime();
      TProofProgressStatus *status = 0;
      if (fPlayer)
         status = fPlayer->GetProgressStatus();
      else {
         Error("GetNextPacket", "no progress status object");
         return 0;
      }
      // the CPU and wallclock proc times are kept in the TProofServ and here
      // added to the status object in the fPlayer.
      if (status->GetEntries() > 0) {
         PDB(kLoop, 2) status->Print(GetOrdinal());
         status->IncProcTime(realtime);
         status->IncCPUTime(cputime);
      }
      // Flag cases with problems in opening files
      if (totalEntries < 0) status->SetBit(TProofProgressStatus::kFileNotOpen);
      // Add to the message
      req << status;
      // Send tree cache information
      Long64_t cacheSize = (fPlayer) ? fPlayer->GetCacheSize() : -1;
      Int_t learnent = (fPlayer) ? fPlayer->GetLearnEntries() : -1;
      req << cacheSize << learnent;

      // Sent over the number of entries in the file, used by packetizer do not relying
      // on initial validation. Also, -1 means that the file could not be open, which is
      // used to flag files as missing
      req << totalEntries;

      PDB(kLoop, 1) {
         PDB(kLoop, 2) status->Print();
         Info("GetNextPacket","cacheSize: %lld, learnent: %d", cacheSize, learnent);
      }
      // Reset the status bits
      status->ResetBit(TProofProgressStatus::kFileNotOpen);
      status->ResetBit(TProofProgressStatus::kFileCorrupted);
      status = 0; // status is owned by the player.
   } else {
      req << fLatency.RealTime() << realtime << cputime
          << bytesRead << totalEntries;
      if (fPlayer)
         req << fPlayer->GetEventsProcessed();
   }

   fLatency.Start();
   Int_t rc = fSocket->Send(req);
   if (rc <= 0) {
      Error("GetNextPacket","Send() failed, returned %d", rc);
      return 0;
   }

   TDSetElement  *e = 0;
   Bool_t notdone = kTRUE;
   while (notdone) {

      TMessage *mess;
      if ((rc = fSocket->Recv(mess)) <= 0) {
         fLatency.Stop();
         Error("GetNextPacket","Recv() failed, returned %d", rc);
         return 0;
      }

      Int_t xrc = 0;
      TString file, dir, obj;

      Int_t what = mess->What();

      switch (what) {
         case kPROOF_GETPACKET:

            fLatency.Stop();
            (*mess) >> e;
            if (e != 0) {
               fCompute.Start();
               PDB(kLoop, 2) Info("GetNextPacket", "'%s' '%s' '%s' %lld %lld",
                                 e->GetFileName(), e->GetDirectory(),
                                 e->GetObjName(), e->GetFirst(),e->GetNum());
            } else {
               PDB(kLoop, 2) Info("GetNextPacket", "Done");
            }
            notdone = kFALSE;
            break;

         case kPROOF_STOPPROCESS:
            // if a kPROOF_STOPPROCESS message is returned to kPROOF_GETPACKET
            // GetNextPacket() will return 0 and the TPacketizer and hence
            // TEventIter will be stopped
            fLatency.Stop();
            PDB(kLoop, 2) Info("GetNextPacket:kPROOF_STOPPROCESS","received");
            break;

         default:
            xrc = HandleSocketInput(mess, kFALSE);
            if (xrc == -1) {
               Error("GetNextPacket", "command %d cannot be executed while processing", what);
            } else if (xrc == -2) {
               Error("GetNextPacket", "unknown command %d ! Protocol error?", what);
            }
            break;
      }

      delete mess;

   }

   // Done
   return e;
}

//______________________________________________________________________________
void TProofServ::GetOptions(Int_t *argc, char **argv)
{
   // Get and handle command line options. Fixed format:
   // "proofserv"|"proofslave" <confdir>

   if (*argc <= 1) {
      Fatal("GetOptions", "Must be started from proofd with arguments");
      exit(1);
   }

   if (!strcmp(argv[1], "proofserv")) {
      fMasterServ = kTRUE;
      fEndMaster = kTRUE;
   } else if (!strcmp(argv[1], "proofslave")) {
      fMasterServ = kFALSE;
      fEndMaster = kFALSE;
   } else {
      Fatal("GetOptions", "Must be started as 'proofserv' or 'proofslave'");
      exit(1);
   }

   fService = argv[1];

   // Confdir
   if (!(gSystem->Getenv("ROOTCONFDIR"))) {
      Fatal("GetOptions", "ROOTCONFDIR shell variable not set");
      exit(1);
   }
   fConfDir = gSystem->Getenv("ROOTCONFDIR");
}

//______________________________________________________________________________
void TProofServ::HandleSocketInput()
{
   // Handle input coming from the client or from the master server.

   // The idle timeout guard: stops the timer and restarts when we return from here
   TIdleTOTimerGuard itg(fIdleTOTimer);

   Bool_t all = (fgRecursive > 0) ? kFALSE : kTRUE;
   fgRecursive++;

   TMessage *mess;
   Int_t rc = 0;
   TString exmsg;

   // Check log file lenght (before the action, so we have the chance to keep the
   // latest logs)
   TruncateLogFile();

   try {
   
      // Get message
      if (fSocket->Recv(mess) <= 0 || !mess) {
         // Pending: do something more intelligent here
         // but at least get a message in the log file
         Error("HandleSocketInput", "retrieving message from input socket");
         Terminate(0);
         return;
      }
      Int_t what = mess->What();
      PDB(kCollect, 1)
         Info("HandleSocketInput", "got type %d from '%s'", what, fSocket->GetTitle());

      fNcmd++;

      if (fProof) fProof->SetActive();

      Bool_t doit = kTRUE;

      while (doit) {

         // Process the message
         rc = HandleSocketInput(mess, all);
         if (rc < 0) {
            TString emsg;
            if (rc == -1) {
               emsg.Form("HandleSocketInput: command %d cannot be executed while processing", what);
            } else if (rc == -3) {
               emsg.Form("HandleSocketInput: message %d undefined! Protocol error?", what);
            } else {
               emsg.Form("HandleSocketInput: unknown command %d! Protocol error?", what);
            }
            SendAsynMessage(emsg.Data());
         } else if (rc == 2) {
            // Add to the queue
            fQueuedMsg->Add(mess);
            PDB(kGlobal, 1)
               Info("HandleSocketInput", "message of type %d enqueued; sz: %d",
                                          what, fQueuedMsg->GetSize());
            mess = 0;
         }

         // Still something to do?
         doit = 0;
         if (fgRecursive == 1 && fQueuedMsg->GetSize() > 0) {
            // Add to the queue
            PDB(kCollect, 1)
               Info("HandleSocketInput", "processing enqueued message of type %d; left: %d",
                                          what, fQueuedMsg->GetSize());
            all = 1;
            SafeDelete(mess);
            mess = (TMessage *) fQueuedMsg->First();
            if (mess) fQueuedMsg->Remove(mess);
            doit = 1;
         }
      }
   
   } catch (std::bad_alloc &) {
      // Memory allocation problem:
      exmsg.Form("caught exception 'bad_alloc' (memory leak?) %s", fgLastMsg.Data());
   } catch (std::exception &exc) {
      // Standard exception caught
      exmsg.Form("caught standard exception '%s' %s", exc.what(), fgLastMsg.Data());
   } catch (int i) {
      // Other exception caught
      exmsg.Form("caught exception throwing %d %s", i, fgLastMsg.Data());
   } catch (const char *str) {
      // Other exception caught
      exmsg.Form("caught exception throwing '%s' %s", str, fgLastMsg.Data());
   } catch (...) {
      // Caught other exception
      exmsg.Form("caught exception <unknown> %s", fgLastMsg.Data());
   }

   // Terminate on exception
   if (!exmsg.IsNull()) {
      // Save info in the log file too
      Error("HandleSocketInput", "%s", exmsg.Data());
      // Try to warn the user
      SendAsynMessage(TString::Format("%s: %s", GetOrdinal(), exmsg.Data()));
      // Terminate
      Terminate(0);
   }
   
   // Terminate also if a high memory footprint was detected before the related
   // exception was thrwon
   if (TestBit(TProofServ::kHighMemory)) {
      // Save info in the log file too
      exmsg.Form("high-memory footprint detected during Process(...) - terminating");
      Error("HandleSocketInput", "%s", exmsg.Data());
      // Try to warn the user
      SendAsynMessage(TString::Format("%s: %s", GetOrdinal(), exmsg.Data()));
      // Terminate
      Terminate(0);
   }

   fgRecursive--;

   if (fProof) {
      // If something wrong went on during processing and we do not have
      // any worker anymore, we shutdown this session
      Bool_t masterOnly = gEnv->GetValue("Proof.MasterOnly", kFALSE);
      Int_t ngwrks = fProof->GetListOfActiveSlaves()->GetSize() + fProof->GetListOfInactiveSlaves()->GetSize();
      if (rc == 0 && ngwrks == 0 && !masterOnly) {
         SendAsynMessage(" *** No workers left: cannot continue! Terminating ... *** ");
         Terminate(0);
      }
      fProof->SetActive(kFALSE);
      // Reset PROOF to running state
      fProof->SetRunStatus(TProof::kRunning);
   }

   // Cleanup
   SafeDelete(mess);
}

//______________________________________________________________________________
Int_t TProofServ::HandleSocketInput(TMessage *mess, Bool_t all)
{
   // Process input coming from the client or from the master server.
   // If 'all' is kFALSE, process only those messages that can be handled
   // during qurey processing.
   // Returns -1 if the message could not be processed, <-1 if something went
   // wrong. Returns 1 if the action may have changed the parallel state.
   // Returns 2 if the message has to be enqueued.
   // Returns 0 otherwise

   static TStopwatch timer;
   char str[2048];
   Bool_t aborted = kFALSE;

   if (!mess) return -3;

   Int_t what = mess->What();
   PDB(kCollect, 1)
      Info("HandleSocketInput", "processing message type %d from '%s'",
                                what, fSocket->GetTitle());

   timer.Start();

   Int_t rc = 0;
   TString slb;
   TString *pslb = (fgLogToSysLog > 0) ? &slb : (TString *)0;

   switch (what) {

      case kMESS_CINT:
         if (all) {
            mess->ReadString(str, sizeof(str));
            // Make sure that the relevant files are available
            TString fn;
            if (TProof::GetFileInCmd(str, fn))
               CopyFromCache(fn, 1);
            if (IsParallel()) {
               fProof->SendCommand(str);
            } else {
               PDB(kGlobal, 1)
                  Info("HandleSocketInput:kMESS_CINT", "processing: %s...", str);
               ProcessLine(str);
            }
            LogToMaster();
         } else {
            rc = -1;
         }
         SendLogFile();
         if (pslb) slb = str;
         break;

      case kMESS_STRING:
         if (all) {
            mess->ReadString(str, sizeof(str));
         } else {
            rc = -1;
         }
         break;

      case kMESS_OBJECT:
         if (all) {
            mess->ReadObject(mess->GetClass());
         } else {
            rc = -1;
         }
         break;

      case kPROOF_GROUPVIEW:
         if (all) {
            mess->ReadString(str, sizeof(str));
            // coverity[secure_coding]
            sscanf(str, "%d %d", &fGroupId, &fGroupSize);
         } else {
            rc = -1;
         }
         break;

      case kPROOF_LOGLEVEL:
         {  UInt_t mask;
            mess->ReadString(str, sizeof(str));
            sscanf(str, "%d %u", &fLogLevel, &mask);
            gProofDebugLevel = fLogLevel;
            gProofDebugMask  = (TProofDebug::EProofDebugMask) mask;
            if (IsMaster())
               fProof->SetLogLevel(fLogLevel, mask);
         }
         break;

      case kPROOF_PING:
         {  if (IsMaster())
               fProof->Ping();
            // do nothing (ping is already acknowledged)
         }
         break;

      case kPROOF_PRINT:
         mess->ReadString(str, sizeof(str));
         Print(str);
         LogToMaster();
         SendLogFile();
         break;

      case kPROOF_RESET:
         if (all) {
            mess->ReadString(str, sizeof(str));
            Reset(str);
         } else {
            rc = -1;
         }
         break;

      case kPROOF_STATUS:
         Warning("HandleSocketInput:kPROOF_STATUS",
               "kPROOF_STATUS message is obsolete");
         fSocket->Send(fProof->GetParallel(), kPROOF_STATUS);
         break;

      case kPROOF_GETSTATS:
         SendStatistics();
         break;

      case kPROOF_GETPARALLEL:
         SendParallel();
         break;

      case kPROOF_STOP:
         if (all) {
            if (IsMaster()) {
               TString ord;
               *mess >> ord;
               PDB(kGlobal, 1)
                  Info("HandleSocketInput:kPROOF_STOP", "request for worker %s", ord.Data());
               if (fProof) fProof->TerminateWorker(ord);
            } else {
               PDB(kGlobal, 1)
                  Info("HandleSocketInput:kPROOF_STOP", "got request to terminate");
               Terminate(0);
            }
         } else {
            rc = -1;
         }
         break;

      case kPROOF_STOPPROCESS:
         if (all) {
            // this message makes only sense when the query is being processed,
            // however the message can also be received if the user pressed
            // ctrl-c, so ignore it!
            PDB(kGlobal, 1) Info("HandleSocketInput:kPROOF_STOPPROCESS","enter");
         } else {
            Long_t timeout = -1;
            (*mess) >> aborted;
            if (fProtocol > 9)
               (*mess) >> timeout;
            PDB(kGlobal, 1)
               Info("HandleSocketInput:kPROOF_STOPPROCESS",
                    "recursive mode: enter %d, %ld", aborted, timeout);
            if (fProof)
               // On the master: propagate further
               fProof->StopProcess(aborted, timeout);
            else
               // Worker: actually stop processing
               if (fPlayer)
                  fPlayer->StopProcess(aborted, timeout);
         }
         break;

      case kPROOF_PROCESS:
         {
            TProofServLogHandlerGuard hg(fLogFile, fSocket, "", fRealTimeLog);
            PDB(kGlobal, 1) Info("HandleSocketInput:kPROOF_PROCESS","enter");
            HandleProcess(mess, pslb);
            // The log file is send either in HandleProcess or HandleSubmergers.
            // The reason is that the order of various messages depend on the
            // processing mode (sync/async) and/or merging mode
         }
         break;

      case kPROOF_QUERYLIST:
         {
            HandleQueryList(mess);
            // Notify
            SendLogFile();
         }
         break;

      case kPROOF_REMOVE:
         {
            HandleRemove(mess, pslb);
            // Notify
            SendLogFile();
         }
         break;

      case kPROOF_RETRIEVE:
         {
            HandleRetrieve(mess, pslb);
            // Notify
            SendLogFile();
         }
         break;

      case kPROOF_ARCHIVE:
         {
            HandleArchive(mess, pslb);
            // Notify
            SendLogFile();
         }
         break;

      case kPROOF_MAXQUERIES:
         {  PDB(kGlobal, 1)
               Info("HandleSocketInput:kPROOF_MAXQUERIES", "Enter");
            TMessage m(kPROOF_MAXQUERIES);
            m << fMaxQueries;
            fSocket->Send(m);
            // Notify
            SendLogFile();
         }
         break;

      case kPROOF_CLEANUPSESSION:
         if (all) {
            PDB(kGlobal, 1)
               Info("HandleSocketInput:kPROOF_CLEANUPSESSION", "Enter");
            TString stag;
            (*mess) >> stag;
            if (fQMgr && fQMgr->CleanupSession(stag) == 0) {
               Printf("Session %s cleaned up", stag.Data());
            } else {
               Printf("Could not cleanup session %s", stag.Data());
            }
         } else {
            rc = -1;
         }
         // Notify
         SendLogFile();
         break;

      case kPROOF_GETENTRIES:
         {  PDB(kGlobal, 1) Info("HandleSocketInput:kPROOF_GETENTRIES", "Enter");
            Bool_t         isTree;
            TString        filename;
            TString        dir;
            TString        objname("undef");
            Long64_t       entries = -1;

            if (all) {
               (*mess) >> isTree >> filename >> dir >> objname;
               PDB(kGlobal, 2) Info("HandleSocketInput:kPROOF_GETENTRIES",
                                    "Report size of object %s (%s) in dir %s in file %s",
                                    objname.Data(), isTree ? "T" : "O",
                                    dir.Data(), filename.Data());
               entries = TDSet::GetEntries(isTree, filename, dir, objname);
               PDB(kGlobal, 2) Info("HandleSocketInput:kPROOF_GETENTRIES",
                                    "Found %lld %s", entries, isTree ? "entries" : "objects");
            } else {
               rc = -1;
            }
            TMessage answ(kPROOF_GETENTRIES);
            answ << entries << objname;
            SendLogFile(); // in case of error messages
            fSocket->Send(answ);
            PDB(kGlobal, 1) Info("HandleSocketInput:kPROOF_GETENTRIES", "Done");
         }
         break;

      case kPROOF_CHECKFILE:
         if (!all && fProtocol <= 19) {
            // Come back later
            rc = 2;
         } else {
            // Handle file checking request
            HandleCheckFile(mess, pslb);
         }
         break;

      case kPROOF_SENDFILE:
         if (!all && fProtocol <= 19) {
            // Come back later
            rc = 2;
         } else {
            mess->ReadString(str, sizeof(str));
            Long_t size;
            Int_t  bin, fw = 1;
            char   name[1024];
            if (fProtocol > 5) {
               sscanf(str, "%1023s %d %ld %d", name, &bin, &size, &fw);
            } else {
               sscanf(str, "%1023s %d %ld", name, &bin, &size);
            }
            TString fnam(name);
            Bool_t copytocache = kTRUE;
            if (fnam.BeginsWith("cache:")) {
               fnam.ReplaceAll("cache:", TString::Format("%s/", fCacheDir.Data()));
               copytocache = kFALSE;
            }
            if (size > 0) {
               ReceiveFile(fnam, bin ? kTRUE : kFALSE, size);
            } else {
               // Take it from the cache
               if (!fnam.BeginsWith(fCacheDir.Data())) {
                  fnam.Insert(0, TString::Format("%s/", fCacheDir.Data()));
               }
            }
            // copy file to cache if not a PAR file
            if (copytocache && size > 0 &&
                strncmp(fPackageDir, name, fPackageDir.Length()))
               CopyToCache(name, 0);
            if (IsMaster() && fw == 1) {
               Int_t opt = TProof::kForward | TProof::kCp;
               if (bin)
                  opt |= TProof::kBinary;
               PDB(kGlobal, 1)
                  Info("HandleSocketInput","forwarding file: %s", fnam.Data());
               if (fProof->SendFile(fnam, opt, (copytocache ? "cache" : "")) < 0) {
                  Error("HandleSocketInput", "forwarding file: %s", fnam.Data());
               }
            }
            if (fProtocol > 19) fSocket->Send(kPROOF_SENDFILE);
         }
         break;

      case kPROOF_LOGFILE:
         {
            Int_t start, end;
            (*mess) >> start >> end;
            PDB(kGlobal, 1)
               Info("HandleSocketInput:kPROOF_LOGFILE",
                    "Logfile request - byte range: %d - %d", start, end);

            LogToMaster();
            SendLogFile(0, start, end);
         }
         break;

      case kPROOF_PARALLEL:
         if (all) {
            if (IsMaster()) {
               Int_t nodes;
               Bool_t random = kFALSE;
               (*mess) >> nodes;
               if ((mess->BufferSize() > mess->Length()))
                  (*mess) >> random;
               if (fProof) fProof->SetParallel(nodes, random);
               rc = 1;
            }
         } else {
            rc = -1;
         }
         // Notify
         SendLogFile();
         break;

      case kPROOF_CACHE:
         if (!all && fProtocol <= 19) {
            // Come back later
            rc = 2;
         } else {
            TProofServLogHandlerGuard hg(fLogFile, fSocket, "", fRealTimeLog);
            PDB(kGlobal, 1) Info("HandleSocketInput:kPROOF_CACHE","enter");
            Int_t status = HandleCache(mess, pslb);
            // Notify
            SendLogFile(status);
         }
         break;

      case kPROOF_WORKERLISTS:
         if (all) {
            if (IsMaster())
               HandleWorkerLists(mess);
            else
               Warning("HandleSocketInput:kPROOF_WORKERLISTS",
                       "Action meaning-less on worker nodes: protocol error?");
         } else {
            rc = -1;
         }
         // Notify
         SendLogFile();
         break;

      case kPROOF_GETSLAVEINFO:
         if (all) {
            PDB(kGlobal, 1) Info("HandleSocketInput:kPROOF_GETSLAVEINFO", "Enter");
            if (IsMaster()) {
               TList *info = fProof->GetListOfSlaveInfos();
               TMessage answ(kPROOF_GETSLAVEINFO);
               answ << info;
               fSocket->Send(answ);
            } else {
               TMessage answ(kPROOF_GETSLAVEINFO);
               TList *info = new TList;
               TSlaveInfo *wi = new TSlaveInfo(GetOrdinal(), TUrl(gSystem->HostName()).GetHostFQDN(), 0, "", GetDataDir());
               SysInfo_t si;
               gSystem->GetSysInfo(&si);
               wi->SetSysInfo(si);
               info->Add(wi);
               answ << (TList *)info;
               fSocket->Send(answ);
               info->SetOwner(kTRUE);
               delete info;
            }

            PDB(kGlobal, 1) Info("HandleSocketInput:kPROOF_GETSLAVEINFO", "Done");
         } else {
            TMessage answ(kPROOF_GETSLAVEINFO);
            answ << (TList *)0;
            fSocket->Send(answ);
            rc = -1;
         }
         break;

      case kPROOF_GETTREEHEADER:
         if (all) {
            PDB(kGlobal, 1) Info("HandleSocketInput:kPROOF_GETTREEHEADER", "Enter");

            TVirtualProofPlayer *p = TVirtualProofPlayer::Create("slave", 0, fSocket);
            p->HandleGetTreeHeader(mess);
            delete p;

            PDB(kGlobal, 1) Info("HandleSocketInput:kPROOF_GETTREEHEADER", "Done");
         } else {
            TMessage answ(kPROOF_GETTREEHEADER);
            answ << TString("Failed") << (TObject *)0;
            fSocket->Send(answ);
            rc = -1;
         }
         break;

      case kPROOF_GETOUTPUTLIST:
         {  PDB(kGlobal, 1) Info("HandleSocketInput:kPROOF_GETOUTPUTLIST", "Enter");
            TList* outputList = 0;
            if (IsMaster()) {
               outputList = fProof->GetOutputList();
               if (!outputList)
                  outputList = new TList();
            } else {
               outputList = new TList();
               if (fProof->GetPlayer()) {
                  TList *olist = fProof->GetPlayer()->GetOutputList();
                  TIter next(olist);
                  TObject *o;
                  while ( (o = next()) ) {
                     outputList->Add(new TNamed(o->GetName(), ""));
                  }
               }
            }
            outputList->SetOwner();
            TMessage answ(kPROOF_GETOUTPUTLIST);
            answ << outputList;
            fSocket->Send(answ);
            delete outputList;
            PDB(kGlobal, 1) Info("HandleSocketInput:kPROOF_GETOUTPUTLIST", "Done");
         }
         break;

      case kPROOF_VALIDATE_DSET:
         if (all) {
            PDB(kGlobal, 1)
               Info("HandleSocketInput:kPROOF_VALIDATE_DSET", "Enter");

            TDSet* dset = 0;
            (*mess) >> dset;

            if (IsMaster()) fProof->ValidateDSet(dset);
            else dset->Validate();

            TMessage answ(kPROOF_VALIDATE_DSET);
            answ << dset;
            fSocket->Send(answ);
            delete dset;
            PDB(kGlobal, 1)
               Info("HandleSocketInput:kPROOF_VALIDATE_DSET", "Done");
         } else {
            rc = -1;
         }
         // Notify
         SendLogFile();
         break;

      case kPROOF_DATA_READY:
         if (all) {
            PDB(kGlobal, 1) Info("HandleSocketInput:kPROOF_DATA_READY", "Enter");
            TMessage answ(kPROOF_DATA_READY);
            if (IsMaster()) {
               Long64_t totalbytes = 0, bytesready = 0;
               Bool_t dataready = fProof->IsDataReady(totalbytes, bytesready);
               answ << dataready << totalbytes << bytesready;
            } else {
               Error("HandleSocketInput:kPROOF_DATA_READY",
                     "This message should not be sent to slaves");
               answ << kFALSE << Long64_t(0) << Long64_t(0);
            }
            fSocket->Send(answ);
            PDB(kGlobal, 1) Info("HandleSocketInput:kPROOF_DATA_READY", "Done");
         } else {
            TMessage answ(kPROOF_DATA_READY);
            answ << kFALSE << Long64_t(0) << Long64_t(0);
            fSocket->Send(answ);
            rc = -1;
         }
         // Notify
         SendLogFile();
         break;

      case kPROOF_DATASETS:
         {  Int_t xrc = -1;
            if (fProtocol > 16) {
               xrc = HandleDataSets(mess, pslb);
            } else {
               Error("HandleSocketInput", "old client: no or incompatible dataset support");
            }
            SendLogFile(xrc);
         }
         break;

      case kPROOF_SUBMERGER:
         {  HandleSubmerger(mess);
         }
         break;

      case kPROOF_LIB_INC_PATH:
         if (all) {
            HandleLibIncPath(mess);
         } else {
            rc = -1;
         }
         // Notify the client
         SendLogFile();
         break;

      case kPROOF_REALTIMELOG:
         {  Bool_t on;
            (*mess) >> on;
            PDB(kGlobal, 1)
               Info("HandleSocketInput:kPROOF_REALTIMELOG",
                    "setting real-time logging %s", (on ? "ON" : "OFF"));
            fRealTimeLog = on;
            // Forward the request to lower levels
            if (IsMaster())
               fProof->SetRealTimeLog(on);
         }
         break;

      case kPROOF_FORK:
         if (all) {
            HandleFork(mess);
            LogToMaster();
         } else {
            rc = -1;
         }
         SendLogFile();
         break;

      case kPROOF_STARTPROCESS:
         if (all) {
            // This message resumes the session; should not come during processing.

            if (WaitingQueries() == 0) {
               Error("HandleSocketInput", "no queries enqueued");
               break;
            }

            // Similar to handle process
            // get the list of workers and start them
            TList *workerList = (fProof->UseDynamicStartup()) ? new TList : (TList *)0;
            Int_t pc = 0;
            EQueryAction retVal = GetWorkers(workerList, pc, kTRUE);

            if (retVal == TProofServ::kQueryOK) {
               Int_t ret = 0;
               if (workerList && (ret = fProof->AddWorkers(workerList)) < 0) {
                  Error("HandleSocketInput", "adding a list of worker nodes returned: %d", ret);
               } else {
                  ProcessNext(pslb);
                  // Set idle
                  SetIdle(kTRUE);
                  // Signal the client that we are idle
                  TMessage m(kPROOF_SETIDLE);
                  Bool_t waiting = (WaitingQueries() > 0) ? kTRUE : kFALSE;
                  m << waiting;
                  fSocket->Send(m);
               }
            } else {
               if (retVal == TProofServ::kQueryStop) {
                  Error("HandleSocketInput", "error getting list of worker nodes");
               } else if (retVal != TProofServ::kQueryEnqueued) {
                  Warning("HandleSocketInput", "query was re-queued!");
               } else {
                  Error("HandleSocketInput", "unexpected answer: %d", retVal);
                  break;
               }
            }

         }
         break;

      case kPROOF_GOASYNC:
         {  // The client requested to switch to asynchronous mode:
            // communicate the sequential number of the running query for later
            // identification, if any
            if (!IsIdle() && fPlayer) {
               // Get query currently being processed
               TProofQueryResult *pq = (TProofQueryResult *) fPlayer->GetCurrentQuery();
               TMessage m(kPROOF_QUERYSUBMITTED);
               m << pq->GetSeqNum() << kFALSE;
               fSocket->Send(m);
            } else {
               // Idle or undefined: nothing to do; ignore
               SendAsynMessage("Processing request to go asynchronous:"
                               " idle or undefined player - ignoring");
            }
         }
         break;

      default:
         Error("HandleSocketInput", "unknown command %d", what);
         rc = -2;
         break;
   }

   fRealTime += (Float_t)timer.RealTime();
   fCpuTime  += (Float_t)timer.CpuTime();

   if (!(slb.IsNull()) || fgLogToSysLog > 1) {
      TString s;
      s.Form("%s %d %.3f %.3f %s", fgSysLogEntity.Data(),
                                   what, timer.RealTime(), timer.CpuTime(), slb.Data());
      gSystem->Syslog(kLogNotice, s.Data());
   }

   // Done
   return rc;
}

//______________________________________________________________________________
Bool_t TProofServ::AcceptResults(Int_t connections, TVirtualProofPlayer *mergerPlayer)
{
   // Accept and merge results from a set of workers

   TMessage *mess = new TMessage();
   Int_t mergedWorkers = 0;

   PDB(kSubmerger, 1)  Info("AcceptResults", "enter");

   // Overall result of this procedure
   Bool_t result = kTRUE;

   fMergingMonitor = new TMonitor();
   fMergingMonitor->Add(fMergingSocket);

   Int_t numworkers = 0;
   while (fMergingMonitor->GetActive() > 0 && mergedWorkers <  connections) {

      TSocket *s = fMergingMonitor->Select();
      if (!s) {
         Info("AcceptResults", "interrupt!");
         result = kFALSE;
         break;
      }

      if (s == fMergingSocket) {
         // New incoming connection
         TSocket *sw = fMergingSocket->Accept();
         fMergingMonitor->Add(sw);

         PDB(kSubmerger, 2)
            Info("AcceptResults", "connection from a worker accepted on merger %s ",
                                  fOrdinal.Data());
         // All assigned workers are connected
         if (++numworkers >= connections)
            fMergingMonitor->Remove(fMergingSocket);
      } else {
         s->Recv(mess);
         PDB(kSubmerger, 2)
            Info("AcceptResults", "message received: %d ", (mess ? mess->What() : 0));
         if (!mess) {
            Error("AcceptResults", "message received: %p ", mess);
            continue;
         }
         Int_t type = 0;

         // Read output objec(s) from the received message
         while ((mess->BufferSize() > mess->Length())) {
            (*mess) >> type;

            PDB(kSubmerger, 2) Info("AcceptResults", " type %d ", type);
            if (type == 2) {
               mergedWorkers++;
               PDB(kSubmerger, 2)
                  Info("AcceptResults",
                       "a new worker has been mergerd. Total merged workers: %d",
                       mergedWorkers);
            }
            TObject *o = mess->ReadObject(TObject::Class());
            if (mergerPlayer->AddOutputObject(o) == 1) {
               // Remove the object if it has been merged
               PDB(kSubmerger, 2)  Info("AcceptResults", "removing %p (has been merged)", o);
               SafeDelete(o);
            } else
               PDB(kSubmerger, 2) Info("AcceptResults", "%p not merged yet", o);
         }
      }
   }
   fMergingMonitor->DeActivateAll();

   TList* sockets = fMergingMonitor->GetListOfDeActives();
   Int_t size = sockets->GetSize();
   for (Int_t i =0; i< size; ++i){
      ((TSocket*)(sockets->At(i)))->Close();
      PDB(kSubmerger, 2) Info("AcceptResults", "closing socket");
      delete ((TSocket*)(sockets->At(i)));
   }

   fMergingMonitor->RemoveAll();
   SafeDelete(fMergingMonitor);

   PDB(kSubmerger, 2) Info("AcceptResults", "exit: %d", result);
   return result;
}

//______________________________________________________________________________
void TProofServ::HandleUrgentData()
{
   // Handle Out-Of-Band data sent by the master or client.

   char  oob_byte;
   Int_t n, nch, wasted = 0;

   const Int_t kBufSize = 1024;
   char waste[kBufSize];

   // Real-time notification of messages
   TProofServLogHandlerGuard hg(fLogFile, fSocket, "", fRealTimeLog);

   PDB(kGlobal, 5)
      Info("HandleUrgentData", "handling oob...");

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

   PDB(kGlobal, 5)
      Info("HandleUrgentData", "got OOB byte: %d\n", oob_byte);

   if (fProof) fProof->SetActive();

   switch (oob_byte) {

      case TProof::kHardInterrupt:
         Info("HandleUrgentData", "*** Hard Interrupt");

         // If master server, propagate interrupt to slaves
         if (IsMaster())
            fProof->Interrupt(TProof::kHardInterrupt);

         // Flush input socket
         while (1) {
            Int_t atmark;

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

         SendLogFile();

         break;

      case TProof::kSoftInterrupt:
         Info("HandleUrgentData", "Soft Interrupt");

         // If master server, propagate interrupt to slaves
         if (IsMaster())
            fProof->Interrupt(TProof::kSoftInterrupt);

         if (wasted) {
            Error("HandleUrgentData", "soft interrupt flushed stream");
            break;
         }

         Interrupt();

         SendLogFile();

         break;

      case TProof::kShutdownInterrupt:
         Info("HandleUrgentData", "Shutdown Interrupt");

         // If master server, propagate interrupt to slaves
         if (IsMaster())
            fProof->Interrupt(TProof::kShutdownInterrupt);

         Terminate(0);

         break;

      default:
         Error("HandleUrgentData", "unexpected OOB byte");
         break;
   }

   if (fProof) fProof->SetActive(kFALSE);
}

//______________________________________________________________________________
void TProofServ::HandleSigPipe()
{
   // Called when the client is not alive anymore (i.e. when kKeepAlive
   // has failed).

   // Real-time notification of messages
   TProofServLogHandlerGuard hg(fLogFile, fSocket, "", fRealTimeLog);

   if (IsMaster()) {
      // Check if we are here because client is closed. Try to ping client,
      // if that works it we are here because some slave died
      if (fSocket->Send(kPROOF_PING | kMESS_ACK) < 0) {
         Info("HandleSigPipe", "keepAlive probe failed");
         // Tell slaves we are going to close since there is no client anymore

         fProof->SetActive();
         fProof->Interrupt(TProof::kShutdownInterrupt);
         fProof->SetActive(kFALSE);
         Terminate(0);
      }
   } else {
      Info("HandleSigPipe", "keepAlive probe failed");
      Terminate(0);  // will not return from here....
   }
}

//______________________________________________________________________________
Bool_t TProofServ::IsParallel() const
{
   // True if in parallel mode.

   if (IsMaster() && fProof)
      return fProof->IsParallel();

   // false in case we are a slave
   return kFALSE;
}

//______________________________________________________________________________
void TProofServ::Print(Option_t *option) const
{
   // Print status of slave server.

   if (IsMaster() && fProof)
      fProof->Print(option);
   else
      Printf("This is worker %s", gSystem->HostName());
}

//______________________________________________________________________________
void TProofServ::RedirectOutput(const char *dir, const char *mode)
{
   // Redirect stdout to a log file. This log file will be flushed to the
   // client or master after each command.

   char logfile[512];

   TString sdir = (dir && strlen(dir) > 0) ? dir : fSessionDir.Data();
   if (IsMaster()) {
      snprintf(logfile, 512, "%s/master-%s.log", sdir.Data(), fOrdinal.Data());
   } else {
      snprintf(logfile, 512, "%s/worker-%s.log", sdir.Data(), fOrdinal.Data());
   }

   if ((freopen(logfile, mode, stdout)) == 0)
      SysError("RedirectOutput", "could not freopen stdout (%s)", logfile);

   if ((dup2(fileno(stdout), fileno(stderr))) < 0)
      SysError("RedirectOutput", "could not redirect stderr");

   if ((fLogFile = fopen(logfile, "r")) == 0)
      SysError("RedirectOutput", "could not open logfile '%s'", logfile);

   // from this point on stdout and stderr are properly redirected
   if (fProtocol < 4 && fWorkDir != TString::Format("~/%s", kPROOF_WorkDir)) {
      Warning("RedirectOutput", "no way to tell master (or client) where"
              " to upload packages");
   }
}

//______________________________________________________________________________
void TProofServ::Reset(const char *dir)
{
   // Reset PROOF environment to be ready for execution of next command.

   // First go to new directory. Check first that we got a reasonable path;
   // in PROOF-Lite it may not be the case
   TString dd(dir);
   if (!dd.BeginsWith("proofserv")) {
      Int_t ic = dd.Index(":");
      if (ic != kNPOS)
         dd.Replace(0, ic, "proofserv");
   }
   gDirectory->cd(dd.Data());

   // Clear interpreter environment.
   gROOT->Reset();

   // Make sure current directory is empty (don't delete anything when
   // we happen to be in the ROOT memory only directory!?)
   if (gDirectory != gROOT) {
      gDirectory->Delete();
   }

   if (IsMaster()) fProof->SendCurrentState();
}

//______________________________________________________________________________
Int_t TProofServ::ReceiveFile(const char *file, Bool_t bin, Long64_t size)
{
   // Receive a file, either sent by a client or a master server.
   // If bin is true it is a binary file, other wise it is an ASCII
   // file and we need to check for Windows \r tokens. Returns -1 in
   // case of error, 0 otherwise.

   if (size <= 0) return 0;

   // open file, overwrite already existing file
   Int_t fd = open(file, O_CREAT | O_TRUNC | O_WRONLY, 0600);
   if (fd < 0) {
      SysError("ReceiveFile", "error opening file %s", file);
      return -1;
   }

   const Int_t kMAXBUF = 16384;  //32768  //16384  //65536;
   char buf[kMAXBUF], cpy[kMAXBUF];

   Int_t    left, r;
   Long64_t filesize = 0;

   while (filesize < size) {
      left = Int_t(size - filesize);
      if (left > kMAXBUF)
         left = kMAXBUF;
      r = fSocket->RecvRaw(&buf, left);
      if (r > 0) {
         char *p = buf;

         filesize += r;
         while (r) {
            Int_t w;

            if (!bin) {
               Int_t k = 0, i = 0, j = 0;
               char *q;
               while (i < r) {
                  if (p[i] == '\r') {
                     i++;
                     k++;
                  }
                  cpy[j++] = buf[i++];
               }
               q = cpy;
               r -= k;
               w = write(fd, q, r);
            } else {
               w = write(fd, p, r);
            }

            if (w < 0) {
               SysError("ReceiveFile", "error writing to file %s", file);
               close(fd);
               return -1;
            }
            r -= w;
            p += w;
         }
      } else if (r < 0) {
         Error("ReceiveFile", "error during receiving file %s", file);
         close(fd);
         return -1;
      }
   }

   close(fd);

   chmod(file, 0644);

   return 0;
}

//______________________________________________________________________________
void TProofServ::Run(Bool_t retrn)
{
   // Main server eventloop.

   // Setup the server
   if (CreateServer() == 0) {

      // Run the main event loop
      TApplication::Run(retrn);
   }
}

//______________________________________________________________________________
void TProofServ::SendLogFile(Int_t status, Int_t start, Int_t end)
{
   // Send log file to master.
   // If start > -1 send only bytes in the range from start to end,
   // if end <= start send everything from start.

   // Determine the number of bytes left to be read from the log file.
   fflush(stdout);

   // On workers we do not send the logs to masters (to avoid duplication of
   // text) unless asked explicitely, e.g. after an Exec(...) request.
   if (!IsMaster()) {
      if (!fSendLogToMaster) {
         FlushLogFile();
      } else {
         // Decide case by case
         LogToMaster(kFALSE);
      }
   }

   off_t ltot=0, lnow=0;
   Int_t left = -1;
   Bool_t adhoc = kFALSE;

   if (fLogFileDes > -1) {
      ltot = lseek(fileno(stdout),   (off_t) 0, SEEK_END);
      lnow = lseek(fLogFileDes, (off_t) 0, SEEK_CUR);

      if (ltot >= 0 && lnow >= 0) {
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
      }
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

         if (fSocket->SendRaw(buf, len) < 0) {
            SysError("SendLogFile", "error sending log file");
            break;
         }

         // Update counters
         left -= len;
         wanted = (left > kMAXBUF) ? kMAXBUF : left;

      } while (len > 0 && left > 0);
   }

   // Restore initial position if partial send
   if (adhoc && lnow >=0 )
      lseek(fLogFileDes, lnow, SEEK_SET);

   TMessage mess(kPROOF_LOGDONE);
   if (IsMaster())
      mess << status << (fProof ? fProof->GetParallel() : 0);
   else
      mess << status << (Int_t) 1;

   fSocket->Send(mess);

   PDB(kGlobal, 1) Info("SendLogFile", "kPROOF_LOGDONE sent");
}

//______________________________________________________________________________
void TProofServ::SendStatistics()
{
   // Send statistics of slave server to master or client.

   Long64_t bytesread = TFile::GetFileBytesRead();
   Float_t cputime = fCpuTime, realtime = fRealTime;
   if (IsMaster()) {
      bytesread = fProof->GetBytesRead();
      cputime = fProof->GetCpuTime();
      realtime = fProof->GetRealTime();
   }

   TMessage mess(kPROOF_GETSTATS);
   TString workdir = gSystem->WorkingDirectory();  // expect TString on other side
   mess << bytesread << realtime << cputime << workdir;
   if (fProtocol >= 4) mess << TString(gProofServ->GetWorkDir());
   mess << TString(gProofServ->GetImage());
   fSocket->Send(mess);
}

//______________________________________________________________________________
void TProofServ::SendParallel(Bool_t async)
{
   // Send number of parallel nodes to master or client.

   Int_t nparallel = 0;
   if (IsMaster()) {
      fProof->AskParallel();
      nparallel = fProof->GetParallel();
   } else {
      nparallel = 1;
   }

   TMessage mess(kPROOF_GETPARALLEL);
   mess << nparallel << async;
   fSocket->Send(mess);
}

//______________________________________________________________________________
Int_t TProofServ::UnloadPackage(const char *package)
{
   // Removes link to package in working directory,
   // removes entry from include path,
   // removes entry from enabled package list,
   // does not currently remove entry from interpreter include path.
   // Returns -1 in case of error, 0 otherwise.

   TObjString *pack = (TObjString *) fEnabledPackages->FindObject(package);
   if (pack) {

      // Remove entry from include path
      TString aclicincpath = gSystem->GetIncludePath();
      TString cintincpath = gInterpreter->GetIncludePath();
      // remove interpreter part of gSystem->GetIncludePath()
      aclicincpath.Remove(aclicincpath.Length() - cintincpath.Length() - 1);
      // remove package's include path
      aclicincpath.ReplaceAll(TString(" -I") + package, "");
      gSystem->SetIncludePath(aclicincpath);

      //TODO reset interpreter include path

      // remove entry from enabled packages list
      delete fEnabledPackages->Remove(pack);
      PDB(kPackage, 1)
         Info("UnloadPackage",
              "package %s successfully unloaded", package);
   }

   // Cleanup the link, if there
   if (!gSystem->AccessPathName(package))
      if (gSystem->Unlink(package) != 0)
         Warning("UnloadPackage", "unable to remove symlink to %s", package);

   // We are done
   return 0;
}

//______________________________________________________________________________
Int_t TProofServ::UnloadPackages()
{
   // Unloads all enabled packages. Returns -1 in case of error, 0 otherwise.

   // Iterate over packages and remove each package
   TIter nextpackage(fEnabledPackages);
   while (TObjString* objstr = dynamic_cast<TObjString*>(nextpackage()))
      if (UnloadPackage(objstr->String()) != 0)
         return -1;

   PDB(kPackage, 1)
      Info("UnloadPackages",
           "packages successfully unloaded");

   return 0;
}

//______________________________________________________________________________
Int_t TProofServ::Setup()
{
   // Print the ProofServ logo on standard output.
   // Return 0 on success, -1 on failure

   char str[512];

   if (IsMaster()) {
      snprintf(str, 512, "**** Welcome to the PROOF server @ %s ****", gSystem->HostName());
   } else {
      snprintf(str, 512, "**** PROOF slave server @ %s started ****", gSystem->HostName());
   }

   if (fSocket->Send(str) != 1+static_cast<Int_t>(strlen(str))) {
      Error("Setup", "failed to send proof server startup message");
      return -1;
   }

   // exchange protocol level between client and master and between
   // master and slave
   Int_t what;
   if (fSocket->Recv(fProtocol, what) != 2*sizeof(Int_t)) {
      Error("Setup", "failed to receive remote proof protocol");
      return -1;
   }
   if (fSocket->Send(kPROOF_Protocol, kROOTD_PROTOCOL) != 2*sizeof(Int_t)) {
      Error("Setup", "failed to send local proof protocol");
      return -1;
   }

   // If old version, setup authentication related stuff
   if (fProtocol < 5) {
      TString wconf;
      if (OldAuthSetup(wconf) != 0) {
         Error("Setup", "OldAuthSetup: failed to setup authentication");
         return -1;
      }
      if (IsMaster()) {
         fConfFile = wconf;
         fWorkDir.Form("~/%s", kPROOF_WorkDir);
      } else {
         if (fProtocol < 4) {
            fWorkDir.Form("~/%s", kPROOF_WorkDir);
         } else {
            fWorkDir = wconf;
            if (fWorkDir.IsNull()) fWorkDir.Form("~/%s", kPROOF_WorkDir);
         }
      }
   } else {

      // Receive some useful information
      TMessage *mess;
      if ((fSocket->Recv(mess) <= 0) || !mess) {
         Error("Setup", "failed to receive ordinal and config info");
         return -1;
      }
      if (IsMaster()) {
         (*mess) >> fUser >> fOrdinal >> fConfFile;
         fWorkDir = gEnv->GetValue("ProofServ.Sandbox", TString::Format("~/%s", kPROOF_WorkDir));
      } else {
         (*mess) >> fUser >> fOrdinal >> fWorkDir;
         if (fWorkDir.IsNull())
            fWorkDir = gEnv->GetValue("ProofServ.Sandbox", TString::Format("~/%s", kPROOF_WorkDir));
      }
      // Set the correct prefix
      if (fOrdinal != "-1")
         fPrefix += fOrdinal;
      TProofServLogHandler::SetDefaultPrefix(fPrefix);
      delete mess;
   }

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
         Info("Setup", "invalid config file %s (missing or unreadable",
                        resources.GetFileName().Data());
      }
   }

   // Set $HOME and $PATH. The HOME directory was already set to the
   // user's home directory by proofd.
   gSystem->Setenv("HOME", gSystem->HomeDirectory());

   // Add user name in case of non default workdir
   if (fWorkDir.BeginsWith("/") &&
      !fWorkDir.BeginsWith(gSystem->HomeDirectory())) {
      if (!fWorkDir.EndsWith("/"))
         fWorkDir += "/";
      UserGroup_t *u = gSystem->GetUserInfo();
      if (u) {
         fWorkDir += u->fUser;
         delete u;
      }
   }

   // Goto to the main PROOF working directory
   char *workdir = gSystem->ExpandPathName(fWorkDir.Data());
   fWorkDir = workdir;
   delete [] workdir;
   if (gProofDebugLevel > 0)
      Info("Setup", "working directory set to %s", fWorkDir.Data());

   // host first name
   TString host = gSystem->HostName();
   if (host.Index(".") != kNPOS)
      host.Remove(host.Index("."));

   // Session tag
   fSessionTag.Form("%s-%s-%ld-%d", fOrdinal.Data(), host.Data(),
                    (Long_t)TTimeStamp().GetSec(),gSystem->GetPid());
   fTopSessionTag = fSessionTag;

   // create session directory and make it the working directory
   fSessionDir = fWorkDir;
   if (IsMaster())
      fSessionDir += "/master-";
   else
      fSessionDir += "/slave-";
   fSessionDir += fSessionTag;

   // Common setup
   if (SetupCommon() != 0) {
      Error("Setup", "common setup failed");
      return -1;
   }

   // Incoming OOB should generate a SIGURG
   fSocket->SetOption(kProcessGroup, gSystem->GetPid());

   // Send packets off immediately to reduce latency
   fSocket->SetOption(kNoDelay, 1);

   // Check every two hours if client is still alive
   fSocket->SetOption(kKeepAlive, 1);

   // Done
   return 0;
}

//______________________________________________________________________________
Int_t TProofServ::SetupCommon()
{
   // Common part (between TProofServ and TXProofServ) of the setup phase.
   // Return 0 on success, -1 on error

   // deny write access for group and world
   gSystem->Umask(022);

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
   if (compiler.Index("is ") != kNPOS)
      compiler.Remove(0, compiler.Index("is ") + 3);
   compiler = gSystem->DirName(compiler);
   if (!bindir.IsNull()) bindir += ":";
   bindir += compiler;
#endif
   if (!bindir.IsNull()) bindir += ":";
   bindir += "/bin:/usr/bin:/usr/local/bin";
   // Add bindir to PATH
   TString path(gSystem->Getenv("PATH"));
   if (!path.IsNull()) path.Insert(0, ":");
   path.Insert(0, bindir);
   gSystem->Setenv("PATH", path);
#endif

   if (gSystem->AccessPathName(fWorkDir)) {
      gSystem->mkdir(fWorkDir, kTRUE);
      if (!gSystem->ChangeDirectory(fWorkDir)) {
         Error("SetupCommon", "can not change to PROOF directory %s",
               fWorkDir.Data());
         return -1;
      }
   } else {
      if (!gSystem->ChangeDirectory(fWorkDir)) {
         gSystem->Unlink(fWorkDir);
         gSystem->mkdir(fWorkDir, kTRUE);
         if (!gSystem->ChangeDirectory(fWorkDir)) {
            Error("SetupCommon", "can not change to PROOF directory %s",
                     fWorkDir.Data());
            return -1;
         }
      }
   }

   // Set group
   fGroup = gEnv->GetValue("ProofServ.ProofGroup", "default");

   // Check and make sure "cache" directory exists
   fCacheDir = gEnv->GetValue("ProofServ.CacheDir",
                               TString::Format("%s/%s", fWorkDir.Data(), kPROOF_CacheDir));
   ResolveKeywords(fCacheDir);
   if (gSystem->AccessPathName(fCacheDir))
      gSystem->mkdir(fCacheDir, kTRUE);
   if (gProofDebugLevel > 0)
      Info("SetupCommon", "cache directory set to %s", fCacheDir.Data());
   fCacheLock =
      new TProofLockPath(TString::Format("%s/%s%s",
                         gSystem->TempDirectory(), kPROOF_CacheLockFile,
                         TString(fCacheDir).ReplaceAll("/","%").Data()));

   // Check and make sure "packages" directory exists
   fPackageDir = gEnv->GetValue("ProofServ.PackageDir",
                                 TString::Format("%s/%s", fWorkDir.Data(), kPROOF_PackDir));
   ResolveKeywords(fPackageDir);
   if (gSystem->AccessPathName(fPackageDir))
      gSystem->mkdir(fPackageDir, kTRUE);
   if (gProofDebugLevel > 0)
      Info("SetupCommon", "package directory set to %s", fPackageDir.Data());
   fPackageLock =
      new TProofLockPath(TString::Format("%s/%s%s",
                         gSystem->TempDirectory(), kPROOF_PackageLockFile,
                         TString(fPackageDir).ReplaceAll("/","%").Data()));

   // Check and make sure "data" directory exists
   fDataDir = gEnv->GetValue("ProofServ.DataDir","");
   if (fDataDir.IsNull()) {
      // Use default
      fDataDir.Form("%s/%s/<ord>/<stag>", fWorkDir.Data(), kPROOF_DataDir);
   }
   ResolveKeywords(fDataDir);
   if (gSystem->AccessPathName(fDataDir))
      gSystem->mkdir(fDataDir, kTRUE);
   if (gProofDebugLevel > 0)
      Info("SetupCommon", "data directory set to %s", fDataDir.Data());

   // List of directories where to look for global packages
   TString globpack = gEnv->GetValue("Proof.GlobalPackageDirs","");
   if (globpack.Length() > 0) {
      Int_t ng = 0;
      Int_t from = 0;
      TString ldir;
      while (globpack.Tokenize(ldir, from, ":")) {
         if (gSystem->AccessPathName(ldir, kReadPermission)) {
            Warning("SetupCommon", "directory for global packages %s does not"
                             " exist or is not readable", ldir.Data());
         } else {
            // Add to the list, key will be "G<ng>", i.e. "G0", "G1", ...
            TString key;
            key.Form("G%d", ng++);
            if (!fGlobalPackageDirList) {
               fGlobalPackageDirList = new THashList();
               fGlobalPackageDirList->SetOwner();
            }
            fGlobalPackageDirList->Add(new TNamed(key,ldir));
            Info("SetupCommon", "directory for global packages %s added to the list",
                          ldir.Data());
            FlushLogFile();
         }
      }
   }

   // Check the session dir
   if (fSessionDir != gSystem->WorkingDirectory()) {
      ResolveKeywords(fSessionDir);
      if (gSystem->AccessPathName(fSessionDir))
         gSystem->mkdir(fSessionDir, kTRUE);
      if (!gSystem->ChangeDirectory(fSessionDir)) {
         Error("SetupCommon", "can not change to working directory '%s'",
                              fSessionDir.Data());
         return -1;
      }
   }
   gSystem->Setenv("PROOF_SANDBOX", fSessionDir);
   if (gProofDebugLevel > 0)
      Info("SetupCommon", "session dir is '%s'", fSessionDir.Data());

   // On masters, check and make sure that "queries" and "datasets"
   // directories exist
   if (IsMaster()) {

      // Make sure that the 'queries' dir exist
      fQueryDir = fWorkDir;
      fQueryDir += TString("/") + kPROOF_QueryDir;
      ResolveKeywords(fQueryDir);
      if (gSystem->AccessPathName(fQueryDir))
         gSystem->mkdir(fQueryDir, kTRUE);
      fQueryDir += TString("/session-") + fTopSessionTag;
      if (gSystem->AccessPathName(fQueryDir))
         gSystem->mkdir(fQueryDir, kTRUE);
      if (gProofDebugLevel > 0)
         Info("SetupCommon", "queries dir is %s", fQueryDir.Data());

      // Create 'queries' locker instance and lock it
      fQueryLock = new TProofLockPath(TString::Format("%s/%s%s-%s",
                       gSystem->TempDirectory(),
                       kPROOF_QueryLockFile, fTopSessionTag.Data(),
                       TString(fQueryDir).ReplaceAll("/","%").Data()));
      fQueryLock->Lock();
      // Create the query manager
      fQMgr = new TQueryResultManager(fQueryDir, fSessionTag, fSessionDir,
                                      fQueryLock, 0);
   }

   // Server image
   fImage = gEnv->GetValue("ProofServ.Image", "");

   // Get the group priority
   if (IsMaster()) {
      // Send session tag to client
      TMessage m(kPROOF_SESSIONTAG);
      m << fTopSessionTag;
      if (GetProtocol() > 24) m << fGroup;
      fSocket->Send(m);
      // Group priority
      fGroupPriority = GetPriority();
      // Dataset manager instance via plug-in
      TPluginHandler *h = 0;
      TString dsms = gEnv->GetValue("Proof.DataSetManager", "");
      if (!dsms.IsNull()) {
         TString dsm;
         Int_t from  = 0;
         dsms.Tokenize(dsm, from, ",");
         // Get plugin manager to load the appropriate TDataSetManager
         if (gROOT->GetPluginManager()) {
            // Find the appropriate handler
            h = gROOT->GetPluginManager()->FindHandler("TDataSetManager", dsm);
            if (h && h->LoadPlugin() != -1) {
               // make instance of the dataset manager
               fDataSetManager =
                  reinterpret_cast<TDataSetManager*>(h->ExecPlugin(3, fGroup.Data(),
                                                          fUser.Data(), dsm.Data()));
            }
         }
         // Check the result of the dataset manager initialization
         if (fDataSetManager && fDataSetManager->TestBit(TObject::kInvalidObject)) {
            Warning("SetupCommon", "dataset manager plug-in initialization failed");
            SendAsynMessage("TXProofServ::SetupCommon: dataset manager plug-in initialization failed");
            SafeDelete(fDataSetManager);
         }
      } else {
         // Initialize the default dataset manager
         TString opts("Av:");
         TString dsetdir = gEnv->GetValue("ProofServ.DataSetDir", "");
         if (dsetdir.IsNull()) {
            // Use the default in the sandbox
            dsetdir.Form("%s/%s", fWorkDir.Data(), kPROOF_DataSetDir);
            if (gSystem->AccessPathName(fDataSetDir))
               gSystem->MakeDirectory(fDataSetDir);
            opts += "Sb:";
         }
         // Find the appropriate handler
         if (!h) {
            h = gROOT->GetPluginManager()->FindHandler("TDataSetManager", "file");
            if (h && h->LoadPlugin() == -1) h = 0;
         }
         if (h) {
            // make instance of the dataset manager
            TString oo = TString::Format("dir:%s opt:%s", dsetdir.Data(), opts.Data());
            fDataSetManager = reinterpret_cast<TDataSetManager*>(h->ExecPlugin(3,
                              fGroup.Data(), fUser.Data(), oo.Data()));
         }
         if (fDataSetManager && fDataSetManager->TestBit(TObject::kInvalidObject)) {
            Warning("SetupCommon", "default dataset manager plug-in initialization failed");
            SafeDelete(fDataSetManager);
         }
      }
   }

   // Quotas
   TString quotas = gEnv->GetValue(TString::Format("ProofServ.UserQuotas.%s", fUser.Data()),"");
   if (quotas.IsNull())
      quotas = gEnv->GetValue(TString::Format("ProofServ.UserQuotasByGroup.%s", fGroup.Data()),"");
   if (quotas.IsNull())
      quotas = gEnv->GetValue("ProofServ.UserQuotas", "");
   if (!quotas.IsNull()) {
      // Parse it; format ("maxquerykept:10 hwmsz:800m maxsz:1g")
      TString tok;
      Ssiz_t from = 0;
      while (quotas.Tokenize(tok, from, " ")) {
         // Set max number of query results to keep
         if (tok.BeginsWith("maxquerykept=")) {
            tok.ReplaceAll("maxquerykept=","");
            if (tok.IsDigit())
               fMaxQueries = tok.Atoi();
            else
               Info("SetupCommon",
                    "parsing 'maxquerykept' :ignoring token %s : not a digit", tok.Data());
         }
         // Set High-Water-Mark or max on the sandbox size
         const char *ksz[2] = {"hwmsz=", "maxsz="};
         for (Int_t j = 0; j < 2; j++) {
            if (tok.BeginsWith(ksz[j])) {
               tok.ReplaceAll(ksz[j],"");
               Long64_t fact = -1;
               if (!tok.IsDigit()) {
                  // Parse (k, m, g)
                  tok.ToLower();
                  const char *s[3] = {"k", "m", "g"};
                  Int_t i = 0, k = 1024;
                  while (fact < 0) {
                     if (tok.EndsWith(s[i]))
                        fact = k;
                     else
                        k *= 1024;
                  }
                  tok.Remove(tok.Length()-1);
               }
               if (tok.IsDigit()) {
                  if (j == 0)
                     fHWMBoxSize = (fact > 0) ? tok.Atoi() * fact : tok.Atoi();
                  else
                     fMaxBoxSize = (fact > 0) ? tok.Atoi() * fact : tok.Atoi();
               } else {
                  TString ssz(ksz[j], strlen(ksz[j])-1);
                  Info("SetupCommon", "parsing '%s' : ignoring token %s", ssz.Data(), tok.Data());
               }
            }
         }
      }
   }

   // Apply quotas, if any
   if (IsMaster() && fQMgr)
      if (fQMgr->ApplyMaxQueries(fMaxQueries) != 0)
         Warning("SetupCommon", "problems applying fMaxQueries");

   // Send "ROOTversion|ArchCompiler" flag
   if (fProtocol > 12) {
      TString vac = gROOT->GetVersion();
      if (gROOT->GetSvnRevision() > 0)
         vac += TString::Format(":r%d", gROOT->GetSvnRevision());
      TString rtag = gEnv->GetValue("ProofServ.RootVersionTag", "");
      if (rtag.Length() > 0)
         vac += TString::Format(":%s", rtag.Data());
      vac += TString::Format("|%s-%s",gSystem->GetBuildArch(), gSystem->GetBuildCompilerVersion());
      TMessage m(kPROOF_VERSARCHCOMP);
      m << vac;
      fSocket->Send(m);
   }

   // Set user vars in TProof
   TString all_vars(gSystem->Getenv("PROOF_ALLVARS"));
   TString name;
   Int_t from = 0;
   while (all_vars.Tokenize(name, from, ",")) {
      if (!name.IsNull()) {
         TString value = gSystem->Getenv(name);
         TProof::AddEnvVar(name, value);
      }
   }

   if (fgLogToSysLog > 0) {
      // Set the syslog entity (all the information is available now)
      if (!(fUser.IsNull()) && !(fGroup.IsNull())) {
         fgSysLogEntity.Form("%s:%s", fUser.Data(), fGroup.Data());
      } else if (!(fUser.IsNull()) && fGroup.IsNull()) {
         fgSysLogEntity.Form("%s:default", fUser.Data());
      } else if (fUser.IsNull() && !(fGroup.IsNull())) {
         fgSysLogEntity.Form("undef:%s", fGroup.Data());
      }
      // Log the beginning of this session
      TString s;
      s.Form("%s 0 %.3f %.3f", fgSysLogEntity.Data(), fRealTime, fCpuTime);
      gSystem->Syslog(kLogNotice, s.Data());
   }

   if (gProofDebugLevel > 0)
      Info("SetupCommon", "successfully completed");

   // Done
   return 0;
}

//______________________________________________________________________________
void TProofServ::Terminate(Int_t status)
{
   // Terminate the proof server.

   if (fgLogToSysLog > 0) {
      TString s;
      s.Form("%s -1 %.3f %.3f %d", fgSysLogEntity.Data(), fRealTime, fCpuTime, status);
      gSystem->Syslog(kLogNotice, s.Data());
   }

   // Notify the memory footprint
   ProcInfo_t pi;
   if (!gSystem->GetProcInfo(&pi)){
      Info("Terminate", "process memory footprint: %ld/%ld kB virtual, %ld/%ld kB resident ",
                        pi.fMemVirtual, fgVirtMemMax, pi.fMemResident, fgResMemMax);
   }

   // Cleanup session directory
   if (status == 0) {
      // make sure we remain in a "connected" directory
      gSystem->ChangeDirectory("/");
      // needed in case fSessionDir is on NFS ?!
      gSystem->MakeDirectory(fSessionDir+"/.delete");
      gSystem->Exec(TString::Format("%s %s", kRM, fSessionDir.Data()));
   }

   // Cleanup queries directory if empty
   if (IsMaster()) {
      if (!(fQMgr && fQMgr->Queries() && fQMgr->Queries()->GetSize())) {
         // make sure we remain in a "connected" directory
         gSystem->ChangeDirectory("/");
         // needed in case fQueryDir is on NFS ?!
         gSystem->MakeDirectory(fQueryDir+"/.delete");
         gSystem->Exec(TString::Format("%s %s", kRM, fQueryDir.Data()));
         // Remove lock file
         if (fQueryLock)
            gSystem->Unlink(fQueryLock->GetName());
      }

      // Unlock the query dir owned by this session
      if (fQueryLock)
         fQueryLock->Unlock();
   }

   // Cleanup data directory if empty
   if (!fDataDir.IsNull() && !gSystem->AccessPathName(fDataDir, kWritePermission)) {
     if (UnlinkDataDir(fDataDir))
        Info("Terminate", "data directory '%s' has been removed", fDataDir.Data());
   }

   // Remove input handler to avoid spurious signals in socket
   // selection for closing activities executed upon exit()
   TIter next(gSystem->GetListOfFileHandlers());
   TObject *fh = 0;
   while ((fh = next())) {
      TProofServInputHandler *ih = dynamic_cast<TProofServInputHandler *>(fh);
      if (ih)
         gSystem->RemoveFileHandler(ih);
   }

   // Stop processing events
   gSystem->ExitLoop();

   // Exit() is called in pmain
}

//______________________________________________________________________________
Bool_t TProofServ::UnlinkDataDir(const char *path)
{
   // Scan recursively the datadir and unlink it if empty
   // Return kTRUE if it can be unlinked, kFALSE otherwise

   if (!path || strlen(path) <= 0) return kFALSE;

   Bool_t dorm = kTRUE;
   void *dirp = gSystem->OpenDirectory(path);
   if (dirp) {
      TString fpath;
      const char *ent = 0;
      while (dorm && (ent = gSystem->GetDirEntry(dirp))) {
         if (!strcmp(ent, ".") || !strcmp(ent, "..")) continue;
         fpath.Form("%s/%s", path, ent);
         FileStat_t st;
         if (gSystem->GetPathInfo(fpath, st) == 0 && R_ISDIR(st.fMode)) {
            dorm = UnlinkDataDir(fpath);
         } else {
            dorm = kFALSE;
         }
      }
      // Close the directory
      gSystem->FreeDirectory(dirp);
   } else {
      // Cannot open the directory
      dorm = kFALSE;
   }

    // Do remove, if required
   if (dorm && gSystem->Unlink(path) != 0)
      Warning("UnlinkDataDir", "data directory '%s' is empty but could not be removed", path);
   // done
   return dorm;
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

//______________________________________________________________________________
Int_t TProofServ::OldAuthSetup(TString &conf)
{
   // Setup authentication related stuff for old versions.
   // Provided for backward compatibility.

   OldProofServAuthSetup_t oldAuthSetupHook = 0;

   if (!oldAuthSetupHook) {
      // Load libraries needed for (server) authentication ...
      TString authlib = "libRootAuth";
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
      // Locate OldProofServAuthSetup
      Func_t f = gSystem->DynFindSymbol(authlib,"OldProofServAuthSetup");
      if (f)
         oldAuthSetupHook = (OldProofServAuthSetup_t)(f);
      else {
         Error("OldAuthSetup", "can't find OldProofServAuthSetup");
         return -1;
      }
   }
   //
   // Setup
   return (*oldAuthSetupHook)(fSocket, IsMaster(), fProtocol,
                              fUser, fOrdinal, conf);
}

//______________________________________________________________________________
TProofQueryResult *TProofServ::MakeQueryResult(Long64_t nent,
                                               const char *opt,
                                               TList *inlist, Long64_t fst,
                                               TDSet *dset, const char *selec,
                                               TObject *elist)
{
   // Create a TProofQueryResult instance for this query.

   // Increment sequential number
   Int_t seqnum = -1;
   if (fQMgr) {
      fQMgr->IncrementSeqNum();
      seqnum = fQMgr->SeqNum();
   }

   // Locally we always use the current streamer
   Bool_t olds = (dset && dset->TestBit(TDSet::kWriteV3)) ? kTRUE : kFALSE;
   if (olds)
      dset->SetWriteV3(kFALSE);

   // Create the instance and add it to the list
   TProofQueryResult *pqr = new TProofQueryResult(seqnum, opt, inlist, nent,
                                                  fst, dset, selec, elist);
   // Title is the session identifier
   pqr->SetTitle(gSystem->BaseName(fQueryDir));

   // Restore old streamer info
   if (olds)
      dset->SetWriteV3(kTRUE);

   return pqr;
}

//______________________________________________________________________________
void TProofServ::SetQueryRunning(TProofQueryResult *pq)
{
   // Set query in running state.

   // Record current position in the log file at start
   fflush(stdout);
   Int_t startlog = lseek(fileno(stdout), (off_t) 0, SEEK_END);

   // Add some header to logs
   Printf(" ");
   Info("SetQueryRunning", "starting query: %d", pq->GetSeqNum());

   // Build the list of loaded PAR packages
   TString parlist = "";
   TIter nxp(fEnabledPackages);
   TObjString *os= 0;
   while ((os = (TObjString *)nxp())) {
      if (parlist.Length() <= 0)
         parlist = os->GetName();
      else
         parlist += TString::Format(";%s",os->GetName());
   }

   if (fProof) {
      // Set in running state
      pq->SetRunning(startlog, parlist, fProof->GetParallel());

      // Bytes and CPU at start (we will calculate the differential at end)
      pq->SetProcessInfo(pq->GetEntries(),
                        fProof->GetCpuTime(), fProof->GetBytesRead());
   } else {
      // Set in running state
      pq->SetRunning(startlog, parlist, -1);

      // Bytes and CPU at start (we will calculate the differential at end)
      pq->SetProcessInfo(pq->GetEntries(), float(0.), 0);
   }
}

//______________________________________________________________________________
void TProofServ::HandleArchive(TMessage *mess, TString *slb)
{
   // Handle archive request.

   PDB(kGlobal, 1)
      Info("HandleArchive", "Enter");

   TString queryref;
   TString path;
   (*mess) >> queryref >> path;

   if (slb) slb->Form("%s %s", queryref.Data(), path.Data());

   // If this is a set default action just save the default
   if (queryref == "Default") {
      fArchivePath = path;
      Info("HandleArchive",
           "default path set to %s", fArchivePath.Data());
      return;
   }

   Int_t qry = -1;
   TString qdir;
   TProofQueryResult *pqr = fQMgr ? fQMgr->LocateQuery(queryref, qry, qdir) : 0;
   TProofQueryResult *pqm = pqr;

   if (path.Length() <= 0) {
      if (fArchivePath.Length() <= 0) {
         Info("HandleArchive",
              "archive paths are not defined - do nothing");
         return;
      }
      if (qry > 0) {
         path.Form("%s/session-%s-%d.root",
                   fArchivePath.Data(), fTopSessionTag.Data(), qry);
      } else {
         path = queryref;
         path.ReplaceAll(":q","-");
         path.Insert(0, TString::Format("%s/",fArchivePath.Data()));
         path += ".root";
      }
   }

   // Build file name for specific query
   if (!pqr || qry < 0) {
      TString fout = qdir;
      fout += "/query-result.root";

      TFile *f = TFile::Open(fout,"READ");
      pqr = 0;
      if (f) {
         f->ReadKeys();
         TIter nxk(f->GetListOfKeys());
         TKey *k =  0;
         while ((k = (TKey *)nxk())) {
            if (!strcmp(k->GetClassName(), "TProofQueryResult")) {
               pqr = (TProofQueryResult *) f->Get(k->GetName());
               if (pqr)
                  break;
            }
         }
         f->Close();
         delete f;
      } else {
         Info("HandleArchive",
              "file cannot be open (%s)",fout.Data());
         return;
      }
   }

   if (pqr) {

      PDB(kGlobal, 1) Info("HandleArchive",
                           "archive path for query #%d: %s",
                           qry, path.Data());
      TFile *farc = 0;
      if (gSystem->AccessPathName(path))
         farc = TFile::Open(path,"NEW");
      else
         farc = TFile::Open(path,"UPDATE");
      if (!farc || !(farc->IsOpen())) {
         Info("HandleArchive",
              "archive file cannot be open (%s)",path.Data());
         return;
      }
      farc->cd();

      // Update query status
      pqr->SetArchived(path);
      if (pqm)
         pqm->SetArchived(path);

      // Write to file
      pqr->Write();

      // Update temporary files too
      if (qry > -1 && fQMgr)
         fQMgr->SaveQuery(pqr);

      // Notify
      Info("HandleArchive",
           "results of query %s archived to file %s",
           queryref.Data(), path.Data());
   }

   // Done
   return;
}

//______________________________________________________________________________
void TProofServ::HandleProcess(TMessage *mess, TString *slb)
{
   // Handle processing request.

   PDB(kGlobal, 1)
      Info("HandleProcess", "Enter");

   // Nothing to do for slaves if we are not idle
   if (!IsTopMaster() && !IsIdle())
      return;

   TDSet *dset;
   TString filename, opt;
   TList *input;
   Long64_t nentries, first;
   TEventList *evl = 0;
   TEntryList *enl = 0;
   Bool_t sync;

   (*mess) >> dset >> filename >> input >> opt >> nentries >> first >> evl >> sync;
   // Get entry list information, if any (support started with fProtocol == 15)
   if ((mess->BufferSize() > mess->Length()) && fProtocol > 14)
      (*mess) >> enl;
   Bool_t hasNoData = (!dset || dset->TestBit(TDSet::kEmpty)) ? kTRUE : kFALSE;

   // Priority to the entry list
   TObject *elist = (enl) ? (TObject *)enl : (TObject *)evl;
   if (enl && evl)
      // Cannot specify both at the same time
      SafeDelete(evl);
   if ((!hasNoData) && elist)
      dset->SetEntryList(elist);

   if (IsTopMaster()) {

      // Make sure the dataset contains the information needed
      if ((!hasNoData) && dset->GetListOfElements()->GetSize() == 0) {
         TString emsg;
         if (TProof::AssertDataSet(dset, input, fDataSetManager, emsg) != 0) {
            SendAsynMessage(TString::Format("AssertDataSet on %s: %s",
                                 fPrefix.Data(), emsg.Data()));
            Error("HandleProcess", "AssertDataSet: %s", emsg.Data());
            // To terminate collection
            if (sync) SendLogFile();
            return;
         }
      }

      TProofQueryResult *pq = 0;

      // Create instance of query results; we set ownership of the input list
      // to the TQueryResult object, to avoid too many instantiations
      pq = MakeQueryResult(nentries, opt, 0, first, 0, filename, 0);

      // Prepare the input list and transfer it into the TQueryResult object
      if (dset) input->Add(dset);
      if (elist) input->Add(elist);
      pq->SetInputList(input, kTRUE);

      // Clear the list
      input->Clear("nodelete");
      SafeDelete(input);

      // Save input data, if any
      TString emsg;
      if (TProof::SaveInputData(pq, fCacheDir.Data(), emsg) != 0)
         Warning("HandleProcess", "could not save input data: %s", emsg.Data());

      // If not a draw action add the query to the main list
      if (!(pq->IsDraw())) {
         if (fQMgr) {
            if (fQMgr->Queries()) fQMgr->Queries()->Add(pq);
            // Also save it to queries dir
            fQMgr->SaveQuery(pq);
         }
      }

      // Add anyhow to the waiting lists
      QueueQuery(pq);

      // Call get Workers
      // if we are not idle the scheduler will just enqueue the query and
      // send a resume message later.

      Bool_t enqueued = kFALSE;
      Int_t pc = 0;
      // if the session does not have workers and is in the dynamic mode
      if (fProof->UseDynamicStartup()) {
         // get the a list of workers and start them
         TList* workerList = new TList();
         EQueryAction retVal = GetWorkers(workerList, pc);
         if (retVal == TProofServ::kQueryStop) {
            Error("HandleProcess", "error getting list of worker nodes");
            // To terminate collection
            if (sync) SendLogFile();
            return;
         } else if (retVal == TProofServ::kQueryEnqueued) {
            // change to an asynchronous query
            enqueued = kTRUE;
            Info("HandleProcess", "query %d enqueued", pq->GetSeqNum());
         } else if (Int_t ret = fProof->AddWorkers(workerList) < 0) {
            Error("HandleProcess", "Adding a list of worker nodes returned: %d",
                  ret);
            // To terminate collection
            if (sync) SendLogFile();
            return;
         }
      } else {
         EQueryAction retVal = GetWorkers(0, pc);
         if (retVal == TProofServ::kQueryStop) {
            Error("HandleProcess", "error getting list of worker nodes");
            // To terminate collection
            if (sync) SendLogFile();
            return;
         } else if (retVal == TProofServ::kQueryEnqueued) {
            // change to an asynchronous query
            enqueued = kTRUE;
            Info("HandleProcess", "query %d enqueued", pq->GetSeqNum());
         } else if (retVal != TProofServ::kQueryOK) {
            Error("HandleProcess", "unknown return value: %d", retVal);
            // To terminate collection
            if (sync) SendLogFile();
            return;
         }
      }

      // If the client submission was asynchronous, signal the submission of
      // the query and communicate the assigned sequential number for later
      // identification
      TMessage m(kPROOF_QUERYSUBMITTED);
      if (!sync || enqueued) {
         m << pq->GetSeqNum() << kFALSE;
         fSocket->Send(m);
      }

      // Nothing more to do if we are not idle
      if (!IsIdle()) {
         // Notify submission
         Info("HandleProcess",
              "query \"%s:%s\" submitted", pq->GetTitle(), pq->GetName());
         return;
      }

      // Process
      // in the static mode, if a session is enqueued it will be processed after current query
      // (there is no way to enqueue if idle).
      // in the dynamic mode we will process here only if the session was idle and got workers!
      Bool_t doprocess = kFALSE;
      while (WaitingQueries() > 0 && !enqueued) {
         doprocess = kTRUE;
         //
         ProcessNext(slb);
         // avoid processing async queries sent during processing in dyn mode
         if (fProof->UseDynamicStartup())
            enqueued = kTRUE;

      } // Loop on submitted queries

      // Set idle
      SetIdle(kTRUE);

      // Reset mergers
      fProof->ResetMergers();

      // kPROOF_SETIDLE sets the client to idle; in asynchronous mode clients monitor
      // TProof::IsIdle for to check the readiness of a query, so we need to send this
      // before to be sure thatn everything about a query is received by the client
      if (!sync) SendLogFile();

      // Signal the client that we are idle
      if (doprocess) {
         m.Reset(kPROOF_SETIDLE);
         Bool_t waiting = (WaitingQueries() > 0) ? kTRUE : kFALSE;
         m << waiting;
         fSocket->Send(m);
      }

      // In synchronous mode TProof::Collect is terminated by the reception of the
      // log file and subsequent submissions are controlled by TProof::IsIdle(), so
      // this must be last one to be sent
      if (sync) SendLogFile();

      // Set idle
      SetIdle(kTRUE);

   } else {

      // Set not idle
      SetIdle(kFALSE);

      // Cleanup the player
      Bool_t deleteplayer = kTRUE;
      MakePlayer();

      // Setup data set
      if (dset && (dset->IsA() == TDSetProxy::Class()))
         ((TDSetProxy*)dset)->SetProofServ(this);

      // Get input data, if any
      TString emsg;
      if (TProof::GetInputData(input, fCacheDir.Data(), emsg) != 0)
         Warning("HandleProcess", "could not get input data: %s", emsg.Data());

      // Get query sequential number
      if (TProof::GetParameter(input, "PROOF_QuerySeqNum", fQuerySeqNum) != 0)
         Warning("HandleProcess", "could not get query sequential number!");

      // Make the ordinal number available in the selector
      TObject *nord = 0;
      while ((nord = input->FindObject("PROOF_Ordinal")))
         input->Remove(nord);
      input->Add(new TNamed("PROOF_Ordinal", GetOrdinal()));

      // Set input
      TIter next(input);
      TObject *o = 0;
      while ((o = next())) {
         PDB(kGlobal, 2) Info("HandleProcess", "adding: %s", o->GetName());
         fPlayer->AddInput(o);
      }

      // Signal the master that we are starting processing
      fSocket->Send(kPROOF_STARTPROCESS);

      // Process
      PDB(kGlobal, 1) Info("HandleProcess", "calling %s::Process()", fPlayer->IsA()->GetName());
      fPlayer->Process(dset, filename, opt, nentries, first);

      // Return number of events processed
      TMessage m(kPROOF_STOPPROCESS);
      Bool_t abort = (fPlayer->GetExitStatus() != TVirtualProofPlayer::kAborted) ? kFALSE : kTRUE;
      if (fProtocol > 18) {
         TProofProgressStatus* status =
            new TProofProgressStatus(fPlayer->GetEventsProcessed(),
                                    gPerfStats?gPerfStats->GetBytesRead():0);
         if (status)
            m << status << abort;
         if (slb)
            slb->Form("%d %lld %lld", fPlayer->GetExitStatus(),
                                      status->GetEntries(), status->GetBytesRead());
         SafeDelete(status);
      } else {
         m << fPlayer->GetEventsProcessed() << abort;
         if (slb)
            slb->Form("%d %lld -1", fPlayer->GetExitStatus(), fPlayer->GetEventsProcessed());
      }

      fSocket->Send(m);
      PDB(kGlobal, 2)
         Info("TProofServ::Handleprocess",
              "worker %s has finished processing with %d objects in output list",
              GetOrdinal(), fPlayer->GetOutputList()->GetEntries());

      // Cleanup the input data set info
      SafeDelete(dset);
      SafeDelete(enl);
      SafeDelete(evl);

      // Check if we are in merging mode (i.e. parameter PROOF_UseMergers exists)
      Bool_t isInMergingMode = kFALSE;
      if (!(TestBit(TProofServ::kHighMemory))) {
         Int_t nm = 0;
         if (TProof::GetParameter(input, "PROOF_UseMergers", nm) == 0) {
            isInMergingMode = (nm >= 0) ? kTRUE : kFALSE;
         }
      }
      PDB(kGlobal, 2) Info("HandleProcess", "merging mode check: %d", isInMergingMode);

      if (!IsMaster() && isInMergingMode &&
          fPlayer->GetExitStatus() != TVirtualProofPlayer::kAborted && fPlayer->GetOutputList()) {
         // Worker in merging mode.
         //----------------------------
         // First, it reports only the size of its output to the master
         // + port on which it can possibly accept outputs from other workers if it becomes a merger
         // Master will later tell it where it should send the output (either to the master or to some merger)
         // or if it should become a merger

         TMessage msg_osize(kPROOF_SUBMERGER);
         msg_osize << Int_t(TProof::kOutputSize);
         msg_osize << fPlayer->GetOutputList()->GetEntries();

         fMergingSocket = new TServerSocket(0);
         Int_t merge_port = 0;
         if (fMergingSocket) {
            PDB(kGlobal, 2)
               Info("HandleProcess", "possible port for merging connections: %d",
                                     fMergingSocket->GetLocalPort());
            merge_port = fMergingSocket->GetLocalPort();
         }
         msg_osize << merge_port;
         fSocket->Send(msg_osize);

         // Set idle
         SetIdle(kTRUE);

         // Do not cleanup the player yet: it will be used in sub-merging activities
         deleteplayer = kFALSE;

         PDB(kSubmerger, 2) Info("HandleProcess", "worker %s has finished", fOrdinal.Data());

      } else {
         // Sub-master OR worker not in merging mode
         // ---------------------------------------------
         if (fPlayer->GetExitStatus() != TVirtualProofPlayer::kAborted && fPlayer->GetOutputList()) {
            PDB(kGlobal, 2)  Info("HandleProcess", "sending result directly to master");
            if (SendResults(fSocket, fPlayer->GetOutputList()) != 0)
               Warning("HandleProcess","problems sending output list");
         } else {
            if (fPlayer->GetExitStatus() != TVirtualProofPlayer::kAborted)
               Warning("HandleProcess","the output list is empty!");
            if (SendResults(fSocket) != 0)
               Warning("HandleProcess", "problems sending output list");
         }

         // Masters reset the mergers, if any
         if (IsMaster()) fProof->ResetMergers();

         // Signal the master that we are idle
         fSocket->Send(kPROOF_SETIDLE);

         // Set idle
         SetIdle(kTRUE);

         // Notify the user
         SendLogFile();
      }
      // Make also sure the input list objects are deleted
      fPlayer->GetInputList()->SetOwner(0);
      input->SetOwner();
      SafeDelete(input);

      // Cleanup if required
      if (deleteplayer) DeletePlayer();
   }

   PDB(kGlobal, 1) Info("HandleProcess", "done");

   // Done
   return;
}

//______________________________________________________________________________
Int_t TProofServ::SendResults(TSocket *sock, TList *outlist, TQueryResult *pq)
{
   // Sends all objects from the given list to the specified socket

   PDB(kOutput, 2) Info("SendResults", "enter");

   TString msg;
   if (fProtocol > 23 && outlist) {
      // Send objects in bunches of max fMsgSizeHWM bytes to optimize transfer
      // Objects are merged one-by-one by the client
      // Messages for objects
      TMessage mbuf(kPROOF_OUTPUTOBJECT);
      // Objects in the output list
      Int_t olsz = outlist->GetSize();
      if (IsTopMaster() && pq) {
         msg.Form("%s: merging output objects ... done                                     ",
                       fPrefix.Data());
         SendAsynMessage(msg.Data());
         // Message for the client
         msg.Form("%s: objects merged; sending output: %d objs", fPrefix.Data(), olsz);
         SendAsynMessage(msg.Data(), kFALSE);
         // Send light query info
         mbuf << (Int_t) 0;
         mbuf.WriteObject(pq);
         if (sock->Send(mbuf) < 0) return -1;
      }
      // Objects in the output list
      Int_t ns = 0, np = 0;
      TIter nxo(outlist);
      TObject *o = 0;
      Int_t totsz = 0, objsz = 0;
      mbuf.Reset();
      while ((o = nxo())) {
         if (mbuf.Length() > fMsgSizeHWM) {
            PDB(kOutput, 1)
               Info("SendResults",
                    "message has %d bytes: limit of %lld bytes reached - sending ...",
                    mbuf.Length(), fMsgSizeHWM);
            // Compress the message, if required; for these messages we do it already
            // here so we get the size; TXSocket does not do it twice.
            if (fCompressMsg > 0) {
               mbuf.SetCompressionLevel(fCompressMsg);
               mbuf.Compress();
               objsz = mbuf.CompLength();
            } else {
               objsz = mbuf.Length();
            }
            totsz += objsz;
            if (IsTopMaster()) {
               msg.Form("%s: objects merged; sending obj %d/%d (%d bytes)   ",
                              fPrefix.Data(), ns, olsz, objsz);
               SendAsynMessage(msg.Data(), kFALSE);
            }
            if (sock->Send(mbuf) < 0) return -1;
            // Reset the message
            mbuf.Reset();
            np = 0;
         }
         ns++;
         np++;
         mbuf << (Int_t) ((ns >= olsz) ? 2 : 1);
         mbuf << o;
      }
      if (np > 0) {
         // Compress the message, if required; for these messages we do it already
         // here so we get the size; TXSocket does not do it twice.
         if (fCompressMsg > 0) {
            mbuf.SetCompressionLevel(fCompressMsg);
            mbuf.Compress();
            objsz = mbuf.CompLength();
         } else {
            objsz = mbuf.Length();
         }
         totsz += objsz;
         if (IsTopMaster()) {
            msg.Form("%s: objects merged; sending obj %d/%d (%d bytes)     ",
                           fPrefix.Data(), ns, olsz, objsz);
            SendAsynMessage(msg.Data(), kFALSE);
         }
         if (sock->Send(mbuf) < 0) return -1;
      }
      if (IsTopMaster()) {
         // Send total size
         msg.Form("%s: grand total: sent %d objects, size: %d bytes                            ",
                                        fPrefix.Data(), olsz, totsz);
         SendAsynMessage(msg.Data());
      }
   } else if (fProtocol > 10 && outlist) {

      // Send objects one-by-one to optimize transfer and merging
      // Messages for objects
      TMessage mbuf(kPROOF_OUTPUTOBJECT);
      // Objects in the output list
      Int_t olsz = outlist->GetSize();
      if (IsTopMaster() && pq) {
         msg.Form("%s: merging output objects ... done                                     ",
                       fPrefix.Data());
         SendAsynMessage(msg.Data());
         // Message for the client
         msg.Form("%s: objects merged; sending output: %d objs", fPrefix.Data(), olsz);
         SendAsynMessage(msg.Data(), kFALSE);
         // Send light query info
         mbuf << (Int_t) 0;
         mbuf.WriteObject(pq);
         if (sock->Send(mbuf) < 0) return -1;
      }

      Int_t ns = 0;
      Int_t totsz = 0, objsz = 0;
      TIter nxo(fPlayer->GetOutputList());
      TObject *o = 0;
      while ((o = nxo())) {
         ns++;
         mbuf.Reset();
         Int_t type = (Int_t) ((ns >= olsz) ? 2 : 1);
         mbuf << type;
         mbuf.WriteObject(o);
         // Compress the message, if required; for these messages we do it already
         // here so we get the size; TXSocket does not do it twice.
         if (fCompressMsg > 0) {
            mbuf.SetCompressionLevel(fCompressMsg);
            mbuf.Compress();
            objsz = mbuf.CompLength();
         } else {
            objsz = mbuf.Length();
         }
         totsz += objsz;
         if (IsTopMaster()) {
            msg.Form("%s: objects merged; sending obj %d/%d (%d bytes)   ",
                           fPrefix.Data(), ns, olsz, objsz);
            SendAsynMessage(msg.Data(), kFALSE);
         }
         if (sock->Send(mbuf) < 0) return -1;
      }
      // Total size
      if (IsTopMaster()) {
         // Send total size
         msg.Form("%s: grand total: sent %d objects, size: %d bytes       ",
                                        fPrefix.Data(), olsz, totsz);
         SendAsynMessage(msg.Data());
      }

   } else if (IsTopMaster() && fProtocol > 6 && outlist) {

      // Buffer to be sent
      TMessage mbuf(kPROOF_OUTPUTLIST);
      mbuf.WriteObject(pq);
      // Sizes
      Int_t blen = mbuf.CompLength();
      Int_t olsz = outlist->GetSize();
      // Message for the client
      msg.Form("%s: sending output: %d objs, %d bytes", fPrefix.Data(), olsz, blen);
      SendAsynMessage(msg.Data(), kFALSE);
      if (sock->Send(mbuf) < 0) return -1;

   } else {
      if (outlist) {
         PDB(kGlobal, 2) Info("SendResults", "sending output list");
      } else {
         PDB(kGlobal, 2) Info("SendResults", "notifying failure or abort");
      }
      if (sock->SendObject(outlist, kPROOF_OUTPUTLIST) < 0) return -1;
   }

   PDB(kOutput,2) Info("SendResults", "done");

   // Done
   return 0;
}

//______________________________________________________________________________
void TProofServ::ProcessNext(TString *slb)
{
   // process the next query from the queue of submitted jobs.
   // to be called on the top master only.

   TDSet *dset = 0;
   TString filename, opt;
   TList *input = 0;
   Long64_t nentries = -1, first = 0;

   TObject *elist = 0;
   TProofQueryResult *pq = 0;

   // Process

   // Get next query info (also removes query from the list)
   pq = NextQuery();
   if (pq) {

      // Set not idle
      SetIdle(kFALSE);
      opt      = pq->GetOptions();
      input    = pq->GetInputList();
      nentries = pq->GetEntries();
      first    = pq->GetFirst();
      filename = pq->GetSelecImp()->GetName();
      Ssiz_t id = opt.Last('#');
      if (id != kNPOS && id < opt.Length() - 1) {
         filename += opt(id + 1, opt.Length());
         // Remove it from 'opt' so user found on the workers what they specified
         opt.Remove(id);
      }
      // Attach to data set and entry- (or event-) list (if any)
      TObject *o = 0;
      if ((o = pq->GetInputObject("TDSet"))) {
         dset = (TDSet *) o;
      } else {
         // Should never get here
         Error("ProcessNext", "no TDset object: cannot continue");
         return;
      }
      elist = 0;
      if ((o = pq->GetInputObject("TEntryList")))
         elist = o;
      else if ((o = pq->GetInputObject("TEventList")))
         elist = o;
      //
      // Expand selector files
      if (pq->GetSelecImp()) {
         gSystem->Exec(TString::Format("%s %s", kRM, pq->GetSelecImp()->GetName()));
         pq->GetSelecImp()->SaveSource(pq->GetSelecImp()->GetName());
      }
      if (pq->GetSelecHdr() &&
          !strstr(pq->GetSelecHdr()->GetName(), "TProofDrawHist")) {
         gSystem->Exec(TString::Format("%s %s", kRM, pq->GetSelecHdr()->GetName()));
         pq->GetSelecHdr()->SaveSource(pq->GetSelecHdr()->GetName());
      }
   } else {
      // Should never get here
      Error("ProcessNext", "empty waiting queries list!");
      return;
   }

   // Set in running state
   SetQueryRunning(pq);

   // Save to queries dir, if not standard draw
   if (fQMgr) {
      if (!(pq->IsDraw()))
         fQMgr->SaveQuery(pq);
      else
         fQMgr->IncrementDrawQueries();
      fQMgr->ResetTime();
   }

   // Signal the client that we are starting a new query
   TMessage m(kPROOF_STARTPROCESS);
   m << TString(pq->GetSelecImp()->GetName())
     << dset->GetNumOfFiles()
     << pq->GetFirst() << pq->GetEntries();
   fSocket->Send(m);

   // Create player
   MakePlayer();

   // Add query results to the player lists
   fPlayer->AddQueryResult(pq);

   // Set query currently processed
   fPlayer->SetCurrentQuery(pq);

   // Setup data set
   if (dset->IsA() == TDSetProxy::Class())
      ((TDSetProxy*)dset)->SetProofServ(this);

   // Add the unique query tag as TNamed object to the input list
   // so that it is available in TSelectors for monitoring
   TString qid = TString::Format("%s:%s",pq->GetTitle(),pq->GetName());
   input->Add(new TNamed("PROOF_QueryTag", qid.Data()));
   //  ... and the sequential number
   fQuerySeqNum = pq->GetSeqNum();
   input->Add(new TParameter<Int_t>("PROOF_QuerySeqNum", fQuerySeqNum));

   // Check whether we have to enforce the use of submergers, but only if the user did
   // not express itself on the subject
   if (gEnv->Lookup("Proof.UseMergers") && !input->FindObject("PROOF_UseMergers")) {
      Int_t smg = gEnv->GetValue("Proof.UseMergers",-1);
      if (smg >= 0) {
         input->Add(new TParameter<Int_t>("PROOF_UseMergers", smg));
         PDB(kSubmerger, 2) Info("ProcessNext", "PROOF_UseMergers set to %d", smg);
      }
   }

   // Set input
   TIter next(input);
   TObject *o = 0;
   while ((o = next())) {
      PDB(kGlobal, 2) Info("ProcessNext", "adding: %s", o->GetName());
      fPlayer->AddInput(o);
   }

   // Remove the list of the missing files from the original list, if any
   if ((o = input->FindObject("MissingFiles"))) input->Remove(o);

   // Process
   PDB(kGlobal, 1) Info("ProcessNext", "calling %s::Process()", fPlayer->IsA()->GetName());
   fPlayer->Process(dset, filename, opt, nentries, first);

   // Return number of events processed
   if (fPlayer->GetExitStatus() != TVirtualProofPlayer::kFinished) {
      Bool_t abort =
         (fPlayer->GetExitStatus() == TVirtualProofPlayer::kAborted) ? kTRUE : kFALSE;
      m.Reset(kPROOF_STOPPROCESS);
      // message sent from worker to the master
      if (fProtocol > 18) {
         TProofProgressStatus* status = fPlayer->GetProgressStatus();
         m << status << abort;
         status = 0; // the status belongs to the player.
      } else if (fProtocol > 8) {
         m << fPlayer->GetEventsProcessed() << abort;
      } else {
         m << fPlayer->GetEventsProcessed();
      }
      fSocket->Send(m);
   }

   // Register any dataset produced during this processing, if required
   if (fDataSetManager && fPlayer->GetOutputList()) {
      TNamed *psr = (TNamed *) fPlayer->GetOutputList()->FindObject("PROOFSERV_RegisterDataSet");
      if (psr) {
         if (RegisterDataSets(input, fPlayer->GetOutputList()) != 0)
            Warning("ProcessNext", "problems registering produced datasets");
         fPlayer->GetOutputList()->Remove(psr);
         delete psr;
      }
   }

   // Complete filling of the TQueryResult instance
   if (fQMgr && !pq->IsDraw()) {
      fProof->AskStatistics();
      if (fQMgr->FinalizeQuery(pq, fProof, fPlayer))
         fQMgr->SaveQuery(pq, fMaxQueries);
   }

   // Send back the results
   TQueryResult *pqr = pq->CloneInfo();
   // At least the TDSet name in the light object
   Info("ProcessNext", "adding info about dataset '%s' in the light query result", dset->GetName());
   TList rin;
   TDSet *ds = new TDSet(dset->GetName(), dset->GetObjName());
   rin.Add(ds);
   pqr->SetInputList(&rin, kTRUE);
   if (fPlayer->GetExitStatus() != TVirtualProofPlayer::kAborted && fPlayer->GetOutputList()) {
      PDB(kGlobal, 2)
         Info("ProcessNext", "sending results");
      TQueryResult *xpq = (fProtocol > 10) ? pqr : pq;
      if (SendResults(fSocket, fPlayer->GetOutputList(), xpq) != 0)
         Warning("ProcessNext", "problems sending output list");
      if (slb) slb->Form("%d %lld %lld %.3f", fPlayer->GetExitStatus(), pq->GetEntries(),
                                              pq->GetBytes(), pq->GetUsedCPU());
   } else {
      if (fPlayer->GetExitStatus() != TVirtualProofPlayer::kAborted)
         Warning("ProcessNext","the output list is empty!");
      if (SendResults(fSocket) != 0)
         Warning("ProcessNext", "problems sending output list");
      if (slb) slb->Form("%d -1 -1 %.3f", fPlayer->GetExitStatus(), pq->GetUsedCPU());
   }

   // Remove aborted queries from the list
   if (fPlayer->GetExitStatus() == TVirtualProofPlayer::kAborted) {
      delete pqr;
      if (fQMgr) fQMgr->RemoveQuery(pq);
   } else {
      // Keep in memory only light infor about a query
      if (!(pq->IsDraw())) {
         if (fQMgr && fQMgr->Queries()) {
            fQMgr->Queries()->Add(pqr);
            // Remove from the fQueries list
            fQMgr->Queries()->Remove(pq);
         }
         // These removes 'pq' from the internal player list and
         // deletes it; in this way we do not attempt a double delete
         // when destroying the player
         fPlayer->RemoveQueryResult(TString::Format("%s:%s",
                                    pq->GetTitle(), pq->GetName()));
      }
   }

   DeletePlayer();
   if (IsMaster() && fProof->UseDynamicStartup())
      // stop the workers
      fProof->RemoveWorkers(0);
}

//______________________________________________________________________________
Int_t TProofServ::RegisterDataSets(TList *in, TList *out)
{
   // Register TFileCollections in 'out' as datasets according to the rules in 'in'

   PDB(kDataset, 1) Info("RegisterDataSets", "enter");

   if (!in || !out) return 0;

   TString msg;
   TIter nxo(out);
   TObject *o = 0;
   while ((o = nxo())) {
      // Only file collections TFileCollection
      TFileCollection *ds = dynamic_cast<TFileCollection*> (o);
      if (ds) {
         // The tag and register option
         TNamed *fcn = 0;
         TString tag = TString::Format("DATASET_%s", ds->GetName());
         if (!(fcn = (TNamed *) out->FindObject(tag))) continue;
         // Register option
         TString regopt(fcn->GetTitle());
         // Register this dataset
         if (fDataSetManager) {
            if (fDataSetManager->TestBit(TDataSetManager::kAllowRegister)) {
               // Extract the list
               if (ds->GetList()->GetSize() > 0) {
                  // Register the dataset (quota checks are done inside here)
                  msg.Form("Registering and verifying dataset '%s' ... ", ds->GetName());
                  SendAsynMessage(msg.Data(), kFALSE);
                  Int_t rc = 0;
                  FlushLogFile();
                  {  TProofServLogHandlerGuard hg(fLogFile,  fSocket);
                     // Always allow verification for this action
                     Bool_t allowVerify = fDataSetManager->TestBit(TDataSetManager::kAllowVerify) ? kTRUE : kFALSE;
                     if (regopt.Contains("V") && !allowVerify)
                        fDataSetManager->SetBit(TDataSetManager::kAllowVerify);
                     rc = fDataSetManager->RegisterDataSet(ds->GetName(), ds, regopt);
                     // Reset to the previous state if needed
                     if (regopt.Contains("V") && !allowVerify)
                        fDataSetManager->ResetBit(TDataSetManager::kAllowVerify);
                  }
                  if (rc != 0) {
                     Warning("RegisterDataSets",
                              "failure registering dataset '%s'", ds->GetName());
                     msg.Form("Registering and verifying dataset '%s' ... failed! See log for more details", ds->GetName());
                  } else {
                     Info("RegisterDataSets", "dataset '%s' successfully registered", ds->GetName());
                     msg.Form("Registering and verifying dataset '%s' ... OK", ds->GetName());
                  }
                  SendAsynMessage(msg.Data(), kTRUE);
                  // Notify
                  PDB(kDataset, 2) {
                     Info("RegisterDataSets","printing collection");
                     ds->Print("F");
                  }
               } else {
                  Warning("RegisterDataSets", "collection '%s' is empty", o->GetName());
               }
            } else {
               Info("RegisterDataSets", "dataset registration not allowed");
               return -1;
            }
         } else {
            Error("RegisterDataSets", "dataset manager is undefined!");
            return -1;
         }
         // Cleanup temporary stuff
         out->Remove(fcn);
         SafeDelete(fcn);
      }
   }

   PDB(kDataset, 1) Info("RegisterDataSets", "exit");
   // Done
   return 0;
}

//______________________________________________________________________________
void TProofServ::HandleQueryList(TMessage *mess)
{
   // Handle request for list of queries.

   PDB(kGlobal, 1)
      Info("HandleQueryList", "Enter");

   Bool_t all;
   (*mess) >> all;

   TList *ql = new TList;
   Int_t ntot = 0, npre = 0, ndraw= 0;
   if (fQMgr) {
      if (all) {
         // Rescan
         TString qdir = fQueryDir;
         Int_t idx = qdir.Index("session-");
         if (idx != kNPOS)
            qdir.Remove(idx);
         fQMgr->ScanPreviousQueries(qdir);
         // Send also information about previous queries, if any
         if (fQMgr->PreviousQueries()) {
            TIter nxq(fQMgr->PreviousQueries());
            TProofQueryResult *pqr = 0;
            while ((pqr = (TProofQueryResult *)nxq())) {
               ntot++;
               pqr->fSeqNum = ntot;
               ql->Add(pqr);
            }
         }
      }

      npre = ntot;
      if (fQMgr->Queries()) {
         // Add info about queries in this session
         TIter nxq(fQMgr->Queries());
         TProofQueryResult *pqr = 0;
         TQueryResult *pqm = 0;
         while ((pqr = (TProofQueryResult *)nxq())) {
            ntot++;
            pqm = pqr->CloneInfo();
            pqm->fSeqNum = ntot;
            ql->Add(pqm);
         }
      }
      // Number of draw queries
      ndraw = fQMgr->DrawQueries();
   }

   TMessage m(kPROOF_QUERYLIST);
   m << npre << ndraw << ql;
   fSocket->Send(m);
   delete ql;

   // Done
   return;
}

//______________________________________________________________________________
void TProofServ::HandleRemove(TMessage *mess, TString *slb)
{
   // Handle remove request.

   PDB(kGlobal, 1)
      Info("HandleRemove", "Enter");

   TString queryref;
   (*mess) >> queryref;

   if (slb) *slb = queryref;

   if (queryref == "cleanupqueue") {
      // Remove pending requests
      Int_t pend = CleanupWaitingQueries();
      // Notify
      Info("HandleRemove", "%d queries removed from the waiting list", pend);
      // We are done
      return;
   }

   if (queryref == "cleanupdir") {

      // Cleanup previous sessions results
      Int_t nd = (fQMgr) ? fQMgr->CleanupQueriesDir() : -1;

      // Notify
      Info("HandleRemove", "%d directories removed", nd);
      // We are done
      return;
   }


   if (fQMgr) {
      TProofLockPath *lck = 0;
      if (fQMgr->LockSession(queryref, &lck) == 0) {

         // Remove query
         TList qtorm;
         fQMgr->RemoveQuery(queryref, &qtorm);
         CleanupWaitingQueries(kFALSE, &qtorm);

         // Unlock and remove the lock file
         if (lck) {
            gSystem->Unlink(lck->GetName());
            SafeDelete(lck);
         }

         // We are done
         return;
      }
   } else {
      Warning("HandleRemove", "query result manager undefined!");
   }

   // Notify failure
   Info("HandleRemove",
        "query %s could not be removed (unable to lock session)", queryref.Data());

   // Done
   return;
}

//______________________________________________________________________________
void TProofServ::HandleRetrieve(TMessage *mess, TString *slb)
{
   // Handle retrieve request.

   PDB(kGlobal, 1)
      Info("HandleRetrieve", "Enter");

   TString queryref;
   (*mess) >> queryref;

   if (slb) *slb = queryref;

   // Parse reference string
   Int_t qry = -1;
   TString qdir;
   if (fQMgr) fQMgr->LocateQuery(queryref, qry, qdir);

   TString fout = qdir;
   fout += "/query-result.root";

   TFile *f = TFile::Open(fout,"READ");
   TProofQueryResult *pqr = 0;
   if (f) {
      f->ReadKeys();
      TIter nxk(f->GetListOfKeys());
      TKey *k =  0;
      while ((k = (TKey *)nxk())) {
         if (!strcmp(k->GetClassName(), "TProofQueryResult")) {
            pqr = (TProofQueryResult *) f->Get(k->GetName());
            // For backward compatibility
            if (fProtocol < 13) {
               TDSet *d = 0;
               TObject *o = 0;
               TIter nxi(pqr->GetInputList());
               while ((o = nxi()))
                  if ((d = dynamic_cast<TDSet *>(o)))
                     break;
               d->SetWriteV3(kTRUE);
            }
            if (pqr) {

               // Message for the client
               Float_t qsz = (Float_t) f->GetSize();
               Int_t ilb = 0;
               static const char *clb[4] = { "bytes", "KB", "MB", "GB" };
               while (qsz > 1000. && ilb < 3) {
                  qsz /= 1000.;
                  ilb++;
               }
               SendAsynMessage(TString::Format("%s: sending result of %s:%s (%.1f %s)",
                                               fPrefix.Data(), pqr->GetTitle(), pqr->GetName(),
                                               qsz, clb[ilb]));
               fSocket->SendObject(pqr, kPROOF_RETRIEVE);
            } else {
               Info("HandleRetrieve",
                    "query not found in file %s",fout.Data());
               // Notify not found
               fSocket->SendObject(0, kPROOF_RETRIEVE);
            }
            break;
         }
      }
      f->Close();
      delete f;
   } else {
      Info("HandleRetrieve",
           "file cannot be open (%s)",fout.Data());
      // Notify not found
      fSocket->SendObject(0, kPROOF_RETRIEVE);
      return;
   }

   // Done
   return;
}

//______________________________________________________________________________
void TProofServ::HandleLibIncPath(TMessage *mess)
{
   // Handle lib, inc search paths modification request

   TString type;
   Bool_t add;
   TString path;
   (*mess) >> type >> add >> path;

   // Check type of action
   if ((type != "lib") && (type != "inc")) {
      Error("HandleLibIncPath","unknown action type: %s", type.Data());
      return;
   }

   // Separators can be either commas or blanks
   path.ReplaceAll(","," ");

   // Decompose lists
   TObjArray *op = 0;
   if (path.Length() > 0 && path != "-") {
      if (!(op = path.Tokenize(" "))) {
         Error("HandleLibIncPath","decomposing path %s", path.Data());
         return;
      }
   }

   if (add) {

      if (type == "lib") {

         // Add libs
         TIter nxl(op, kIterBackward);
         TObjString *lib = 0;
         while ((lib = (TObjString *) nxl())) {
            // Expand path
            TString xlib = lib->GetName();
            gSystem->ExpandPathName(xlib);
            // Add to the dynamic lib search path if it exists and can be read
            if (!gSystem->AccessPathName(xlib, kReadPermission)) {
               TString newlibpath = gSystem->GetDynamicPath();
               // In the first position after the working dir
               Int_t pos = 0;
               if (newlibpath.BeginsWith(".:"))
                  pos = 2;
               if (newlibpath.Index(xlib) == kNPOS) {
                  newlibpath.Insert(pos,TString::Format("%s:", xlib.Data()));
                  gSystem->SetDynamicPath(newlibpath);
               }
            } else {
               Info("HandleLibIncPath",
                    "libpath %s does not exist or cannot be read - not added", xlib.Data());
            }
         }

         // Forward the request, if required
         if (IsMaster())
            fProof->AddDynamicPath(path);

      } else {

         // Add incs
         TIter nxi(op);
         TObjString *inc = 0;
         while ((inc = (TObjString *) nxi())) {
            // Expand path
            TString xinc = inc->GetName();
            gSystem->ExpandPathName(xinc);
            // Add to the dynamic lib search path if it exists and can be read
            if (!gSystem->AccessPathName(xinc, kReadPermission)) {
               TString curincpath = gSystem->GetIncludePath();
               if (curincpath.Index(xinc) == kNPOS)
                  gSystem->AddIncludePath(TString::Format("-I%s", xinc.Data()));
            } else
               Info("HandleLibIncPath",
                    "incpath %s does not exist or cannot be read - not added", xinc.Data());
         }

         // Forward the request, if required
         if (IsMaster())
            fProof->AddIncludePath(path);
      }


   } else {

      if (type == "lib") {

         // Remove libs
         TIter nxl(op);
         TObjString *lib = 0;
         while ((lib = (TObjString *) nxl())) {
            // Expand path
            TString xlib = lib->GetName();
            gSystem->ExpandPathName(xlib);
            // Remove from the dynamic lib search path
            TString newlibpath = gSystem->GetDynamicPath();
            newlibpath.ReplaceAll(TString::Format("%s:", xlib.Data()),"");
            gSystem->SetDynamicPath(newlibpath);
         }

         // Forward the request, if required
         if (IsMaster())
            fProof->RemoveDynamicPath(path);

      } else {

         // Remove incs
         TIter nxi(op);
         TObjString *inc = 0;
         while ((inc = (TObjString *) nxi())) {
            TString newincpath = gSystem->GetIncludePath();
            newincpath.ReplaceAll(TString::Format("-I%s", inc->GetName()),"");
            // Remove the interpreter path (added anyhow internally)
            newincpath.ReplaceAll(gInterpreter->GetIncludePath(),"");
            gSystem->SetIncludePath(newincpath);
         }

         // Forward the request, if required
         if (IsMaster())
            fProof->RemoveIncludePath(path);
      }
   }
}

//______________________________________________________________________________
void TProofServ::HandleCheckFile(TMessage *mess, TString *slb)
{
   // Handle file checking request.

   TString filenam;
   TMD5    md5;
   UInt_t  opt = TProof::kUntar;

   TMessage reply(kPROOF_CHECKFILE);

   // Parse message
   (*mess) >> filenam >> md5;
   if ((mess->BufferSize() > mess->Length()) && (fProtocol > 8))
      (*mess) >> opt;

   if (slb) *slb = filenam;

   if (filenam.BeginsWith("-")) {
      // install package:
      // compare md5's, untar, store md5 in PROOF-INF, remove par file
      Int_t  st  = 0;
      Bool_t err = kFALSE;
      filenam = filenam.Strip(TString::kLeading, '-');
      TString packnam = filenam;
      packnam.Remove(packnam.Length() - 4);  // strip off ".par"
      // compare md5's to check if transmission was ok
      fPackageLock->Lock();
      TMD5 *md5local = TMD5::FileChecksum(fPackageDir + "/" + filenam);
      if (md5local && md5 == (*md5local)) {
         if ((opt & TProof::kRemoveOld)) {
            // remove any previous package directory with same name
            st = gSystem->Exec(TString::Format("%s %s/%s", kRM, fPackageDir.Data(),
                               packnam.Data()));
            if (st)
               Error("HandleCheckFile", "failure executing: %s %s/%s",
                     kRM, fPackageDir.Data(), packnam.Data());
         }
         // find gunzip...
         char *gunzip = gSystem->Which(gSystem->Getenv("PATH"), kGUNZIP,
                                       kExecutePermission);
         if (gunzip) {
            // untar package
            st = gSystem->Exec(TString::Format(kUNTAR, gunzip, fPackageDir.Data(),
                               filenam.Data(), fPackageDir.Data()));
            if (st)
               Error("HandleCheckFile", "failure executing: %s",
                     TString::Format(kUNTAR, gunzip, fPackageDir.Data(),
                          filenam.Data(), fPackageDir.Data()).Data());
            delete [] gunzip;
         } else
            Error("HandleCheckFile", "%s not found", kGUNZIP);
         // check that fPackageDir/packnam now exists
         if (gSystem->AccessPathName(fPackageDir + "/" + packnam, kWritePermission)) {
            // par file did not unpack itself in the expected directory, failure
            reply << (Int_t)0;
            if (fProtocol <= 19) reply.Reset(kPROOF_FATAL);
            err = kTRUE;
            Error("HandleCheckFile", "package %s did not unpack into %s",
                                     filenam.Data(), packnam.Data());
         } else {
            // store md5 in package/PROOF-INF/md5.txt
            TString md5f = fPackageDir + "/" + packnam + "/PROOF-INF/md5.txt";
            TMD5::WriteChecksum(md5f, md5local);
            // Notify the client
            reply << (Int_t)1;
            PDB(kPackage, 1)
               Info("HandleCheckFile",
                    "package %s installed on node", filenam.Data());
         }
      } else {
         reply << (Int_t)0;
         if (fProtocol <= 19) reply.Reset(kPROOF_FATAL);
         err = kTRUE;
         PDB(kPackage, 1)
            Info("HandleCheckFile",
                 "package %s not yet on node", filenam.Data());
      }

      // Note: Originally an fPackageLock->Unlock() call was made
      // after the if-else statement below. With multilevel masters,
      // submasters still check to make sure the package exists with
      // the correct md5 checksum and need to do a read lock there.
      // As yet locking is not that sophisicated so the lock must
      // be released below before the call to fProof->UploadPackage().
      if (err) {
         // delete par file in case of error
         gSystem->Exec(TString::Format("%s %s/%s", kRM, fPackageDir.Data(),
                       filenam.Data()));
         fPackageLock->Unlock();
      } else if (IsMaster()) {
         // forward to workers
         fPackageLock->Unlock();
         fProof->UploadPackage(fPackageDir + "/" + filenam, (TProof::EUploadPackageOpt)opt);
      } else {
         // Unlock in all cases
         fPackageLock->Unlock();
      }
      delete md5local;
      fSocket->Send(reply);

   } else if (filenam.BeginsWith("+")) {
      // check file in package directory
      filenam = filenam.Strip(TString::kLeading, '+');
      TString packnam = filenam;
      packnam.Remove(packnam.Length() - 4);  // strip off ".par"
      TString md5f = fPackageDir + "/" + packnam + "/PROOF-INF/md5.txt";
      fPackageLock->Lock();
      TMD5 *md5local = TMD5::ReadChecksum(md5f);
      fPackageLock->Unlock();
      if (md5local && md5 == (*md5local)) {
         // package already on server, unlock directory
         reply << (Int_t)1;
         PDB(kPackage, 1)
            Info("HandleCheckFile",
                 "package %s already on node", filenam.Data());
         if (IsMaster())
            fProof->UploadPackage(fPackageDir + "/" + filenam);
      } else {
         reply << (Int_t)0;
         if (fProtocol <= 19) reply.Reset(kPROOF_FATAL);
         PDB(kPackage, 1)
            Info("HandleCheckFile",
                 "package %s not yet on node", filenam.Data());
      }
      delete md5local;
      fSocket->Send(reply);

   } else if (filenam.BeginsWith("=")) {
      // check file in package directory, do not lock if it is the wrong file
      filenam = filenam.Strip(TString::kLeading, '=');
      TString packnam = filenam;
      packnam.Remove(packnam.Length() - 4);  // strip off ".par"
      TString md5f = fPackageDir + "/" + packnam + "/PROOF-INF/md5.txt";
      fPackageLock->Lock();
      TMD5 *md5local = TMD5::ReadChecksum(md5f);
      fPackageLock->Unlock();
      if (md5local && md5 == (*md5local)) {
         // package already on server, unlock directory
         reply << (Int_t)1;
         PDB(kPackage, 1)
            Info("HandleCheckFile",
                 "package %s already on node", filenam.Data());
         if (IsMaster())
            fProof->UploadPackage(fPackageDir + "/" + filenam);
      } else {
         reply << (Int_t)0;
         if (fProtocol <= 19) reply.Reset(kPROOF_FATAL);
         PDB(kPackage, 1)
            Info("HandleCheckFile",
                 "package %s not yet on node", filenam.Data());
      }
      delete md5local;
      fSocket->Send(reply);

   } else {
      // check file in cache directory
      TString cachef = fCacheDir + "/" + filenam;
      fCacheLock->Lock();
      TMD5 *md5local = TMD5::FileChecksum(cachef);

      if (md5local && md5 == (*md5local)) {
         // copy file from cache to working directory
         Bool_t cp = ((opt & TProof::kCp || opt & TProof::kCpBin) || (fProtocol <= 19)) ? kTRUE : kFALSE;
         if (cp) {
            Bool_t cpbin = (opt & TProof::kCpBin) ? kTRUE : kFALSE;
            CopyFromCache(filenam, cpbin);
         }
         reply << (Int_t)1;
         PDB(kCache, 1)
            Info("HandleCheckFile", "file %s already on node", filenam.Data());
      } else {
         reply << (Int_t)0;
         if (fProtocol <= 19) reply.Reset(kPROOF_FATAL);
         PDB(kCache, 1)
            Info("HandleCheckFile", "file %s not yet on node", filenam.Data());
      }
      delete md5local;
      fSocket->Send(reply);
      fCacheLock->Unlock();
   }
}

//______________________________________________________________________________
Int_t TProofServ::HandleCache(TMessage *mess, TString *slb)
{
   // Handle here all cache and package requests.

   PDB(kGlobal, 1)
      Info("HandleCache", "Enter");

   Int_t status = 0;
   Int_t type = 0;
   Bool_t all = kFALSE;
   TMessage msg;
   Bool_t fromglobal = kFALSE;

   // Notification message
   TString noth;
   const char *k = (IsMaster()) ? "Mst" : "Wrk";
   noth.Form("%s-%s", k, fOrdinal.Data());

   TList *optls = 0;
   TString packagedir(fPackageDir), package, pdir, ocwd, file;
   (*mess) >> type;
   switch (type) {
      case TProof::kShowCache:
         (*mess) >> all;
         printf("*** File cache %s:%s ***\n", gSystem->HostName(),
                fCacheDir.Data());
         fflush(stdout);
         PDB(kCache, 1) {
            gSystem->Exec(TString::Format("%s -a %s", kLS, fCacheDir.Data()));
         } else {
            gSystem->Exec(TString::Format("%s %s", kLS, fCacheDir.Data()));
         }
         if (IsMaster() && all)
            fProof->ShowCache(all);
         LogToMaster();
         if (slb) slb->Form("%d %d", type, all);
         break;
      case TProof::kClearCache:
         file = "";
         if ((mess->BufferSize() > mess->Length())) (*mess) >> file;
         fCacheLock->Lock();
         if (file.IsNull() || file == "*") {
            gSystem->Exec(TString::Format("%s %s/* %s/.*.binversion", kRM, fCacheDir.Data(), fCacheDir.Data()));
         } else {
            gSystem->Exec(TString::Format("%s %s/%s", kRM, fCacheDir.Data(), file.Data()));
         }
         fCacheLock->Unlock();
         if (IsMaster())
            fProof->ClearCache(file);
         if (slb) slb->Form("%d %s", type, file.Data());
         break;
      case TProof::kShowPackages:
         (*mess) >> all;
         if (fGlobalPackageDirList && fGlobalPackageDirList->GetSize() > 0) {
            // Scan the list of global packages dirs
            TIter nxd(fGlobalPackageDirList);
            TNamed *nm = 0;
            while ((nm = (TNamed *)nxd())) {
               printf("*** Global Package cache %s %s:%s ***\n",
                      nm->GetName(), gSystem->HostName(), nm->GetTitle());
               fflush(stdout);
               gSystem->Exec(TString::Format("%s %s", kLS, nm->GetTitle()));
               printf("\n");
               fflush(stdout);
            }
         }
         printf("*** Package cache %s:%s ***\n", gSystem->HostName(),
                fPackageDir.Data());
         fflush(stdout);
         gSystem->Exec(TString::Format("%s %s", kLS, fPackageDir.Data()));
         if (IsMaster() && all)
            fProof->ShowPackages(all);
         LogToMaster();
         if (slb) slb->Form("%d %d", type, all);
         break;
      case TProof::kClearPackages:
         status = UnloadPackages();
         if (status == 0) {
            fPackageLock->Lock();
            gSystem->Exec(TString::Format("%s %s/*", kRM, fPackageDir.Data()));
            fPackageLock->Unlock();
            if (IsMaster())
               status = fProof->ClearPackages();
         }
         if (slb) slb->Form("%d %d", type, status);
         break;
      case TProof::kClearPackage:
         (*mess) >> package;
         status = UnloadPackage(package);
         if (status == 0) {
            fPackageLock->Lock();
            // remove package directory and par file
            gSystem->Exec(TString::Format("%s %s/%s", kRM, fPackageDir.Data(),
                          package.Data()));
            if (IsMaster())
               gSystem->Exec(TString::Format("%s %s/%s.par", kRM, fPackageDir.Data(),
                             package.Data()));
            fPackageLock->Unlock();
            if (IsMaster())
               status = fProof->ClearPackage(package);
         }
         if (slb) slb->Form("%d %s %d", type, package.Data(), status);
         break;
      case TProof::kBuildPackage:
         (*mess) >> package;

         // always follows BuildPackage so no need to check for PROOF-INF
         pdir = fPackageDir + "/" + package;

         fromglobal = kFALSE;
         if (gSystem->AccessPathName(pdir, kReadPermission) ||
             gSystem->AccessPathName(pdir + "/PROOF-INF", kReadPermission)) {
            // Is there a global package with this name?
            if (fGlobalPackageDirList && fGlobalPackageDirList->GetSize() > 0) {
               // Scan the list of global packages dirs
               TIter nxd(fGlobalPackageDirList);
               TNamed *nm = 0;
               while ((nm = (TNamed *)nxd())) {
                  pdir.Form("%s/%s", nm->GetTitle(), package.Data());
                  if (!gSystem->AccessPathName(pdir, kReadPermission) &&
                      !gSystem->AccessPathName(pdir + "/PROOF-INF", kReadPermission)) {
                     // Package found, stop searching
                     fromglobal = kTRUE;
                     packagedir = nm->GetTitle();
                     break;
                  }
                  pdir = "";
               }
               if (pdir.Length() <= 0) {
                  // Package not found
                  SendAsynMessage(TString::Format("%s: kBuildPackage: failure locating %s ...",
                                       noth.Data(), package.Data()));
                  break;
               }
            }
         }

         if (IsMaster() && !fromglobal) {
            // make sure package is available on all slaves, even new ones
            fProof->UploadPackage(pdir + ".par");
         }
         fPackageLock->Lock();

         if (!status) {

            PDB(kPackage, 1)
               Info("HandleCache",
                    "kBuildPackage: package %s exists and has PROOF-INF directory", package.Data());

            ocwd = gSystem->WorkingDirectory();
            gSystem->ChangeDirectory(pdir);

            // forward build command to slaves, but don't wait for results
            if (IsMaster())
               fProof->BuildPackage(package, TProof::kBuildOnSlavesNoWait);

            // check for BUILD.sh and execute
            if (!gSystem->AccessPathName("PROOF-INF/BUILD.sh")) {
               // Notify the upper level
               SendAsynMessage(TString::Format("%s: building %s ...", noth.Data(), package.Data()));

               // read version from file proofvers.txt, and if current version is
               // not the same do a "BUILD.sh clean"
               Bool_t savever = kFALSE;
               TString v;
               Int_t rev = -1;
               FILE *f = fopen("PROOF-INF/proofvers.txt", "r");
               if (f) {
                  TString r;
                  v.Gets(f);
                  r.Gets(f);
                  rev = (!r.IsNull() && r.IsDigit()) ? r.Atoi() : -1;
                  fclose(f);
               }
               if (!f || v != gROOT->GetVersion() ||
                  (gROOT->GetSvnRevision() > 0 && rev != gROOT->GetSvnRevision())) {
                  if (!fromglobal || !gSystem->AccessPathName(pdir, kWritePermission)) {
                     savever = kTRUE;
                     SendAsynMessage(TString::Format("%s: %s: version change (current: %s:%d,"
                                          " build: %s:%d): cleaning ... ",
                                          noth.Data(), package.Data(), gROOT->GetVersion(),
                                          gROOT->GetSvnRevision(), v.Data(), rev));
                     // Hard cleanup: go up the dir tree
                     gSystem->ChangeDirectory(packagedir);
                     // remove package directory
                     gSystem->Exec(TString::Format("%s %s", kRM, pdir.Data()));
                     // find gunzip...
                     char *gunzip = gSystem->Which(gSystem->Getenv("PATH"), kGUNZIP,
                                                   kExecutePermission);
                     if (gunzip) {
                        TString par;
                        par.Form("%s.par", pdir.Data());
                        // untar package
                        TString cmd;
                        cmd.Form(kUNTAR3, gunzip, par.Data());
                        status = gSystem->Exec(cmd);
                        if (status) {
                           Error("HandleCache", "kBuildPackage: failure executing: %s", cmd.Data());
                        } else {
                           // Store md5 in package/PROOF-INF/md5.txt
                           TMD5 *md5local = TMD5::FileChecksum(par);
                           if (md5local) {
                              TString md5f = packagedir + "/" + package + "/PROOF-INF/md5.txt";
                              TMD5::WriteChecksum(md5f, md5local);
                              // Go down to the package directory
                              gSystem->ChangeDirectory(pdir);
                              // Cleanup
                              SafeDelete(md5local);
                           } else {
                              Error("HandleCache", "kBuildPackage: failure calculating MD5sum for '%s'", par.Data());
                           }
                        }
                        delete [] gunzip;
                     } else
                        Error("HandleCache", "kBuildPackage: %s not found", kGUNZIP);
                  } else {
                     SendAsynMessage(TString::Format("%s: %s: ROOT version inconsistency (current: %s, build: %s):"
                                          " global package: cannot re-build!!! ",
                                          noth.Data(), package.Data(), gROOT->GetVersion(), v.Data()));
                  }
               }

               if (!status) {
                  // To build the package we execute PROOF-INF/BUILD.sh via a pipe
                  // so that we can send back the log in (almost) real-time to the
                  // (impatient) client. Note that this operation will block, so
                  // the messages from builds on the workers will reach the client
                  // shortly after the master ones.
                  TString ipath(gSystem->GetIncludePath());
                  ipath.ReplaceAll("\"","");
                  TString cmd;
                  cmd.Form("export ROOTINCLUDEPATH=\"%s\" ; PROOF-INF/BUILD.sh", ipath.Data());
                  {
                     TProofServLogHandlerGuard hg(cmd, fSocket);
                  }
                  if (!(status = TProofServLogHandler::GetCmdRtn())) {
                     // Success: write version file
                     if (savever) {
                        f = fopen("PROOF-INF/proofvers.txt", "w");
                        if (f) {
                           fputs(gROOT->GetVersion(), f);
                           fputs(TString::Format("\n%d",gROOT->GetSvnRevision()), f);
                           fclose(f);
                        }
                     }
                  }
               }
            } else {
               // Notify the user
               PDB(kPackage, 1)
                  Info("HandleCache", "no PROOF-INF/BUILD.sh found for package %s", package.Data());
            }
            gSystem->ChangeDirectory(ocwd);
         }

         fPackageLock->Unlock();

         if (status) {
            // Notify the upper level
            SendAsynMessage(TString::Format("%s: failure building %s ... (status: %d)", noth.Data(), package.Data(), status));
         } else {
            // collect built results from slaves
            if (IsMaster())
               status = fProof->BuildPackage(package, TProof::kCollectBuildResults);
            PDB(kPackage, 1)
               Info("HandleCache", "package %s successfully built", package.Data());
         }
         if (slb) slb->Form("%d %s %d", type, package.Data(), status);
         break;
      case TProof::kLoadPackage:
         (*mess) >> package;

         // If already loaded don't do it again
         if (fEnabledPackages->FindObject(package)) {
            Info("HandleCache",
                 "package %s already loaded", package.Data());
            break;
         }

         // always follows BuildPackage so no need to check for PROOF-INF
         pdir = fPackageDir + "/" + package;

         if (gSystem->AccessPathName(pdir, kReadPermission)) {
            // Is there a global package with this name?
            if (fGlobalPackageDirList && fGlobalPackageDirList->GetSize() > 0) {
               // Scan the list of global packages dirs
               TIter nxd(fGlobalPackageDirList);
               TNamed *nm = 0;
               while ((nm = (TNamed *)nxd())) {
                  pdir.Form("%s/%s", nm->GetTitle(), package.Data());
                  if (!gSystem->AccessPathName(pdir, kReadPermission)) {
                     // Package found, stop searching
                     break;
                  }
                  pdir = "";
               }
               if (pdir.Length() <= 0) {
                  // Package not found
                  SendAsynMessage(TString::Format("%s: kLoadPackage: failure locating %s ...",
                                       noth.Data(), package.Data()));
                  break;
               }
            }
         }

         ocwd = gSystem->WorkingDirectory();
         gSystem->ChangeDirectory(pdir);

         // We have to be atomic here
         fPackageLock->Lock();

         // Check for SETUP.C and execute
         if (!gSystem->AccessPathName("PROOF-INF/SETUP.C")) {
            // We need to change the name of the function to avoid problems when we load more packages
            TString setup, setupfn;
            setup.Form("SETUP_%x", package.Hash());
            // Remove special characters
            setupfn.Form("%s/%s.C", gSystem->TempDirectory(), setup.Data());
            TMacro setupmc("PROOF-INF/SETUP.C");
            TObjString *setupline = setupmc.GetLineWith("SETUP(");
            if (setupline) {
               TString setupstring(setupline->GetString());
               setupstring.ReplaceAll("SETUP(", TString::Format("%s(", setup.Data()));
               setupline->SetString(setupstring);
            } else {
               // Macro does not contain SETUP()
               SendAsynMessage(TString::Format("%s: warning: macro '%s/PROOF-INF/SETUP.C' does not contain a SETUP()"
                                               " function", noth.Data(), package.Data()));
            }
            setupmc.SaveSource(setupfn.Data());
            // Load the macro
            if (gROOT->LoadMacro(setupfn.Data()) != 0) {
               // Macro could not be loaded
               SendAsynMessage(TString::Format("%s: error: macro '%s/PROOF-INF/SETUP.C' could not be loaded:"
                                                " cannot continue",
                                                noth.Data(), package.Data()));
               status = -1;
            } else {
               // Check the signature
               TFunction *fun = (TFunction *) gROOT->GetListOfGlobalFunctions()->FindObject(setup);
               if (!fun) {
                  // Notify the upper level
                  SendAsynMessage(TString::Format("%s: error: function SETUP() not found in macro '%s/PROOF-INF/SETUP.C':"
                                                   " cannot continue",
                                                   noth.Data(), package.Data()));
                  status = -1;
               } else {
                  TMethodCall callEnv;
                  // Check the number of arguments
                  if (fun->GetNargs() == 0) {
                     // No arguments (basic signature)
                     callEnv.InitWithPrototype(setup.Data(),"");
                     if ((mess->BufferSize() > mess->Length())) {
                        (*mess) >> optls;
                        SendAsynMessage(TString::Format("%s: warning: loaded SETUP() does not take any argument:"
                                                        " the specified argument will be ignored", noth.Data()));
                     }
                  } else if (fun->GetNargs() == 1) {
                     TMethodArg *arg = (TMethodArg *) fun->GetListOfMethodArgs()->First();
                     if (arg) {
                        // Get argument
                        if ((mess->BufferSize() > mess->Length())) (*mess) >> optls;
                        // Check argument type
                        TString argsig(arg->GetTitle());
                        if (argsig.BeginsWith("TList")) {
                           callEnv.InitWithPrototype(setup.Data(),"TList *");
                           callEnv.ResetParam();
                           callEnv.SetParam((Long_t) optls);
                        } else if (argsig.BeginsWith("const char")) {
                           callEnv.InitWithPrototype(setup.Data(),"const char *");
                           callEnv.ResetParam();
                           TObjString *os = optls ? dynamic_cast<TObjString *>(optls->First()) : 0;
                           if (os) {
                              callEnv.SetParam((Long_t) os->GetName());
                           } else {
                              if (optls && optls->First()) {
                                 SendAsynMessage(TString::Format("%s: warning: found object argument of type %s:"
                                                                 " SETUP expects 'const char *': ignoring",
                                                                 noth.Data(), optls->First()->ClassName()));
                              }
                              callEnv.SetParam((Long_t) 0);
                           }
                        } else {
                           // Notify the upper level
                           SendAsynMessage(TString::Format("%s: error: unsupported SETUP signature: SETUP(%s)"
                                                            " cannot continue", noth.Data(), arg->GetTitle()));
                           status = -1;
                        }
                     } else {
                        // Notify the upper level
                        SendAsynMessage(TString::Format("%s: error: cannot get information about the SETUP() argument:"
                                                         " cannot continue", noth.Data()));
                        status = -1;
                     }
                  } else if (fun->GetNargs() > 1) {
                     // Notify the upper level
                     SendAsynMessage(TString::Format("%s: error: function SETUP() can have at most a 'TList *' argument:"
                                                      " cannot continue", noth.Data()));
                     status = -1;
                  }
                  // Execute
                  Long_t setuprc = (status == 0) ? 0 : -1;
                  if (status == 0) {
                     callEnv.Execute(setuprc);
                     if (setuprc < 0) status = -1;
                  }
               }
            }
            if (!gSystem->AccessPathName(setupfn.Data())) gSystem->Unlink(setupfn.Data());
         }

         // End of atomicity
         fPackageLock->Unlock();

         gSystem->ChangeDirectory(ocwd);

         if (status < 0) {

            // Notify the upper level
            SendAsynMessage(TString::Format("%s: failure loading %s ...", noth.Data(), package.Data()));

         } else {

            // create link to package in working directory
            gSystem->Symlink(pdir, package);

            // add package to list of include directories to be searched
            // by ACliC
            gSystem->AddIncludePath(TString("-I") + package);

            // add package to list of include directories to be searched by CINT
            gROOT->ProcessLine(TString(".include ") + package);

            // if successful add to list and propagate to slaves
            fEnabledPackages->Add(new TObjString(package));
            if (IsMaster()) {
               if (optls && optls->GetSize() > 0) {
                  // List argument
                  status = fProof->LoadPackage(package, kFALSE, optls);
               } else {
                  // No argument
                  status = fProof->LoadPackage(package);
               }
            }

            PDB(kPackage, 1)
               Info("HandleCache", "package %s successfully loaded", package.Data());
         }
         if (slb) slb->Form("%d %s %d", type, package.Data(), status);
         break;
      case TProof::kShowEnabledPackages:
         (*mess) >> all;
         if (IsMaster()) {
            if (all)
               printf("*** Enabled packages on master %s on %s\n",
                      fOrdinal.Data(), gSystem->HostName());
            else
               printf("*** Enabled packages ***\n");
         } else {
            printf("*** Enabled packages on slave %s on %s\n",
                   fOrdinal.Data(), gSystem->HostName());
         }
         {
            TIter next(fEnabledPackages);
            while (TObjString *str = (TObjString*) next())
               printf("%s\n", str->GetName());
         }
         if (IsMaster() && all)
            fProof->ShowEnabledPackages(all);
         LogToMaster();
         if (slb) slb->Form("%d %d", type, all);
         break;
      case TProof::kShowSubCache:
         (*mess) >> all;
         if (IsMaster() && all)
            fProof->ShowCache(all);
         LogToMaster();
         if (slb) slb->Form("%d %d", type, all);
         break;
      case TProof::kClearSubCache:
         file = "";
         if ((mess->BufferSize() > mess->Length())) (*mess) >> file;
         if (IsMaster())
            fProof->ClearCache(file);
         if (slb) slb->Form("%d %s", type, file.Data());
         break;
      case TProof::kShowSubPackages:
         (*mess) >> all;
         if (IsMaster() && all)
            fProof->ShowPackages(all);
         LogToMaster();
         if (slb) slb->Form("%d %d", type, all);
         break;
      case TProof::kDisableSubPackages:
         if (IsMaster())
            fProof->DisablePackages();
         if (slb) slb->Form("%d", type);
         break;
      case TProof::kDisableSubPackage:
         (*mess) >> package;
         if (IsMaster())
            fProof->DisablePackage(package);
         if (slb) slb->Form("%d %s", type, package.Data());
         break;
      case TProof::kBuildSubPackage:
         (*mess) >> package;
         if (IsMaster())
            fProof->BuildPackage(package);
         if (slb) slb->Form("%d %s", type, package.Data());
         break;
      case TProof::kUnloadPackage:
         (*mess) >> package;
         status = UnloadPackage(package);
         if (IsMaster() && status == 0)
            status = fProof->UnloadPackage(package);
         if (slb) slb->Form("%d %s %d", type, package.Data(), status);
         break;
      case TProof::kDisablePackage:
         (*mess) >> package;
         fPackageLock->Lock();
         // remove package directory and par file
         gSystem->Exec(TString::Format("%s %s/%s", kRM, fPackageDir.Data(),
                       package.Data()));
         gSystem->Exec(TString::Format("%s %s/%s.par", kRM, fPackageDir.Data(),
                       package.Data()));
         fPackageLock->Unlock();
         if (IsMaster())
            fProof->DisablePackage(package);
         if (slb) slb->Form("%d %s", type, package.Data());
         break;
      case TProof::kUnloadPackages:
         status = UnloadPackages();
         if (IsMaster() && status == 0)
            status = fProof->UnloadPackages();
         if (slb) slb->Form("%d %s %d", type, package.Data(), status);
         break;
      case TProof::kDisablePackages:
         fPackageLock->Lock();
         gSystem->Exec(TString::Format("%s %s/*", kRM, fPackageDir.Data()));
         fPackageLock->Unlock();
         if (IsMaster())
            fProof->DisablePackages();
         if (slb) slb->Form("%d %s", type, package.Data());
         break;
      case TProof::kListEnabledPackages:
         msg.Reset(kPROOF_PACKAGE_LIST);
         msg << type << fEnabledPackages;
         fSocket->Send(msg);
         if (slb) slb->Form("%d", type);
         break;
      case TProof::kListPackages:
         {
            TList *pack = new TList;
            void *dir = gSystem->OpenDirectory(fPackageDir);
            if (dir) {
               TString pac(gSystem->GetDirEntry(dir));
               while (pac.Length() > 0) {
                  if (pac.EndsWith(".par")) {
                     pac.ReplaceAll(".par","");
                     pack->Add(new TObjString(pac.Data()));
                  }
                  pac = gSystem->GetDirEntry(dir);
               }
            }
            gSystem->FreeDirectory(dir);
            msg.Reset(kPROOF_PACKAGE_LIST);
            msg << type << pack;
            fSocket->Send(msg);
         }
         if (slb) slb->Form("%d", type);
         break;
      case TProof::kLoadMacro:

         (*mess) >> package;

         // By first forwarding the load command to the unique workers
         // and only then loading locally we load/build in parallel
         if (IsMaster())
            fProof->Load(package, kFALSE, kTRUE);

         // Atomic action
         fCacheLock->Lock();

         // Load locally; the implementation and header files (and perhaps
         // the binaries) are already in the cache
         CopyFromCache(package, kTRUE);

         // Load the macro
         Info("HandleCache", "loading macro %s ...", package.Data());
         gROOT->ProcessLine(TString::Format(".L %s", package.Data()));

         // Cache binaries, if any new
         CopyToCache(package, 1);

         // Release atomicity
         fCacheLock->Unlock();

         // Now we collect the result from the unique workers and send the load request
         // to the other workers (no compilation)
         if (IsMaster())
            fProof->Load(package, kFALSE, kFALSE);

         // Notify the upper level
         LogToMaster();

         if (slb) slb->Form("%d %s", type, package.Data());
         break;
      default:
         Error("HandleCache", "unknown type %d", type);
         break;
   }

   // We are done
   return status;
}

//______________________________________________________________________________
void TProofServ::HandleWorkerLists(TMessage *mess)
{
   // Handle here all requests to modify worker lists

   PDB(kGlobal, 1)
      Info("HandleWorkerLists", "Enter");

   Int_t type = 0;
   TString ord;

   (*mess) >> type;

   switch (type) {
      case TProof::kActivateWorker:
         (*mess) >> ord;
         if (fProof) {
            Int_t nact = fProof->GetListOfActiveSlaves()->GetSize();
            Int_t nactmax = fProof->GetListOfSlaves()->GetSize() -
                            fProof->GetListOfBadSlaves()->GetSize();
            if (nact < nactmax) {
               fProof->ActivateWorker(ord);
               Int_t nactnew = fProof->GetListOfActiveSlaves()->GetSize();
               if (ord == "*") {
                  if (nactnew == nactmax) {
                     Info("HandleWorkerList","all workers (re-)activated");
                  } else {
                     Info("HandleWorkerList","%d workers could not be (re-)activated", nactmax - nactnew);
                  }
               } else {
                  if (nactnew == (nact + 1)) {
                     Info("HandleWorkerList","worker %s (re-)activated", ord.Data());
                  } else {
                     Info("HandleWorkerList","worker %s could not be (re-)activated;"
                                             " # of actives: %d --> %d", ord.Data(), nact, nactnew);
                  }
               }
            } else {
               Info("HandleWorkerList","all workers are already active");
            }
         } else {
            Warning("HandleWorkerList","undefined PROOF session: protocol error?");
         }
         break;
      case TProof::kDeactivateWorker:
         (*mess) >> ord;
         if (fProof) {
            Int_t nact = fProof->GetListOfActiveSlaves()->GetSize();
            if (nact > 0) {
               fProof->DeactivateWorker(ord);
               Int_t nactnew = fProof->GetListOfActiveSlaves()->GetSize();
               if (ord == "*") {
                  if (nactnew == 0) {
                     Info("HandleWorkerList","all workers deactivated");
                  } else {
                     Info("HandleWorkerList","%d workers could not be deactivated", nactnew);
                  }
               } else {
                  if (nactnew == (nact - 1)) {
                     Info("HandleWorkerList","worker %s deactivated", ord.Data());
                  } else {
                     Info("HandleWorkerList","worker %s could not be deactivated:"
                                             " # of actives: %d --> %d", ord.Data(), nact, nactnew);
                  }
               }
            } else {
               Info("HandleWorkerList","all workers are already inactive");
            }
         } else {
            Warning("HandleWorkerList","undefined PROOF session: protocol error?");
         }
         break;
      default:
         Warning("HandleWorkerList","unknown action type (%d)", type);
   }
}

//______________________________________________________________________________
TProofServ::EQueryAction TProofServ::GetWorkers(TList *workers,
                                                Int_t & /* prioritychange */,
                                                Bool_t /* resume */)
{
   // Get list of workers to be used from now on.
   // The list must be provided by the caller.

   // Parse the config file
   TProofResourcesStatic *resources =
      new TProofResourcesStatic(fConfDir, fConfFile);
   fConfFile = resources->GetFileName(); // Update the global file name (with path)
   PDB(kGlobal,1)
         Info("GetWorkers", "using PROOF config file: %s", fConfFile.Data());

   // Get the master
   TProofNodeInfo *master = resources->GetMaster();
   if (!master) {
      PDB(kAll,1)
         Info("GetWorkers",
              "no appropriate master line found in %s", fConfFile.Data());
      return kQueryStop;
   } else {
      // Set image if not yet done and available
      if (fImage.IsNull() && strlen(master->GetImage()) > 0)
         fImage = master->GetImage();
   }

   // Fill submaster or worker list
   if (workers) {
      if (resources->GetSubmasters() && resources->GetSubmasters()->GetSize() > 0) {
         PDB(kAll,1)
            resources->GetSubmasters()->Print();
         TProofNodeInfo *ni = 0;
         TIter nw(resources->GetSubmasters());
         while ((ni = (TProofNodeInfo *) nw()))
            workers->Add(new TProofNodeInfo(*ni));
      } else if (resources->GetWorkers() && resources->GetWorkers()->GetSize() > 0) {
         PDB(kAll,1)
            resources->GetWorkers()->Print();
         TProofNodeInfo *ni = 0;
         TIter nw(resources->GetWorkers());
         while ((ni = (TProofNodeInfo *) nw()))
            workers->Add(new TProofNodeInfo(*ni));
      }
   }

   // We are done
   return kQueryOK;
}

//______________________________________________________________________________
FILE *TProofServ::SetErrorHandlerFile(FILE *ferr)
{
   // Set the file stream where to log (default stderr).
   // If ferr == 0 the default is restored.
   // Returns current setting.

   FILE *oldferr = fgErrorHandlerFile;
   fgErrorHandlerFile = (ferr) ? ferr : stderr;
   return oldferr;
}

//______________________________________________________________________________
void TProofServ::ErrorHandler(Int_t level, Bool_t abort, const char *location,
                              const char *msg)
{
   // The PROOF error handler function. It prints the message on fgErrorHandlerFile and
   // if abort is set it aborts the application.

   if (gErrorIgnoreLevel == kUnset) {
      gErrorIgnoreLevel = 0;
      if (gEnv) {
         TString lvl = gEnv->GetValue("Root.ErrorIgnoreLevel", "Print");
         if (!lvl.CompareTo("Print", TString::kIgnoreCase))
            gErrorIgnoreLevel = kPrint;
         else if (!lvl.CompareTo("Info", TString::kIgnoreCase))
            gErrorIgnoreLevel = kInfo;
         else if (!lvl.CompareTo("Warning", TString::kIgnoreCase))
            gErrorIgnoreLevel = kWarning;
         else if (!lvl.CompareTo("Error", TString::kIgnoreCase))
            gErrorIgnoreLevel = kError;
         else if (!lvl.CompareTo("Break", TString::kIgnoreCase))
            gErrorIgnoreLevel = kBreak;
         else if (!lvl.CompareTo("SysError", TString::kIgnoreCase))
            gErrorIgnoreLevel = kSysError;
         else if (!lvl.CompareTo("Fatal", TString::kIgnoreCase))
            gErrorIgnoreLevel = kFatal;
      }
   }

   if (level < gErrorIgnoreLevel)
      return;

   // Always communicate errors via SendLogFile
   if (level >= kError && gProofServ)
      gProofServ->LogToMaster();

   Bool_t tosyslog = (fgLogToSysLog > 2) ? kTRUE : kFALSE;

   const char *type   = 0;
   ELogLevel loglevel = kLogInfo;

   Int_t ipos = (location) ? strlen(location) : 0;

   if (level >= kPrint) {
      loglevel = kLogInfo;
      type = "Print";
   }
   if (level >= kInfo) {
      loglevel = kLogInfo;
      char *ps = location ? (char *) strrchr(location, '|') : (char *)0;
      if (ps) {
         ipos = (int)(ps - (char *)location);
         type = "SvcMsg";
      } else {
         type = "Info";
      }
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


   TString buf;

   // Time stamp
   TTimeStamp ts;
   TString st(ts.AsString("lc"),19);

   if (!location || ipos == 0 ||
       (level >= kPrint && level < kInfo) ||
       (level >= kBreak && level < kSysError)) {
      fprintf(fgErrorHandlerFile, "%s %5d %s | %s: %s\n", st(11,8).Data(),
                                  gSystem->GetPid(),
                                 (gProofServ ? gProofServ->GetPrefix() : "proof"),
                                  type, msg);
      if (tosyslog)
         buf.Form("%s: %s:%s", fgSysLogEntity.Data(), type, msg);
   } else {
      fprintf(fgErrorHandlerFile, "%s %5d %s | %s in <%.*s>: %s\n", st(11,8).Data(),
                                  gSystem->GetPid(),
                                 (gProofServ ? gProofServ->GetPrefix() : "proof"),
                                  type, ipos, location, msg);
      if (tosyslog)
         buf.Form("%s: %s:<%.*s>: %s", fgSysLogEntity.Data(), type, ipos, location, msg);
   }
   fflush(fgErrorHandlerFile);
   
   if (tosyslog)
      gSystem->Syslog(loglevel, buf);
   
   if (abort) {

      static Bool_t recursive = kFALSE;

      if (gProofServ != 0 && !recursive) {
         recursive = kTRUE;
         gProofServ->GetSocket()->Send(kPROOF_FATAL);
         recursive = kFALSE;
      }

      fprintf(fgErrorHandlerFile, "aborting\n");
      fflush(fgErrorHandlerFile);
      gSystem->StackTrace();
      gSystem->Abort();
   }
}

//______________________________________________________________________________
Int_t TProofServ::CopyFromCache(const char *macro, Bool_t cpbin)
{
   // Retrieve any files related to 'macro' from the cache directory.
   // If 'cpbin' is true, the associated binaries are retrieved as well.
   // Returns 0 on success, -1 otherwise

   if (!macro || strlen(macro) <= 0)
      // Invalid inputs
      return -1;

   // Split out the aclic mode, if any
   TString name = macro;
   TString acmode, args, io;
   name = gSystem->SplitAclicMode(name, acmode, args, io);

   PDB(kGlobal,1)
      Info("CopyFromCache","enter: names: %s, %s", macro, name.Data());

   // Atomic action
   Bool_t locked = (fCacheLock->IsLocked()) ? kTRUE : kFALSE;
   if (!locked) fCacheLock->Lock();

   // Get source from the cache
   Bool_t assertfile = kFALSE;
   TString srcname(name);
   Int_t dot = srcname.Last('.');
   if (dot != kNPOS) {
      srcname.Remove(dot);
      srcname += ".*";
   } else {
      assertfile = kTRUE;
   }
   srcname.Insert(0, TString::Format("%s/",fCacheDir.Data()));
   dot = (dot != kNPOS) ? srcname.Last('.') : dot;
   // Assert the file if asked (to silence warnings from 'cp')
   if (assertfile) {
      if (gSystem->AccessPathName(srcname)) {
         PDB(kCache,1)
            Info("CopyFromCache", "file %s not in cache", srcname.Data());
         if (!locked) fCacheLock->Unlock();
         return 0;
      }
   }
   PDB(kCache,1)
      Info("CopyFromCache", "retrieving %s from cache", srcname.Data());
   gSystem->Exec(TString::Format("%s %s .", kCP, srcname.Data()));

   // Check if we are done
   if (!cpbin) {
      // End of atomicity
      if (!locked) fCacheLock->Unlock();
      return 0;
   }

   // Create binary name template
   TString binname = name;
   dot = binname.Last('.');
   if (dot != kNPOS) {
      binname.Replace(dot,1,"_");
      binname += ".";
   } else {
      PDB(kCache,1)
         Info("CopyFromCache",
              "non-standard name structure: %s ('.' missing)", name.Data());
      // Done
      if (!locked) fCacheLock->Unlock();
      return 0;
   }

   // Binary version file name
   TString vername;
   vername.Form(".%s", name.Data());
   Int_t dotv = vername.Last('.');
   if (dotv != kNPOS)
      vername.Remove(dotv);
   vername += ".binversion";

   // Check binary version
   TString v;
   Int_t rev = -1;
   Bool_t okfil = kFALSE;
   FILE *f = fopen(TString::Format("%s/%s", fCacheDir.Data(), vername.Data()), "r");
   if (f) {
      TString r;
      v.Gets(f);
      r.Gets(f);
      rev = (!r.IsNull() && r.IsDigit()) ? r.Atoi() : -1;
      fclose(f);
      okfil = kTRUE;
   }

   Bool_t okver = (v != gROOT->GetVersion()) ? kFALSE : kTRUE;
   Bool_t okrev = (gROOT->GetSvnRevision() > 0 && rev != gROOT->GetSvnRevision()) ? kFALSE : kTRUE;
   if (!okfil || !okver || !okrev) {
   PDB(kCache,1)
      Info("CopyFromCache",
           "removing binaries: 'file': %s, 'ROOT version': %s, 'ROOT revision': %s",
           (okfil ? "OK" : "not OK"), (okver ? "OK" : "not OK"), (okrev ? "OK" : "not OK") );
      // Remove all existing binaries
      binname += "*";
      gSystem->Exec(TString::Format("%s %s/%s", kRM, fCacheDir.Data(), binname.Data()));
      // ... and the binary version file
      gSystem->Exec(TString::Format("%s %s/%s", kRM, fCacheDir.Data(), vername.Data()));
      // Done
      if (!locked) fCacheLock->Unlock();
      return 0;
   }

   // Retrieve existing binaries, if any
   void *dirp = gSystem->OpenDirectory(fCacheDir);
   if (dirp) {
      const char *e = 0;
      while ((e = gSystem->GetDirEntry(dirp))) {
         if (!strncmp(e, binname.Data(), binname.Length())) {
            TString fncache;
            fncache.Form("%s/%s", fCacheDir.Data(), e);
            Bool_t docp = kTRUE;
            FileStat_t stlocal, stcache;
            if (!gSystem->GetPathInfo(fncache, stcache)) {
               Int_t rc = gSystem->GetPathInfo(e, stlocal);
               if (rc == 0 && (stlocal.fMtime >= stcache.fMtime))
                  docp = kFALSE;
               // If a copy candidate, check also the MD5
               if (docp) {
                  TMD5 *md5local = TMD5::FileChecksum(e);
                  TMD5 *md5cache = TMD5::FileChecksum(fncache);
                  if (md5local && md5cache && md5local == md5cache) docp = kFALSE;
                  SafeDelete(md5local);
                  SafeDelete(md5cache);
               }
               // Copy the file, if needed
               if (docp) {
                  gSystem->Exec(TString::Format("%s %s", kRM, e));
                  PDB(kCache,1)
                     Info("CopyFromCache",
                          "retrieving %s from cache", fncache.Data());
                  gSystem->Exec(TString::Format("%s %s %s", kCP, fncache.Data(), e));
               }
            }
         }
      }
      gSystem->FreeDirectory(dirp);
   }

   // End of atomicity
   if (!locked) fCacheLock->Unlock();

   // Done
   return 0;
}

//______________________________________________________________________________
Int_t TProofServ::CopyToCache(const char *macro, Int_t opt)
{
   // Copy files related to 'macro' to the cache directory.
   // Action depends on 'opt':
   //
   //    opt = 0         copy 'macro' to cache and delete from cache any binary
   //                    related to name; e.g. if macro = bla.C, the binaries are
   //                    bla_C.so, bla_C.rootmap, ...
   //    opt = 1         copy the binaries related to macro to the cache
   //
   // Returns 0 on success, -1 otherwise

   if (!macro || strlen(macro) <= 0 || opt < 0 || opt > 1)
      // Invalid inputs
      return -1;

   // Split out the aclic mode, if any
   TString name = macro;
   TString acmode, args, io;
   name = gSystem->SplitAclicMode(name, acmode, args, io);

   PDB(kGlobal,1)
      Info("CopyToCache","enter: opt: %d, names: %s, %s", opt, macro, name.Data());

   // Create binary name template
   TString binname = name;
   Int_t dot = binname.Last('.');
   if (dot != kNPOS)
      binname.Replace(dot,1,"_");

   // Create version file name template
   TString vername;
   vername.Form(".%s", name.Data());
   dot = vername.Last('.');
   if (dot != kNPOS)
      vername.Remove(dot);
   vername += ".binversion";
   Bool_t savever = kFALSE;

   // Atomic action
   Bool_t locked = (fCacheLock->IsLocked()) ? kTRUE : kFALSE;
   if (!locked) fCacheLock->Lock();

   // Action depends on 'opt'
   if (opt == 0) {
      // Save name to cache
      PDB(kCache,1)
         Info("CopyToCache",
              "caching %s/%s ...", fCacheDir.Data(), name.Data());
      gSystem->Exec(TString::Format("%s %s %s", kCP, name.Data(), fCacheDir.Data()));
      // If needed, remove from the cache any existing binary related to 'name'
      if (dot != kNPOS) {
         binname += ".*";
         PDB(kCache,1)
            Info("CopyToCache", "opt = 0: removing binaries '%s'", binname.Data());
         gSystem->Exec(TString::Format("%s %s/%s", kRM, fCacheDir.Data(), binname.Data()));
         gSystem->Exec(TString::Format("%s %s/%s", kRM, fCacheDir.Data(), vername.Data()));
      }
   } else if (opt == 1) {
      // If needed, copy to the cache any existing binary related to 'name'.
      if (dot != kNPOS) {
         binname += ".";
         void *dirp = gSystem->OpenDirectory(".");
         if (dirp) {
            const char *e = 0;
            while ((e = gSystem->GetDirEntry(dirp))) {
               if (!strncmp(e, binname.Data(), binname.Length())) {
                  Bool_t docp = kTRUE;
                  FileStat_t stlocal, stcache;
                  if (!gSystem->GetPathInfo(e, stlocal)) {
                     TString fncache;
                     fncache.Form("%s/%s", fCacheDir.Data(), e);
                     Int_t rc = gSystem->GetPathInfo(fncache, stcache);
                     if (rc == 0 && (stlocal.fMtime <= stcache.fMtime)) {
                        docp = kFALSE;
                        if (rc == 0) rc = -1;
                     }
                     // If a copy candidate, check also the MD5
                     if (docp) {
                        TMD5 *md5local = TMD5::FileChecksum(e);
                        TMD5 *md5cache = TMD5::FileChecksum(fncache);
                        if (md5local && md5cache && md5local == md5cache) docp = kFALSE;
                        SafeDelete(md5local);
                        SafeDelete(md5cache);
                        if (!docp) rc = -2;
                     }
                     // Copy the file, if needed
                     if (docp) {
                        gSystem->Exec(TString::Format("%s %s", kRM, fncache.Data()));
                        PDB(kCache,1)
                           Info("CopyToCache","caching %s ... (reason: %d)", e, rc);
                        gSystem->Exec(TString::Format("%s %s %s", kCP, e, fncache.Data()));
                        savever = kTRUE;
                     }
                  }
               }
            }
            gSystem->FreeDirectory(dirp);
         }
         // Save binary version if requested
         if (savever) {
            PDB(kCache,1)
               Info("CopyToCache","updating version file %s ...", vername.Data());
            FILE *f = fopen(TString::Format("%s/%s", fCacheDir.Data(), vername.Data()), "w");
            if (f) {
               fputs(gROOT->GetVersion(), f);
               fputs(TString::Format("\n%d",gROOT->GetSvnRevision()), f);
               fclose(f);
            }
         }
      }
   }

   // End of atomicity
   if (!locked) fCacheLock->Unlock();

   // Done
   return 0;
}

//______________________________________________________________________________
void TProofServ::MakePlayer()
{
   // Make player instance.

   TVirtualProofPlayer *p = 0;

   // Cleanup first
   DeletePlayer();

   if (IsParallel()) {
      // remote mode
      p = fProof->MakePlayer();
   } else {
      // slave or sequential mode
      p = TVirtualProofPlayer::Create("slave", 0, fSocket);
      if (IsMaster())
         fProof->SetPlayer(p);
   }

   // set player
   fPlayer = p;
}

//______________________________________________________________________________
void TProofServ::DeletePlayer()
{
   // Delete player instance.

   if (IsMaster()) {
      if (fProof) fProof->SetPlayer(0);
   } else {
      SafeDelete(fPlayer);
   }
   fPlayer = 0;
}

//______________________________________________________________________________
Int_t TProofServ::GetPriority()
{
   // Get the processing priority for the group the user belongs too. This
   // prioroty is a number (0 - 100) determined by a scheduler (third
   // party process) based on some basic priority the group has, e.g.
   // we might want to give users in a specific group (e.g. promptana)
   // a higher priority than users in other groups, and on the analysis
   // of historical logging data (i.e. usage of CPU by the group in a
   // previous time slot, as recorded in TPerfStats::WriteQueryLog()).
   //
   // Currently the group priority is obtained by a query in a SQL DB
   // table proofpriority, which has the format:
   // CREATE TABLE proofpriority (
   //   id            INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
   //   group         VARCHAR(32) NOT NULL,
   //   priority      INT
   //)

   TString sqlserv = gEnv->GetValue("ProofServ.QueryLogDB","");
   TString sqluser = gEnv->GetValue("ProofServ.QueryLogUser","");
   TString sqlpass = gEnv->GetValue("ProofServ.QueryLogPasswd","");

   Int_t priority = 100;

   if (sqlserv == "")
      return priority;

   TString sql;
   sql.Form("SELECT priority WHERE group='%s' FROM proofpriority", fGroup.Data());

   // open connection to SQL server
   TSQLServer *db =  TSQLServer::Connect(sqlserv, sqluser, sqlpass);

   if (!db || db->IsZombie()) {
      Error("GetPriority", "failed to connect to SQL server %s as %s %s",
            sqlserv.Data(), sqluser.Data(), sqlpass.Data());
      printf("%s\n", sql.Data());
   } else {
      TSQLResult *res = db->Query(sql);

      if (!res) {
         Error("GetPriority", "query into proofpriority failed");
         printf("%s\n", sql.Data());
      } else {
         TSQLRow *row = res->Next();   // first row is header
         priority = atoi(row->GetField(0));
         delete row;
      }
      delete res;
   }
   delete db;

   return priority;
}

//______________________________________________________________________________
Int_t TProofServ::SendAsynMessage(const char *msg, Bool_t lf)
{
   // Send an asychronous message to the master / client .
   // Masters will forward up the message to the client.
   // The client prints 'msg' of stderr and adds a '\n'/'\r' depending on
   // 'lf' being kTRUE (default) or kFALSE.
   // Returns the return value from TSocket::Send(TMessage &) .
   static TMessage m(kPROOF_MESSAGE);

   // To leave a track in the output file ... if requested
   // (clients will be notified twice)
   PDB(kAsyn,1)
      Info("SendAsynMessage","%s", (msg ? msg : "(null)"));

   if (fSocket && msg) {
      m.Reset(kPROOF_MESSAGE);
      m << TString(msg) << lf;
      return fSocket->Send(m);
   }

   // No message
   return -1;
}

//______________________________________________________________________________
void TProofServ::FlushLogFile()
{
   // Reposition the read pointer in the log file to the very end.
   // This allows to "hide" useful debug messages during normal operations
   // while preserving the possibility to have them in case of problems.

   off_t lend = lseek(fileno(stdout), (off_t)0, SEEK_END);
   if (lend >= 0) lseek(fLogFileDes, lend, SEEK_SET);
}

//______________________________________________________________________________
void TProofServ::TruncateLogFile()
{
   // Truncate the log file to the 80% of the required max size if this
   // is set.
#ifndef WIN32
   TString emsg;
   if (fLogFileMaxSize > 0 && fLogFileDes > 0) {
      fflush(stdout);
      struct stat st;
      if (fstat(fLogFileDes, &st) == 0) {
         if (st.st_size >= fLogFileMaxSize) {
            off_t truncsz = (off_t) (( fLogFileMaxSize * 80 ) / 100 );
            if (truncsz < 100) {
               emsg.Form("+++ WARNING +++: %s: requested truncate size too small"
                         " (%lld,%lld) - ignore ", fPrefix.Data(), (Long64_t) truncsz, fLogFileMaxSize);
               SendAsynMessage(emsg.Data());
               return;
            }
            TSystem::ResetErrno();
            while (ftruncate(fileno(stdout), truncsz) != 0 &&
                   (TSystem::GetErrno() == EINTR)) {
               TSystem::ResetErrno();
            }
            if (TSystem::GetErrno() > 0) {
               Error("TruncateLogFile", "truncating to %lld bytes; file size is %lld bytes (errno: %d)",
                                        (Long64_t)truncsz, (Long64_t)st.st_size, TSystem::GetErrno());
               emsg.Form("+++ WARNING +++: %s: problems truncating log file to %lld bytes; file size is %lld bytes"
                         " (errno: %d)", fPrefix.Data(), (Long64_t)truncsz, (Long64_t)st.st_size, TSystem::GetErrno());
               SendAsynMessage(emsg.Data());
            } else {
               Info("TruncateLogFile", "file truncated to %lld bytes (80%% of %lld); file size was %lld bytes ",
                                       (Long64_t)truncsz, fLogFileMaxSize, (Long64_t)st.st_size);
               emsg.Form("+++ WARNING +++: %s: log file truncated to %lld bytes (80%% of %lld)",
                                       fPrefix.Data(), (Long64_t)truncsz, fLogFileMaxSize);
               SendAsynMessage(emsg.Data());
            }
         }
      } else {
         emsg.Form("+++ WARNING +++: %s: could not stat log file descriptor"
                   " for truncation (errno: %d)", fPrefix.Data(), TSystem::GetErrno());
         SendAsynMessage(emsg.Data());
      }
   }
#endif
}

//______________________________________________________________________________
void TProofServ::HandleException(Int_t sig)
{
   // Exception handler: we do not try to recover here, just exit.

   Error("HandleException", "caugth exception triggered by signal '%d' %s",
                            sig, fgLastMsg.Data());
   // Description
   TString emsg;
   emsg.Form("%s: caught exception triggered by signal '%d' %s",
             GetOrdinal(), sig, fgLastMsg.Data());
   // Try to warn the user
   SendAsynMessage(emsg.Data());

   gSystem->Exit(sig);
}

//______________________________________________________________________________
Int_t TProofServ::HandleDataSets(TMessage *mess, TString *slb)
{
   // Handle here requests about datasets.

   if (gDebug > 0)
      Info("HandleDataSets", "enter");

   // We need a dataset manager
   if (!fDataSetManager) {
      Warning("HandleDataSets", "no data manager is available to fullfil the request");
      return -1;
   }

   // Used in most cases
   TString dsUser, dsGroup, dsName, dsTree, uri, opt;
   Int_t rc = 0;

   // Message type
   Int_t type = 0;
   (*mess) >> type;

   switch (type) {
      case TProof::kCheckDataSetName:
         //
         // Check whether this dataset exist
         {
            (*mess) >> uri;
            if (slb) slb->Form("%d %s", type, uri.Data());
            if (fDataSetManager->ExistsDataSet(uri))
               // Dataset name does exist
               return -1;
         }
         break;
      case TProof::kRegisterDataSet:
         // list size must be above 0
         {
            if (fDataSetManager->TestBit(TDataSetManager::kAllowRegister)) {
               (*mess) >> uri;
               (*mess) >> opt;
               if (slb) slb->Form("%d %s %s", type, uri.Data(), opt.Data());
               // Extract the list
               TFileCollection *dataSet =
                  dynamic_cast<TFileCollection*> ((mess->ReadObject(TFileCollection::Class())));
               if (!dataSet || dataSet->GetList()->GetSize() == 0) {
                  Error("HandleDataSets", "can not save an empty list.");
                  return -1;
               }
               // Register the dataset (quota checks are done inside here)
               rc = fDataSetManager->RegisterDataSet(uri, dataSet, opt);
               delete dataSet;
               return rc;
            } else {
               Info("HandleDataSets", "dataset registration not allowed");
               if (slb) slb->Form("%d notallowed", type);
               return -1;
            }
         }
         break;

      case TProof::kShowDataSets:
         {
            (*mess) >> uri >> opt;
            if (slb) slb->Form("%d %s %s", type, uri.Data(), opt.Data());
            // Show content
            fDataSetManager->ShowDataSets(uri, opt);
         }
         break;

      case TProof::kGetDataSets:
         {
            (*mess) >> uri >> opt;
            if (slb) slb->Form("%d %s %s", type, uri.Data(), opt.Data());
            // Get the datasets and fill a map
            UInt_t omsk = (UInt_t)TDataSetManager::kExport;
            Ssiz_t kLite = opt.Index(":lite:", 0, TString::kIgnoreCase);
            if (kLite != kNPOS) {
               omsk |= (UInt_t)TDataSetManager::kReadShort;
               opt.Remove(kLite, strlen(":lite:"));
            }
            TMap *returnMap = fDataSetManager->GetDataSets(uri, omsk);
            // If defines, option gives the name of a server for which to extract the information
            if (returnMap && !opt.IsNull()) {
               // The return map will be in the form   </group/user/datasetname> --> <dataset>
               TMap *rmap = new TMap;
               TObject *k = 0;
               TFileCollection *fc = 0, *xfc = 0;
               TIter nxd(returnMap);
               while ((k = nxd()) && (fc = (TFileCollection *) returnMap->GetValue(k))) {
                  // Get subset on specified server, if any
                  if ((xfc = fc->GetFilesOnServer(opt.Data()))) {
                     rmap->Add(new TObjString(k->GetName()), xfc);
                  }
               }
               returnMap->DeleteAll();
               if (rmap->GetSize() > 0) {
                  returnMap = rmap;
               } else {
                  Info("HandleDataSets", "no dataset found on server '%s'", opt.Data());
                  delete rmap;
                  returnMap = 0;
               }
            }
            if (returnMap) {
               // Send them back
               fSocket->SendObject(returnMap, kMESS_OK);
               returnMap->DeleteAll();
            } else {
               // Failure
               return -1;
            }
         }
         break;
      case TProof::kGetDataSet:
         {
            (*mess) >> uri >> opt;
            if (slb) slb->Form("%d %s %s", type, uri.Data(), opt.Data());
            // Get the list
            TFileCollection *fileList = fDataSetManager->GetDataSet(uri,opt);
            if (fileList) {
               fSocket->SendObject(fileList, kMESS_OK);
               delete fileList;
            } else {
               // Failure
               return -1;
            }
         }
         break;
      case TProof::kRemoveDataSet:
         {
            if (fDataSetManager->TestBit(TDataSetManager::kAllowRegister)) {
               (*mess) >> uri;
               if (slb) slb->Form("%d %s", type, uri.Data());
               if (!fDataSetManager->RemoveDataSet(uri)) {
                  // Failure
                  return -1;
               }
            } else {
               Info("HandleDataSets", "dataset creation / removal not allowed");
               if (slb) slb->Form("%d notallowed", type);
               return -1;
            }
         }
         break;
      case TProof::kVerifyDataSet:
         {
            if (fDataSetManager->TestBit(TDataSetManager::kAllowVerify)) {
               (*mess) >> uri >> opt;
               if (slb) slb->Form("%d %s %s", type, uri.Data(), opt.Data());
               TProofServLogHandlerGuard hg(fLogFile,  fSocket);
               rc = fDataSetManager->ScanDataSet(uri, opt);
               // TODO: verify in parallel:
               //  - dataset = GetDataSet(uri)
               //  - TList flist; TDataSetManager::ScanDataSet(dataset, ..., &flist)
               //  - fPlayer->Process( ... flist ...) // needs to be developed
               //  - dataset->Integrate(flist) (perhaps automatic; flist object owned by dataset)
               //  - RegisterDataSet(uri, dataset, "OT")
            } else {
               Info("HandleDataSets", "dataset verification not allowed");
               return -1;
            }
         }
         break;
      case TProof::kGetQuota:
         {
            if (fDataSetManager->TestBit(TDataSetManager::kCheckQuota)) {
               if (slb) slb->Form("%d", type);
               TMap *groupQuotaMap = fDataSetManager->GetGroupQuotaMap();
               if (groupQuotaMap) {
                  // Send result
                  fSocket->SendObject(groupQuotaMap, kMESS_OK);
               } else {
                  return -1;
               }
            } else {
               Info("HandleDataSets", "quota control disabled");
               if (slb) slb->Form("%d disabled", type);
               return -1;
            }
         }
         break;
      case TProof::kShowQuota:
         {
            if (fDataSetManager->TestBit(TDataSetManager::kCheckQuota)) {
               if (slb) slb->Form("%d", type);
               (*mess) >> opt;
               // Display quota information
               fDataSetManager->ShowQuota(opt);
            } else {
               Info("HandleDataSets", "quota control disabled");
               if (slb) slb->Form("%d disabled", type);
            }
         }
         break;
      case TProof::kSetDefaultTreeName:
         {
            if (fDataSetManager->TestBit(TDataSetManager::kAllowRegister)) {
               (*mess) >> uri;
               if (slb) slb->Form("%d %s", type, uri.Data());
               rc = fDataSetManager->ScanDataSet(uri, (UInt_t)TDataSetManager::kSetDefaultTree);
            } else {
               Info("HandleDataSets", "kSetDefaultTreeName: modification of dataset info not allowed");
               if (slb) slb->Form("%d notallowed", type);
               return -1;
            }
         }
         break;
      case TProof::kCache:
         {
            (*mess) >> uri >> opt;
            if (slb) slb->Form("%d %s %s", type, uri.Data(), opt.Data());
            if (opt == "show") {
               // Show cache content
               fDataSetManager->ShowCache(uri);
            } else if (opt == "clear") {
               // Clear cache content
               fDataSetManager->ClearCache(uri);
            } else {
               Error("HandleDataSets", "kCache: unknown action: %s", opt.Data());
            }
         }
         break;
      default:
         rc = -1;
         Error("HandleDataSets", "unknown type %d", type);
         break;
   }

   // We are done
   return rc;
}

//______________________________________________________________________________
void TProofServ::HandleSubmerger(TMessage *mess)
{
   // Handle a message of type kPROOF_SUBMERGER

   // Message type
   Int_t type = 0;
   (*mess) >> type;

   TString msg;
   switch (type) {
      case TProof::kOutputSize:
         break;

      case TProof::kSendOutput:
         {
            Bool_t deleteplayer = kTRUE;
            if (!IsMaster()) {
               if (fMergingMonitor) {
                  Info("HandleSubmerger", "kSendOutput: interrupting ...");
                  fMergingMonitor->Interrupt();
               }
               if (fMergingSocket) {
                  if (fMergingMonitor) fMergingMonitor->Remove(fMergingSocket);
                  fMergingSocket->Close();
                  SafeDelete(fMergingSocket);
               }

               TString name;
               Int_t port = 0;
               Int_t merger_id = -1;
               (*mess) >> merger_id >> name >> port;
               PDB(kSubmerger, 1)
                  Info("HandleSubmerger","worker %s redirected to merger #%d %s:%d", fOrdinal.Data(), merger_id, name.Data(), port);

               TSocket *t = 0;
               if (name.Length() > 0 && port > 0 && (t = new TSocket(name, port)) && t->IsValid()) {

                  PDB(kSubmerger, 2) Info("HandleSubmerger",
                                          "kSendOutput: worker asked for sending output to merger #%d %s:%d",
                                          merger_id, name.Data(), port);

                  if (SendResults(t, fPlayer->GetOutputList()) != 0) {
                     msg.Form("worker %s cannot send results to merger #%d at %s:%d", GetPrefix(), merger_id, name.Data(), port);
                     PDB(kSubmerger, 2) Info("HandleSubmerger",
                                             "kSendOutput: %s - inform the master", msg.Data());
                     SendAsynMessage(msg);
                     // Results not send
                     TMessage answ(kPROOF_SUBMERGER);
                     answ << Int_t(TProof::kMergerDown);
                     answ << merger_id;
                     fSocket->Send(answ);
                  } else {
                     // Worker informs master that it had sent its output to the merger
                     TMessage answ(kPROOF_SUBMERGER);
                     answ << Int_t(TProof::kOutputSent);
                     answ << merger_id;
                     fSocket->Send(answ);

                     PDB(kSubmerger, 2) Info("HandleSubmerger", "kSendOutput: worker sent its output");
                     fSocket->Send(kPROOF_SETIDLE);
                     SetIdle(kTRUE);
                     SendLogFile();
                  }
               } else {

                  if (name == "master") {
                     PDB(kSubmerger, 2) Info("HandleSubmerger",
                                             "kSendOutput: worker was asked for sending output to master");
                     if (SendResults(fSocket, fPlayer->GetOutputList()) != 0)
                        Warning("HandleSubmerger", "problems sending output list");
                     // Signal the master that we are idle
                     fSocket->Send(kPROOF_SETIDLE);
                     SetIdle(kTRUE);
                     SendLogFile();

                  } else if (!t || !(t->IsValid())) {
                     msg.Form("worker %s could not open a valid socket to merger #%d at %s:%d",
                              GetPrefix(), merger_id, name.Data(), port);
                     PDB(kSubmerger, 2) Info("HandleSubmerger",
                                             "kSendOutput: %s - inform the master", msg.Data());
                     SendAsynMessage(msg);
                     // Results not send
                     TMessage answ(kPROOF_SUBMERGER);
                     answ << Int_t(TProof::kMergerDown);
                     answ << merger_id;
                     fSocket->Send(answ);
                     deleteplayer = kFALSE;
                  }

                  if (t) SafeDelete(t);

               }

            } else {
               Error("HandleSubmerger", "kSendOutput: received not on worker");
            }

            // Cleanup
            if (deleteplayer) DeletePlayer();
         }
         break;
      case TProof::kBeMerger:
         {
            Bool_t deleteplayer = kTRUE;
            if (!IsMaster()) {
               Int_t merger_id = -1;
               //Int_t merger_port = 0;
               Int_t connections = 0;
               (*mess) >> merger_id  >> connections;
               PDB(kSubmerger, 2)
                  Info("HandleSubmerger", "worker %s established as merger", fOrdinal.Data());

               PDB(kSubmerger, 2)
                  Info("HandleSubmerger",
                       "kBeMerger: worker asked for being merger #%d for %d connections",
                       merger_id, connections);

               TVirtualProofPlayer *mergerPlayer =  TVirtualProofPlayer::Create("remote",fProof,0);
               PDB(kSubmerger, 2) Info("HandleSubmerger",
                                       "kBeMerger: mergerPlayer created (%p) ", mergerPlayer);

               // Accept results from assigned workers
               if (AcceptResults(connections, mergerPlayer)) {
                  PDB(kSubmerger, 2)
                     Info("HandleSubmerger", "kBeMerger: all outputs from workers accepted");

                  PDB(kSubmerger, 2)
                     Info("","adding own output to the list on %s", fOrdinal.Data());

                  // Add own results to the output list.
                  // On workers the player does not own the output list, which is owned
                  // by the selector and deleted in there
                  // On workers the player does not own the output list, which is owned
                  // by the selector and deleted in there
                  TIter nxo(fPlayer->GetOutputList());
                  TObject * o = 0;
                  while ((o = nxo())) {
                     if ((mergerPlayer->AddOutputObject(o) != 1)) {
                        // Remove the object if it has not been merged: it is owned
                        // now by the merger player (in its output list)
                        PDB(kSubmerger, 2) Info("HandleSocketInput", "removing merged object (%p)", o);
                        fPlayer->GetOutputList()->Remove(o);
                     }
                  }
                  PDB(kSubmerger, 2) Info("HandleSubmerger","kBeMerger: own outputs added");
                  PDB(kSubmerger, 2) Info("HandleSubmerger","starting delayed merging on %s", fOrdinal.Data());

                  // Delayed merging if neccessary
                  mergerPlayer->MergeOutput();

                  PDB(kSubmerger, 2) Info("HandleSubmerger", "delayed merging on %s finished ", fOrdinal.Data());
                  PDB(kSubmerger, 2) Info("HandleSubmerger", "%s sending results to master ", fOrdinal.Data());
                  // Send merged results to master
                  if (SendResults(fSocket, mergerPlayer->GetOutputList()) != 0)
                     Warning("HandleSubmerger","kBeMerger: problems sending output list");
                  mergerPlayer->GetOutputList()->SetOwner(kTRUE);
                  delete mergerPlayer;

                  PDB(kSubmerger, 2) Info("HandleSubmerger","kBeMerger: results sent to master");
                  // Signal the master that we are idle
                  fSocket->Send(kPROOF_SETIDLE);
                  SetIdle(kTRUE);
                  SendLogFile();
               } else {
                  // Results from all assigned workers not accepted
                  TMessage answ(kPROOF_SUBMERGER);
                  answ << Int_t(TProof::kMergerDown);
                  answ << merger_id;
                  fSocket->Send(answ);
                  deleteplayer = kFALSE;
               }
            } else {
               Error("HandleSubmerger","kSendOutput: received not on worker");
            }

            // Cleanup
            if (deleteplayer) DeletePlayer();
         }
         break;

      case TProof::kMergerDown:
         break;

      case TProof::kStopMerging:
         {
            // Received only in case of forced termination of merger by master
            PDB(kSubmerger, 2)  Info("HandleSubmerger", "kStopMerging");
            if (fMergingMonitor) {
               Info("HandleSubmerger", "kStopMerging: interrupting ...");
               fMergingMonitor->Interrupt();
            }
         }
         break;

      case TProof::kOutputSent:
         break;
   }
}

//______________________________________________________________________________
void TProofServ::HandleFork(TMessage *)
{
   // Cloning itself via fork. Not implemented

   Info("HandleFork", "fork cloning not implemented");
}

//______________________________________________________________________________
Int_t TProofServ::Fork()
{
   // Fork a child.
   // If successful, return 0 in the child process and the child pid in the parent
   // process. The child pid is registered for reaping.
   // Return <0 in the parent process in case of failure.

#ifndef WIN32
   // Fork
   pid_t pid;
   if ((pid = fork()) < 0) {
      Error("Fork", "failed to fork");
      return pid;
   }

   // Nothing else to do in the child
   if (!pid) return pid;

   // Make sure that the reaper timer is started
   if (!fReaperTimer) {
      fReaperTimer = new TReaperTimer(1000);
      fReaperTimer->Start(-1);
   }

   // Register the new child
   fReaperTimer->AddPid(pid);

   // Done
   return pid;
#else
   Warning("Fork", "Functionality not provided under windows");
   return -1;
#endif
}

//______________________________________________________________________________
void TProofServ::ResolveKeywords(TString &fname, const char *path)
{
   // Replace <ord>, <user>, <u>, <group>, <stag>, <qnum> and <file> placeholders in fname

   // Replace <user>, if any
   if (fname.Contains("<user>")) {
      if (gProofServ && gProofServ->GetUser() && strlen(gProofServ->GetUser())) {
         fname.ReplaceAll("<user>", gProofServ->GetUser());
      } else {
         fname.ReplaceAll("<user>", "nouser");
      }
   }
   // Replace <us>, if any
   if (fname.Contains("<u>")) {
      if (gProofServ && gProofServ->GetUser() && strlen(gProofServ->GetUser())) {
         TString u(gProofServ->GetUser()[0]);
         fname.ReplaceAll("<u>", u);
      } else {
         fname.ReplaceAll("<u>", "n");
      }
   }
   // Replace <group>, if any
   if (fname.Contains("<group>")) {
      if (gProofServ && gProofServ->GetGroup() && strlen(gProofServ->GetGroup()))
         fname.ReplaceAll("<group>", gProofServ->GetGroup());
      else
         fname.ReplaceAll("<group>", "default");
   }
   // Replace <stag>, if any
   if (fname.Contains("<stag>")) {
      if (gProofServ && gProofServ->GetSessionTag() && strlen(gProofServ->GetSessionTag()))
         fname.ReplaceAll("<stag>", gProofServ->GetSessionTag());
      else
         ::Warning("TProofServ::ResolveKeywords", "session tag undefined: ignoring");
   }
   // Replace <ord>, if any
   if (fname.Contains("<ord>")) {
      if (gProofServ && gProofServ->GetOrdinal() && strlen(gProofServ->GetOrdinal()))
         fname.ReplaceAll("<ord>", gProofServ->GetOrdinal());
      else
         ::Warning("TProofServ::ResolveKeywords", "ordinal number undefined: ignoring");
   }
   // Replace <qnum>, if any
   if (fname.Contains("<qnum>")) {
      if (gProofServ && gProofServ->GetQuerySeqNum() && gProofServ->GetQuerySeqNum() > 0)
         fname.ReplaceAll("<qnum>", TString::Format("%d", gProofServ->GetQuerySeqNum()).Data());
      else
         ::Warning("TProofServ::ResolveKeywords", "query seqeuntial number undefined: ignoring");
   }
   // Replace <file>, if any
   if (fname.Contains("<file>") && path && strlen(path) > 0) {
      fname.ReplaceAll("<file>", path);
   }
}

//______________________________________________________________________________
Int_t TProofServ::GetSessionStatus()
{
   // Return the status of this session:
   //     0     idle
   //     1     running
   //     2     being terminated  (currently unused)
   //     3     queued
   //     4     idle timed-out (not set in here but in TIdleTOTimer::Notify)
   // This is typically run in the reader thread, so access needs to be protected

   R__LOCKGUARD(fQMtx);
   Int_t st = (fIdle) ? 0 : 1;
   if (fIdle && fWaitingQueries->GetSize() > 0) st = 3;
   return st;
}

//______________________________________________________________________________
Int_t TProofServ::UpdateSessionStatus(Int_t xst)
{
   // Update the session status in the relevant file. The status is taken from
   // GetSessionStatus() unless xst >= 0, in which case xst is used.
   // Return 0 on success, -errno if the file could not be opened.

   FILE *fs = fopen(fAdminPath.Data(), "w");
   if (fs) {
      Int_t st = (xst < 0) ? GetSessionStatus() : xst;
      fprintf(fs, "%d", st);
      fclose(fs);
      PDB(kGlobal, 2)
         Info("UpdateSessionStatus", "status (=%d) update in path: %s", st, fAdminPath.Data());
   } else {
      return -errno;
   }
   // Done
   return 0;
}

//______________________________________________________________________________
Bool_t TProofServ::IsIdle()
{
   // Return the idle status
   R__LOCKGUARD(fQMtx);
   return fIdle;
}

//______________________________________________________________________________
void TProofServ::SetIdle(Bool_t st)
{
   // Change the idle status
   R__LOCKGUARD(fQMtx);
   fIdle = st;
}

//______________________________________________________________________________
Bool_t TProofServ::IsWaiting()
{
   // Return kTRUE if the session is waiting for the OK to start processing
   R__LOCKGUARD(fQMtx);
   if (fIdle && fWaitingQueries->GetSize() > 0) return kTRUE;
   return kFALSE;
}

//______________________________________________________________________________
Int_t TProofServ::WaitingQueries()
{
   // Return the number of waiting queries
   R__LOCKGUARD(fQMtx);
   return fWaitingQueries->GetSize();
}

//______________________________________________________________________________
Int_t TProofServ::QueueQuery(TProofQueryResult *pq)
{
   // Add a query to the waiting list
   // Returns the number of queries in the list
   R__LOCKGUARD(fQMtx);
   fWaitingQueries->Add(pq);
   return fWaitingQueries->GetSize();
}

//______________________________________________________________________________
TProofQueryResult *TProofServ::NextQuery()
{
   // Get the next query from the waiting list.
   // The query is removed from the list.
   R__LOCKGUARD(fQMtx);
   TProofQueryResult *pq = (TProofQueryResult *) fWaitingQueries->First();
   fWaitingQueries->Remove(pq);
   return pq;
}

//______________________________________________________________________________
Int_t TProofServ::CleanupWaitingQueries(Bool_t del, TList *qls)
{
   // Cleanup the waiting queries list. The objects are deleted if 'del' is true.
   // If 'qls' is non null, only objects in 'qls' are removed.
   // Returns the number of cleanup queries
   R__LOCKGUARD(fQMtx);
   Int_t ncq = 0;
   if (qls) {
      TIter nxq(qls);
      TObject *o = 0;
      while ((o = nxq())) {
         if (fWaitingQueries->FindObject(o)) ncq++;
         fWaitingQueries->Remove(o);
         if (del) delete o;
      }
   } else {
      ncq = fWaitingQueries->GetSize();
      fWaitingQueries->SetOwner(del);
      fWaitingQueries->Delete();
   }
   // Done
   return ncq;
}

//______________________________________________________________________________
void TProofServ::SetLastMsg(const char *lastmsg)
{
   // Set the message to be sent back in case of exceptions

   fgLastMsg = lastmsg;
}

//______________________________________________________________________________
Long_t TProofServ::GetVirtMemMax()
{
   // VirtMemMax getter
   return fgVirtMemMax;
}
//______________________________________________________________________________
Long_t TProofServ::GetResMemMax()
{
   // ResMemMax getter
   return fgResMemMax;
}
//______________________________________________________________________________
Float_t TProofServ::GetMemHWM()
{
   // MemHWM getter
   return fgMemHWM;
}
//______________________________________________________________________________
Float_t TProofServ::GetMemStop()
{
   // MemStop getter
   return fgMemStop;
}

//______________________________________________________________________________
Int_t TProofLockPath::Lock()
{
   // Locks the directory. Waits if lock is hold by an other process.
   // Returns 0 on success, -1 in case of error.

   const char *pname = GetName();

   if (gSystem->AccessPathName(pname))
      fLockId = open(pname, O_CREAT|O_RDWR, 0644);
   else
      fLockId = open(pname, O_RDWR);

   if (fLockId == -1) {
      SysError("Lock", "cannot open lock file %s", pname);
      return -1;
   }

   PDB(kPackage, 2)
      Info("Lock", "%d: locking file %s ...", gSystem->GetPid(), pname);
   // lock the file
#if !defined(R__WIN32) && !defined(R__WINGCC)
   if (lockf(fLockId, F_LOCK, (off_t) 1) == -1) {
      SysError("Lock", "error locking %s", pname);
      close(fLockId);
      fLockId = -1;
      return -1;
   }
#endif

   PDB(kPackage, 2)
      Info("Lock", "%d: file %s locked", gSystem->GetPid(), pname);

   return 0;
}

//______________________________________________________________________________
Int_t TProofLockPath::Unlock()
{
   // Unlock the directory. Returns 0 in case of success,
   // -1 in case of error.

   if (!IsLocked())
      return 0;

   PDB(kPackage, 2)
      Info("Lock", "%d: unlocking file %s ...", gSystem->GetPid(), GetName());
   // unlock the file
   lseek(fLockId, 0, SEEK_SET);
#if !defined(R__WIN32) && !defined(R__WINGCC)
   if (lockf(fLockId, F_ULOCK, (off_t)1) == -1) {
      SysError("Unlock", "error unlocking %s", GetName());
      close(fLockId);
      fLockId = -1;
      return -1;
   }
#endif

   PDB(kPackage, 2)
      Info("Unlock", "%d: file %s unlocked", gSystem->GetPid(), GetName());

   close(fLockId);
   fLockId = -1;

   return 0;
}
