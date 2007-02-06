// @(#)root/proof:$Name:  $:$Id: TProofServ.cxx,v 1.162 2007/02/05 23:12:28 rdm Exp $
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

#ifdef R__HAVE_CONFIG
#include "RConfigure.h"
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
#include "TAuthenticate.h"
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
#include "TSocket.h"
#include "TStopwatch.h"
#include "TSystem.h"
#include "TTimeStamp.h"
#include "TUrl.h"
#include "TTree.h"
#include "TPluginManager.h"
#include "TObjString.h"
#include "compiledata.h"
#include "TProofResourcesStatic.h"
#include "TProofNodeInfo.h"
#include "TFileInfo.h"
#include "TMutex.h"

// global proofserv handle
TProofServ *gProofServ = 0;

// debug hook
static volatile Int_t gProofServDebug = 1;

// Max number of queries kept (-1 to disable)
Int_t TProofServ::fgMaxQueries = -1;


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
//______________________________________________________________________________
TProofServLogHandler::TProofServLogHandler(const char *cmd,
                                             TSocket *s, const char *pfx)
                     : TFileHandler(-1, 1), fSocket(s), fPfx(pfx)
{
   // Execute 'cmd' in a pipe and handle output messages from the related file

   ResetBit(kFileIsPipe);
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

   if (TestBit(kFileIsPipe) && fFile)
      gSystem->ClosePipe(fFile);
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
         // Send the message one level up
         m.Reset(kPROOF_MESSAGE);
         if (fPfx.Length() > 0) {
            // Prepend prefix specific to this instance
            m << TString(Form("%s: %s", fPfx.Data(), line));
         } else if (fgPfx.Length() > 0) {
            // Prepend default prefix
            m << TString(Form("%s: %s", fgPfx.Data(), line));
         } else {
            // Nothing to prepend
            m << TString(line);
         }
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

//--- Special timer to constrol delayed shutdowns ----------------------------//
//______________________________________________________________________________
Bool_t TShutdownTimer::Notify()
{
   // Handle expiration of the shutdown timer. The Terminate() method is called
   // which will exit the main loop.

   if (gDebug > 0)
      Info ("Notify","called!");

   fProofServ->HandleTermination();
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

   // abort on higher than kSysError's and set error handler
   gErrorAbortLevel = kSysError + 1;
   SetErrorHandler(ErrorHandler);

   fNcmd            = 0;
   fInterrupt       = kFALSE;
   fProtocol        = 0;
   fOrdinal         = "-1";
   if (gSystem->Getenv("ROOTPROOFORDINAL"))
      fOrdinal      = gSystem->Getenv("ROOTPROOFORDINAL");
   fGroupId         = -1;
   fGroupSize       = 0;
   fRealTime        = 0.0;
   fCpuTime         = 0.0;
   fProof           = 0;
   fPlayer          = 0;
   fSocket          = 0;
   fEnabledPackages = new TList;
   fEnabledPackages->SetOwner();

   fLogFile         = flog;
   fLogFileDes      = -1;

   fArchivePath     = "";

   // Init lockers
   fPackageLock     = 0;
   fCacheLock       = 0;
   fQueryLock       = 0;
   fDataSetLock     = 0;

   fSeqNum          = 0;
   fDrawQueries     = 0;
   fKeptQueries     = 0;
   fQueries         = new TList;
   fWaitingQueries  = new TList;
   fPreviousQueries = 0;
   fIdle            = kTRUE;

   fRealTimeLog     = kFALSE;

   fShutdownWhenIdle = kTRUE;
   fShutdownTimer   = 0;
   fShutdownTimerMtx = 0;

   if (gSystem->Getenv("ROOTPROOFLOGLEVEL"))
      gProofDebugLevel = atoi(gSystem->Getenv("ROOTPROOFLOGLEVEL"));
   else
      gProofDebugLevel = gEnv->GetValue("Proof.DebugLevel",0);
   fLogLevel = gProofDebugLevel;

   gProofDebugMask = (TProofDebug::EProofDebugMask) gEnv->GetValue("Proof.DebugMask",~0);
   if (gProofDebugLevel > 0)
      Info("TProofServ", "DebugLevel %d Mask 0x%x", gProofDebugLevel, gProofDebugMask);

   // Parse options
   GetOptions(argc, argv);

   // Default prefix in the form '<role>-<ordinal>'
   TString pfx = (IsMaster() ? "master-" : "worker-");
   if (fOrdinal != "-1")
      pfx += fOrdinal;
   TProofServLogHandler::SetDefaultPrefix(pfx);

   // Set global to this instance
   gProofServ = this;
}

//______________________________________________________________________________
Int_t TProofServ::CreateServer()
{
   // Finalize the server setup. If master, create the TProof instance to talk
   // the worker or submaster nodes.
   // Return 0 on success, -1 on error

   // get socket to be used (setup in proofd)
   if (!(gSystem->Getenv("ROOTOPENSOCK"))) {
      Fatal("CreateServer", "Socket setup by proofd undefined");
      return -1;
   }
   Int_t sock = strtol(gSystem->Getenv("ROOTOPENSOCK"), (char **)0, 10);
   if (sock <= 0) {
      Fatal("CreateServer", "Invalid socket descriptor number (%d)", sock);
      return -1;
   }
   fSocket = new TSocket(sock);

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
      SendLogFile();
      Terminate(0);
      return -1;
   }

   // Set the default prefix in the form '<role>-<ordinal>' (it was already done
   // in the constructor, but for standard PROOF the ordinal number is only set in
   // Setup(), so we need to do it again here)
   TString pfx = (IsMaster() ? "master-" : "worker-");
   pfx += GetOrdinal();
   TProofServLogHandler::SetDefaultPrefix(pfx);

   if (!fLogFile) {
      RedirectOutput();
      // If for some reason we failed setting a redirection fole for the logs
      // we cannot continue
      if (!fLogFile || (fLogFileDes = fileno(fLogFile)) < 0) {
         SendLogFile(-98);
         Terminate(0);
         return -1;
      }
   } else {
      // Use the file already open by pmain
      if ((fLogFileDes = fileno(fLogFile)) < 0) {
         SendLogFile(-98);
         Terminate(0);
         return -1;
      }
   }

   // Send message of the day to the client
   if (IsMaster()) {
      if (CatMotd() == -1) {
         SendLogFile(-99);
         Terminate(0);
         return -1;
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

   // The following libs are also useful to have, make sure they are loaded...
   //gROOT->LoadClass("TMinuit",     "Minuit");
   //gROOT->LoadClass("TPostScript", "Postscript");

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
   gSystem->AddSignalHandler(new TProofServInterruptHandler(this));
   gSystem->AddFileHandler(new TProofServInputHandler(this, sock));

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
      fProof = reinterpret_cast<TProof*>(h->ExecPlugin(4, master.Data(),
                                                          fConfFile.Data(),
                                                          GetConfDir(),
                                                          fLogLevel));
      if (!fProof || !fProof->IsValid()) {
         Error("CreateServer", "plugin for TProof could not be executed");
         delete fProof;
         fProof = 0;
         SendLogFile(-99);
         Terminate(0);
         return -1;
      }
      // Find out if we are a master in direct contact only with workers
      fEndMaster = fProof->IsEndMaster();

      SendLogFile();
   }

   // Done
   return 0;
}

//______________________________________________________________________________
TProofServ::~TProofServ()
{
   // Cleanup. Not really necessary since after this dtor there is no
   // live anyway.

   SafeDelete(fQueries);
   SafeDelete(fPreviousQueries);
   SafeDelete(fWaitingQueries);
   SafeDelete(fEnabledPackages);
   SafeDelete(fSocket);
   SafeDelete(fPackageLock);
   SafeDelete(fCacheLock);
   SafeDelete(fQueryLock);
   SafeDelete(fDataSetLock);
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
   close(fd);
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
TDSetElement *TProofServ::GetNextPacket(Long64_t totalEntries)
{
   // Get next range of entries to be processed on this server.

   Long64_t bytesRead = 0;

   if (gPerfStats != 0) bytesRead = gPerfStats->GetBytesRead();

   if (fCompute.Counter() > 0)
      fCompute.Stop();

   TMessage req(kPROOF_GETPACKET);
   req << fLatency.RealTime() << fCompute.RealTime()
       << fCompute.CpuTime() << bytesRead
       << totalEntries;
   if (fPlayer)
       req << fPlayer->GetEventsProcessed();

   fLatency.Start();
   Int_t rc = fSocket->Send(req);
   if (rc <= 0) {
      Error("GetNextPacket","Send() failed, returned %d", rc);
      return 0;
   }

   TMessage *mess;
   if ((rc = fSocket->Recv(mess)) <= 0) {
      fLatency.Stop();
      Error("GetNextPacket","Recv() failed, returned %d", rc);
      return 0;
   }

   fLatency.Stop();

   TDSetElement  *e = 0;
   TString        file;
   TString        dir;
   TString        obj;

   Int_t what = mess->What();

   switch (what) {
      case kPROOF_GETPACKET:

         (*mess) >> e;
         if (e != 0) {
            fCompute.Start();
            PDB(kLoop, 2) Info("GetNextPacket", "'%s' '%s' '%s' %lld %lld",
                               e->GetFileName(), e->GetDirectory(),
                               e->GetObjName(), e->GetFirst(),e->GetNum());
         } else {
            PDB(kLoop, 2) Info("GetNextPacket", "Done");
         }

         delete mess;

         return e;

      case kPROOF_STOPPROCESS:
         // if a kPROOF_STOPPROCESS message is returned to kPROOF_GETPACKET
         // GetNextPacket() will return 0 and the TPacketizer and hence
         // TEventIter will be stopped
         PDB(kLoop, 2) Info("GetNextPacket:kPROOF_STOPPROCESS","received");
         break;

      default:
         Error("GetNextPacket","unexpected answer message type: %d",what);
         break;
   }

   delete mess;

   return 0;
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

   static Int_t recursive = 0;

   if (recursive > 0) {
      HandleSocketInputDuringProcess();
      return;
   }
   recursive++;

   static TStopwatch timer;

   TMessage *mess;
   char      str[2048];
   Int_t     what;

   if (fSocket->Recv(mess) <= 0) {
      // Pending: do something more intelligent here
      // but at least get a message in the log file
      Error("HandleSocketInput", "retrieving message from input socket");
      Terminate(0);
      return;
   }

   what = mess->What();

   timer.Start();
   fNcmd++;

   if (fProof) fProof->SetActive();

   switch (what) {

      case kMESS_CINT:
         mess->ReadString(str, sizeof(str));
         if (IsMaster() && IsParallel()) {
            fProof->SendCommand(str);
         } else {
            PDB(kGlobal, 1)
               Info("HandleSocketInput:kMESS_CINT", "processing: %s...", str);
            ProcessLine(str);
         }
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
         {
            UInt_t mask;
            mess->ReadString(str, sizeof(str));
            sscanf(str, "%d %u", &fLogLevel, &mask);
            gProofDebugLevel = fLogLevel;
            gProofDebugMask  = (TProofDebug::EProofDebugMask) mask;
            if (IsMaster())
               fProof->SetLogLevel(fLogLevel, mask);
         }
         break;

      case kPROOF_PING:
         if (IsMaster())
            fProof->Ping();
         // do nothing (ping is already acknowledged)
         break;

      case kPROOF_PRINT:
         mess->ReadString(str, sizeof(str));
         Print(str);
         SendLogFile();
         break;

      case kPROOF_RESET:
         mess->ReadString(str, sizeof(str));
         Reset(str);
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
         Terminate(0);
         break;

      case kPROOF_STOPPROCESS:
         // this message makes only sense when the query is being processed,
         // however the message can also be received if the user pressed
         // ctrl-c, so ignore it!
         PDB(kGlobal, 1) Info("HandleSocketInput:kPROOF_STOPPROCESS","enter");
         break;

      case kPROOF_PROCESS:
         {  TProofServLogHandlerGuard hg(fLogFile, fSocket, "", fRealTimeLog);

            PDB(kGlobal, 1) Info("HandleSocketInput:kPROOF_PROCESS","enter");
            HandleProcess(mess);
         }
         // Notify
         SendLogFile();
         break;

      case kPROOF_QUERYLIST:

         HandleQueryList(mess);

         // Notify
         SendLogFile();
         break;

      case kPROOF_REMOVE:

         HandleRemove(mess);

         // Notify
         SendLogFile();
         break;

      case kPROOF_RETRIEVE:

         HandleRetrieve(mess);

         // Notify
         SendLogFile();
         break;

      case kPROOF_ARCHIVE:

         HandleArchive(mess);

         // Notify
         SendLogFile();
         break;


      case kPROOF_MAXQUERIES:
         {
            PDB(kGlobal, 1)
               Info("HandleSocketInput:kPROOF_MAXQUERIES", "Enter");
            TMessage m(kPROOF_MAXQUERIES);
            m << fgMaxQueries;
            fSocket->Send(m);
         }
         // Notify
         SendLogFile();
         break;

      case kPROOF_CLEANUPSESSION:
         {
            PDB(kGlobal, 1)
               Info("HandleSocketInput:kPROOF_CLEANUPSESSION", "Enter");
            TString stag;
            (*mess) >> stag;
            if (CleanupSession(stag) == 0) {
               Printf("Session %s cleaned up", stag.Data());
            } else {
               Printf("Could not cleanup session %s", stag.Data());
            }
         }
         // Notify
         SendLogFile();
         break;

      case kPROOF_GETENTRIES:
         {
            PDB(kGlobal, 1) Info("HandleSocketInput:kPROOF_GETENTRIES", "Enter");
            Bool_t         isTree;
            TString        filename;
            TString        dir;
            TString        objname;
            Long64_t       entries;

            (*mess) >> isTree >> filename >> dir >> objname;

            PDB(kGlobal, 2) Info("HandleSocketInput:kPROOF_GETENTRIES",
                                 "Report size of object %s (%s) in dir %s in file %s",
                                 objname.Data(), isTree ? "T" : "O",
                                 dir.Data(), filename.Data());

            entries = TDSet::GetEntries(isTree, filename, dir, objname);

            PDB(kGlobal, 2) Info("HandleSocketInput:kPROOF_GETENTRIES",
                                 "Found %lld %s", entries, isTree ? "entries" : "objects");

            TMessage answ(kPROOF_GETENTRIES);
            answ << entries;
            SendLogFile(); // in case of error messages
            fSocket->Send(answ);
            PDB(kGlobal, 1) Info("HandleSocketInput:kPROOF_GETENTRIES", "Done");
         }
         break;

      case kPROOF_CHECKFILE:

         // Handle file checking request
         HandleCheckFile(mess);
         break;

      case kPROOF_SENDFILE:
         mess->ReadString(str, sizeof(str));
         {
            Long_t size;
            Int_t  bin, fw = 1;
            char   name[1024];
            if (fProtocol > 5)
               sscanf(str, "%s %d %ld %d", name, &bin, &size, &fw);
            else
               sscanf(str, "%s %d %ld", name, &bin, &size);
            ReceiveFile(name, bin ? kTRUE : kFALSE, size);
            // copy file to cache if not a PAR file
            if (size > 0) {
               if (strncmp(fPackageDir, name, fPackageDir.Length())) {
                  fCacheLock->Lock();
                  gSystem->Exec(Form("%s %s %s", kCP, name, fCacheDir.Data()));
                  fCacheLock->Unlock();
               }
            }
            if (IsMaster() && fw == 1)
               fProof->SendFile(name, bin);
         }
         break;

      case kPROOF_LOGFILE:
         {
            Int_t start, end;
            (*mess) >> start >> end;
            PDB(kGlobal, 1)
               Info("HandleSocketInput:kPROOF_LOGFILE",
                    "Logfile request - byte range: %d - %d", start, end);

            SendLogFile(0, start, end);
         }
         break;

      case kPROOF_PARALLEL:
         if (IsMaster()) {
            Int_t nodes;
            Bool_t random = kFALSE;
            (*mess) >> nodes;
            if ((mess->BufferSize() > mess->Length()))
               (*mess) >> random;
            fProof->SetParallel(nodes, random);
            SendLogFile();
         }
         break;

      case kPROOF_CACHE:
         {  TProofServLogHandlerGuard hg(fLogFile, fSocket, "", fRealTimeLog);
            PDB(kGlobal, 1) Info("HandleSocketInput:kPROOF_CACHE","enter");

            Int_t status = HandleCache(mess);

            // Notify
            SendLogFile(status);
         }
         break;

      case kPROOF_WORKERLISTS:
         {
            if (IsMaster())
               HandleWorkerLists(mess);
            else
               Warning("HandleSocketInput:kPROOF_WORKERLISTS",
                       "Action meaning-less on worker nodes: protocol error?");
            // Notify
            SendLogFile();
         }
         break;

      case kPROOF_GETSLAVEINFO:
         {
            PDB(kGlobal, 1) Info("HandleSocketInput:kPROOF_GETSLAVEINFO", "Enter");

            if (IsMaster()) {
               TList *info = fProof->GetSlaveInfo();

               TMessage answ(kPROOF_GETSLAVEINFO);
               answ << info;
               fSocket->Send(answ);
            } else {
               TMessage answ(kPROOF_GETSLAVEINFO);
               answ << (TList *)0;
               fSocket->Send(answ);
            }

            PDB(kGlobal, 1) Info("HandleSocketInput:kPROOF_GETSLAVEINFO", "Done");
         }
         break;
      case kPROOF_GETTREEHEADER:
         {
            PDB(kGlobal, 1) Info("HandleSocketInput:kPROOF_GETTREEHEADER", "Enter");
            TMessage answ(kMESS_OBJECT);

            TDSet* dset;
            (*mess) >> dset;
            dset->Reset();
            TDSetElement *e = dset->Next();
            Long64_t entries = 0;
            TFile *f = 0;
            TTree *t = 0;
            if (!e) {
               PDB(kGlobal, 1) Info("HandleSocketInput:kPROOF_GETTREEHEADER",
                                    "empty TDSet");
            } else {
               f = TFile::Open(e->GetFileName());
               t = 0;
               if (f) {
                  t = (TTree*) f->Get(e->GetObjName());
                  if (t) {
                     t->SetMaxVirtualSize(0);
                     t->DropBaskets();
                     entries = t->GetEntries();

                     // compute #entries in all the files
                     while ((e = dset->Next()) != 0) {
                        TFile *f1 = TFile::Open(e->GetFileName());
                        if (f1) {
                           TTree* t1 = (TTree*) f1->Get(e->GetObjName());
                           if (t1) {
                              entries += t1->GetEntries();
                              delete t1;
                           }
                           delete f1;
                        }
                     }
                     t->SetMaxEntryLoop(entries);   // this field will hold the total number of entries ;)
                  }
               }
            }
            if (t)
               answ << TString("Success") << t;
            else
               answ << TString("Failed") << t;

            fSocket->Send(answ);

            SafeDelete(t);
            SafeDelete(f);

            PDB(kGlobal, 1) Info("HandleSocketInput:kPROOF_GETTREEHEADER", "Done");
         }
         break;
      case kPROOF_GETOUTPUTLIST:
         {
            PDB(kGlobal, 1) Info("HandleSocketInput:kPROOF_GETOUTPUTLIST", "Enter");
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
         {
            PDB(kGlobal, 1) Info("HandleSocketInput:kPROOF_VALIDATE_DSET", "Enter");

            TDSet* dset = 0;
            (*mess) >> dset;

            if (IsMaster()) fProof->ValidateDSet(dset);
            else dset->Validate();

            TMessage answ(kPROOF_VALIDATE_DSET);
            answ << dset;
            fSocket->Send(answ);
            delete dset;
            PDB(kGlobal, 1) Info("HandleSocketInput:kPROOF_VALIDATE_DSET", "Done");
            SendLogFile();
         }
         break;

      case kPROOF_DATA_READY:
         {
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
            SendLogFile();
         }
         break;

      case kPROOF_DATASETS:
         {
            HandleDataSets(mess);
            SendLogFile();
         }
         break;
      case kPROOF_LIB_INC_PATH:

         HandleLibIncPath(mess);

         // Notify the client
         SendLogFile();
         break;

      case kPROOF_REALTIMELOG:
         {
            Bool_t on;
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

      default:
         Error("HandleSocketInput", "unknown command %d", what);
         break;
   }

   recursive--;

   if (fProof) fProof->SetActive(kFALSE);

   fRealTime += (Float_t)timer.RealTime();
   fCpuTime  += (Float_t)timer.CpuTime();

   // Check if we have been asked to shutdown
   // (we will do nothing if not set)
   SetShutdownTimer(kTRUE, -1);

   delete mess;
}

//______________________________________________________________________________
void TProofServ::HandleSocketInputDuringProcess()
{
   // Handle messages that might arrive during processing while being in
   // HandleSocketInput(). This avoids recursive calls into HandleSocketInput().

   PDB(kGlobal,1) Info("HandleSocketInputDuringProcess", "enter");

   TMessage *mess;
   char      str[2048];
   Int_t     what;
   Bool_t    aborted = kFALSE;

   if (fSocket->Recv(mess) <= 0) {
      // Pending: do something more intelligent here
      // but at least get a message in the log file
      Error("HandleSocketInputDuringProcess", "retrieving message from input socket");
      Terminate(0);
      return;
   }

   what = mess->What();
   switch (what) {

      case kPROOF_PROCESS:

         {  TProofServLogHandlerGuard hg(fLogFile, fSocket, "", fRealTimeLog);

            HandleProcess(mess);

            // Notify
            SendLogFile();
         }
         break;

      case kPROOF_GETSTATS:
         // Send statistics
         SendStatistics();
         break;

      case kPROOF_LOGFILE:
         {
            Int_t start, end;
            (*mess) >> start >> end;
            PDB(kGlobal, 1)
               Info("HandleSocketInputDuringProcess:kPROOF_LOGFILE",
                    "Logfile request - byte range: %d - %d", start, end);

            SendLogFile(0, start, end);
         }
         break;

      case kPROOF_QUERYLIST:

         HandleQueryList(mess);

         // Notify
         SendLogFile();
         break;

      case kPROOF_ARCHIVE:

         HandleArchive(mess);

         // Notify
         SendLogFile();
         break;

      case kPROOF_REMOVE:

         HandleRemove(mess);

         // Notify
         SendLogFile();
         break;

      case kPROOF_RETRIEVE:

         HandleRetrieve(mess);

         // Notify
         SendLogFile();
         break;

      case kPROOF_STOPPROCESS:
         {  Long_t timeout = -1;
            (*mess) >> aborted;
            if (fProtocol > 9)
               (*mess) >> timeout;
            PDB(kGlobal, 1)
               Info("HandleSocketInputDuringProcess:kPROOF_STOPPROCESS",
                    "enter %d, %d", aborted, timeout);
            if (fProof)
               fProof->StopProcess(aborted, timeout);
            else
               if (fPlayer)
                  fPlayer->StopProcess(aborted, timeout);
         }
         break;

      case kPROOF_CACHE:
         {  TProofServLogHandlerGuard hg(fLogFile, fSocket, "", fRealTimeLog);
            Int_t status = HandleCache(mess);
            // Notify
            SendLogFile(status);
         }
         break;

      case kPROOF_CHECKFILE:

         // Handle file checking request
         HandleCheckFile(mess);
         break;

      case kPROOF_SENDFILE:
         mess->ReadString(str, sizeof(str));
         {
            Long_t size;
            Int_t  bin, fw = 1;
            char   name[1024];
            if (fProtocol > 5)
               sscanf(str, "%s %d %ld %d", name, &bin, &size, &fw);
            else
               sscanf(str, "%s %d %ld", name, &bin, &size);
            ReceiveFile(name, bin ? kTRUE : kFALSE, size);
            // copy file to cache
            if (size > 0) {
               fCacheLock->Lock();
               gSystem->Exec(Form("%s %s %s", kCP, name, fCacheDir.Data()));
               fCacheLock->Unlock();
            }
            if (IsMaster() && fw == 1)
               fProof->SendFile(name, bin);
         }
         break;

      case kPROOF_DATASETS:
         {
            HandleDataSets(mess);
            SendLogFile();
         }
         break;

      default:
         Error("HandleSocketInputDuringProcess", "unknown command %d", what);
         break;

   }
   delete mess;
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

   if (IsMaster())
      return fProof->IsParallel();

   //kFALSE is returned if we are a slave
   return kFALSE;
}

//______________________________________________________________________________
void TProofServ::Print(Option_t *option) const
{
   // Print status of slave server.

   if (IsMaster())
      fProof->Print(option);
   else
      Printf("This is slave %s", gSystem->HostName());
}

//______________________________________________________________________________
void TProofServ::RedirectOutput()
{
   // Redirect stdout to a log file. This log file will be flushed to the
   // client or master after each command.

   char logfile[512];

   if (IsMaster()) {
      sprintf(logfile, "%s/master.log", fSessionDir.Data());
   } else {
      sprintf(logfile, "%s/slave-%s.log", fSessionDir.Data(), fOrdinal.Data());
   }

   if ((freopen(logfile, "w", stdout)) == 0)
      SysError("RedirectOutput", "could not freopen stdout");

   if ((dup2(fileno(stdout), fileno(stderr))) < 0)
      SysError("RedirectOutput", "could not redirect stderr");

   if ((fLogFile = fopen(logfile, "r")) == 0)
      SysError("RedirectOutput", "could not open logfile");

   // from this point on stdout and stderr are properly redirected
   if (fProtocol < 4 && fWorkDir != kPROOF_WorkDir) {
      Warning("RedirectOutput", "no way to tell master (or client) where"
              " to upload packages");
   }
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

   off_t ltot=0, lnow=0;
   Int_t left = -1;
   Bool_t adhoc = kFALSE;

   if (fLogFileDes > -1) {
      ltot = lseek(fileno(stdout),   (off_t) 0, SEEK_END);
      lnow = lseek(fLogFileDes, (off_t) 0, SEEK_CUR);

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
void TProofServ::SendStatistics()
{
   // Send statistics of slave server to master or client.

   Long64_t bytesread = 0;
   if (IsMaster()) bytesread = fProof->GetBytesRead();
   else bytesread = TFile::GetFileBytesRead();

   TMessage mess(kPROOF_GETSTATS);
   TString workdir = gSystem->WorkingDirectory();  // expect TString on other side
   mess << bytesread << fRealTime << fCpuTime << workdir;
   if (fProtocol >= 4) mess << TString(gProofServ->GetWorkDir());
   fSocket->Send(mess);
}

//______________________________________________________________________________
void TProofServ::SendParallel()
{
   // Send number of parallel nodes to master or client.

   Int_t nparallel = 0;
   if (IsMaster()) {
      fProof->AskParallel();
      nparallel = fProof->GetParallel();
   } else {
      nparallel = 1;
   }

   fSocket->Send(nparallel, kPROOF_GETPARALLEL);
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
      sprintf(str, "**** Welcome to the PROOF server @ %s ****", gSystem->HostName());
   } else {
      sprintf(str, "**** PROOF slave server @ %s started ****", gSystem->HostName());
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
         fWorkDir = kPROOF_WorkDir;
      } else {
         if (fProtocol < 4) {
            fWorkDir = kPROOF_WorkDir;
         } else {
            fWorkDir = wconf;
            if (fWorkDir.IsNull()) fWorkDir = kPROOF_WorkDir;
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
         if (gSystem->Getenv("ROOTPROOFWORKDIR"))
            fWorkDir = gSystem->Getenv("ROOTPROOFWORKDIR");
         else
            fWorkDir = kPROOF_WorkDir;
      } else {
         (*mess) >> fUser >> fOrdinal >> fWorkDir;
         if (fWorkDir.IsNull())
            if (gSystem->Getenv("ROOTPROOFWORKDIR"))
               fWorkDir = gSystem->Getenv("ROOTPROOFWORKDIR");
            else
               fWorkDir = kPROOF_WorkDir;
      }
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

   // goto to the main PROOF working directory
   char *workdir = gSystem->ExpandPathName(fWorkDir.Data());
   fWorkDir = workdir;
   delete [] workdir;

   // deny write access for group and world
   gSystem->Umask(022);

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
   // Add bindir to PATH
   TString path(gSystem->Getenv("PATH"));
   if (!path.IsNull()) path.Insert(0, ":");
   path.Insert(0, bindir);
   gSystem->Setenv("PATH", path);
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

   fCacheLock =
      new TProofLockPath(Form("%s%s", kPROOF_CacheLockFile, fUser.Data()));

   // check and make sure "packages" directory exists
   fPackageDir = fWorkDir;
   fPackageDir += TString("/") + kPROOF_PackDir;
   if (gSystem->AccessPathName(fPackageDir))
      gSystem->MakeDirectory(fPackageDir);

   fPackageLock =
      new TProofLockPath(Form("%s%s", kPROOF_PackageLockFile, fUser.Data()));

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

   // On masters, check and make sure that "queries" and "datasets"
   // directories exist
   if (IsMaster()) {
      // 'queries'
      fQueryDir = fWorkDir;
      fQueryDir += TString("/") + kPROOF_QueryDir;
      if (gSystem->AccessPathName(fQueryDir))
         gSystem->MakeDirectory(fQueryDir);
      fQueryDir += TString("/session-") + fSessionTag;
      if (gSystem->AccessPathName(fQueryDir))
         gSystem->MakeDirectory(fQueryDir);

      // Create 'queries' locker instance and lock it
      fQueryLock = new TProofLockPath(Form("%s%s-%s",
                       kPROOF_QueryLockFile,fSessionTag.Data(),fUser.Data()));
      fQueryLock->Lock();

      // 'datasets'
      fDataSetDir = fWorkDir;
      fDataSetDir += TString("/") + kPROOF_DataSetDir;
      if (gSystem->AccessPathName(fDataSetDir))
         gSystem->MakeDirectory(fDataSetDir);

      fDataSetLock =
         new TProofLockPath(Form("%s%s", kPROOF_DataSetLockFile, fUser.Data()));

      // Send session tag, if a recent client
      if (fProtocol > 6) {
         TMessage m(kPROOF_SESSIONTAG);
         m << fSessionTag;
         fSocket->Send(m);
      }
   }

   // Incoming OOB should generate a SIGURG
   fSocket->SetOption(kProcessGroup, gSystem->GetPid());

   // Send messages off immediately to reduce latency
   fSocket->SetOption(kNoDelay, 1);

   // Check every two hours if client is still alive
   fSocket->SetOption(kKeepAlive, 1);

   // Install SigPipe handler to handle kKeepAlive failure
   gSystem->AddSignalHandler(new TProofServSigPipeHandler(this));

   TString all_vars(gSystem->Getenv("PROOF_ALLVARS"));
   TString name;
   Int_t from = 0;
   while (all_vars.Tokenize(name, from, ",")) {
      if (!name.IsNull()) {
         TString value = gSystem->Getenv(name);
         TProof::AddEnvVar(name, value);
      }
   }

   // Done
   return 0;
}

//______________________________________________________________________________
void TProofServ::Terminate(Int_t status)
{
   // Terminate the proof server.

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
   if (oldAuthSetupHook) {
      return (*oldAuthSetupHook)(fSocket, IsMaster(), fProtocol,
                                 fUser, fOrdinal, conf);
   } else {
      Error("OldAuthSetup",
            "hook to method OldProofServAuthSetup is undefined");
      return -1;
   }
}

//______________________________________________________________________________
TProofQueryResult *TProofServ::MakeQueryResult(Long64_t nent, const char *opt,
                                               TList *inlist, Long64_t fst,
                                               TDSet *dset, const char *selec,
                                               TEventList *evl)
{
   // Create a TProofQueryResult instance for this query.

   // Increment sequential number
   fSeqNum++;

   // Create the instance and add it to the list
   TProofQueryResult *pqr =
      new TProofQueryResult(fSeqNum, opt, inlist, nent, fst, dset, selec, evl);

   // Title is the session identifier
   pqr->SetTitle(gSystem->BaseName(fQueryDir));

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
         parlist += Form(";%s",os->GetName());
   }

   // Set in running state
   pq->SetRunning(startlog, parlist);

   // Bytes and CPU at start (we will calculate the differential at end)
   pq->SetProcessInfo(pq->GetEntries(),
                      fProof->GetCpuTime(), fProof->GetBytesRead());
}

//______________________________________________________________________________
void TProofServ::AddLogFile(TProofQueryResult *pq)
{
   // Add part of log file concerning TQueryResult pq to its macro
   // container.

   if (!pq)
      return;

   // Make sure everything is written to file
   fflush(stdout);

   // Save current position
   off_t lnow = lseek(fLogFileDes, (off_t) 0, SEEK_CUR);

   // The range we are interested in
   Int_t start = pq->fStartLog;
   if (start > -1)
      lseek(fLogFileDes, (off_t) start, SEEK_SET);

   // Read the lines and add then to the internal container
   const Int_t kMAXBUF = 4096;
   char line[kMAXBUF];
   while (fgets(line, sizeof(line), fLogFile)) {
      if (line[strlen(line)-1] == '\n')
         line[strlen(line)-1] = 0;
      pq->AddLogLine((const char *)line);
   }

   // Restore initial position if partial send
   lseek(fLogFileDes, lnow, SEEK_SET);
}

//______________________________________________________________________________
void TProofServ::FinalizeQuery(TProofPlayer *p, TProofQueryResult *pq)
{
   // Final steps after Process() to complete the TQueryResult instance.

   if (!pq || !p) {
      Warning("FinalizeQuery",
              "bad inputs: query = %p, player = %p ", pq ? pq : 0, p ? p : 0);
      return;
   }

   Int_t qn = pq->GetSeqNum();
   Long64_t np = p->GetEventsProcessed();
   TProofPlayer::EExitStatus est = p->GetExitStatus();
   TList *out = p->GetOutputList();

   fProof->AskStatistics();

   Float_t cpu = fProof->GetCpuTime();
   Long64_t bytes = fProof->GetBytesRead();

   TQueryResult::EQueryStatus st = TQueryResult::kAborted;

   PDB(kGlobal, 2) Info("FinalizeQuery","query #%d", qn);

   PDB(kGlobal, 1)
      Info("FinalizeQuery","%.1f %lld", cpu, bytes);

   // Some notification (useful in large logs)
   Bool_t save = kTRUE;
   switch (est) {
   case TProofPlayer::kAborted:
      PDB(kGlobal, 1)
         Info("FinalizeQuery", "query %d has been ABORTED <====", qn);
      out = 0;
      save = kFALSE;
      break;
   case TProofPlayer::kStopped:
      PDB(kGlobal, 1)
         Info("FinalizeQuery",
              "query %d has been STOPPED: %d events processed", qn, np);
      st = TQueryResult::kStopped;
      break;
   case TProofPlayer::kFinished:
      PDB(kGlobal, 1)
         Info("FinalizeQuery",
              "query %d has been completed: %d events processed", qn, np);
      st = TQueryResult::kCompleted;
      break;
   default:
      Warning("FinalizeQuery",
              "query %d: unknown exit status (%d)", qn, p->GetExitStatus());
   }

   // Fill some variables
   pq->SetProcessInfo(np, cpu - pq->GetUsedCPU());
   pq->RecordEnd(st, out);

   // Save the logs into the query result instance
   AddLogFile(pq);

   // Update/Save entry in specific query dir
   if (save) {

      // We may need some cleanup
      if (fgMaxQueries > -1) {
         if (fQueries && fKeptQueries >= fgMaxQueries) {
            // Find oldest completed and archived query
            TQueryResult *fcom = 0;
            TQueryResult *farc = 0;
            TIter nxq(fQueries);
            TQueryResult *qr = 0;
            while ((qr = (TQueryResult *) nxq())) {
               if (qr->IsArchived()) {
                  if (qr->GetOutputList() && !farc)
                     farc = qr;
               } else if (qr->GetStatus() > TQueryResult::kRunning && !fcom) {
                  fcom = qr;
               }
               if (farc && fcom)
                  break;
            }
            if (farc) {
               RemoveQuery(farc, kTRUE);
               fKeptQueries--;
            } else if (fcom) {
               RemoveQuery(fcom);
               fKeptQueries--;
            }
         }
         if (fKeptQueries < fgMaxQueries) {
            SaveQuery(pq);
            fKeptQueries++;
         }
      } else {
         SaveQuery(pq);
         fKeptQueries++;
      }
   }

   // Done!
}

//______________________________________________________________________________
Int_t TProofServ::CleanupQueriesDir()
{
   // Remove all queries results referring to previous sessions

   Int_t nd = 0;

   // Cleanup previous stuff
   if (fPreviousQueries) {
      fPreviousQueries->Delete();
      SafeDelete(fPreviousQueries);
   }

   // Loop over session dirs
   TString queriesdir = fQueryDir;
   queriesdir = queriesdir.Remove(queriesdir.Index(kPROOF_QueryDir) +
                                  strlen(kPROOF_QueryDir));
   void *dirs = gSystem->OpenDirectory(queriesdir);
   char *sess = 0;
   while ((sess = (char *) gSystem->GetDirEntry(dirs))) {

      // We are interested only in "session-..." subdirs
      if (strlen(sess) < 7 || strncmp(sess,"session",7))
         continue;

      // We do not want this session at this level
      if (strstr(sess, fSessionTag))
         continue;

      // Remove the directory
      TString qdir = Form("%s/%s", queriesdir.Data(), sess);
      Info("RemoveQuery", "removing directory: %s", qdir.Data());
      gSystem->Exec(Form("%s %s", kRM, qdir.Data()));
      nd++;
   }

   // Done
   return nd;
}

//______________________________________________________________________________
void TProofServ::ScanPreviousQueries(const char *dir)
{
   // Scan the queries directory for the results of previous queries.
   // The headers of the query results found are loaded in fPreviousQueries.
   // The full query result can be retrieved via TProof::Retrieve.

   // Cleanup previous stuff
   if (fPreviousQueries) {
      fPreviousQueries->Delete();
      SafeDelete(fPreviousQueries);
   }

   // Loop over session dirs
   void *dirs = gSystem->OpenDirectory(dir);
   char *sess = 0;
   while ((sess = (char *) gSystem->GetDirEntry(dirs))) {

      // We are interested only in "session-..." subdirs
      if (strlen(sess) < 7 || strncmp(sess,"session",7))
         continue;

      // We do not want this session at this level
      if (strstr(sess, fSessionTag))
         continue;

      // Loop over query dirs
      void *dirq = gSystem->OpenDirectory(Form("%s/%s", dir, sess));
      char *qry = 0;
      while ((qry = (char *) gSystem->GetDirEntry(dirq))) {

         // We are interested only in "n/" subdirs
         if (qry[0] == '.')
            continue;

         // File with the query result
         TString fn = Form("%s/%s/%s/query-result.root", dir, sess, qry);
         TFile *f = TFile::Open(fn);
         if (f) {
            f->ReadKeys();
            TIter nxk(f->GetListOfKeys());
            TKey *k =  0;
            TProofQueryResult *pqr = 0;
            while ((k = (TKey *)nxk())) {
               if (!strcmp(k->GetClassName(), "TProofQueryResult")) {
                  pqr = (TProofQueryResult *) f->Get(k->GetName());
                  if (pqr) {
                     TQueryResult *qr = pqr->CloneInfo();
                     if (!fPreviousQueries)
                        fPreviousQueries = new TList;
                     if (qr->GetStatus() > TQueryResult::kRunning) {
                        fPreviousQueries->Add(qr);
                     } else {
                        // (For the time being) remove a non completed
                        // query if not owned by anybody
                        TProofLockPath *lck = 0;
                        if (LockSession(qr->GetTitle(), &lck) == 0) {
                           RemoveQuery(qr);
                           // Unlock and remove the lock file
                           SafeDelete(lck);
                        }
                     }
                  }
               }
            }
            f->Close();
            delete f;
         }
      }
      gSystem->FreeDirectory(dirq);
   }
   gSystem->FreeDirectory(dirs);
}

//______________________________________________________________________________
Int_t TProofServ::LockSession(const char *sessiontag, TProofLockPath **lck)
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
   TRegexp re("session-.*-.*-.*-.*");
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
Int_t TProofServ::CleanupSession(const char *sessiontag)
{
   // Cleanup query dir qdir.

   if (!sessiontag) {
      Info("CleanupSession","session tag undefined");
      return -1;
   }

   // Query dir
   TString qdir = fQueryDir;
   qdir.ReplaceAll(Form("session-%s", fSessionTag.Data()), sessiontag);
   Int_t idx = qdir.Index(":q");
   if (idx != kNPOS)
      qdir.Remove(idx);
   if (gSystem->AccessPathName(qdir)) {
      Info("CleanupSession","query dir %s does not exist", qdir.Data());
      return -1;
   }

   TProofLockPath *lck = 0;
   if (LockSession(sessiontag, &lck) == 0) {

      // Cleanup now
      gSystem->Exec(Form("%s %s", kRM, qdir.Data()));

      // Unlock and remove the lock file
      if (lck) {
         gSystem->Unlink(lck->GetName());
         SafeDelete(lck);  // Unlocks, if necessary
      }

      // We are done
      return 0;
   }

   // Notify failure
   Info("CleanupSession", "could not lock session %s", sessiontag);
   return -1;
}

//______________________________________________________________________________
void TProofServ::SaveQuery(TQueryResult *qr, const char *fout)
{
   // Save current status of query 'qr' to file name fout.
   // If fout == 0 (default) use the default name.

   if (!qr || qr->IsDraw())
      return;

   // Create dir for specific query
   TString querydir = Form("%s/%d",fQueryDir.Data(), qr->GetSeqNum());

   // Create dir, if needed
   if (gSystem->AccessPathName(querydir))
      gSystem->MakeDirectory(querydir);
   TString ofn = fout ? fout : Form("%s/query-result.root", querydir.Data());

   // Recreate file and save query in its current status
   TFile *f = TFile::Open(ofn, "RECREATE");
   if (f) {
      f->cd();
      if (!(qr->IsArchived()))
         qr->fResultFile = ofn;
      qr->Write();
      f->Close();
      delete f;
   }
}

//______________________________________________________________________________
void TProofServ::RemoveQuery(const char *queryref)
{
   // Remove everything about query queryref.

   PDB(kGlobal, 1)
      Info("RemoveQuery", "Enter");

   // Parse reference string
   Int_t qry = -1;
   TString qdir;
   TProofQueryResult *pqr = LocateQuery(queryref, qry, qdir);
   // Remove instance in memory
   if (pqr) {
      if (qry > -1) {
         fQueries->Remove(pqr);
         fWaitingQueries->Remove(pqr);
      } else
         fPreviousQueries->Remove(pqr);
      delete pqr;
      pqr = 0;
   }

   // Remove the directory
   Info("RemoveQuery", "removing directory: %s", qdir.Data());
   gSystem->Exec(Form("%s %s", kRM, qdir.Data()));

   // Done
   return;
}

//______________________________________________________________________________
void TProofServ::RemoveQuery(TQueryResult *qr, Bool_t soft)
{
   // Remove everything about query qr. If soft = TRUE leave a track
   // in memory with the relevant info

   PDB(kGlobal, 1)
      Info("RemoveQuery", "Enter");

   if (!qr)
      return;

   // Remove the directory
   TString qdir = fQueryDir;
   qdir = qdir.Remove(qdir.Index(kPROOF_QueryDir)+strlen(kPROOF_QueryDir));
   qdir = Form("%s/%s/%d", qdir.Data(), qr->GetTitle(), qr->GetSeqNum());
   PDB(kGlobal, 1)
      Info("RemoveQuery", "removing directory: %s", qdir.Data());
   gSystem->Exec(Form("%s %s", kRM, qdir.Data()));

   // Remove from memory lists
   if (soft) {
      TQueryResult *qrn = qr->CloneInfo();
      Int_t idx = fQueries->IndexOf(qr);
      if (idx > -1)
         fQueries->AddAt(qrn, idx);
      else
         SafeDelete(qrn);
   }
   fQueries->Remove(qr);
   SafeDelete(qr);

   // Done
   return;
}

//______________________________________________________________________________
TProofQueryResult *TProofServ::LocateQuery(TString queryref,
                                           Int_t &qry, TString &qdir)
{
   // Locate query referenced by queryref. Return pointer to instance
   // in memory, if any, or 0. Fills qdir with the query specific directory
   // and qry with the query number for queries processed by this session.

   TProofQueryResult *pqr = 0;

   // Find out if the request is a for a local query or for a
   // previously processed one
   qry = -1;
   if (queryref.IsDigit()) {
      qry = queryref.Atoi();
   } else if (queryref.Contains(fSessionTag)) {
      Int_t i1 = queryref.Index(":q");
      if (i1 != kNPOS) {
         queryref.Remove(0,i1+2);
         qry = queryref.Atoi();
      }
   }

   // Build dir name for specific query
   qdir = "";
   if (qry > -1) {

      PDB(kGlobal, 1)
         Info("LocateQuery", "local query: %d", qry);

      // Remove query from memory list
      if (fQueries) {
         TIter nxq(fQueries);
         while ((pqr = (TProofQueryResult *) nxq())) {
            if (pqr->GetSeqNum() == qry) {
               // Dir for specific query
               qdir = Form("%s/%d", fQueryDir.Data(), qry);
               break;
            }
         }
      }

   } else {
      PDB(kGlobal, 1)
         Info("LocateQuery", "previously processed query: %s", queryref.Data());

      // Remove query from memory list
      if (fPreviousQueries) {
         TIter nxq(fPreviousQueries);
         while ((pqr = (TProofQueryResult *) nxq())) {
            if (queryref.Contains(pqr->GetTitle()) &&
                queryref.Contains(pqr->GetName()))
               break;
         }
      }

      queryref.ReplaceAll(":q","/");
      qdir = fQueryDir;
      qdir = qdir.Remove(qdir.Index(kPROOF_QueryDir)+strlen(kPROOF_QueryDir));
      qdir = Form("%s/%s", qdir.Data(), queryref.Data());
   }

   // We are done
   return pqr;
}

//______________________________________________________________________________
void TProofServ::HandleArchive(TMessage *mess)
{
   // Handle archive request.

   PDB(kGlobal, 1)
      Info("HandleArchive", "Enter");

   TString queryref;
   TString path;
   (*mess) >> queryref >> path;

   // If this is a set default action just save the default
   if (queryref == "Default") {
      fArchivePath = path;
      Info("HandleArchive",
           "default path set to %s", fArchivePath.Data());
      return;
   }

   Int_t qry = -1;
   TString qdir;
   TProofQueryResult *pqr = LocateQuery(queryref, qry, qdir);
   TProofQueryResult *pqm = pqr;

   if (path.Length() <= 0) {
      if (fArchivePath.Length() <= 0) {
         Info("HandleArchive",
              "archive paths are not defined - do nothing");
         return;
      }
      if (qry > 0) {
         path = Form("%s/session-%s-%d.root",
                     fArchivePath.Data(), fSessionTag.Data(), qry);
      } else {
         path = queryref;
         path.ReplaceAll(":q","-");
         path.Insert(0, Form("%s/",fArchivePath.Data()));
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
      if (qry > -1)
         SaveQuery(pqr);

      // Notify
      Info("HandleArchive",
           "results of query %s archived to file %s",
           queryref.Data(), path.Data());
   }

   // Done
   return;
}

//______________________________________________________________________________
void TProofServ::HandleProcess(TMessage *mess)
{
   // Handle processing request.

   PDB(kGlobal, 1)
      Info("HandleProcess", "Enter");

   // Nothing to do for slaves if we are not idle
   if (!IsTopMaster() && !fIdle)
      return;

   TDSet *dset;
   TString filename, opt;
   TList *input;
   Long64_t nentries, first;
   TEventList *evl;
   Bool_t sync;

   (*mess) >> dset >> filename >> input >> opt >> nentries >> first >> evl >> sync;

   if (evl)
      dset->SetEventList(evl);

   TProofPlayer *p = 0;

   if (IsTopMaster()) {

      TProofQueryResult *pq = 0;

      // Create instance of query results
      pq = MakeQueryResult(nentries, opt, input, first, dset, filename, evl);

      // Make sure that cleanup is done when required
      input->SetOwner();
      SafeDelete(input);

      // If not a draw action add the query to the main list
      if (!(pq->IsDraw())) {
         fQueries->Add(pq);
         // Also save it to queries dir
         SaveQuery(pq);
      }

      // Add anyhow to the waiting lists
      fWaitingQueries->Add(pq);

      // If the client submission was asynchronous, signal the submission of
      // the query and communicate the assigned sequential number for later
      // identification
      if (!sync) {
         TMessage m(kPROOF_QUERYSUBMITTED);
         m << pq->GetSeqNum();
         fSocket->Send(m);
      }

      // Nothing more to do if we are not idle
      if (!fIdle) {
         // Notify submission
         Info("HandleProcess",
              "query \"%s:%s\" submitted", pq->GetTitle(), pq->GetName());
         return;
      }

      // Process
      while (fWaitingQueries->GetSize() > 0) {
         //
         // Set not idle
         fIdle = kFALSE;
         Reset("");
         //
         // Get query info
         pq = (TProofQueryResult *)(fWaitingQueries->First());
         if (pq) {
            opt      = pq->GetOptions();
            input    = pq->GetInputList();
            nentries = pq->GetEntries();
            first    = pq->GetFirst();
            // Attach to data set and event list (if any)
            TObject *o = 0;
            if ((o = pq->GetInputObject("TDSet")))
               dset = (TDSet *) o;
            if ((o = pq->GetInputObject("TEventList")))
               evl = (TEventList *) o;
            //
            // Expand selector files
            if (pq->GetSelecImp()) {
               gSystem->Exec(Form("%s %s", kRM, pq->GetSelecImp()->GetName()));
               pq->GetSelecImp()->SaveSource(pq->GetSelecImp()->GetName());
            }
            if (pq->GetSelecHdr() &&
                !strstr(pq->GetSelecHdr()->GetName(), "TProofDrawHist")) {
               gSystem->Exec(Form("%s %s", kRM, pq->GetSelecHdr()->GetName()));
               pq->GetSelecHdr()->SaveSource(pq->GetSelecHdr()->GetName());
            }
            //
            // Remove processed query from the list
            fWaitingQueries->Remove(pq);
         } else {
            // Should never get here
            Error("HandleProcess", "empty query in queue!");
            continue;
         }

         // Set in running state
         SetQueryRunning(pq);

         // Save to queries dir, if not standard draw
         if (!(pq->IsDraw()))
            SaveQuery(pq);
         else
            fDrawQueries++;

         // Signal the client that we are starting a new query
         TMessage m(kPROOF_STARTPROCESS);
         m << TString(pq->GetSelecImp()->GetName())
           << dset->GetListOfElements()->GetSize()
           << pq->GetFirst() << pq->GetEntries();
         fSocket->Send(m);

         // Create player
         if (IsParallel()) {
            p = fProof->MakePlayer();
         } else {
            // sequential mode
            p = new TProofPlayerSlave(fSocket);
            fProof->SetPlayer(p);
         }

         // Add query results to the player lists
         p->AddQueryResult(pq);

         // Set query currently processed
         p->SetCurrentQuery(pq);

         // Set player
         fPlayer = p;

         // Setup data set
         if (dset->IsA() == TDSetProxy::Class())
            ((TDSetProxy*)dset)->SetProofServ(this);

         // Set input
         TIter next(input);
         for (TObject *o; (o = next()); ) {
            PDB(kGlobal, 2) Info("HandleProcess", "adding: %s", o->GetName());
            p->AddInput(o);
         }

         // Add the unique query tag as TNamed object to the input list
         // so that it is available in TSelectors for monitoring
         p->AddInput(new TNamed("PROOF_QueryTag",Form("%s:%s",pq->GetTitle(),pq->GetName())));

         // Process
         PDB(kGlobal, 1) Info("HandleProcess", "calling TProofPlayerRemote::Process()");
         p->Process(dset, filename, opt, nentries, first);

         // Return number of events processed
         if (p->GetExitStatus() != TProofPlayer::kFinished) {
            Bool_t abort =
              (p->GetExitStatus() == TProofPlayer::kAborted) ? kTRUE : kFALSE;
            TMessage m(kPROOF_STOPPROCESS);
            if (fProtocol > 8) {
               m << p->GetEventsProcessed() << abort;
            } else {
               m << p->GetEventsProcessed();
            }
            fSocket->Send(m);
         }

         // Complete filling of the TQueryResult instance
         FinalizeQuery(p, pq);

         // Send back the results
         TQueryResult *pqr = pq->CloneInfo();
         if (p->GetExitStatus() != TProofPlayer::kAborted && p->GetOutputList()) {

            PDB(kGlobal, 2) Info("HandleProcess","Sending results");
            if (fProtocol > 10) {
               // Send objects one-by-one to optimize transfer and merging
               TMessage m(kPROOF_MESSAGE);
               TMessage mbuf(kPROOF_OUTPUTOBJECT);
               // Objects in the output list
               Int_t olsz = p->GetOutputList()->GetSize();
               // Message for the client
               m << TString(Form("master-%s: sending output: %d objs",
                                   fOrdinal.Data(), olsz));
               m << (Bool_t) kFALSE;
               fSocket->Send(m);
               // Send light query info
               mbuf << (Int_t) 0;
               mbuf.WriteObject(pqr);
               fSocket->Send(mbuf);

               Int_t ns = 0;
               Int_t totsz = 0;
               TIter nxo(p->GetOutputList());
               TObject *o = 0;
               while ((o = nxo())) {
                  ns++;
                  mbuf.Reset();
                  Int_t type = (Int_t) ((ns >= olsz) ? 2 : 1);
                  mbuf << type;
                  mbuf.WriteObject(o);
                  totsz += mbuf.Length();
                  m.Reset();
                  m << TString(Form("master-%s: sending obj %d/%d (%d bytes)",
                                    fOrdinal.Data(), ns, olsz, mbuf.Length()));
                  m << (Bool_t) kFALSE;
                  fSocket->Send(m);
                  fSocket->Send(mbuf);
               }
               // Total size
               m.Reset();
               m << TString(Form("master-%s: grand total: sent %d objects, size: %d bytes",
                                      fOrdinal.Data(), olsz, totsz));
               m << (Bool_t) kTRUE;
               fSocket->Send(m);
            } else if (fProtocol > 6) {

               // Buffer to be sent
               TMessage mbuf(kPROOF_OUTPUTLIST);
               mbuf.WriteObject(pq);
               // Sizes
               Int_t blen = mbuf.Length();
               Int_t olsz = p->GetOutputList()->GetSize();
               // Message for the client
               TString cmsg = Form("master-%s: sending output: %d objs, %d bytes",
                                   fOrdinal.Data(), olsz, blen);
               TMessage m(kPROOF_MESSAGE);
               m << cmsg;
               fSocket->Send(m);
               fSocket->Send(mbuf);

            } else {
               // TQueryResult unknow to client: send the output list only
               PDB(kGlobal, 2) Info("HandleProcess","Sending output list");
               fSocket->SendObject(p->GetOutputList(), kPROOF_OUTPUTLIST);
            }
         } else {
            if (p->GetExitStatus() != TProofPlayer::kAborted)
               Warning("HandleProcess","The output list is empty!");
            fSocket->SendObject(0, kPROOF_OUTPUTLIST);
         }

         // Remove aborted queries from the list
         if (p->GetExitStatus() == TProofPlayer::kAborted) {
            RemoveQuery(pq);
         } else {
            // Keep in memory only light infor about a query
            if (!(pq->IsDraw())) {
               if (pqr)
                  fQueries->Add(pqr);
               // Remove from the fQueries list
               fQueries->Remove(pq);
               SafeDelete(pq);
            }
         }

         // Player cleanup
         fProof->SetPlayer(0);
         SafeDelete(p);

      } // Loop on submitted queries

      // Signal the client that we are idle
      fSocket->Send(kPROOF_SETIDLE);

   } else {

      // Set not idle
      fIdle = kFALSE;

      // Create player
      if (IsMaster() && IsParallel()) {
         // NOTE: fProof->SetPlayer(0) should be called after Process()
         p = fProof->MakePlayer();
      } else {
         // slave or sequential mode
         p = new TProofPlayerSlave(fSocket);
         if (IsMaster())
            fProof->SetPlayer(p);
      }

      // Set player
      fPlayer = p;

      // Setup data set
      if (dset->IsA() == TDSetProxy::Class())
         ((TDSetProxy*)dset)->SetProofServ(this);

      // Set input
      TIter next(input);
      for (TObject *o; (o = next()); ) {
         PDB(kGlobal, 2) Info("HandleProcess", "adding: %s", o->GetName());
         p->AddInput(o);
      }

      // Process
      PDB(kGlobal, 1) Info("HandleProcess", "calling TProofPlayer::Process()");
      p->Process(dset, filename, opt, nentries, first);

      // Return number of events processed
      TMessage m(kPROOF_STOPPROCESS);
      m << p->GetEventsProcessed();
      fSocket->Send(m);

      // Send back the results
      if (p->GetExitStatus() != TProofPlayer::kAborted && p->GetOutputList()) {
         if (fProtocol > 10) {
            // Send objects one-by-one to optimize transfer and merging
            // Messages for objects
            TMessage mbuf(kPROOF_OUTPUTOBJECT);
            // Objects in the output list
            Int_t ns = 0;
            Int_t olsz = p->GetOutputList()->GetSize();
            TIter nxo(p->GetOutputList());
            TObject *o = 0;
            while ((o = nxo())) {
               ns++;
               mbuf.Reset();
               mbuf << (Int_t) ((ns >= olsz) ? 2 : 1);
               mbuf.WriteObject(o);
               fSocket->Send(mbuf);
            }
         } else {
            PDB(kGlobal, 2) Info("HandleProcess","Sending output list");
            fSocket->SendObject(p->GetOutputList(), kPROOF_OUTPUTLIST);
         }
      } else {
         fSocket->SendObject(0,kPROOF_OUTPUTLIST);
      }

      // Cleanup
      SafeDelete(dset);
      p->GetInputList()->SetOwner();  // Make sure the input list objects are deleted

      // Player cleanup
      if (IsMaster())
         fProof->SetPlayer(0);
      SafeDelete(p);
   }

   fPlayer = 0;

   // Set idle
   fIdle = kTRUE;

   PDB(kGlobal, 1) Info("HandleProcess", "Done");

   // Done
   return;
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
   Int_t ntot = 0;
   if (all) {
      // Rescan
      TString qdir = fQueryDir;
      Int_t idx = qdir.Index("session-");
      if (idx != kNPOS)
         qdir.Remove(idx);
      ScanPreviousQueries(qdir);
      // Send also information about previous queries, if any
      if (fPreviousQueries) {
         TIter nxq(fPreviousQueries);
         TProofQueryResult *pqr = 0;
         while ((pqr = (TProofQueryResult *)nxq())) {
            ntot++;
            pqr->fSeqNum = ntot;
            ql->Add(pqr);
         }
      }
   }

   Int_t npre = ntot;
   if (fQueries) {
      // Add info about queries in this session
      TIter nxq(fQueries);
      TProofQueryResult *pqr = 0;
      TQueryResult *pqm = 0;
      while ((pqr = (TProofQueryResult *)nxq())) {
         ntot++;
         pqm = pqr->CloneInfo();
         pqm->fSeqNum = ntot;
         ql->Add(pqm);
      }
   }

   TMessage m(kPROOF_QUERYLIST);
   m << npre << fDrawQueries << ql;
   fSocket->Send(m);
   delete ql;

   // Done
   return;
}

//______________________________________________________________________________
void TProofServ::HandleRemove(TMessage *mess)
{
   // Handle remove request.

   PDB(kGlobal, 1)
      Info("HandleRemove", "Enter");

   TString queryref;
   (*mess) >> queryref;

   if (queryref == "cleanupqueue") {
      Int_t pend = fWaitingQueries->GetSize();
      // Remove pending requests
      fWaitingQueries->Delete();
      // Notify
      Info("HandleRemove", "%d queries removed from the waiting list", pend);
      // We are done
      return;
   }

   if (queryref == "cleanupdir") {

      // Cleanup previous sessions results
      Int_t nd = CleanupQueriesDir();

      // Notify
      Info("HandleRemove", "%d directories removed", nd);
      // We are done
      return;
   }



   TProofLockPath *lck = 0;
   if (LockSession(queryref, &lck) == 0) {

      // Remove query
      RemoveQuery(queryref);

      // Unlock and remove the lock file
      if (lck) {
         gSystem->Unlink(lck->GetName());
         SafeDelete(lck);
      }

      // We are done
      return;
   }

   // Notify failure
   Info("HandleRemove",
        "query %s could not be removed (unable to lock session)", queryref.Data());

   // Done
   return;
}

//______________________________________________________________________________
void TProofServ::HandleRetrieve(TMessage *mess)
{
   // Handle retrieve request.

   PDB(kGlobal, 1)
      Info("HandleRetrieve", "Enter");

   TString queryref;
   (*mess) >> queryref;

   // Parse reference string
   Int_t qry = -1;
   TString qdir;
   TProofQueryResult *pqr = LocateQuery(queryref, qry, qdir);

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
            if (pqr) {

               TMessage m(kPROOF_MESSAGE);
               // Message for the client
               Float_t qsz = (Float_t) f->GetSize();
               Int_t ilb = 0;
               static const char *clb[4] = { "bytes", "KB", "MB", "GB" };
               while (qsz > 1000. && ilb < 3) {
                  qsz /= 1000.;
                  ilb++;
               }
               m << TString(Form("master-%s: sending result of %s:%s (%'.1f %s)",
                                  fOrdinal.Data(), pqr->GetTitle(), pqr->GetName(),
                                  qsz, clb[ilb]));
               m << (Bool_t) kTRUE;
               fSocket->Send(m);
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
                  newlibpath.Insert(pos,Form("%s:", xlib.Data()));
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
                  gSystem->AddIncludePath(Form("-I%s", xinc.Data()));
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
            newlibpath.ReplaceAll(Form("%s:", xlib.Data()),"");
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
            newincpath.ReplaceAll(Form("-I%s", inc->GetName()),"");
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
void TProofServ::HandleCheckFile(TMessage *mess)
{
   // Handle file checking request.

   TString filenam;
   TMD5    md5;
   UInt_t  opt = TProof::kUntar;

   // Parse message
   (*mess) >> filenam >> md5;
   if ((mess->BufferSize() > mess->Length()) && (fProtocol > 8))
      (*mess) >> opt;

   if (filenam.BeginsWith("-")) {
      // install package:
      // compare md5's, untar, store md5 in PROOF-INF, remove par file
      Int_t  st  = 0;
      Bool_t err = kFALSE;
      filenam = filenam.Strip(TString::kLeading, '-');
      TString packnam = filenam;
      packnam.Remove(packnam.Length() - 4);  // strip off ".par"
      // compare md5's to check if transmission was ok
      TMD5 *md5local = TMD5::FileChecksum(fPackageDir + "/" + filenam);
      if (md5local && md5 == (*md5local)) {
         if ((opt & TProof::kRemoveOld)) {
            // remove any previous package directory with same name
            st = gSystem->Exec(Form("%s %s/%s", kRM, fPackageDir.Data(),
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
            st = gSystem->Exec(Form(kUNTAR, gunzip, fPackageDir.Data(),
                               filenam.Data(), fPackageDir.Data()));
            if (st)
               Error("HandleCheckFile", "failure executing: %s",
                     Form(kUNTAR, gunzip, fPackageDir.Data(),
                          filenam.Data(), fPackageDir.Data()));
            delete [] gunzip;
         } else
            Error("HandleCheckFile", "%s not found", kGUNZIP);
         // check that fPackageDir/packnam now exists
         if (gSystem->AccessPathName(fPackageDir + "/" + packnam, kWritePermission)) {
            // par file did not unpack itself in the expected directory, failure
            fSocket->Send(kPROOF_FATAL);
            err = kTRUE;
            PDB(kPackage, 1)
               Info("HandleCheckFile",
                    "package %s did not unpack into %s", filenam.Data(),
                    packnam.Data());
         } else {
            // store md5 in package/PROOF-INF/md5.txt
            TString md5f = fPackageDir + "/" + packnam + "/PROOF-INF/md5.txt";
            TMD5::WriteChecksum(md5f, md5local);
            fSocket->Send(kPROOF_CHECKFILE);
            PDB(kPackage, 1)
               Info("HandleCheckFile",
                    "package %s installed on node", filenam.Data());
         }
      } else {
         fSocket->Send(kPROOF_FATAL);
         err = kTRUE;
      }

      // Note: Originally an fPackageLock->Unlock() call was made
      // after the if-else statement below. With multilevel masters,
      // submasters still check to make sure the package exists with
      // the correct md5 checksum and need to do a read lock there.
      // As yet locking is not that sophisicated so the lock must
      // be released below before the call to fProof->UploadPackage().
      if (!IsMaster() || err) {
         // delete par file when on slave or in case of error
         gSystem->Exec(Form("%s %s/%s", kRM, fPackageDir.Data(),
                       filenam.Data()));
         fPackageLock->Unlock();
      } else {
         // forward to slaves
         fPackageLock->Unlock();
         fProof->UploadPackage(fPackageDir + "/" + filenam, (TProof::EUploadPackageOpt)opt);
      }
      delete md5local;
   } else if (filenam.BeginsWith("+")) {
      // check file in package directory
      filenam = filenam.Strip(TString::kLeading, '+');
      TString packnam = filenam;
      packnam.Remove(packnam.Length() - 4);  // strip off ".par"
      TString md5f = fPackageDir + "/" + packnam + "/PROOF-INF/md5.txt";
      fPackageLock->Lock();
      TMD5 *md5local = TMD5::ReadChecksum(md5f);
      if (md5local && md5 == (*md5local)) {
         // package already on server, unlock directory
         fPackageLock->Unlock();
         fSocket->Send(kPROOF_CHECKFILE);
         PDB(kPackage, 1)
            Info("HandleCheckFile",
                 "package %s already on node", filenam.Data());
         if (IsMaster())
            fProof->UploadPackage(fPackageDir + "/" + filenam);
      } else {
         fSocket->Send(kPROOF_FATAL);
         PDB(kPackage, 1)
            Info("HandleCheckFile",
                 "package %s not yet on node", filenam.Data());
      }
      delete md5local;
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
         fSocket->Send(kPROOF_CHECKFILE);
         PDB(kPackage, 1)
            Info("HandleCheckFile",
                 "package %s already on node", filenam.Data());
         if (IsMaster())
            fProof->UploadPackage(fPackageDir + "/" + filenam);
      } else {
         fSocket->Send(kPROOF_FATAL);
         PDB(kPackage, 1)
            Info("HandleCheckFile",
                 "package %s not yet on node", filenam.Data());
      }
      delete md5local;
   } else {
      // check file in cache directory
      TString cachef = fCacheDir + "/" + filenam;
      fCacheLock->Lock();
      TMD5 *md5local = TMD5::FileChecksum(cachef);
      if (md5local && md5 == (*md5local)) {
         // copy file from cache to working directory
         gSystem->Exec(Form("%s %s .", kCP, cachef.Data()));
         fSocket->Send(kPROOF_CHECKFILE);
         PDB(kPackage, 1)
            Info("HandleCheckFile", "file %s already on node", filenam.Data());
      } else {
         fSocket->Send(kPROOF_FATAL);
         PDB(kPackage, 1)
            Info("HandleCheckFile", "file %s not yet on node", filenam.Data());
      }
      delete md5local;
      fCacheLock->Unlock();
   }
}

//______________________________________________________________________________
Int_t TProofServ::HandleCache(TMessage *mess)
{
   // Handle here all cache and package requests.

   PDB(kGlobal, 1)
      Info("HandleCache", "Enter");

   Int_t status = 0;
   Int_t type = 0;
   Bool_t all = kFALSE;
   TMessage msg;

   // Notification message
   TMessage notm(kPROOF_MESSAGE);
   TString noth = Form("worker-%s", fOrdinal.Data());
   if (IsMaster())
      noth.ReplaceAll("worker", "master");
   Bool_t notln = kTRUE;

   TString package, pdir, ocwd;
   (*mess) >> type;
   switch (type) {
      case TProof::kShowCache:
         (*mess) >> all;
         printf("*** File cache %s:%s ***\n", gSystem->HostName(),
                fCacheDir.Data());
         fflush(stdout);
         gSystem->Exec(Form("%s %s", kLS, fCacheDir.Data()));
         if (IsMaster() && all)
            fProof->ShowCache(all);
         break;
      case TProof::kClearCache:
         fCacheLock->Lock();
         gSystem->Exec(Form("%s %s/*", kRM, fCacheDir.Data()));
         fCacheLock->Unlock();
         if (IsMaster())
            fProof->ClearCache();
         break;
      case TProof::kShowPackages:
         (*mess) >> all;
         printf("*** Package cache %s:%s ***\n", gSystem->HostName(),
                fPackageDir.Data());
         fflush(stdout);
         gSystem->Exec(Form("%s %s", kLS, fPackageDir.Data()));
         if (IsMaster() && all)
            fProof->ShowPackages(all);
         break;
      case TProof::kClearPackages:
         status = UnloadPackages();
         if (status == 0) {
            fPackageLock->Lock();
            gSystem->Exec(Form("%s %s/*", kRM, fPackageDir.Data()));
            fPackageLock->Unlock();
            if (IsMaster())
               status = fProof->ClearPackages();
         }
         break;
      case TProof::kClearPackage:
         (*mess) >> package;
         status = UnloadPackage(package);
         if (status == 0) {
            fPackageLock->Lock();
            // remove package directory and par file
            gSystem->Exec(Form("%s %s/%s", kRM, fPackageDir.Data(),
                          package.Data()));
            if (IsMaster())
               gSystem->Exec(Form("%s %s/%s.par", kRM, fPackageDir.Data(),
                             package.Data()));
            fPackageLock->Unlock();
            if (IsMaster())
               status = fProof->ClearPackage(package);
         }
         break;
      case TProof::kBuildPackage:
         (*mess) >> package;
         if (IsMaster()) {
            // make sure package is available on all slaves, even new ones
            fProof->UploadPackage(fPackageDir + "/" + package + ".par");
         }
         fPackageLock->Lock();
         // check that package and PROOF-INF directory exists
         pdir = fPackageDir + "/" + package;
         if (gSystem->AccessPathName(pdir)) {
            Error("HandleCache", "package %s does not exist",
                  package.Data());
            status = -1;
         } else if (gSystem->AccessPathName(pdir + "/PROOF-INF")) {
            Error("HandleCache", "package %s does not have a PROOF-INF directory",
                  package.Data());
            status = -1;
         }

         if (!status) {

            PDB(kPackage, 1)
               Info("HandleCache",
                    "package %s exists and has PROOF-INF directory", package.Data());

            ocwd = gSystem->WorkingDirectory();
            gSystem->ChangeDirectory(pdir);

            // forward build command to slaves, but don't wait for results
            if (IsMaster())
               fProof->BuildPackage(package, TProof::kBuildOnSlavesNoWait);

            // check for BUILD.sh and execute
            if (!gSystem->AccessPathName("PROOF-INF/BUILD.sh")) {
               // Notify the upper level
               notm.Reset();
               notm << TString(Form("%s: building %s ...", noth.Data(), package.Data())) << notln;
               fSocket->Send(notm);

               // read version from file proofvers.txt, and if current version is
               // not the same do a "BUILD.sh clean"
               FILE *f = fopen("PROOF-INF/proofvers.txt", "r");
               if (f) {
                  TString v;
                  v.Gets(f);
                  fclose(f);
                  if (v != gROOT->GetVersion()) {
                     if (gSystem->Exec("PROOF-INF/BUILD.sh clean"))
                        status = -1;
                  }
               }

               // To build the package we execute PROOF-INF/BUILD.sh via a pipe
               // so that we can send back the log in (almost) real-time to the
               // (impatient) client. Note that this operation will block, so
               // the messages from builds on the workers will reach the client
               // shortly after the master ones.
               { TProofServLogHandlerGuard hg("PROOF-INF/BUILD.sh", fSocket);
               }

               // write version file
               f = fopen("PROOF-INF/proofvers.txt", "w");
               if (f) {
                  fputs(gROOT->GetVersion(), f);
                  fclose(f);
               }
            }

            gSystem->ChangeDirectory(ocwd);

         }

         fPackageLock->Unlock();

         if (status) {
            // Notify the upper level
            notm.Reset();
            notm << TString(Form("%s: failure building %s ...", noth.Data(), package.Data())) << notln;
            fSocket->Send(notm);
         } else {
            // collect built results from slaves
            if (IsMaster())
               fProof->BuildPackage(package, TProof::kCollectBuildResults);
            PDB(kPackage, 1)
               Info("HandleCache", "package %s successfully built", package.Data());
         }
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

         ocwd = gSystem->WorkingDirectory();
         gSystem->ChangeDirectory(pdir);

         // check for SETUP.C and execute
         if (!gSystem->AccessPathName("PROOF-INF/SETUP.C")) {
            Int_t err = 0;
            Int_t errm = gROOT->Macro("PROOF-INF/SETUP.C", &err);
            if (errm < 0)
               status = -1;
            if (err > TInterpreter::kNoError && err <= TInterpreter::kFatal)
               status = -1;
         }

         gSystem->ChangeDirectory(ocwd);

         if (status) {

            // Notify the upper level
            notm.Reset();
            notm << TString(Form("%s: failure loading %s ...", noth.Data(), package.Data())) << notln;
            fSocket->Send(notm);

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
            if (IsMaster())
               fProof->LoadPackage(package);

            PDB(kPackage, 1)
               Info("HandleCache", "package %s successfully loaded", package.Data());
         }
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
         break;
      case TProof::kShowSubCache:
         (*mess) >> all;
         if (IsMaster() && all)
            fProof->ShowCache(all);
         break;
      case TProof::kClearSubCache:
         if (IsMaster())
            fProof->ClearCache();
         break;
      case TProof::kShowSubPackages:
         (*mess) >> all;
         if (IsMaster() && all)
            fProof->ShowPackages(all);
         break;
      case TProof::kDisableSubPackages:
         if (IsMaster())
            fProof->DisablePackages();
         break;
      case TProof::kDisableSubPackage:
         (*mess) >> package;
         if (IsMaster())
            fProof->DisablePackage(package);
         break;
      case TProof::kBuildSubPackage:
         (*mess) >> package;
         if (IsMaster())
            fProof->BuildPackage(package);
         break;
      case TProof::kUnloadPackage:
         (*mess) >> package;
         status = UnloadPackage(package);
         if (IsMaster() && status == 0)
            status = fProof->UnloadPackage(package);
         break;
      case TProof::kDisablePackage:
         (*mess) >> package;
         fPackageLock->Lock();
         // remove package directory and par file
         gSystem->Exec(Form("%s %s/%s", kRM, fPackageDir.Data(),
                       package.Data()));
         gSystem->Exec(Form("%s %s/%s.par", kRM, fPackageDir.Data(),
                       package.Data()));
         fPackageLock->Unlock();
         if (IsMaster())
            fProof->DisablePackage(package);
         break;
      case TProof::kUnloadPackages:
         status = UnloadPackages();
         if (IsMaster() && status == 0)
            status = fProof->UnloadPackages();
         break;
      case TProof::kDisablePackages:
         fPackageLock->Lock();
         gSystem->Exec(Form("%s %s/*", kRM, fPackageDir.Data()));
         fPackageLock->Unlock();
         if (IsMaster())
            fProof->DisablePackages();
         break;
      case TProof::kListEnabledPackages:
         msg.Reset(kPROOF_PACKAGE_LIST);
         msg << type << fEnabledPackages;
         fSocket->Send(msg);
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
                     Info("HandleWorkerList","worker %s could not be (re-)activated:"
                                             " check the ordinal number", ord.Data());
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
                                             " check the ordinal number", ord.Data());
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
Int_t TProofServ::HandleDataSets(TMessage *mess)
{
   // Handle here all dataset requests

   PDB(kGlobal, 1)
      Info("HandleDataSets", "Enter");
   TList *previousDataSet = 0; //used when appending datasets
   TString dataSetName; // used in most cases
   Int_t type = 0;
   (*mess) >> type;
   switch (type) {
      case TProof::kCheckDataSetName:
         //
         // Check whether this dataset exist
         // Communication Summary
         //   Client                              Master
         //     |------------>DataSetName----------->|
         //     |<-------kMESS_OK/kMESS_NOTOK<-------| (Name OK/file exists)
         {
            TString fileListName;
            (*mess) >> fileListName;
            char *fileListPath =
               Form("%s/%s.root", fDataSetDir.Data(), fileListName.Data());

            // Lock the directory
            // TProofLockPathGuard dslguard(fDataSetLock);

            if (gSystem->AccessPathName(fileListPath, kFileExists) == kFALSE) {
               //Dataset name does exist
               fSocket->Send(kMESS_NOTOK);
            } else {
               fSocket->Send("", kMESS_OK);
            }
         }
         break;
      case TProof::kAppendDataSet:
         {
            (*mess) >> dataSetName;
            previousDataSet = GetDataSet(dataSetName.Data());
         }
         // NO break => continuing with kCreateDataSet
      case TProof::kCreateDataSet:
         // list size must be above 0
         {
            if (type == TProof::kCreateDataSet) {
               // if not kAppendDataSet
               (*mess) >> dataSetName;
            }
            char *fileListPath =
               Form("%s/%s.root", fDataSetDir.Data(), dataSetName.Data());

            // We would overwrite a dataset if it existed by this name
            TList *fileList =
               (TList *) (mess->ReadObject(TList::Class()));
            // if we started with kAppendDataSet
            if (previousDataSet) {
               TIter nextOldFile(previousDataSet);
               while (TFileInfo *obj = (TFileInfo*)nextOldFile())
                  fileList->Add(obj);
               delete previousDataSet;
            }

            // (re)create file and save dataset in its current status
            if (fileList->GetSize() > 0) {
               // We will save a sorted list
               fileList->Sort();
               // Removing repeated files (also when it's a new dataset name!
               TList *newFileList = new TList();
               TIter nextFile(fileList);
               TFileInfo *prevFile = (TFileInfo*)nextFile();
               newFileList->Add(prevFile);
               while (TFileInfo *obj = (TFileInfo*)nextFile())
                  if (prevFile->Compare(obj)) {
                     newFileList->Add(obj);
                     prevFile = obj;
                  }
               if (gSystem->AccessPathName(gSystem->DirName(fileListPath))) {
                  //the public dir or it's subdir does not exist
                  TString dirname = gSystem->DirName(fileListPath);
                  if (gSystem->mkdir(dirname, kTRUE))
                     Error("HandleDataSets",
                           "Error creating a datasets subdirectory: %s",
                           dirname.Data());
               }
               TFile *f = TFile::Open(fileListPath, "RECREATE");
               if (f) {
                  f->cd();
                  newFileList->Write("fileList", TObject::kSingleKey);
                  f->Close();
                  //TODO should depend on what Write returns
                  fSocket->Send(kMESS_OK);
               } else {
                  fSocket->Send(kMESS_NOTOK);
                  Error("HandleDataSets",
                        "can't open dataset file for writing");
               }
               delete f;
               delete newFileList;
               fileList->SetOwner();
               delete fileList;
            } else {
               fSocket->Send(kMESS_NOTOK);
               Printf("Can not save an empty list.");
            } // if (fileList->GetSize() > 0)
         }
         break;
      case TProof::kGetDataSets:
         {
            TString dir;
            (*mess) >> dir;
            TString dataSetDirPath;
            void *dataSetDir;
            if (dir.Length())
               if (strstr(dir, "public") == dir)
                  // list user own public datasets
                  dataSetDirPath = fDataSetDir + "/public/";
               else {
                  char *userName = (char *)malloc(strlen(dir));
                  strcpy(userName, dir.Data() + 1); //dir starts with '~'
                  strtok(userName, "/");
                  dataSetDirPath = fWorkDir + "/../" + userName + "/" +
                                     kPROOF_DataSetDir + "/public/";
               }
            else
               dataSetDirPath = fDataSetDir;
            if ((dataSetDir = gSystem->OpenDirectory(dataSetDirPath))) {
               TRegexp rg(".*.root"); //check that it is a root file
               TList *fileList = new TList();
               const char *ent;
               while ((ent = gSystem->GetDirEntry(dataSetDir))) {
                  if (TString(ent).Index(rg) != kNPOS)
                     //Matching dir entry
                     fileList->Add(new TObjString(TString(ent, strlen(ent) - 5)));
               }
               fileList->Sort();
               fSocket->SendObject(fileList, kMESS_OBJECT);
               fileList->SetOwner();
               delete fileList;
            } else {
               Printf("Can not open the dataset directory.");
               fSocket->Send(kMESS_NOTOK);
            }
         }
         break;
      case TProof::kGetDataSet:
         {
            TString name;
            (*mess) >> name;
            if (TList *fileList = GetDataSet(name.Data())) {
               fSocket->SendObject(fileList, kMESS_OK);
               delete fileList;
            } else                   // no such dataset
               fSocket->Send(kMESS_NOTOK);
         }
         break;
      case TProof::kRemoveDataSet:
         {
            TString name;
            (*mess) >> name;

            const char *fileListPath = Form("%s/%s.root",fDataSetDir.Data(),name.Data());

            // Lock the directory
            TProofLockPathGuard dslguard(fDataSetLock);

            if (gSystem->AccessPathName(fileListPath, kFileExists) == kFALSE) {
               if (gSystem->Unlink(fileListPath)) {
                  Printf("Error removing dataset %s", name.Data());
                  fSocket->Send(kMESS_NOTOK);
               } else
                  fSocket->Send(kMESS_OK);
            } else {
               Printf("The dataset does not exist");
               fSocket->Send(kMESS_NOTOK);
            }
         }
         break;
      case TProof::kVerifyDataSet:
         {
            TString name;
            (*mess) >> name;
            if (TList *fileList = GetDataSet(name.Data())) {
               TList *missingFileList = new TList();
               TIter next(fileList);
               TFileInfo *fileInfo;
               while ((fileInfo = (TFileInfo *)next())) {
                  if (gSystem->AccessPathName(fileInfo->GetFirstUrl()->GetUrl(),
                                              kFileExists) != kFALSE)
                     missingFileList->Add(fileInfo);
               }
               fSocket->SendObject(missingFileList, kMESS_OK);
            } else
               fSocket->Send(kMESS_NOTOK); //dataset does not exist
         }
         break;
      default:
         Error("HandleDataSets", "unknown type %d", type);
         break;
   }

   // We are done
   return 0;
}

//______________________________________________________________________________
TProofServ::EQueryAction TProofServ::GetWorkers(TList *workers,
                                                Int_t & /* prioritychange */)
{
   // Get list of workers to be used from now on.
   // The list must be provide by the caller.

   // Needs a list where to store the info
   if (!workers) {
      Error("GetWorkers", "output list undefined");
      return kQueryStop;
   }

   // Parse the config file
   TProofResourcesStatic *resources =
      new TProofResourcesStatic(fConfDir, fConfFile);
   fConfFile = resources->GetFileName(); // Update the global file name (with path)
   PDB(kGlobal,1)
         Info("GetWorkers", "using PROOF config file: %s", fConfFile.Data());

   // Get the master
   TProofNodeInfo *master = resources->GetMaster();
   if (master)
      fImage = master->GetImage();
   if (!master || (fImage.Length() == 0)) {
      PDB(kAll,1)
         Info("GetWorkers",
              "no appropriate master line found in %s", fConfFile.Data());
      return kQueryStop;
   }

   // Fill worker list
   if (resources->GetWorkers()) {
      TProofNodeInfo *ni = 0;
      TIter nw(resources->GetWorkers());
      while ((ni = (TProofNodeInfo *) nw()))
         workers->Add(new TProofNodeInfo(*ni));
   }

   // We are done
   return kQueryOK;
}

//______________________________________________________________________________
TList *TProofServ::GetDataSet(const char *name)
{
   // Utility function used in various methods for user dataset upload.

   TString fileListPath;
   if (strchr(name, '~') == name) {
      char *nameCopy = new char[strlen(name)];
      strcpy(nameCopy, name + 1);
      char *userName = strtok(nameCopy, "/");
      if (strcmp(strtok(0, "/"), "public"))
         return 0;
      fileListPath = fWorkDir + "/../" + userName + "/"
                     + kPROOF_DataSetDir + "/public/";
      delete[] nameCopy;
   } else if (strchr(name, '/') && strstr(name, "public") != name) {
      Printf("Dataset name should be of form [[~user/]public/]dataset");
      return 0;
   } else
      fileListPath = fDataSetDir + "/" + name + ".root";
   TList *fileList = 0;
   if (gSystem->AccessPathName(fileListPath.Data(), kFileExists) == kFALSE) {
      TFile *f = TFile::Open(fileListPath);
      f->cd();
      fileList = (TList *) f->Get("fileList");
      f->Close();
      delete f;
      if (strchr(name, '~') == name)  // not when allocated with Form
         delete[] fileListPath;
   }
   return fileList;
}

//______________________________________________________________________________
void TProofServ::ErrorHandler(Int_t level, Bool_t abort, const char *location,
                              const char *msg)
{
   // The PROOF error handler function. It prints the message on stderr and
   // if abort is set it aborts the application.

   if (gErrorIgnoreLevel == kUnset) {
      gErrorIgnoreLevel = 0;
      if (gEnv) {
         TString level = gEnv->GetValue("Root.ErrorIgnoreLevel", "Print");
         if (!level.CompareTo("Print", TString::kIgnoreCase))
            gErrorIgnoreLevel = kPrint;
         else if (!level.CompareTo("Info", TString::kIgnoreCase))
            gErrorIgnoreLevel = kInfo;
         else if (!level.CompareTo("Warning", TString::kIgnoreCase))
            gErrorIgnoreLevel = kWarning;
         else if (!level.CompareTo("Error", TString::kIgnoreCase))
            gErrorIgnoreLevel = kError;
         else if (!level.CompareTo("Break", TString::kIgnoreCase))
            gErrorIgnoreLevel = kBreak;
         else if (!level.CompareTo("SysError", TString::kIgnoreCase))
            gErrorIgnoreLevel = kSysError;
         else if (!level.CompareTo("Fatal", TString::kIgnoreCase))
            gErrorIgnoreLevel = kFatal;
      }
   }

   if (level < gErrorIgnoreLevel)
      return;

   static TString syslogService;

   if (syslogService.IsNull()) {
      syslogService = gProofServ != 0 ? gProofServ->GetService() : "proof";
      gSystem->Openlog(syslogService, kLogPid | kLogCons, kLogLocal5);

   } else if (gProofServ != 0 && syslogService != gProofServ->GetService()) {
      // re-initialize if proper service is now know
      syslogService = gProofServ->GetService();
      gSystem->Openlog(syslogService, kLogPid | kLogCons, kLogLocal5);
   }

   const char *type   = 0;
   ELogLevel loglevel = kLogInfo;

   if (level >= kPrint) {
      loglevel = kLogInfo;
      type = "Print";
   }
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

   TString node = "proof";
   TString user = "unknown";

   if (gProofServ) {
      node = gProofServ->IsMaster() ? "master" : "slave";
      node += gProofServ->GetOrdinal();
      user = gProofServ->GetUser();
   }

   TString buf;

   if (!location || strlen(location) == 0 ||
       (level >= kPrint && level < kInfo) ||
       (level >= kBreak && level < kSysError)) {
      fprintf(stderr, "%s on %s: %s\n", type, node.Data(), msg);
      buf.Form("%s:%s:%s:%s", user.Data(), node.Data(), type, msg);
   } else {
      fprintf(stderr, "%s in <%s> on %s: %s\n", type, location, node.Data(), msg);
      buf.Form("%s:%s:%s:<%s>:%s", user.Data(), node.Data(), type, location, msg);
   }
   fflush(stderr);

   gSystem->Syslog(loglevel, buf);

   if (abort) {

      static Bool_t recursive = kFALSE;

      if (gProofServ != 0 && !recursive) {
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
      Info("Lock", "file %s locked", pname);

   return 0;
}

//______________________________________________________________________________
Int_t TProofLockPath::Unlock()
{
   // Unlock the directory. Returns 0 in case of success,
   // -1 in case of error.

   if (!IsLocked())
      return 0;

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
      Info("Unlock", "file %s unlocked", GetName());

   close(fLockId);
   fLockId = -1;

   return 0;
}
