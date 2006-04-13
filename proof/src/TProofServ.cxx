// @(#)root/proof:$Name:  $:$Id: TProofServ.cxx,v 1.116 2006/03/31 07:42:20 rdm Exp $
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
#include "TSysEvtHandler.h"
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

// global proofserv handle
TProofServ *gProofServ;

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
   fServ->HandleSocketInput();
   return kTRUE;
}

// Max number of queries kept (-1 to disable)
Int_t TProofServ::fgMaxQueries = -1;

//______________________________________________________________________________
static TList *GetDataSet(const char *name)
{
   // Utility function used in various methods for user dataset upload.

   const char *fileListPath = Form("%s/%s/%s.root",
                                   gSystem->ExpandPathName(kPROOF_WorkDir),
                                   kPROOF_DataSetDir,
                                   name);
   if (gSystem->AccessPathName(fileListPath, kFileExists) == kFALSE) {
      TFile *f = TFile::Open(fileListPath);
      f->cd();
      TList *fileList = (TList *) f->Get("fileList");
      f->Close();
      delete f;
      return fileList;
   } else
      return 0;
}

ClassImp(TProofServ)

//______________________________________________________________________________
TProofServ::TProofServ(Int_t *argc, char **argv)
       : TApplication("proofserv", argc, argv, 0, -1)
{
   // Main constructor. Create an application environment. The TProofServ
   // environment provides an eventloop via inheritance of TApplication.
   // Actual server creation work is done in CreateServer() to allow
   // overloading.

   // crude check on number of arguments
   if (*argc < 2) {
     Fatal("TProofServ", "Must have at least 1 arguments (see  proofd).");
     exit(1);
   }

   // Parse options
   GetOptions(argc, argv);
}

//______________________________________________________________________________
void TProofServ::CreateServer()
{
   // Finalize the server setup. If master, create the TProof instance to talk
   // the worker or submaster nodes.

   // wait (loop) to allow debugger to connect
   if (gEnv->GetValue("Proof.GdbHook",0) == 3) {
      while (gProofServDebug)
         ;
   }

   // get socket to be used (setup in proofd)
   if (!(gSystem->Getenv("ROOTOPENSOCK"))) {
     Fatal("CreateServer", "Socket setup by proofd undefined");
     exit(1);
   }
   Int_t sock = strtol(gSystem->Getenv("ROOTOPENSOCK"), (char **)0, 10);
   if (sock <= 0) {
     Fatal("CreateServer", "Invalid socket descriptor number (%d)", sock);
     exit(1);
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
   fSocket          = new TSocket(sock);
   fEnabledPackages = new TList;
   fEnabledPackages->SetOwner();

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

   // The following libs are also useful to have, make sure they are loaded...
   gROOT->LoadClass("TMinuit",     "Minuit");
   gROOT->LoadClass("TPostScript", "Postscript");

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

   gProofServ = this;

   // if master, start slave servers
   if (IsMaster()) {
      TString master = "proof://__master__";
      TInetAddress a = gSystem->GetSockName(sock);
      if (a.IsValid()) {
         master += ":";
         master += a.GetPort();
      }

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
TProofServ::~TProofServ()
{
   // Cleanup. Not really necessary since after this dtor there is no
   // live anyway.

   SafeDelete(fQueries);
   SafeDelete(fPreviousQueries);
   SafeDelete(fWaitingQueries);
   delete fEnabledPackages;
   delete fSocket;
}

//______________________________________________________________________________
Int_t TProofServ::CatMotd()
{
   // Print message of the day (in fConfDir/etc/proof/motd). The motd
   // is not shown more than once a dat. If the file fConfDir/etc/proof/noproof
   // exists, show its contents and close the connection.

   TString motdname;
   TString lastname;
   FILE   *motd;
   Bool_t  show = kFALSE;

   motdname = fConfDir + "/etc/proof/noproof";
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

   motdname = fConfDir + "/etc/proof/motd";
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
       << fCompute.CpuTime() << bytesRead << totalEntries;

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
      fService = argv[1];
      fMasterServ = kTRUE;
   } else if (!strcmp(argv[1], "proofslave")) {
      fMasterServ = kFALSE;
      fService = argv[1];
   } else {
      Fatal("GetOptions", "Must be started as proofmaster or proofslave");
      exit(1);
   }

   // Confdir
   if (!(gSystem->Getenv("ROOTCONFDIR"))) {
     Fatal("GetOptions", "Must specify a config directory");
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

   if (fSocket->Recv(mess) <= 0)
      Terminate(0);               // do something more intelligent here

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
            PDB(kGlobal, 1) Info("HandleSocketInput:kMESS_CINT", "processing: %s...", str);
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

         HandleProcess(mess);

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
         {
            TString filenam;
            TMD5    md5;
            (*mess) >> filenam >> md5;
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
                  // remove any previous package directory with same name
                  st = gSystem->Exec(Form("%s %s/%s", kRM, fPackageDir.Data(),
                                     packnam.Data()));
                  if (st)
                     Error("HandleSocketInput:kPROOF_CHECKFILE", "failure executing: %s %s/%s",
                           kRM, fPackageDir.Data(), packnam.Data());
                  // find gunzip...
                  char *gunzip = gSystem->Which(gSystem->Getenv("PATH"),kGUNZIP,
                                                kExecutePermission);
                  if (gunzip) {
                     // untar package
                     st = gSystem->Exec(Form(kUNTAR, gunzip, fPackageDir.Data(),
                                        filenam.Data(), fPackageDir.Data()));
                     if (st)
                        Error("HandleSocketInput:kPROOF_CHECKFILE", "failure executing: %s",
                              Form(kUNTAR, gunzip, fPackageDir.Data(),
                                   filenam.Data(), fPackageDir.Data()));
                     delete [] gunzip;
                  } else
                     Error("HandleSocketInput:kPROOF_CHECKFILE", "%s not found",
                           kGUNZIP);
                  // check that fPackageDir/packnam now exists
                  if (gSystem->AccessPathName(fPackageDir + "/" + packnam, kWritePermission)) {
                     // par file did not unpack itself in the expected directory, failure
                     fSocket->Send(kPROOF_FATAL);
                     err = kTRUE;
                     PDB(kPackage, 1)
                        Info("HandleSocketInput:kPROOF_CHECKFILE",
                             "package %s did not unpack into %s", filenam.Data(),
                             packnam.Data());
                  } else {
                     // store md5 in package/PROOF-INF/md5.txt
                     TString md5f = fPackageDir + "/" + packnam + "/PROOF-INF/md5.txt";
                     TMD5::WriteChecksum(md5f, md5local);
                     fSocket->Send(kPROOF_CHECKFILE);
                     PDB(kPackage, 1)
                        Info("HandleSocketInput:kPROOF_CHECKFILE",
                             "package %s installed on node", filenam.Data());
                  }
               } else {
                  fSocket->Send(kPROOF_FATAL);
                  err = kTRUE;
               }

               // Note: Originally an UnlockPackage() call was made at the
               // after the if-else statement below. With multilevel masters,
               // submasters still check to make sure the package exists with
               // the correct md5 checksum and need to do a read lock there.
               // As yet locking is not that sophisicated so the lock must
               // be released below before the call to fProof->UploadPackage().
               if (!IsMaster() || err) {
                  // delete par file when on slave or in case of error
                  gSystem->Exec(Form("%s %s/%s", kRM, fPackageDir.Data(),
                                filenam.Data()));
                  UnlockPackage();
               } else {
                  // forward to slaves
                  UnlockPackage();
                  fProof->UploadPackage(fPackageDir + "/" + filenam);
               }
               delete md5local;
            } else if (filenam.BeginsWith("+")) {
               // check file in package directory
               filenam = filenam.Strip(TString::kLeading, '+');
               TString packnam = filenam;
               packnam.Remove(packnam.Length() - 4);  // strip off ".par"
               TString md5f = fPackageDir + "/" + packnam + "/PROOF-INF/md5.txt";
               LockPackage();
               TMD5 *md5local = TMD5::ReadChecksum(md5f);
               if (md5local && md5 == (*md5local)) {
                  // package already on server, unlock directory
                  UnlockPackage();
                  fSocket->Send(kPROOF_CHECKFILE);
                  PDB(kPackage, 1)
                     Info("HandleSocketInput:kPROOF_CHECKFILE",
                          "package %s already on node", filenam.Data());
                  if (IsMaster())
                     fProof->UploadPackage(fPackageDir + "/" + filenam);
               } else {
                  fSocket->Send(kPROOF_FATAL);
                  PDB(kPackage, 1)
                     Info("HandleSocketInput:kPROOF_CHECKFILE",
                          "package %s not yet on node", filenam.Data());
               }
               delete md5local;
            } else if (filenam.BeginsWith("=")) {
               // check file in package directory, do not lock if it is the wrong file
               filenam = filenam.Strip(TString::kLeading, '=');
               TString packnam = filenam;
               packnam.Remove(packnam.Length() - 4);  // strip off ".par"
               TString md5f = fPackageDir + "/" + packnam + "/PROOF-INF/md5.txt";
               LockPackage();
               TMD5 *md5local = TMD5::ReadChecksum(md5f);
               UnlockPackage();
               if (md5local && md5 == (*md5local)) {
                  // package already on server, unlock directory
                  fSocket->Send(kPROOF_CHECKFILE);
                  PDB(kPackage, 1)
                     Info("HandleSocketInput:kPROOF_CHECKFILE",
                          "package %s already on node", filenam.Data());
                  if (IsMaster())
                     fProof->UploadPackage(fPackageDir + "/" + filenam);
               } else {
                  fSocket->Send(kPROOF_FATAL);
                  PDB(kPackage, 1)
                     Info("HandleSocketInput:kPROOF_CHECKFILE",
                          "package %s not yet on node", filenam.Data());
               }
               delete md5local;
            } else {
               // check file in cache directory
               TString cachef = fCacheDir + "/" + filenam;
               LockCache();
               TMD5 *md5local = TMD5::FileChecksum(cachef);
               if (md5local && md5 == (*md5local)) {
                  // copy file from cache to working directory
                  gSystem->Exec(Form("%s %s .", kCP, cachef.Data()));
                  fSocket->Send(kPROOF_CHECKFILE);
                  PDB(kPackage, 1)
                     Info("HandleSocketInput:kPROOF_CHECKFILE", "file %s already on node", filenam.Data());
               } else {
                  fSocket->Send(kPROOF_FATAL);
                  PDB(kPackage, 1)
                     Info("HandleSocketInput:kPROOF_CHECKFILE", "file %s not yet on node", filenam.Data());
               }
               delete md5local;
               UnlockCache();
            }
         }
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
               LockCache();
               gSystem->Exec(Form("%s %s %s", kCP, name, fCacheDir.Data()));
               UnlockCache();
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
            (*mess) >> nodes;
            fProof->SetParallel(nodes);
            SendLogFile();
         }
         break;

      case kPROOF_CACHE:
         {
            // handle here all cache and package requests
            Int_t  status = 0;
            Int_t  type;
            Bool_t all;  //build;
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
                  LockCache();
                  gSystem->Exec(Form("%s %s/*", kRM, fCacheDir.Data()));
                  UnlockCache();
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
                     LockPackage();
                     gSystem->Exec(Form("%s %s/*", kRM, fPackageDir.Data()));
                     UnlockPackage();
                     if (IsMaster())
                        status = fProof->ClearPackages();
                  }
                  break;
               case TProof::kClearPackage:
                  (*mess) >> package;
                  status = UnloadPackage(package);
                  if (status == 0) {
                     LockPackage();
                     // remove package directory and par file
                     gSystem->Exec(Form("%s %s/%s", kRM, fPackageDir.Data(),
                                   package.Data()));
                     if (IsMaster())
                        gSystem->Exec(Form("%s %s/%s.par", kRM, fPackageDir.Data(),
                                      package.Data()));
                     UnlockPackage();
                     if (IsMaster())
                        status = fProof->ClearPackage(package);
                  }
                  break;
               case TProof::kBuildPackage:
                  (*mess) >> package;
                  LockPackage();
                  // check that package and PROOF-INF directory exists
                  pdir = fPackageDir + "/" + package;
                  if (gSystem->AccessPathName(pdir)) {
                     Error("HandleSocketInput:kPROOF_CACHE", "package %s does not exist",
                           package.Data());
                     status = -1;
                  } else if (gSystem->AccessPathName(pdir + "/PROOF-INF")) {
                     Error("HandleSocketInput:kPROOF_CACHE", "package %s does not have a PROOF-INF directory",
                           package.Data());
                     status = -1;
                  }

                  if (!status) {

                     PDB(kPackage, 1)
                        Info("HandleSocketInput:kPROOF_CACHE",
                             "package %s exists and has PROOF-INF directory", package.Data());

                     ocwd = gSystem->WorkingDirectory();
                     gSystem->ChangeDirectory(pdir);

                     // check for BUILD.sh and execute
                     if (!gSystem->AccessPathName(pdir + "/PROOF-INF/BUILD.sh")) {
                         if (gSystem->Exec("PROOF-INF/BUILD.sh"))
                            status = -1;
                     }

                     gSystem->ChangeDirectory(ocwd);

                  }
                  UnlockPackage();
                  // if built successful propagate to slaves
                  if (!status) {
                     if (IsMaster())
                        fProof->BuildPackage(package);

                     PDB(kPackage, 1)
                        Info("HandleSocketInput:kPROOF_CACHE",
                             "package %s successfully built", package.Data());
                  }
                  break;
               case TProof::kLoadPackage:
                  (*mess) >> package;
                  // always follows BuildPackage so no need to check for PROOF-INF
                  pdir = fPackageDir + "/" + package;

                  ocwd = gSystem->WorkingDirectory();
                  gSystem->ChangeDirectory(pdir);

                  // check for SETUP.C and execute
                  if (!gSystem->AccessPathName(pdir + "/PROOF-INF/SETUP.C")) {
                     gROOT->Macro("PROOF-INF/SETUP.C");
                  }

                  gSystem->ChangeDirectory(ocwd);

                  // create link to package in working directory
                  gSystem->Symlink(pdir, package);

                  // add package to list of include directories to be searched
                  // by ACliC
                  gSystem->AddIncludePath(TString("-I") + package);

                  // add package to list of include directories to be searched
                  // by CINT
                  gROOT->ProcessLine(TString(".include ") + package);

                  // if successful add to list and propagate to slaves
                  if (!status) {
                     fEnabledPackages->Add(new TObjString(package));
                     if (IsMaster())
                        fProof->LoadPackage(package);

                     PDB(kPackage, 1)
                         Info("HandleSocketInput:kPROOF_CACHE",
                              "package %s successfully loaded", package.Data());
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
                  LockPackage();
                  // remove package directory and par file
                  gSystem->Exec(Form("%s %s/%s", kRM, fPackageDir.Data(),
                                package.Data()));
                  gSystem->Exec(Form("%s %s/%s.par", kRM, fPackageDir.Data(),
                                package.Data()));
                  UnlockPackage();
                  if (IsMaster())
                     fProof->DisablePackage(package);
                  break;
               case TProof::kUnloadPackages:
                  status = UnloadPackages();
                  if (IsMaster() && status == 0)
                     status = fProof->UnloadPackages();
                  break;
               case TProof::kDisablePackages:
                  LockPackage();
                  gSystem->Exec(Form("%s %s/*", kRM, fPackageDir.Data()));
                  UnlockPackage();
                  if (IsMaster())
                     fProof->DisablePackages();
                  break;
               default:
                  Error("HandleSocketInput:kPROOF_CACHE", "unknown type %d", type);
                  break;
            }
            SendLogFile(status);
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

      case kPROOF_UPLOAD_DATASET:
         //
         // Communication Summary
         //   Client                             Master
         //     |------------>DataSetName----------->|
         //     |<-------kMESS_OK/kMESS_NOTOK<-------| (Name OK/file exist)
         //     |-->TList of TFileInfo/kMESS_NOTOK-->| (dataset/Cancel)
         //  (*)|<-------kMESS_OK/kMESS_NOTOK<-------| (transaction complete?)
         //  if the message with dataset is of type kPROOF_APPEND_DATASET => we append
         //  (*) - optional
         {
            TString fileListName;
            (*mess) >> fileListName;
            TMessage *retMess;
            Bool_t goodName = kFALSE;
            char *fileListPath = Form("%s/%s/%s.root",
                                      gSystem->ExpandPathName(kPROOF_WorkDir),
                                      kPROOF_DataSetDir,
                                      fileListName.Data());
            if (gSystem->AccessPathName(fileListPath, kFileExists) == kFALSE) {
               //Dataset name does exist
               fSocket->Send(kMESS_NOTOK);
            } else {
               fSocket->Send("", kMESS_OK);
               goodName = kTRUE;
            }

            fSocket->Recv(retMess);
            Int_t kind = retMess->What();
            if (kind == kMESS_OK || kind == kPROOF_APPEND_DATASET)
               //User wants to overwrite or append
               goodName = kTRUE;
            else if (kind != kMESS_NOTOK)
               Error("HandleSocketInput", "Wrong message type (%d)", kind);

            if (goodName && kind != kMESS_NOTOK) {
               // The fileName had existed user agreed to overwrite
               // and some TList was sent.
               TList *fileList =
                   (TList *) (retMess->ReadObject(TList::Class()));
               // (re)create file and save dataset in its current status
               if (retMess->What() == kPROOF_APPEND_DATASET) {
                  TList *oldFileList = GetDataSet(fileListName.Data());
                  TIter nextOldFile(oldFileList);
                  while (TFileInfo *obj = (TFileInfo*)nextOldFile())
                     fileList->Add(obj);
                  delete oldFileList;
               }
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
                  TFile *f = TFile::Open(fileListPath, "RECREATE");
                  if (f) {
                     f->cd();
                     newFileList->Write("fileList", TObject::kSingleKey);
                     f->Close();
                     fSocket->Send(kMESS_OK);
                     //should depend on what Write returns
                  } else {
                     fSocket->Send(kMESS_NOTOK);
                     Error("HandleSocketInput",
                           "can't open dataset file for writing");
                  }
                  delete f;
                  delete newFileList;
                  fileList->SetOwner();
                  delete fileList;
               } else {
                  fSocket->Send(kMESS_NOTOK);
                  Printf("No files were copied.");
               } // if (fileList->GetSize() > 0)
            } // if (goodName)
            delete retMess;
            // Notify
            SendLogFile();
         }
         break;
      case kPROOF_QUERY_DATASETS:
         {
            const char *ent;
            TList *fileList = new TList();
            void *dataSetDir = gSystem->OpenDirectory(
                                 Form("%s/%s",
                                 gSystem->ExpandPathName(kPROOF_WorkDir), //fWorkDir??
                                 kPROOF_DataSetDir));
            TRegexp rg(".*.root"); //check that it is a root file
            while ((ent = gSystem->GetDirEntry(dataSetDir))) {
               if (TString(ent).Index(rg) != kNPOS)
                  //Matching dir entry
                  fileList->Add(new TObjString(TString(ent, strlen(ent) - 5)));
            }
            fileList->Sort();
            fSocket->SendObject(fileList, kMESS_OBJECT);
            fileList->SetOwner();
            delete fileList;
            SendLogFile();
         }
         break;
      case kPROOF_GET_DATASET:
         {
            TString name;
            (*mess) >> name;
            if (TList *fileList = GetDataSet(name.Data())) {
               fSocket->SendObject(fileList, kMESS_OK);
               delete fileList; 
            } else                   // no such dataset
               fSocket->Send(kMESS_NOTOK);
            SendLogFile();
         }
         break;
      case kPROOF_RM_DATASET:
         {
            TString name;
            (*mess) >> name;
            const char *fileListPath = Form("%s/%s/%s.root",
                                            gSystem->
                                            ExpandPathName(kPROOF_WorkDir),
                                            kPROOF_DataSetDir, name.Data());
            if (gSystem->AccessPathName(fileListPath, kFileExists) == kFALSE) {
               if (gSystem->Unlink(fileListPath)) {
                  Printf("Error removing dataset %s", name.Data());
                  fSocket->Send(kMESS_NOTOK);
                  SendLogFile();
               } else
                  fSocket->Send(kMESS_OK);
            } else {
               Printf("The dataset does not exist");
               fSocket->Send(kMESS_NOTOK);
               SendLogFile();
            }
         }
         break;
      case kPROOF_VERIFY_DATASET:
         {
            TString name;
            (*mess) >> name;
            if (TList *fileList = GetDataSet(name.Data())) {
               TList *missingFileList = new TList();
               TIter next(fileList);
               TFileInfo *fileInfo;
               while ((fileInfo = (TFileInfo *)next())) {
                  if (gSystem->AccessPathName(fileInfo->GetFirstUrl()->GetUrl(),
                                              kFileExists)
                      != kFALSE)
                     missingFileList->Add(fileInfo);
               }
               fSocket->SendObject(missingFileList, kMESS_OK);
            } else
               fSocket->Send(kMESS_NOTOK); //dataset does not exist

            SendLogFile();
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

   delete mess;
}

//______________________________________________________________________________
void TProofServ::HandleSocketInputDuringProcess()
{
   // Handle messages that might arrive during processing while being in
   // HandleSocketInput(). This avoids recursive calls into HandleSocketInput().

   PDB(kGlobal,1) Info("HandleSocketInputDuringProcess", "enter");

   TMessage *mess;
   Int_t     what;
   Bool_t    aborted = kFALSE;

   if (fSocket->Recv(mess) <= 0)
      Terminate(0);               // do something more intelligent here

   what = mess->What();
   switch (what) {

      case kPROOF_PROCESS:

         HandleProcess(mess);

         // Notify
         SendLogFile();
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
         (*mess) >> aborted;
         PDB(kGlobal, 1)
            Info("HandleSocketInputDuringProcess:kPROOF_STOPPROCESS", "enter %d", aborted);
         if (fProof)
            fProof->StopProcess(aborted);
         else
            if(fPlayer)
               fPlayer->StopProcess(aborted);
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

         break;

      case TProof::kShutdownInterrupt:
         Info("HandleUrgentData", "Shutdown Interrupt");

         // If master server, propagate interrupt to slaves
         if (IsMaster())
            fProof->Interrupt(TProof::kShutdownInterrupt);

         Terminate(0);  // will not return from here....

         break;

      default:
         Error("HandleUrgentData", "unexpected OOB byte");
         break;
   }

   SendLogFile();

   if (fProof) fProof->SetActive(kFALSE);
}

//______________________________________________________________________________
void TProofServ::HandleSigPipe()
{
   // Called when the client is not alive anymore (i.e. when kKeepAlive
   // has failed).

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
Int_t TProofServ::LockDir(const TString &lock)
{
   // Lock a directory. Waits if lock is hold by an other process.
   // Returns 0 on success, -1 in case of error.

   Int_t *fid;
   if (lock == fCacheLock)
      fid = &fCacheLockId;
   else if (lock == fPackageLock)
      fid = &fPackageLockId;
   else if (lock == fQueryLock)
      fid = &fQueryLockId;
   else {
      Error("LockDir", "unknown lock file specified %s", lock.Data());
      return -1;
   }

   const char *lfile = lock;

   if (gSystem->AccessPathName(lfile))
      *fid = open(lfile, O_CREAT|O_RDWR, 0644);
   else
      *fid = open(lfile, O_RDWR);

   if (*fid == -1) {
      SysError("LockDir", "cannot open lock file %s", lfile);
      return -1;
   }

   // lock the file
#if !defined(R__WIN32) && !defined(R__WINGCC)
   if (lockf(*fid, F_LOCK, (off_t) 1) == -1) {
      SysError("LockDir", "error locking %s", lfile);
      close(*fid);
      *fid = -1;
      return -1;
   }
#endif

   PDB(kPackage, 2)
      Info("LockDir", "file %s locked", lfile);

   return 0;
}

//______________________________________________________________________________
Int_t TProofServ::UnlockDir(const TString &lock)
{
   // Unlock a directory. Returns 0 in case of success,
   // -1 in case of error.

   Int_t *fid;
   if (lock == fCacheLock)
      fid = &fCacheLockId;
   else if (lock == fPackageLock)
      fid = &fPackageLockId;
   else if (lock == fQueryLock)
      fid = &fQueryLockId;
   else {
      Error("UnlockDir", "unknown lock file specified %s", lock.Data());
      return -1;
   }

   if (*fid == -1) return 0;

   // unlock the file
   lseek(*fid, 0, SEEK_SET);
#if !defined(R__WIN32) && !defined(R__WINGCC)
   if (lockf(*fid, F_ULOCK, (off_t)1) == -1) {
      SysError("UnlockDir", "error unlocking %s", lock.Data());
      close(*fid);
      *fid = -1;
      return -1;
   }
#endif

   PDB(kPackage, 2)
      Info("UnlockDir", "file %s unlocked", lock.Data());

   close(*fid);
   *fid = -1;

   return 0;
}

//______________________________________________________________________________
Int_t TProofServ::LockQueryFile(const char *qlock)
{
   // Try to lock the query lock file "qlock" or fQueryLock if qlock = 0.
   // Does not block. Returns file descriptor on success on success and
   // -1 in case of any other error.

   Int_t fid = -1;

   const char *lfile = qlock ? qlock : fQueryLock.Data();

   if (gSystem->AccessPathName(lfile)) {
      SysError("LockQueryFile", "query lock file %s does not exist", lfile);
      return -1;
   } else {
      if ((fid = open(lfile, O_RDWR)) == -1) {
         SysError("LockQueryFile", "cannot open query lock file %s", lfile);
         return -1;
      }
   }

   // lock the file
#if !defined(R__WIN32) && !defined(R__WINGCC)
   if (lockf(fid, F_TLOCK, (off_t) 1) == -1) {
      Int_t rc = -1;
      if (TSystem::GetErrno() == EAGAIN)
         PDB(kPackage, 2)
            Info("LockDir", "query lock file %s already locked", lfile);
      else
         SysError("LockDir", "error locking %s", lfile);

      close(fid);
      return rc;
   }
#endif

   PDB(kPackage, 2)
      Info("LockDir", "query lock file %s locked", lfile);

   return fid;
}

//______________________________________________________________________________
Int_t TProofServ::UnlockQueryFile(Int_t fid)
{
   // Unlock the query lock file "qlock" open at file descriptor fid.
   // Returns 0 in case of success and -1 in case of any other error.

   if (fid <= -1)
      return 0;

   // unlock the file
   lseek(fid, 0, SEEK_SET);
#if !defined(R__WIN32) && !defined(R__WINGCC)
   if (lockf(fid, F_ULOCK, (off_t)1) == -1) {
      SysError("UnlockQueryFile", "error unlocking fd: %d", fid);
      close(fid);
      return -1;
   }
#endif

   PDB(kPackage, 2)
      Info("UnlockQueryFile", "file desc %d unlocked", fid);

   close(fid);

   return 0;
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

   TApplication::Run(retrn);
}

//______________________________________________________________________________
void TProofServ::SendLogFile(Int_t status, Int_t start, Int_t end)
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
      lseek(fileno(fLogFile), lnow, SEEK_SET);

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

   // remove link to package in working directory
   if (gSystem->Unlink(package) != 0) {
      Error("UnloadPackage",
            "unload of package %s unsuccessful", package);
      return -1;
   } else {
      // remove entry from include path

      TString aclicincpath = gSystem->GetIncludePath();
      TString cintincpath = gInterpreter->GetIncludePath();
      // remove interpreter part of gSystem->GetIncludePath()
      aclicincpath.Remove(aclicincpath.Length() - cintincpath.Length() - 1);
      // remove package's include path
      aclicincpath.ReplaceAll(TString(" -I") + package, "");
      gSystem->SetIncludePath(aclicincpath);

      //TODO reset interpreter include path

      // remove entry from enabled packages list
      delete fEnabledPackages->Remove(fEnabledPackages->FindObject(package));
      PDB(kPackage, 1)
         Info("UnloadPackage",
              "package %s successfully unloaded", package);
      return 0;
   }
}

//______________________________________________________________________________
Int_t TProofServ::UnloadPackages()
{
   // Unloads all enabled packages.

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
void TProofServ::Setup()
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

   // If old version, setup authentication related stuff
   if (fProtocol < 5) {
      TString wconf;
      if (OldAuthSetup(wconf) != 0) {
         Error("Setup", "OldAuthSetup: failed to setup authentication");
         gSystem->Exit(1);
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

      // Send session tag, if a recent client
      if (fProtocol > 6) {
         TMessage m(kPROOF_SESSIONTAG);
         m << fSessionTag;
         fSocket->Send(m);
      }
   }

   // Incoming OOB should generate a SIGURG
   fSocket->SetOption(kProcessGroup, gSystem->GetPid());

   // Send packages off immediately to reduce latency
   fSocket->SetOption(kNoDelay, 1);

   // Check every two hours if client is still alive
   fSocket->SetOption(kKeepAlive, 1);

   // Install SigPipe handler to handle kKeepAlive failure
   gSystem->AddSignalHandler(new TProofServSigPipeHandler(this));
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
   if (!(fQueries->GetSize())) {
      // make sure we remain in a "connected" directory
      gSystem->ChangeDirectory("/");
      // needed in case fQueryDir is on NFS ?!
      gSystem->MakeDirectory(fQueryDir+"/.delete");
      gSystem->Exec(Form("%s %s", kRM, fQueryDir.Data()));
      // Remove lock file
      gSystem->Unlink(fQueryLock);
   }

   // Unlock the query dir owned by this session
   UnlockQueryFile(fQueryLockId);

   // Remove input handler to avoid spurious signals in socket
   // selection for closing activities executed upon exit()
   TIter next(gSystem->GetListOfFileHandlers());
   TObject *fh = 0;
   while ((fh = next())) {
      TProofServInputHandler *ih = dynamic_cast<TProofServInputHandler *>(fh);
      if (ih)
         gSystem->RemoveFileHandler(ih);
   }

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

   // Save curren position
   off_t lnow = lseek(fileno(fLogFile), (off_t) 0, SEEK_CUR);

   // The range we are interested in
   Int_t start = pq->fStartLog;
   if (start > -1)
      lseek(fileno(fLogFile), (off_t) start, SEEK_SET);

   // Read the lines and add then to the internal container
   const Int_t kMAXBUF = 4096;
   char line[kMAXBUF];
   while (fgets(line, sizeof(line), fLogFile)) {
      if (line[strlen(line)-1] == '\n')
         line[strlen(line)-1] = 0;
      pq->AddLogLine((const char *)line);
   }

   // Restore initial position if partial send
   lseek(fileno(fLogFile), lnow, SEEK_SET);
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
   pq->SetProcessInfo(np, cpu - pq->GetUsedCPU(), bytes - pq->GetBytes());
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
                        Int_t fid = -1;
                        TString qlock;
                        if (LockSession(qr->GetTitle(), fid, qlock) == 0) {
                           RemoveQuery(qr);
                           // Unlock and remove the lock file
                           if (fid > -1)
                              UnlockQueryFile(fid);
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
Int_t TProofServ::LockSession(const char *sessiontag, Int_t &fid, TString &qlock)
{
   // Try locking query area of session tagged sessiontag.
   // The id of the locking file is returned in fid and must be
   // unlocked via UnlockQueryFile(fid).

   fid = -1;
   qlock = "";

   // We do not need to lock our own session
   if (strstr(sessiontag, fSessionTag))
      return 0;

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
   qlock = fQueryLock;
   qlock.ReplaceAll(fSessionTag, stag);

   if (!gSystem->AccessPathName(qlock)) {
      if ((fid = LockQueryFile(qlock)) < 0) {
         Info("LockSession","problems locking query lock file");
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

   Int_t fid = -1;
   TString qlock;
   if (LockSession(sessiontag, fid, qlock) == 0) {

      // Cleanup now
      gSystem->Exec(Form("%s %s", kRM, qdir.Data()));

      // Unlock and remove the lock file
      if (fid > -1) {
         UnlockQueryFile(fid);
         gSystem->Unlink(qlock);
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
            dset     = pq->GetDSet();
            opt      = pq->GetOptions();
            input    = pq->GetInputList();
            nentries = pq->GetEntries();
            first    = pq->GetFirst();
            evl      = pq->GetEventList();
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
           << pq->GetDSet()->GetListOfElements()->GetSize()
           << pq->GetFirst() << pq->GetEntries();
         fSocket->Send(m);

         // Create player
         if (IsParallel()) {
            // NOTE: fProof->SetPlayer(0) should be called after Process()
            p = fProof->MakePlayer();
         } else {
            // sequential mode
            p = new TProofPlayerSlave(fSocket);
            fProof->SetPlayer(p);
         }

         // Add query results to the player lists
         p->AddQueryResult(pq);

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
         if (p->GetExitStatus() != TProofPlayer::kAborted) {
            if (fProtocol > 6) {
               PDB(kGlobal, 2) Info("HandleProcess","Sending results");
               fSocket->SendObject(pq, kPROOF_OUTPUTLIST);
            } else {
               PDB(kGlobal, 2) Info("HandleProcess","Sending output list");
               fSocket->SendObject(p->GetOutputList(), kPROOF_OUTPUTLIST);
            }
         } else {
            fSocket->SendObject(0,kPROOF_OUTPUTLIST);
         }

         // Remove aborted queries from the list
         if (p->GetExitStatus() == TProofPlayer::kAborted)
            RemoveQuery(pq);

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
      p->Process(dset, filename, opt, nentries, first);

      // Return number of events processed
      TMessage m(kPROOF_STOPPROCESS);
      m << p->GetEventsProcessed();
      fSocket->Send(m);

      // Send back the results
      if (p->GetExitStatus() != TProofPlayer::kAborted) {
         PDB(kGlobal, 2) Info("HandleProcess","Sending output list");
         fSocket->SendObject(p->GetOutputList(), kPROOF_OUTPUTLIST);
      } else {
         fSocket->SendObject(0,kPROOF_OUTPUTLIST);
      }

      if (dset)
         delete dset;
      if (input)
         delete input;
   }

   // Set idle
   fIdle = kTRUE;

   // Cleanup
   if (fProof != 0)
      // ensure player is no longer referenced
      fProof->SetPlayer(0);
   delete p;
   fPlayer = 0;

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

   Int_t fid = -1;
   TString qlock;
   if (LockSession(queryref, fid, qlock) == 0) {

      // Remove query
      RemoveQuery(queryref);

      // Unlock and remove the lock file
      if (fid > -1) {
         UnlockQueryFile(fid);
         gSystem->Unlink(qlock);
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
               if (pqr) {
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
   } else {
      // Send the object
      fSocket->SendObject(pqr, kPROOF_RETRIEVE);
   }

   // Done
   return;
}
