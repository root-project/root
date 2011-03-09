// @(#)root/proof:$Id$
// Author: G. Ganis March 2008

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofLite                                                           //
//                                                                      //
// This class starts a PROOF session on the local machine: no daemons,  //
// client and master merged, communications via UNIX-like sockets.      //
// By default the number of workers started is NumberOfCores+1; a       //
// different number can be forced on construction.                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TProofLite.h"

#ifdef WIN32
#   include <io.h>
#   include "snprintf.h"
#endif
#include "RConfigure.h"
#include "TDSet.h"
#include "TEnv.h"
#include "TError.h"
#include "TFile.h"
#include "TFileCollection.h"
#include "TFileInfo.h"
#include "THashList.h"
#include "TMessage.h"
#include "TMonitor.h"
#include "TObjString.h"
#include "TPluginManager.h"
#include "TDataSetManager.h"
#include "TProofQueryResult.h"
#include "TProofServ.h"
#include "TQueryResultManager.h"
#include "TROOT.h"
#include "TServerSocket.h"
#include "TSlave.h"
#include "TSortedList.h"
#include "TTree.h"
#include "TVirtualProofPlayer.h"
#include "TSelector.h"

ClassImp(TProofLite)

Int_t TProofLite::fgWrksMax = -2; // Unitialized max number of workers

//______________________________________________________________________________
TProofLite::TProofLite(const char *url, const char *conffile, const char *confdir,
                       Int_t loglevel, const char *alias, TProofMgr *mgr)
{
   // Create a PROOF environment. Starting PROOF involves either connecting
   // to a master server, which in turn will start a set of slave servers, or
   // directly starting as master server (if master = ""). Masterurl is of
   // the form: [proof[s]://]host[:port]. Conffile is the name of the config
   // file describing the remote PROOF cluster (this argument alows you to
   // describe different cluster configurations).
   // The default is proof.conf. Confdir is the directory where the config
   // file and other PROOF related files are (like motd and noproof files).
   // Loglevel is the log level (default = 1). User specified custom config
   // files will be first looked for in $HOME/.conffile.

   fUrl.SetUrl(url);

   // This may be needed during init
   fManager = mgr;

   // Default server type
   fServType = TProofMgr::kProofLite;

   // Default query mode
   fQueryMode = kSync;

   // Client and master are merged
   fMasterServ = kTRUE;
   SetBit(TProof::kIsClient);
   SetBit(TProof::kIsMaster);

   // Flag that we are a client
   if (!gSystem->Getenv("ROOTPROOFCLIENT")) gSystem->Setenv("ROOTPROOFCLIENT","");
   
   // Protocol and Host
   fUrl.SetProtocol("proof");
   fUrl.SetHost("__lite__");
   fUrl.SetPort(1093);

   // User
   if (strlen(fUrl.GetUser()) <= 0) {
      // Get user logon name
      UserGroup_t *pw = gSystem->GetUserInfo();
      if (pw) {
         fUrl.SetUser(pw->fUser);
         delete pw;
      }
   }
   fMaster = gSystem->HostName();

   // Analysise the conffile field
   ParseConfigField(conffile);

   // Determine the number of workers giving priority to users request.
   // Otherwise use the system information, if available, or just start
   // the minimal number, i.e. 2 .
   if ((fNWorkers = GetNumberOfWorkers(url)) > 0) {

      Printf(" +++ Starting PROOF-Lite with %d workers +++", fNWorkers);
      // Init the session now
      Init(url, conffile, confdir, loglevel, alias);
   }

   // For final cleanup
   if (!gROOT->GetListOfProofs()->FindObject(this))
      gROOT->GetListOfProofs()->Add(this);

   // Still needed by the packetizers: needs to be changed
   gProof = this;
}

//______________________________________________________________________________
Int_t TProofLite::Init(const char *, const char *conffile,
                       const char *confdir, Int_t loglevel, const char *)
{
   // Start the PROOF environment. Starting PROOF involves either connecting
   // to a master server, which in turn will start a set of slave servers, or
   // directly starting as master server (if master = ""). For a description
   // of the arguments see the TProof ctor. Returns the number of started
   // master or slave servers, returns 0 in case of error, in which case
   // fValid remains false.

   R__ASSERT(gSystem);

   fValid = kFALSE;

   if (TestBit(TProof::kIsMaster)) {
      // Fill default conf file and conf dir
      if (!conffile || strlen(conffile) == 0)
         fConfFile = kPROOF_ConfFile;
      if (!confdir  || strlen(confdir) == 0)
         fConfDir  = kPROOF_ConfDir;
   } else {
      fConfDir     = confdir;
      fConfFile    = conffile;
   }

   // The sandbox for this session
   if (CreateSandbox() != 0) {
      Error("Init", "could not create/assert sandbox for this session");
      return 0;
   }

   // UNIX path for communication with workers
   TString sockpathdir = gEnv->GetValue("ProofLite.SockPathDir", gSystem->TempDirectory());
   if (sockpathdir.IsNull()) sockpathdir = gSystem->TempDirectory();
   if (sockpathdir(sockpathdir.Length()-1) == '/') sockpathdir.Remove(sockpathdir.Length()-1);
   fSockPath.Form("%s/plite-%d", sockpathdir.Data(), gSystem->GetPid());
   if (fSockPath.Length() > 104) {
      // Sort of hardcoded limit length for Unix systems
      Error("Init", "Unix socket path '%s' is too long (%d bytes):",
                    fSockPath.Data(), fSockPath.Length());
      Error("Init", "use 'ProofLite.SockPathDir' to create it under a directory different"
                    " from '%s'", sockpathdir.Data());
      return 0;
   }

   fLogLevel       = loglevel;
   fProtocol       = kPROOF_Protocol;
   fSendGroupView  = kTRUE;
   fImage          = "<local>";
   fIntHandler     = 0;
   fStatus         = 0;
   fRecvMessages   = new TList;
   fRecvMessages->SetOwner(kTRUE);
   fSlaveInfo      = 0;
   fChains         = new TList;
   fAvailablePackages = 0;
   fEnabledPackages = 0;
   fEndMaster      = TestBit(TProof::kIsMaster) ? kTRUE : kFALSE;
   fInputData      = 0;
   ResetBit(TProof::kNewInputData);

   // Timeout for some collect actions
   fCollectTimeout = gEnv->GetValue("Proof.CollectTimeout", -1);

   fProgressDialog        = 0;
   fProgressDialogStarted = kFALSE;

   // Client logging of messages from the workers
   fRedirLog = kFALSE;
   if (TestBit(TProof::kIsClient)) {
      fLogFileName = Form("%s/session-%s.log", fWorkDir.Data(), GetName());
      if ((fLogFileW = fopen(fLogFileName.Data(), "w")) == 0)
         Error("Init", "could not create temporary logfile %s", fLogFileName.Data());
      if ((fLogFileR = fopen(fLogFileName.Data(), "r")) == 0)
         Error("Init", "could not open logfile %s for reading", fLogFileName.Data());
   }
   fLogToWindowOnly = kFALSE;

   fCacheLock = new TProofLockPath(TString::Format("%s/%s%s", gSystem->TempDirectory(),
                                   kPROOF_CacheLockFile,
                                   TString(fCacheDir).ReplaceAll("/","%").Data()));

   // Create 'queries' locker instance and lock it
   fQueryLock = new TProofLockPath(TString::Format("%s/%s%s-%s", gSystem->TempDirectory(),
                                   kPROOF_QueryLockFile, GetName(),
                                   TString(fQueryDir).ReplaceAll("/","%").Data()));
   fQueryLock->Lock();
   // Create the query manager
   fQMgr = new TQueryResultManager(fQueryDir, GetName(), fWorkDir,
                                   fQueryLock, fLogFileW);

   // Apply quotas, if any
   if (fQMgr && fQMgr->ApplyMaxQueries(10) != 0)
      Warning("Init", "problems applying fMaxQueries");

   if (InitDataSetManager() != 0)
      Warning("Init", "problems initializing the dataset manager");

   // Status of cluster
   fNotIdle = 0;

   // Query type
   fSync = kTRUE;

   // List of queries
   fQueries = 0;
   fOtherQueries = 0;
   fDrawQueries = 0;
   fMaxDrawQueries = 1;
   fSeqNum = 0;

   // Remote ID of the session
   fSessionID = -1;

   // Part of active query
   fWaitingSlaves = 0;

   // Make remote PROOF player
   fPlayer = 0;
   MakePlayer("lite");

   fFeedback = new TList;
   fFeedback->SetOwner();
   fFeedback->SetName("FeedbackList");
   AddInput(fFeedback);

   // Sort workers by descending performance index
   fSlaves           = new TSortedList(kSortDescending);
   fActiveSlaves     = new TList;
   fInactiveSlaves   = new TList;
   fUniqueSlaves     = new TList;
   fAllUniqueSlaves  = new TList;
   fNonUniqueMasters = new TList;
   fBadSlaves        = new TList;
   fAllMonitor       = new TMonitor;
   fActiveMonitor    = new TMonitor;
   fUniqueMonitor    = new TMonitor;
   fAllUniqueMonitor = new TMonitor;
   fCurrentMonitor   = 0;
   fServSock         = 0;

   // Control how to start the workers; copy-on-write (fork) is *very*
   // experimental and available on Unix only.
   fForkStartup      = kFALSE;
   if (gEnv->GetValue("ProofLite.ForkStartup", 0) != 0) {
#ifndef WIN32
      fForkStartup   = kTRUE;
#else
      Warning("Init", "fork-based workers startup is not available on Windows - ignoring");
#endif
   }

   fPackageLock             = 0;
   fEnabledPackagesOnClient = 0;
   fLoadedMacros            = 0;
   fGlobalPackageDirList    = 0;
   if (TestBit(TProof::kIsClient)) {

      // List of directories where to look for global packages
      TString globpack = gEnv->GetValue("Proof.GlobalPackageDirs","");
      if (globpack.Length() > 0) {
         Int_t ng = 0;
         Int_t from = 0;
         TString ldir;
         while (globpack.Tokenize(ldir, from, ":")) {
            if (gSystem->AccessPathName(ldir, kReadPermission)) {
               Warning("Init", "directory for global packages %s does not"
                               " exist or is not readable", ldir.Data());
            } else {
               // Add to the list, key will be "G<ng>", i.e. "G0", "G1", ...
               TString key = Form("G%d", ng++);
               if (!fGlobalPackageDirList) {
                  fGlobalPackageDirList = new THashList();
                  fGlobalPackageDirList->SetOwner();
               }
               fGlobalPackageDirList->Add(new TNamed(key,ldir));
            }
         }
      }

      TString lockpath(fPackageDir);
      lockpath.ReplaceAll("/", "%");
      lockpath.Insert(0, TString::Format("%s/%s", gSystem->TempDirectory(), kPROOF_PackageLockFile));
      fPackageLock = new TProofLockPath(lockpath.Data());

      fEnabledPackagesOnClient = new TList;
      fEnabledPackagesOnClient->SetOwner();
   }

   // Start workers
   if (SetupWorkers(0) != 0) {
      Error("Init", "problems setting up workers");
      return 0;
   }

   // we are now properly initialized
   fValid = kTRUE;

   // De-activate monitor (will be activated in Collect)
   fAllMonitor->DeActivateAll();

   // By default go into parallel mode
   GoParallel(9999, kFALSE);

   // Send relevant initial state to slaves
   SendInitialState();

   SetActive(kFALSE);

   if (IsValid()) {
      // Activate input handler
      ActivateAsyncInput();
      // Set PROOF to running state
      SetRunStatus(TProof::kRunning);
   }
   // We register the session as a socket so that cleanup is done properly
   R__LOCKGUARD2(gROOTMutex);
   gROOT->GetListOfSockets()->Add(this);

   return fActiveSlaves->GetSize();
}
//______________________________________________________________________________
TProofLite::~TProofLite()
{
   // Destructor

   // Shutdown the workers
   RemoveWorkers(0);

   if (!(fQMgr && fQMgr->Queries() && fQMgr->Queries()->GetSize())) {
      // needed in case fQueryDir is on NFS ?!
      gSystem->MakeDirectory(fQueryDir+"/.delete");
      gSystem->Exec(Form("%s %s", kRM, fQueryDir.Data()));
   }

   // Remove lock file
   if (fQueryLock) {
      gSystem->Unlink(fQueryLock->GetName());
      fQueryLock->Unlock();
   }

   // Cleanup the socket
   SafeDelete(fServSock);
   gSystem->Unlink(fSockPath);
}

//______________________________________________________________________________
Int_t TProofLite::GetNumberOfWorkers(const char *url)
{
   // Static method to determine the number of workers giving priority to users request.
   // Otherwise use the system information, if available, or just start
   // the minimal number, i.e. 2 .

   Bool_t notify = kFALSE;
   if (fgWrksMax == -2) {
      // Find the max number of workers, if any
      TString sysname = "system.rootrc";
#ifdef ROOTETCDIR
      char *s = gSystem->ConcatFileName(ROOTETCDIR, sysname);
#else
      TString etc = gRootDir;
#ifdef WIN32
      etc += "\\etc";
#else
      etc += "/etc";
#endif
      char *s = gSystem->ConcatFileName(etc, sysname);
#endif
      TEnv sysenv(0);
      sysenv.ReadFile(s, kEnvGlobal);
      fgWrksMax = sysenv.GetValue("ProofLite.MaxWorkers", -1);
      // Notify once the user if its will is changed
      notify = kTRUE;
      if (s) delete[] s;
   }
   if (fgWrksMax == 0) {
      ::Error("TProofLite::GetNumberOfWorkers",
              "PROOF-Lite disabled by the system administrator: sorry!");
      return 0;
   }

   Int_t nWorkers = -1;
   if (url && strlen(url)) {
      TString o(url);
      Int_t in = o.Index("workers=");
      if (in != kNPOS) {
         o.Remove(0, in + strlen("workers="));
         while (!o.IsDigit())
            o.Remove(o.Length()-1);
         nWorkers = (!o.IsNull()) ? o.Atoi() : nWorkers;
      }
   }
   if (nWorkers <= 0) {
      nWorkers = gEnv->GetValue("ProofLite.Workers", -1);
      if (nWorkers <= 0) {
         SysInfo_t si;
         if (gSystem->GetSysInfo(&si) == 0 && si.fCpus > 2) {
            nWorkers = si.fCpus;
         } else {
            // Two workers by default
            nWorkers = 2;
         }
         if (notify) notify = kFALSE;
      }
   }
   // Apply the max, if any
   if (fgWrksMax > 0 && fgWrksMax < nWorkers) {
      if (notify)
         ::Warning("TProofLite::GetNumberOfWorkers", "number of PROOF-Lite workers limited by"
                                                     " the system administrator to %d", fgWrksMax);
      nWorkers = fgWrksMax;
   }

   // Done
   return nWorkers;
}

//______________________________________________________________________________
Int_t TProofLite::SetupWorkers(Int_t opt, TList *startedWorkers)
{
   // Start up PROOF workers.

   // Create server socket on the assigned UNIX sock path
   if (!fServSock) {
      if ((fServSock = new TServerSocket(fSockPath))) {
         R__LOCKGUARD2(gROOTMutex);
         // Remove from the list so that cleanup can be done in the correct order
         gROOT->GetListOfSockets()->Remove(fServSock);
      }
   }
   if (!fServSock || !fServSock->IsValid()) {
      Error("SetupWorkers",
            "unable to create server socket for internal communications");
      SetBit(kInvalidObject);
      return -1;
   }

   // Create a monitor and add the socket to it
   TMonitor *mon = new TMonitor;
   mon->Add(fServSock);

   TList started;
   TSlave *wrk = 0;
   Int_t nWrksDone = 0, nWrksTot = -1;
   TString fullord;

   if (opt == 0) {
      nWrksTot = fForkStartup ? 1 : fNWorkers;
      // Now we create the worker applications which will call us back to finalize
      // the setup
      Int_t ord = 0;
      for (; ord < nWrksTot; ord++) {

         // Ordinal for this worker server
         fullord = Form("0.%d", ord);

         // Create environment files
         SetProofServEnv(fullord);

         // Create worker server and add to the list
         if ((wrk = CreateSlave("lite", fullord, 100, fImage, fWorkDir)))
            started.Add(wrk);

         // Notify
         NotifyStartUp("Opening connections to workers", ++nWrksDone, nWrksTot);

      } //end of worker loop
   } else {
      if (!fForkStartup) {
         Warning("SetupWorkers", "standard startup: workers already started");
         return -1;
      }
      nWrksTot = fNWorkers - 1;
      // Now we create the worker applications which will call us back to finalize
      // the setup
      TString clones;
      Int_t ord = 0;
      for (; ord < nWrksTot; ord++) {

         // Ordinal for this worker server
         fullord.Form("0.%d", ord + 1);
         if (!clones.IsNull()) clones += " ";
         clones += fullord;

         // Create worker server and add to the list
         if ((wrk = CreateSlave("lite", fullord, -1, fImage, fWorkDir)))
            started.Add(wrk);

         // Notify
         NotifyStartUp("Opening connections to workers", ++nWrksDone, nWrksTot);

      } //end of worker loop

      // Send the request
      TMessage m(kPROOF_FORK);
      m << clones;
      Broadcast(m, kActive);
   }

   // Wait for call backs
   nWrksDone = 0;
   nWrksTot = started.GetSize();
   Int_t nSelects = 0;
   Int_t to = gEnv->GetValue("ProofLite.StartupTimeOut", 5) * 1000;
   while (started.GetSize() > 0 && nSelects < nWrksTot) {

      // Wait for activity on the socket for max 5 secs
      TSocket *xs = mon->Select(to);

      // Count attempts and check
      nSelects++;
      if (xs == (TSocket *) -1) continue;

      // Get the connection
      TSocket *s = fServSock->Accept();
      if (s && s->IsValid()) {
         // Receive ordinal
         TMessage *msg = 0;
         s->Recv(msg);
         if (msg) {
            TString ord;
            *msg >> ord;
            // Find who is calling back
            if ((wrk = (TSlave *) started.FindObject(ord))) {
               // Remove it from the started list
               started.Remove(wrk);

               // Assign tis socket the selected worker
               wrk->SetSocket(s);
               // Remove socket from global TROOT socket list. Only the TProof object,
               // representing all worker sockets, will be added to this list. This will
               // ensure the correct termination of all proof servers in case the
               // root session terminates.
               {  R__LOCKGUARD2(gROOTMutex);
                  gROOT->GetListOfSockets()->Remove(s);
               }
               if (wrk->IsValid()) {
                  // Set the input handler
                  wrk->SetInputHandler(new TProofInputHandler(this, wrk->GetSocket()));
                  // Set fParallel to 1 for workers since they do not
                  // report their fParallel with a LOG_DONE message
                  wrk->fParallel = 1;
                  // Finalize setup of the server
                  wrk->SetupServ(TSlave::kSlave, 0);
               }

               // Monitor good workers
               if (wrk->IsValid()) {
                  fSlaves->Add(wrk);
                  if (opt == 1) fActiveSlaves->Add(wrk);
                  fAllMonitor->Add(wrk->GetSocket());
                  // Record also in the list for termination
                  if (startedWorkers) startedWorkers->Add(wrk);
                  // Notify startup operations
                  NotifyStartUp("Setting up worker servers", ++nWrksDone, nWrksTot);
               } else {
                  // Flag as bad
                  fBadSlaves->Add(wrk);
               }
            }
         }
      }
   }

   // Cleanup the monitor and the server socket
   mon->DeActivateAll();
   delete mon;

   // Create Progress dialog, if needed
   if (!gROOT->IsBatch() && !fProgressDialog) {
      if ((fProgressDialog =
         gROOT->GetPluginManager()->FindHandler("TProofProgressDialog")))
         if (fProgressDialog->LoadPlugin() == -1)
            fProgressDialog = 0;
   }

   if (opt == 1) {
      // Collect replies
      Collect(kActive);
      // Update group view
      SendGroupView();
      // By default go into parallel mode
      SetParallel(9999, 0);
   }
   // Done
   return 0;
}

//______________________________________________________________________________
void TProofLite::NotifyStartUp(const char *action, Int_t done, Int_t tot)
{
   // Notify setting-up operation message

   Int_t frac = (Int_t) (done*100.)/tot;
   char msg[512] = {0};
   if (frac >= 100) {
      snprintf(msg, 512, "%s: OK (%d workers)                 \n",
                   action, tot);
   } else {
      snprintf(msg, 512, "%s: %d out of %d (%d %%)\r",
                   action, done, tot, frac);
   }
   fprintf(stderr,"%s", msg);
}

//______________________________________________________________________________
Int_t TProofLite::SetProofServEnv(const char *ord)
{
   // Create environment files for worker 'ord'

   // Check input
   if (!ord || strlen(ord) <= 0) {
      Error("SetProofServEnv", "ordinal string undefined");
      return -1;
   }

   // ROOT env file
   TString rcfile(Form("%s/worker-%s.rootrc", fWorkDir.Data(), ord));
   FILE *frc = fopen(rcfile.Data(), "w");
   if (!frc) {
      Error("SetProofServEnv", "cannot open rc file %s", rcfile.Data());
      return -1;
   }

   // The session working dir depends on the role
   fprintf(frc,"# The session working dir\n");
   fprintf(frc,"ProofServ.SessionDir: %s/worker-%s\n", fWorkDir.Data(), ord);

   // The session unique tag
   fprintf(frc,"# Session tag\n");
   fprintf(frc,"ProofServ.SessionTag: %s\n", GetName());

   // Log / Debug level
   fprintf(frc,"# Proof Log/Debug level\n");
   fprintf(frc,"Proof.DebugLevel: %d\n", gDebug);

   // Ordinal number
   fprintf(frc,"# Ordinal number\n");
   fprintf(frc,"ProofServ.Ordinal: %s\n", ord);

   // ROOT Version tag
   fprintf(frc,"# ROOT Version tag\n");
   fprintf(frc,"ProofServ.RootVersionTag: %s\n", gROOT->GetVersion());

   // Work dir
   TString sandbox = gEnv->GetValue("ProofServ.Sandbox", fSandbox);
   gSystem->ExpandPathName(sandbox);
   fprintf(frc,"# Users sandbox\n");
   fprintf(frc, "ProofServ.Sandbox: %s\n", sandbox.Data());

   // Cache dir
   fprintf(frc,"# Users cache\n");
   fprintf(frc, "ProofServ.CacheDir: %s\n", fCacheDir.Data());

   // Package dir
   fprintf(frc,"# Users packages\n");
   fprintf(frc, "ProofServ.PackageDir: %s\n", fPackageDir.Data());

   // Image
   fprintf(frc,"# Server image\n");
   fprintf(frc, "ProofServ.Image: %s\n", fImage.Data());

   // Set Open socket
   fprintf(frc,"# Open socket\n");
   fprintf(frc, "ProofServ.OpenSock: %s\n", fSockPath.Data());

   // Client Protocol
   fprintf(frc,"# Client Protocol\n");
   fprintf(frc, "ProofServ.ClientVersion: %d\n", kPROOF_Protocol);

   // ROOT env file created
   fclose(frc);

   // System env file
   TString envfile(Form("%s/worker-%s.env", fWorkDir.Data(), ord));
   FILE *fenv = fopen(envfile.Data(), "w");
   if (!fenv) {
      Error("SetProofServEnv", "cannot open env file %s", envfile.Data());
      return -1;
   }
   // ROOTSYS
#ifdef R__HAVE_CONFIG
   fprintf(fenv, "export ROOTSYS=%s\n", ROOTPREFIX);
#else
   fprintf(fenv, "export ROOTSYS=%s\n", gSystem->Getenv("ROOTSYS"));
#endif
   // Conf dir
#ifdef R__HAVE_CONFIG
   fprintf(fenv, "export ROOTCONFDIR=%s\n", ROOTETCDIR);
#else
   fprintf(fenv, "export ROOTCONFDIR=%s\n", gSystem->Getenv("ROOTSYS"));
#endif
   // TMPDIR
   fprintf(fenv, "export TMPDIR=%s\n", gSystem->TempDirectory());
   // Log file in the log dir
   TString logfile(Form("%s/worker-%s.log", fWorkDir.Data(), ord));
   fprintf(fenv, "export ROOTPROOFLOGFILE=%s\n", logfile.Data());
   // RC file
   fprintf(fenv, "export ROOTRCFILE=%s\n", rcfile.Data());
   // ROOT version tag (needed in building packages)
   fprintf(fenv, "export ROOTVERSIONTAG=%s\n", gROOT->GetVersion());
   // This flag can be used to identify the type of worker; for example, in BUILD.sh or SETUP.C ...
   fprintf(fenv, "export ROOTPROOFLITE=%d\n", fNWorkers);
   // Set the user envs
   if (fgProofEnvList) {
      TString namelist;
      TIter nxenv(fgProofEnvList);
      TNamed *env = 0;
      while ((env = (TNamed *)nxenv())) {
         TString senv(env->GetTitle());
         ResolveKeywords(senv, logfile.Data());
         fprintf(fenv, "export %s=%s\n", env->GetName(), senv.Data());
         if (namelist.Length() > 0)
            namelist += ',';
         namelist += env->GetName();
      }
      fprintf(fenv, "export PROOF_ALLVARS=%s\n", namelist.Data());
   }

   // System env file created
   fclose(fenv);

   // Done
   return 0;
}

//__________________________________________________________________________
void TProofLite::ResolveKeywords(TString &s, const char *logfile)
{
   // Resolve some keywords in 's'
   //    <logfileroot>, <user>, <rootsys>

   if (!logfile) return;

   // Log file
   if (s.Contains("<logfilewrk>") && logfile) {
      TString lfr(logfile);
      if (lfr.EndsWith(".log")) lfr.Remove(lfr.Last('.'));
      s.ReplaceAll("<logfilewrk>", lfr.Data());
   }

   // user
   if (gSystem->Getenv("USER") && s.Contains("<user>")) {
      s.ReplaceAll("<user>", gSystem->Getenv("USER"));
   }

   // rootsys
   if (gSystem->Getenv("ROOTSYS") && s.Contains("<rootsys>")) {
      s.ReplaceAll("<rootsys>", gSystem->Getenv("ROOTSYS"));
   }
}

//______________________________________________________________________________
Int_t TProofLite::CreateSandbox()
{
   // Create the sandbox for this session

   // Make sure the sandbox area exist and is writable
   fSandbox = gEnv->GetValue("ProofLite.Sandbox", "");
   if (fSandbox.IsNull())
      fSandbox = gEnv->GetValue("Proof.Sandbox", TString::Format("~/%s", kPROOF_WorkDir));
   gSystem->ExpandPathName(fSandbox);
   if (AssertPath(fSandbox, kTRUE) != 0) return -1;

   // Package Dir
   fPackageDir = gEnv->GetValue("Proof.PackageDir", "");
   if (fPackageDir.IsNull())
      fPackageDir.Form("%s/%s", fSandbox.Data(), kPROOF_PackDir);
   if (AssertPath(fPackageDir, kTRUE) != 0) return -1;

   // Cache Dir
   fCacheDir = gEnv->GetValue("Proof.CacheDir", "");
   if (fCacheDir.IsNull())
      fCacheDir.Form("%s/%s", fSandbox.Data(), kPROOF_CacheDir);
   if (AssertPath(fCacheDir, kTRUE) != 0) return -1;

   // Data Set Dir
   fDataSetDir = gEnv->GetValue("Proof.DataSetDir", "");
   if (fDataSetDir.IsNull())
      fDataSetDir.Form("%s/%s", fSandbox.Data(), kPROOF_DataSetDir);
   if (AssertPath(fDataSetDir, kTRUE) != 0) return -1;

   // Session unique tag (name of this TProof instance)
   TString stag;
   stag.Form("%s-%d-%d", gSystem->HostName(), (int)time(0), gSystem->GetPid());
   SetName(stag.Data());

   // Subpath for this session in the fSandbox (<sandbox>/path-to-working-dir)
   TString sessdir(gSystem->WorkingDirectory());
   sessdir.ReplaceAll(gSystem->HomeDirectory(),"");
   sessdir.ReplaceAll("/","-");
   sessdir.Replace(0,1,"/",1);
   sessdir.Insert(0, fSandbox.Data());

   // Session working and queries dir
   fWorkDir.Form("%s/session-%s", sessdir.Data(), stag.Data());
   if (AssertPath(fWorkDir, kTRUE) != 0) return -1;

   // Create symlink to the last session
   TString lastsess;
   lastsess.Form("%s/last-lite-session", sessdir.Data());
   gSystem->Unlink(lastsess);
   gSystem->Symlink(fWorkDir, lastsess);

   // Queries Dir: local to the working dir, unless required differently
   fQueryDir = gEnv->GetValue("Proof.QueryDir", "");
   if (fQueryDir.IsNull())
      fQueryDir.Form("%s/%s", sessdir.Data(), kPROOF_QueryDir);
   if (AssertPath(fQueryDir, kTRUE) != 0) return -1;

   // Cleanup old sessions dirs
   CleanupSandbox();

   // Done
   return 0;
}

//______________________________________________________________________________
void TProofLite::Print(Option_t *option) const
{
   // Print status of PROOF-Lite cluster.

   if (IsParallel())
      Printf("*** PROOF-Lite cluster (parallel mode, %d workers):", GetParallel());
   else
      Printf("*** PROOF-Lite cluster (sequential mode)");

   Printf("Host name:                  %s", gSystem->HostName());
   Printf("User:                       %s", GetUser());
   TString ver(gROOT->GetVersion());
   if (gROOT->GetSvnRevision() > 0)
      ver += Form("|r%d", gROOT->GetSvnRevision());
   if (gSystem->Getenv("ROOTVERSIONTAG"))
      ver += Form("|%s", gSystem->Getenv("ROOTVERSIONTAG"));
   Printf("ROOT version|rev|tag:       %s", ver.Data());
   Printf("Architecture-Compiler:      %s-%s", gSystem->GetBuildArch(),
                                               gSystem->GetBuildCompilerVersion());
   Printf("Protocol version:           %d", GetClientProtocol());
   Printf("Working directory:          %s", gSystem->WorkingDirectory());
   Printf("Communication path:         %s", fSockPath.Data());
   Printf("Log level:                  %d", GetLogLevel());
   Printf("Number of workers:          %d", GetNumberOfSlaves());
   Printf("Number of active workers:   %d", GetNumberOfActiveSlaves());
   Printf("Number of unique workers:   %d", GetNumberOfUniqueSlaves());
   Printf("Number of inactive workers: %d", GetNumberOfInactiveSlaves());
   Printf("Number of bad workers:      %d", GetNumberOfBadSlaves());
   Printf("Total MB's processed:       %.2f", float(GetBytesRead())/(1024*1024));
   Printf("Total real time used (s):   %.3f", GetRealTime());
   Printf("Total CPU time used (s):    %.3f", GetCpuTime());
   if (TString(option).Contains("a", TString::kIgnoreCase) && GetNumberOfSlaves()) {
      Printf("List of workers:");
      TIter nextslave(fSlaves);
      while (TSlave* sl = dynamic_cast<TSlave*>(nextslave())) {
         if (sl->IsValid())
            sl->Print(option);
      }
   }
}

//______________________________________________________________________________
TProofQueryResult *TProofLite::MakeQueryResult(Long64_t nent, const char *opt,
                                               Long64_t fst, TDSet *dset,
                                               const char *selec)
{
   // Create a TProofQueryResult instance for this query.

   // Increment sequential number
   Int_t seqnum = -1;
   if (fQMgr) {
      fQMgr->IncrementSeqNum();
      seqnum = fQMgr->SeqNum();
   }

   // Create the instance and add it to the list
   TProofQueryResult *pqr = new TProofQueryResult(seqnum, opt,
                                                  fPlayer->GetInputList(), nent,
                                                  fst, dset, selec,
                                                  (dset ? dset->GetEntryList() : 0));
   // Title is the session identifier
   pqr->SetTitle(GetName());

   return pqr;
}

//______________________________________________________________________________
void TProofLite::SetQueryRunning(TProofQueryResult *pq)
{
   // Set query in running state.

   // Record current position in the log file at start
   fflush(fLogFileW);
   Int_t startlog = lseek(fileno(fLogFileW), (off_t) 0, SEEK_END);

   // Add some header to logs
   Printf(" ");
   Info("SetQueryRunning", "starting query: %d", pq->GetSeqNum());

   // Build the list of loaded PAR packages
   TString parlist = "";
   TIter nxp(fEnabledPackagesOnClient);
   TObjString *os= 0;
   while ((os = (TObjString *)nxp())) {
      if (parlist.Length() <= 0)
         parlist = os->GetName();
      else
         parlist += Form(";%s",os->GetName());
   }

   // Set in running state
   pq->SetRunning(startlog, parlist, GetParallel());

   // Bytes and CPU at start (we will calculate the differential at end)
   pq->SetProcessInfo(pq->GetEntries(), GetCpuTime(), GetBytesRead());
}

//______________________________________________________________________________
Long64_t TProofLite::DrawSelect(TDSet *dset, const char *varexp,
                                const char *selection, Option_t *option,
                                Long64_t nentries, Long64_t first)
{
   // Execute the specified drawing action on a data set (TDSet).
   // Event- or Entry-lists should be set in the data set object using
   // TDSet::SetEntryList.
   // Returns -1 in case of error or number of selected events otherwise.

   if (!IsValid()) return -1;

   // Make sure that asynchronous processing is not active
   if (!IsIdle()) {
      Info("DrawSelect","not idle, asynchronous Draw not supported");
      return -1;
   }
   TString opt(option);
   Int_t idx = opt.Index("ASYN", 0, TString::kIgnoreCase);
   if (idx != kNPOS)
      opt.Replace(idx,4,"");

   // Prepare the description string
   TString q;
   q.Form("draw:\"%s\"%s\"", varexp, selection);

   return Process(dset, q.Data(), opt, nentries, first);
}

//______________________________________________________________________________
Long64_t TProofLite::Process(TDSet *dset, const char *selector, Option_t *option,
                             Long64_t nentries, Long64_t first)
{
   // Process a data set (TDSet) using the specified selector (.C) file.
   // Entry- or event-lists should be set in the data set object using
   // TDSet::SetEntryList.
   // The return value is -1 in case of error and TSelector::GetStatus() in
   // in case of success.

   // For the time being cannot accept other queries if not idle, even if in async
   // mode; needs to set up an event handler to manage that

   // Resolve query mode
   fSync = (GetQueryMode(option) == kSync);
   if (!fSync) {
      Info("Process","asynchronous mode not yet supported in PROOF-Lite");
      return -1;
   }

   if (!IsIdle()) {
      // Notify submission
      Info("Process", "not idle: cannot accept queries");
      return -1;
   }

   // Cleanup old temporary datasets
   if (IsIdle() && fRunningDSets && fRunningDSets->GetSize() > 0) {
      fRunningDSets->SetOwner(kTRUE);
      fRunningDSets->Delete();
   }

   if (!IsValid() || !fQMgr || !fPlayer) {
      Error("Process", "invalid sesion or query-result manager undefined!");
      return -1;
   }

   // Make sure that all enabled workers get some work, unless stated
   // differently
   if (!fPlayer->GetInputList()->FindObject("PROOF_MaxSlavesPerNode"))
      SetParameter("PROOF_MaxSlavesPerNode", (Long_t)fNWorkers);

   Bool_t hasNoData = (!dset || (dset && dset->TestBit(TDSet::kEmpty))) ? kTRUE : kFALSE;

   // If just a name was given to identify the dataset, retrieve it from the
   // local files
   // Make sure the dataset contains the information needed
   if ((!hasNoData) && dset->GetListOfElements()->GetSize() == 0) {
      TString emsg;
      if (TProof::AssertDataSet(dset, fPlayer->GetInputList(), fDataSetManager, emsg) != 0) {
         Error("Process", "from AssertDataSet: %s", emsg.Data());
         return -1;
      }
      if (dset->GetListOfElements()->GetSize() == 0) {
         Error("Process", "no files to process!");
         return -1;
      }
   }

   TString selec(selector), varexp, selection, objname;
   // If a draw query, extract the relevant info
   if (selec.BeginsWith("draw:")) {
      TString ss(selector), s;
      Ssiz_t from = 0;
      ss.Tokenize(s, from, ":\"");
      // The expression is mandatory
      if (!ss.Tokenize(varexp, from, "\"")) {
         Error("Process", "draw query: badly formed expression: %s", selector);
         return -1;
      }
      // Check if a section is present
      ss.Tokenize(selection, from, "\"");
      // Decode now the expression
      if (fPlayer->GetDrawArgs(varexp, selection, option, selec, objname) != 0) {
         Error("Process", "draw query: error parsing arguments '%s', '%s', '%s'",
                          varexp.Data(), selection.Data(), option);
         return -1;
      }
   }

   // Create instance of query results (the data set is added after Process)
   TProofQueryResult *pq = MakeQueryResult(nentries, option, first, 0, selec);

   // If not a draw action add the query to the main list
   if (!(pq->IsDraw())) {
      if (fQMgr->Queries()) fQMgr->Queries()->Add(pq);
      // Also save it to queries dir
      fQMgr->SaveQuery(pq);
   }

   // Set the query number
   fSeqNum = pq->GetSeqNum();

   // Set in running state
   SetQueryRunning(pq);

   // Save to queries dir, if not standard draw
   if (!(pq->IsDraw()))
      fQMgr->SaveQuery(pq);
   else
      fQMgr->IncrementDrawQueries();

   // Start or reset the progress dialog
   if (!gROOT->IsBatch()) {
      Int_t dsz = (dset && dset->GetListOfElements()) ? dset->GetListOfElements()->GetSize() : -1;
      if (fProgressDialog &&
          !TestBit(kUsingSessionGui) && TestBit(kUseProgressDialog)) {
         if (!fProgressDialogStarted) {
            fProgressDialog->ExecPlugin(5, this, selec.Data(), dsz,
                                           first, nentries);
            fProgressDialogStarted = kTRUE;
         } else {
            ResetProgressDialog(selec.Data(), dsz, first, nentries);
         }
      }
      ResetBit(kUsingSessionGui);
   }

   // Add query results to the player lists
   if (!(pq->IsDraw()))
      fPlayer->AddQueryResult(pq);

   // Set query currently processed
   fPlayer->SetCurrentQuery(pq);

   // Make sure the unique query tag is available as TNamed object in the
   // input list so that it can be used in TSelectors for monitoring
   TNamed *qtag = (TNamed *) fPlayer->GetInputList()->FindObject("PROOF_QueryTag");
   if (qtag) {
      qtag->SetTitle(Form("%s:%s",pq->GetTitle(),pq->GetName()));
   } else {
      TObject *o = fPlayer->GetInputList()->FindObject("PROOF_QueryTag");
      if (o) fPlayer->GetInputList()->Remove(o);
      fPlayer->AddInput(new TNamed("PROOF_QueryTag",
                                   Form("%s:%s",pq->GetTitle(),pq->GetName())));
   }

   // Set PROOF to running state
   SetRunStatus(TProof::kRunning);

   // deactivate the default application interrupt handler
   // ctrl-c's will be forwarded to PROOF to stop the processing
   TSignalHandler *sh = 0;
   if (fSync) {
      if (gApplication)
         sh = gSystem->RemoveSignalHandler(gApplication->GetSignalHandler());
   }

   // Start the additional workers now if using fork-based startup
   TList *startedWorkers = 0;
   if (fForkStartup) {
      startedWorkers = new TList;
      startedWorkers->SetOwner(kFALSE);
      SetupWorkers(1, startedWorkers);
   }

   Long64_t rv = 0;
   if (!(pq->IsDraw())) {
      rv = fPlayer->Process(dset, selec, option, nentries, first);
   } else {
      rv = fPlayer->DrawSelect(dset, varexp, selection, option, nentries, first);
   }

   if (fSync) {

      // Terminate additional workers if using fork-based startup
      if (fForkStartup && startedWorkers) {
         RemoveWorkers(startedWorkers);
         SafeDelete(startedWorkers);
      }

      // reactivate the default application interrupt handler
      if (sh)
         gSystem->AddSignalHandler(sh);

      // Return number of events processed
      if (fPlayer->GetExitStatus() != TVirtualProofPlayer::kFinished) {
         Bool_t abort = (fPlayer->GetExitStatus() == TVirtualProofPlayer::kAborted)
                     ? kTRUE : kFALSE;
         if (abort) fPlayer->StopProcess(kTRUE);
         Emit("StopProcess(Bool_t)", abort);
      }

      // In PROOFLite this has to be done once only in TProofLite::Process
      pq->SetOutputList(fPlayer->GetOutputList(), kFALSE);
      // If the last object, notify the GUI that the result arrived
      QueryResultReady(Form("%s:%s", pq->GetTitle(), pq->GetName()));
      // Processing is over
      UpdateDialog();

      // Save the data set into the TQueryResult (should be done after Process to avoid
      // improper deletion during collection)
      if (rv == 0 && dset && pq->GetInputList()) {
         pq->GetInputList()->Add(dset);
         if (dset->GetEntryList())
            pq->GetInputList()->Add(dset->GetEntryList());
      }

      // Register any dataset produced during this processing, if required
      if (fDataSetManager && fPlayer->GetOutputList()) {
         TNamed *psr = (TNamed *) fPlayer->GetOutputList()->FindObject("PROOFSERV_RegisterDataSet");
         if (psr) {
            if (RegisterDataSets(fPlayer->GetInputList(), fPlayer->GetOutputList()) != 0)
               Warning("ProcessNext", "problems registering produced datasets");
            fPlayer->GetOutputList()->Remove(psr);
            delete psr;
         }
      }

      // Complete filling of the TQueryResult instance
      AskStatistics();
      if (!(pq->IsDraw())) {
         if (fQMgr->FinalizeQuery(pq, this, fPlayer)) {
            // Automatic saving is controlled by ProofLite.AutoSaveQueries
            if (!strcmp(gEnv->GetValue("ProofLite.AutoSaveQueries", "off"), "on"))
               fQMgr->SaveQuery(pq, -1);
         }
      }

      // Remove aborted queries from the list
      if (fPlayer && fPlayer->GetExitStatus() == TVirtualProofPlayer::kAborted) {
         if (fPlayer->GetListOfResults()) fPlayer->GetListOfResults()->Remove(pq);
         if (fQMgr) fQMgr->RemoveQuery(pq);
      } else {
         // If the last object, notify the GUI that the result arrived
         QueryResultReady(Form("%s:%s", pq->GetTitle(), pq->GetName()));
         // Keep in memory only light info about a query
         if (!(pq->IsDraw())) {
            if (fQMgr && fQMgr->Queries())
               // Remove from the fQueries list
               fQMgr->Queries()->Remove(pq);
         }
         // To get the prompt back
         TString msg;
         msg.Form("Lite-0: all output objects have been merged                                                         ");
         fprintf(stderr, "%s\n", msg.Data());
      }

   }

   // Done
   return rv;
}

//______________________________________________________________________________
Int_t TProofLite::CreateSymLinks(TList *files)
{
   // Create in each worker sandbox symlinks to the files in the list
   // Used to make the caceh information available to workers.

   Int_t rc = 0;
   if (files) {
      TIter nxf(files);
      TObjString *os = 0;
      while ((os = (TObjString *) nxf())) {
         // Expand target
         TString tgt(os->GetName());
         gSystem->ExpandPathName(tgt);
         // Loop over active workers
         TIter nxw(fActiveSlaves);
         TSlave *wrk = 0;
         while ((wrk = (TSlave *) nxw())) {
            // Link name
            TString lnk = Form("%s/%s", wrk->GetWorkDir(), gSystem->BaseName(os->GetName()));
            gSystem->Unlink(lnk);
            if (gSystem->Symlink(tgt, lnk) != 0) {
               rc++;
               Warning("CreateSymLinks", "problems creating sym link: %s", lnk.Data());
            }
         }
      }
   } else {
      Warning("CreateSymLinks", "files list is undefined");
   }
   // Done
   return rc;
}

//______________________________________________________________________________
Int_t TProofLite::InitDataSetManager()
{
   // Initialize the dataset manager from directives or from defaults
   // Return 0 on success, -1 on failure

   fDataSetManager = 0;

   // Default user and group
   TString user("???"), group("default");
   UserGroup_t *pw = gSystem->GetUserInfo();
   if (pw) {
      user = pw->fUser;
      delete pw;
   }

   // Dataset manager instance via plug-in
   TPluginHandler *h = 0;
   TString dsm = gEnv->GetValue("Proof.DataSetManager", "");
   if (!dsm.IsNull()) {
      // Get plugin manager to load the appropriate TDataSetManager
      if (gROOT->GetPluginManager()) {
         // Find the appropriate handler
         h = gROOT->GetPluginManager()->FindHandler("TDataSetManager", dsm);
         if (h && h->LoadPlugin() != -1) {
            // make instance of the dataset manager
            fDataSetManager =
               reinterpret_cast<TDataSetManager*>(h->ExecPlugin(3, group.Data(),
                                                         user.Data(), dsm.Data()));
         }
      }
   }
   if (fDataSetManager && fDataSetManager->TestBit(TObject::kInvalidObject)) {
      Warning("InitDataSetManager", "dataset manager plug-in initialization failed");
      SafeDelete(fDataSetManager);
   }

   // If no valid dataset manager has been created we instantiate the default one
   if (!fDataSetManager) {
      TString opts("Av:");
      TString dsetdir = gEnv->GetValue("ProofServ.DataSetDir", "");
      if (dsetdir.IsNull()) {
         // Use the default in the sandbox
         dsetdir = fDataSetDir;
         opts += "Sb:";
      }
      // Find the appropriate handler
      if (!h) {
         h = gROOT->GetPluginManager()->FindHandler("TDataSetManager", "file");
         if (h && h->LoadPlugin() == -1) h = 0;
      }
      if (h) {
         // make instance of the dataset manager
         fDataSetManager = reinterpret_cast<TDataSetManager*>(h->ExecPlugin(3,
                           group.Data(), user.Data(),
                           Form("dir:%s opt:%s", dsetdir.Data(), opts.Data())));
      }
      if (fDataSetManager && fDataSetManager->TestBit(TObject::kInvalidObject)) {
         Warning("InitDataSetManager", "default dataset manager plug-in initialization failed");
         SafeDelete(fDataSetManager);
      }
   }

   if (gDebug > 0 && fDataSetManager) {
      Info("InitDataSetManager", "datasetmgr Cq: %d, Ar: %d, Av: %d, Ti: %d, Sb: %d",
            fDataSetManager->TestBit(TDataSetManager::kCheckQuota),
            fDataSetManager->TestBit(TDataSetManager::kAllowRegister),
            fDataSetManager->TestBit(TDataSetManager::kAllowVerify),
            fDataSetManager->TestBit(TDataSetManager::kTrustInfo),
            fDataSetManager->TestBit(TDataSetManager::kIsSandbox));
   }

   // Done
   return (fDataSetManager ? 0 : -1);
}

//______________________________________________________________________________
void TProofLite::ShowCache(Bool_t)
{
   // List contents of file cache. If all is true show all caches also on
   // slaves. If everything is ok all caches are to be the same.

   if (!IsValid()) return;

   Printf("*** Local file cache %s ***", fCacheDir.Data());
   gSystem->Exec(Form("%s %s", kLS, fCacheDir.Data()));
}

//______________________________________________________________________________
void TProofLite::ClearCache(const char *file)
{
    // Remove files from all file caches.

   if (!IsValid()) return;

   fCacheLock->Lock();
   if (!file || strlen(file) <= 0) {
      gSystem->Exec(Form("%s %s/*", kRM, fCacheDir.Data()));
   } else {
      gSystem->Exec(Form("%s %s/%s", kRM, fCacheDir.Data(), file));
   }
   fCacheLock->Unlock();
}

//______________________________________________________________________________
Int_t TProofLite::Load(const char *macro, Bool_t notOnClient, Bool_t uniqueOnly,
                       TList *wrks)
{
   // Copy the specified macro in the cache directory. The macro file is
   // uploaded if new or updated. If existing, the corresponding header
   // basename(macro).h or .hh, is also uploaded. For the other arguments
   // see TProof::Load().
   // Returns 0 in case of success and -1 in case of error.

   if (!IsValid()) return -1;

   if (!macro || !strlen(macro)) {
      Error("Load", "need to specify a macro name");
      return -1;
   }

   TString macs(macro), mac;
   Int_t from = 0;
   while (macs.Tokenize(mac, from, ",")) {
      if (CopyMacroToCache(mac) < 0) return -1;
   }

   return TProof::Load(macro, notOnClient, uniqueOnly, wrks);
}

//______________________________________________________________________________
Int_t TProofLite::CopyMacroToCache(const char *macro, Int_t headerRequired,
                                   TSelector **selector, Int_t opt)
{
   // Copy a macro, and its possible associated .h[h] file,
   // to the cache directory, from where the workers can get the file.
   // If headerRequired is 1, return -1 in case the header is not found.
   // If headerRequired is 0, try to copy header too.
   // If headerRequired is -1, don't look for header, only copy macro.
   // If the selector pionter is not 0, consider the macro to be a selector
   // and try to load the selector and set it to the pointer.
   // The mask 'opt' is an or of ESendFileOpt:
   //       kCpBin   (0x8)     Retrieve from the cache the binaries associated
   //                          with the file
   //       kCp      (0x10)    Retrieve the files from the cache
   // Return -1 in case of error, 0 otherwise.

   // Relevant pointers
   TString cacheDir = fCacheDir;
   gSystem->ExpandPathName(cacheDir);
   TProofLockPath *cacheLock = fCacheLock;

   // Split out the aclic mode, if any
   TString name = macro;
   TString acmode, args, io;
   name = gSystem->SplitAclicMode(name, acmode, args, io);

   PDB(kGlobal,1)
      Info("CopyMacroToCache", "enter: names: %s, %s", macro, name.Data());

   // Make sure that the file exists
   if (gSystem->AccessPathName(name, kReadPermission)) {
      Error("CopyMacroToCache", "file %s not found or not readable", name.Data());
      return -1;
   }

   // Update the macro path
   TString mp(TROOT::GetMacroPath());
   TString np(gSystem->DirName(name));
   if (!np.IsNull()) {
      np += ":";
      if (!mp.BeginsWith(np) && !mp.Contains(":"+np)) {
         Int_t ip = (mp.BeginsWith(".:")) ? 2 : 0;
         mp.Insert(ip, np);
         TROOT::SetMacroPath(mp);
         if (gDebug > 0)
            Info("CopyMacroToCache", "macro path set to '%s'", TROOT::GetMacroPath());
      }
   }

   // Check the header file
   Int_t dot = name.Last('.');
   const char *hext[] = { ".h", ".hh", "" };
   TString hname, checkedext;
   Int_t i = 0;
   while (strlen(hext[i]) > 0) {
      hname = name(0, dot);
      hname += hext[i];
      if (!gSystem->AccessPathName(hname, kReadPermission))
         break;
      if (!checkedext.IsNull()) checkedext += ",";
      checkedext += hext[i];
      hname = "";
      i++;
   }
   if (hname.IsNull() && headerRequired == 1) {
      Error("CopyMacroToCache", "header file for %s not found or not readable "
            "(checked extensions: %s)", name.Data(), checkedext.Data());
      return -1;
   }
   if (headerRequired < 0)
      hname = "";

   cacheLock->Lock();

   // Check these files with those in the cache (if any)
   Bool_t useCacheBinaries = kFALSE;
   TString cachedname = Form("%s/%s", cacheDir.Data(), gSystem->BaseName(name));
   TString cachedhname;
   if (!hname.IsNull())
      cachedhname = Form("%s/%s", cacheDir.Data(), gSystem->BaseName(hname));
   if (!gSystem->AccessPathName(cachedname, kReadPermission)) {
      TMD5 *md5 = TMD5::FileChecksum(name);
      TMD5 *md5cache = TMD5::FileChecksum(cachedname);
      if (md5 && md5cache && (*md5 == *md5cache))
         useCacheBinaries = kTRUE;
      if (!hname.IsNull()) {
         if (!gSystem->AccessPathName(cachedhname, kReadPermission)) {
            TMD5 *md5h = TMD5::FileChecksum(hname);
            TMD5 *md5hcache = TMD5::FileChecksum(cachedhname);
            if (md5h && md5hcache && (*md5h != *md5hcache))
               useCacheBinaries = kFALSE;
            SafeDelete(md5h);
            SafeDelete(md5hcache);
         }
      }
      SafeDelete(md5);
      SafeDelete(md5cache);
   }

   // Create version file name template
   TString vername(Form(".%s", name.Data()));
   dot = vername.Last('.');
   if (dot != kNPOS)
      vername.Remove(dot);
   vername += ".binversion";
   Bool_t savever = kFALSE;

   // Check binary version
   if (useCacheBinaries) {
      TString v;
      Int_t rev = -1;
      FILE *f = fopen(Form("%s/%s", cacheDir.Data(), vername.Data()), "r");
      if (f) {
         TString r;
         v.Gets(f);
         r.Gets(f);
         rev = (!r.IsNull() && r.IsDigit()) ? r.Atoi() : -1;
         fclose(f);
      }
      if (!f || v != gROOT->GetVersion() ||
          (gROOT->GetSvnRevision() > 0 && rev != gROOT->GetSvnRevision()))
         useCacheBinaries = kFALSE;
   }

   // Create binary name template
   TString binname = gSystem->BaseName(name);
   dot = binname.Last('.');
   if (dot != kNPOS)
      binname.Replace(dot,1,"_");
   binname += ".";

   FileStat_t stlocal, stcache;
   void *dirp = 0;
   if (useCacheBinaries) {
      // Loop over binaries in the cache and copy them locally if newer then the local
      // versions or there is no local version
      dirp = gSystem->OpenDirectory(cacheDir);
      if (dirp) {
         const char *e = 0;
         while ((e = gSystem->GetDirEntry(dirp))) {
            if (!strncmp(e, binname.Data(), binname.Length())) {
               TString fncache = Form("%s/%s", cacheDir.Data(), e);
               Bool_t docp = kTRUE;
               if (!gSystem->GetPathInfo(fncache, stcache)) {
                  Int_t rc = gSystem->GetPathInfo(e, stlocal);
                  if (rc == 0 && (stlocal.fMtime >= stcache.fMtime))
                     docp = kFALSE;
                  // Copy the file, if needed
                  if (docp) {
                     gSystem->Exec(Form("%s %s", kRM, e));
                     PDB(kGlobal,2)
                        Info("CopyMacroToCache",
                             "retrieving %s from cache", fncache.Data());
                     gSystem->Exec(Form("%s %s %s", kCP, fncache.Data(), e));
                  }
               }
            }
         }
         gSystem->FreeDirectory(dirp);
      }
   }
   cacheLock->Unlock();

   if (selector) {
      // Now init the selector in optimized way
      if (!(*selector = TSelector::GetSelector(macro))) {
         Error("CopyMacroToCache", "could not create a selector from %s", macro);
         return -1;
      }
   }

   cacheLock->Lock();

   TList *cachedFiles = new TList;
   // Save information in the cache now for later usage
   dirp = gSystem->OpenDirectory(".");
   if (dirp) {
      const char *e = 0;
      while ((e = gSystem->GetDirEntry(dirp))) {
         if (!strncmp(e, binname.Data(), binname.Length())) {
            Bool_t docp = kTRUE;
            if (!gSystem->GetPathInfo(e, stlocal)) {
               TString fncache = Form("%s/%s", cacheDir.Data(), e);
               Int_t rc = gSystem->GetPathInfo(fncache, stcache);
               if (rc == 0 && (stlocal.fMtime <= stcache.fMtime))
                  docp = kFALSE;
               // Copy the file, if needed
               if (docp) {
                  gSystem->Exec(Form("%s %s", kRM, fncache.Data()));
                  PDB(kGlobal,2)
                     Info("CopyMacroToCache","caching %s ...", e);
                  gSystem->Exec(Form("%s %s %s", kCP, e, fncache.Data()));
                  savever = kTRUE;
               }
               if (opt & kCpBin)
                  cachedFiles->Add(new TObjString(fncache.Data()));
            }
         }
      }
      gSystem->FreeDirectory(dirp);
   }

   // Save binary version if requested
   if (savever) {
      FILE *f = fopen(Form("%s/%s", cacheDir.Data(), vername.Data()), "w");
      if (f) {
         fputs(gROOT->GetVersion(), f);
         fputs(Form("\n%d",gROOT->GetSvnRevision()), f);
         fclose(f);
      }
   }

   // Save also the selector info, if needed
   if (!useCacheBinaries) {
      gSystem->Exec(Form("%s %s", kRM, cachedname.Data()));
      PDB(kGlobal,2)
         Info("CopyMacroToCache","caching %s ...", name.Data());
      gSystem->Exec(Form("%s %s %s", kCP, name.Data(), cachedname.Data()));
      if (!hname.IsNull()) {
         gSystem->Exec(Form("%s %s", kRM, cachedhname.Data()));
         PDB(kGlobal,2)
            Info("CopyMacroToCache","caching %s ...", hname.Data());
         gSystem->Exec(Form("%s %s %s", kCP, hname.Data(), cachedhname.Data()));
      }
   }
   if (opt & kCp) {
      cachedFiles->Add(new TObjString(cachedname.Data()));
      if (!hname.IsNull())
         cachedFiles->Add(new TObjString(cachedhname.Data()));
   }

   cacheLock->Unlock();

   // Create symlinks
   if (opt & (kCp | kCpBin))
      CreateSymLinks(cachedFiles);

   cachedFiles->SetOwner();
   delete cachedFiles;

   return 0;
}

//______________________________________________________________________________
Int_t TProofLite::CleanupSandbox()
{
   // Remove old sessions dirs keep at most 'Proof.MaxOldSessions' (default 10)

   Int_t maxold = gEnv->GetValue("Proof.MaxOldSessions", 10);

   if (maxold < 0) return 0;

   TSortedList *olddirs = new TSortedList(kFALSE);

   TString sandbox = gSystem->DirName(fWorkDir.Data());

   void *dirp = gSystem->OpenDirectory(sandbox);
   if (dirp) {
      const char *e = 0;
      while ((e = gSystem->GetDirEntry(dirp))) {
         if (!strncmp(e, "session-", 8) && !strstr(e, GetName())) {
            TString d(e);
            Int_t i = d.Last('-');
            if (i != kNPOS) d.Remove(i);
            i = d.Last('-');
            if (i != kNPOS) d.Remove(0,i+1);
            TString path = Form("%s/%s", sandbox.Data(), e);
            olddirs->Add(new TNamed(d, path));
         }
      }
      gSystem->FreeDirectory(dirp);
   }

   // Clean it up, if required
   Bool_t notify = kTRUE;
   while (olddirs->GetSize() > maxold) {
      if (notify && gDebug > 0)
         Printf("Cleaning sandbox at: %s", sandbox.Data());
      notify = kFALSE;
      TNamed *n = (TNamed *) olddirs->Last();
      if (n) {
         gSystem->Exec(Form("%s %s", kRM, n->GetTitle()));
         olddirs->Remove(n);
         delete n;
      }
   }

   // Cleanup
   olddirs->SetOwner();
   delete olddirs;

   // Done
   return 0;
}

//______________________________________________________________________________
TList *TProofLite::GetListOfQueries(Option_t *opt)
{
   // Get the list of queries.

   Bool_t all = ((strchr(opt,'A') || strchr(opt,'a'))) ? kTRUE : kFALSE;

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
         // Gather also information about previous queries, if any
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

   fOtherQueries = npre;
   fDrawQueries = ndraw;
   if (fQueries) {
      fQueries->Delete();
      delete fQueries;
      fQueries = 0;
   }
   fQueries = ql;

   // This should have been filled by now
   return fQueries;
}

//______________________________________________________________________________
Bool_t TProofLite::RegisterDataSet(const char *uri,
                                   TFileCollection *dataSet, const char* optStr)
{
   // Register the 'dataSet' on the cluster under the current
   // user, group and the given 'dataSetName'.
   // Fails if a dataset named 'dataSetName' already exists, unless 'optStr'
   // contains 'O', in which case the old dataset is overwritten.
   // If 'optStr' contains 'V' the dataset files are verified (default no
   // verification).
   // Returns kTRUE on success.

   if (!fDataSetManager) {
      Info("RegisterDataSet", "dataset manager not available");
      return kFALSE;
   }

   if (!uri || strlen(uri) <= 0) {
      Info("RegisterDataSet", "specifying a dataset name is mandatory");
      return kFALSE;
   }

   Bool_t result = kTRUE;
   if (fDataSetManager->TestBit(TDataSetManager::kAllowRegister)) {
      // Check the list
      if (!dataSet || dataSet->GetList()->GetSize() == 0) {
         Error("RegisterDataSet", "can not save an empty list.");
         result = kFALSE;
      }
      // Register the dataset (quota checks are done inside here)
      result = (fDataSetManager->RegisterDataSet(uri, dataSet, optStr) == 0)
             ? kTRUE : kFALSE;
   } else {
      Info("RegisterDataSets", "dataset registration not allowed");
      result = kFALSE;
   }

   if (!result)
      Error("RegisterDataSet", "dataset was not saved");

   // Done
   return result;
}

//______________________________________________________________________________
Int_t TProofLite::SetDataSetTreeName(const char *dataset, const char *treename)
{
   // Set/Change the name of the default tree. The tree name may contain
   // subdir specification in the form "subdir/name".
   // Returns 0 on success, -1 otherwise.

   if (!fDataSetManager) {
      Info("ExistsDataSet", "dataset manager not available");
      return kFALSE;
   }

   if (!dataset || strlen(dataset) <= 0) {
      Info("SetDataSetTreeName", "specifying a dataset name is mandatory");
      return -1;
   }

   if (!treename || strlen(treename) <= 0) {
      Info("SetDataSetTreeName", "specifying a tree name is mandatory");
      return -1;
   }

   TUri uri(dataset);
   TString fragment(treename);
   if (!fragment.BeginsWith("/")) fragment.Insert(0, "/");
   uri.SetFragment(fragment);

   return fDataSetManager->ScanDataSet(uri.GetUri().Data(),
                                      (UInt_t)TDataSetManager::kSetDefaultTree);
}

//______________________________________________________________________________
Bool_t TProofLite::ExistsDataSet(const char *uri)
{
   // Returns kTRUE if 'dataset' described by 'uri' exists, kFALSE otherwise

   if (!fDataSetManager) {
      Info("ExistsDataSet", "dataset manager not available");
      return kFALSE;
   }

   if (!uri || strlen(uri) <= 0) {
      Error("ExistsDataSet", "dataset name missing");
      return kFALSE;
   }

   // Check if the dataset exists
   return fDataSetManager->ExistsDataSet(uri);
}

//______________________________________________________________________________
TMap *TProofLite::GetDataSets(const char *uri, const char *srvex)
{
   // lists all datasets that match given uri

   if (!fDataSetManager) {
      Info("GetDataSets", "dataset manager not available");
      return (TMap *)0;
   }

   // Get the datasets and return the map
   if (srvex && strlen(srvex) > 0) {
      return fDataSetManager->GetSubDataSets(uri, srvex);
   } else {
      UInt_t opt = (UInt_t)TDataSetManager::kExport;
      return fDataSetManager->GetDataSets(uri, opt);
   }
}

//______________________________________________________________________________
void TProofLite::ShowDataSets(const char *uri, const char *opt)
{
   // Shows datasets in locations that match the uri
   // By default shows the user's datasets and global ones

   if (!fDataSetManager) {
      Info("GetDataSet", "dataset manager not available");
      return;
   }

   fDataSetManager->ShowDataSets(uri, opt);
}

//______________________________________________________________________________
TFileCollection *TProofLite::GetDataSet(const char *uri, const char *)
{
   // Get a list of TFileInfo objects describing the files of the specified
   // dataset.

   if (!fDataSetManager) {
      Info("GetDataSet", "dataset manager not available");
      return (TFileCollection *)0;
   }

   if (!uri || strlen(uri) <= 0) {
      Info("GetDataSet", "specifying a dataset name is mandatory");
      return 0;
   }

   // Return the list
   return fDataSetManager->GetDataSet(uri);
}

//______________________________________________________________________________
Int_t TProofLite::RemoveDataSet(const char *uri, const char *)
{
   // Remove the specified dataset from the PROOF cluster.
   // Files are not deleted.

   if (!fDataSetManager) {
      Info("RemoveDataSet", "dataset manager not available");
      return -1;
   }

   if (fDataSetManager->TestBit(TDataSetManager::kAllowRegister)) {
      if (!fDataSetManager->RemoveDataSet(uri)) {
         // Failure
         return -1;
      }
   } else {
      Info("RemoveDataSet", "dataset creation / removal not allowed");
      return -1;
   }

   // Done
   return 0;
}

//______________________________________________________________________________
Int_t TProofLite::VerifyDataSet(const char *uri, const char *)
{
   // Verify if all files in the specified dataset are available.
   // Print a list and return the number of missing files.

   if (!fDataSetManager) {
      Info("VerifyDataSet", "dataset manager not available");
      return -1;
   }

   Int_t rc = -1;
   if (fDataSetManager->TestBit(TDataSetManager::kAllowVerify)) {
      rc = fDataSetManager->ScanDataSet(uri);
   } else {
      Info("VerifyDataSet", "dataset verification not allowed");
      return -1;
   }

   // Done
   return rc;
}

//______________________________________________________________________________
void TProofLite::ClearDataSetCache(const char *dataset)
{
   // Clear the content of the dataset cache, if any (matching 'dataset', if defined).

   if (fDataSetManager) fDataSetManager->ClearCache(dataset);
   // Done
   return;
}

//______________________________________________________________________________
void TProofLite::ShowDataSetCache(const char *dataset)
{
   // Display the content of the dataset cache, if any (matching 'dataset', if defined).

   // For PROOF-Lite act locally
   if (fDataSetManager) fDataSetManager->ShowCache(dataset);
   // Done
   return;
}

//______________________________________________________________________________
void TProofLite::SendInputDataFile()
{
   // Make sure that the input data objects are available to the workers in a
   // dedicated file in the cache; the objects are taken from the dedicated list
   // and / or the specified file.
   // If the fInputData is empty the specified file is sent over.
   // If there is no specified file, a file named "inputdata.root" is created locally
   // with the content of fInputData and sent over to the master.
   // If both fInputData and the specified file are not empty, a copy of the file
   // is made locally and augmented with the content of fInputData.

   // Prepare the file
   TString dataFile;
   PrepareInputDataFile(dataFile);

   // Make sure it is in the cache, if not empty
   if (dataFile.Length() > 0) {

      if (!dataFile.BeginsWith(fCacheDir)) {
         // Destination
         TString dst;
         dst.Form("%s/%s", fCacheDir.Data(), gSystem->BaseName(dataFile));
         // Remove it first if it exists
         if (!gSystem->AccessPathName(dst))
            gSystem->Unlink(dst);
         // Copy the file
         gSystem->CopyFile(dataFile, dst);
      }

      // Set the name in the input list so that the workers can find it
      AddInput(new TNamed("PROOF_InputDataFile", Form("%s", gSystem->BaseName(dataFile))));
   }
}

//______________________________________________________________________________
Int_t TProofLite::Remove(const char *ref, Bool_t all)
{
   // Handle remove request.

   PDB(kGlobal, 1)
      Info("Remove", "Enter: %s, %d", ref, all);

   if (all) {
      // Remove also local copies, if any
      if (fPlayer)
         fPlayer->RemoveQueryResult(ref);
   }

   TString queryref(ref);

   if (queryref == "cleanupdir") {

      // Cleanup previous sessions results
      Int_t nd = (fQMgr) ? fQMgr->CleanupQueriesDir() : -1;

      // Notify
      Info("Remove", "%d directories removed", nd);
      // We are done
      return 0;
   }


   if (fQMgr) {
      TProofLockPath *lck = 0;
      if (fQMgr->LockSession(queryref, &lck) == 0) {

         // Remove query
         fQMgr->RemoveQuery(queryref, 0);

         // Unlock and remove the lock file
         if (lck) {
            gSystem->Unlink(lck->GetName());
            SafeDelete(lck);
         }

         // We are done
         return 0;
      }
   } else {
      Warning("Remove", "query result manager undefined!");
   }

   // Notify failure
   Info("Remove",
        "query %s could not be removed (unable to lock session)", queryref.Data());

   // Done
   return -1;
}

//______________________________________________________________________________
TTree *TProofLite::GetTreeHeader(TDSet *dset)
{
   // Creates a tree header (a tree with nonexisting files) object for
   // the DataSet.

   TTree *t = 0;
   if (!dset) {
      Error("GetTreeHeader", "undefined TDSet");
      return t;
   }

   dset->Reset();
   TDSetElement *e = dset->Next();
   Long64_t entries = 0;
   TFile *f = 0;
   if (!e) {
      PDB(kGlobal, 1) Info("GetTreeHeader", "empty TDSet");
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
                  TTree *t1 = (TTree*) f1->Get(e->GetObjName());
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
   // Done
   return t;
}

//______________________________________________________________________________
void TProofLite::FindUniqueSlaves()
{
   // Add to the fUniqueSlave list the active slaves that have a unique
   // (user) file system image. This information is used to transfer files
   // only once to nodes that share a file system (an image). Submasters
   // which are not in fUniqueSlaves are put in the fNonUniqueMasters
   // list. That list is used to trigger the transferring of files to
   // the submaster's unique slaves without the need to transfer the file
   // to the submaster.

   fUniqueSlaves->Clear();
   fUniqueMonitor->RemoveAll();
   fAllUniqueSlaves->Clear();
   fAllUniqueMonitor->RemoveAll();
   fNonUniqueMasters->Clear();

   if (fActiveSlaves->GetSize() <= 0) return;

   TSlave *wrk = dynamic_cast<TSlave*>(fActiveSlaves->First());
   if (!wrk) {
      Error("FindUniqueSlaves", "first object in fActiveSlaves not a TSlave: embarrasing!");
      return;
   }
   fUniqueSlaves->Add(wrk);
   fAllUniqueSlaves->Add(wrk);
   fUniqueMonitor->Add(wrk->GetSocket());
   fAllUniqueMonitor->Add(wrk->GetSocket());

   // will be actiavted in Collect()
   fUniqueMonitor->DeActivateAll();
   fAllUniqueMonitor->DeActivateAll();
}

//______________________________________________________________________________
Int_t TProofLite::RegisterDataSets(TList *in, TList *out)
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
                  Info("RegisterDataSets", "%s", msg.Data());
                  // Always allow verification for this action
                  Bool_t allowVerify = fDataSetManager->TestBit(TDataSetManager::kAllowVerify) ? kTRUE : kFALSE;
                  if (regopt.Contains("V") && !allowVerify)
                     fDataSetManager->SetBit(TDataSetManager::kAllowVerify);
                  Int_t rc = fDataSetManager->RegisterDataSet(ds->GetName(), ds, regopt);
                  // Reset to the previous state if needed
                  if (regopt.Contains("V") && !allowVerify)
                     fDataSetManager->ResetBit(TDataSetManager::kAllowVerify);
                  if (rc != 0) {
                     Warning("RegisterDataSets",
                              "failure registering dataset '%s'", ds->GetName());
                     msg.Form("Registering and verifying dataset '%s' ... failed! See log for more details", ds->GetName());
                  } else {
                     Info("RegisterDataSets", "dataset '%s' successfully registered", ds->GetName());
                     msg.Form("Registering and verifying dataset '%s' ... OK", ds->GetName());
                  }
                  Info("RegisterDataSets", "%s", msg.Data());
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
