// @(#)root/proof:$Id: 7735e42a1b96a9f40ae76bd884acac883a178dee $
// Author: G. Ganis March 2008

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TProofLite
\ingroup proofkernel

This class starts a PROOF session on the local machine: no daemons,
client and master merged, communications via UNIX-like sockets.
By default the number of workers started is NumberOfCores+1; a
different number can be forced on construction.

*/

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
#include "TDataSetManagerFile.h"
#include "TParameter.h"
#include "TPRegexp.h"
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
#include "TPackMgr.h"

ClassImp(TProofLite);

Int_t TProofLite::fgWrksMax = -2; // Unitialized max number of workers

////////////////////////////////////////////////////////////////////////////////
/// Create a PROOF environment. Starting PROOF involves either connecting
/// to a master server, which in turn will start a set of slave servers, or
/// directly starting as master server (if master = ""). Masterurl is of
/// the form: [proof[s]://]host[:port]. Conffile is the name of the config
/// file describing the remote PROOF cluster (this argument alows you to
/// describe different cluster configurations).
/// The default is proof.conf. Confdir is the directory where the config
/// file and other PROOF related files are (like motd and noproof files).
/// Loglevel is the log level (default = 1). User specified custom config
/// files will be first looked for in $HOME/.conffile.

TProofLite::TProofLite(const char *url, const char *conffile, const char *confdir,
                       Int_t loglevel, const char *alias, TProofMgr *mgr)
{
   fUrl.SetUrl(url);

   // Default initializations
   fServSock = 0;
   fCacheLock = 0;
   fQueryLock = 0;
   fQMgr = 0;
   fDataSetManager = 0;
   fDataSetStgRepo = 0;
   fReInvalid = new TPMERegexp("[^A-Za-z0-9._-]");
   InitMembers();

   // This may be needed during init
   fManager = mgr;

   // Default server type
   fServType = TProofMgr::kProofLite;

   // Default query mode
   fQueryMode = kSync;

   // Client and master are merged
   fMasterServ = kTRUE;
   if (fManager) SetBit(TProof::kIsClient);
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

      TString stup;
      if (gProofServ) {
         Int_t port = gEnv->GetValue("ProofServ.XpdPort", 1093);
         stup.Form("%s @ %s:%d ", gProofServ->GetOrdinal(), gSystem->HostName(), port);
      }
      Printf(" +++ Starting PROOF-Lite %swith %d workers +++", stup.Data(), fNWorkers);
      // Init the session now
      Init(url, conffile, confdir, loglevel, alias);
   }

   // For final cleanup
   if (!gROOT->GetListOfProofs()->FindObject(this))
      gROOT->GetListOfProofs()->Add(this);

   // Still needed by the packetizers: needs to be changed
   gProof = this;
}

////////////////////////////////////////////////////////////////////////////////
/// Start the PROOF environment. Starting PROOF involves either connecting
/// to a master server, which in turn will start a set of slave servers, or
/// directly starting as master server (if master = ""). For a description
/// of the arguments see the TProof ctor. Returns the number of started
/// master or slave servers, returns 0 in case of error, in which case
/// fValid remains false.

Int_t TProofLite::Init(const char *, const char *conffile,
                       const char *confdir, Int_t loglevel, const char *)
{
   R__ASSERT(gSystem);

   fValid = kFALSE;

   // Connected to terminal?
   fTty = (isatty(0) == 0 || isatty(1) == 0) ? kFALSE : kTRUE;

   if (TestBit(TProof::kIsMaster)) {
      // Fill default conf file and conf dir
      if (!conffile || !conffile[0])
         fConfFile = kPROOF_ConfFile;
      if (!confdir  || !confdir[0])
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

   fEnabledPackagesOnCluster = new TList;
   fEnabledPackagesOnCluster->SetOwner();

   // Timeout for some collect actions
   fCollectTimeout = gEnv->GetValue("Proof.CollectTimeout", -1);

   // Should the workers be started dynamically; default: no
   fDynamicStartup = kFALSE;
   fDynamicStartupStep = -1;
   fDynamicStartupNMax = -1;
   TString dynconf = gEnv->GetValue("Proof.SimulateDynamicStartup", "");
   if (dynconf.Length() > 0) {
      fDynamicStartup = kTRUE;
      fLastPollWorkers_s = time(0);
      // Extract parameters
      Int_t from = 0;
      TString p;
      if (dynconf.Tokenize(p, from, ":"))
         if (p.IsDigit()) fDynamicStartupStep = p.Atoi();
      if (dynconf.Tokenize(p, from, ":"))
         if (p.IsDigit()) fDynamicStartupNMax = p.Atoi();
   }


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
   Int_t maxq = gEnv->GetValue("ProofLite.MaxQueriesSaved", 10);
   if (fQMgr && fQMgr->ApplyMaxQueries(maxq) != 0)
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

   fTerminatedSlaveInfos = new TList;
   fTerminatedSlaveInfos->SetOwner(kTRUE);

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

   fLoadedMacros            = 0;
   if (TestBit(TProof::kIsClient)) {

      // List of directories where to look for global packages
      TString globpack = gEnv->GetValue("Proof.GlobalPackageDirs","");
      TProofServ::ResolveKeywords(globpack);
      Int_t nglb = TPackMgr::RegisterGlobalPath(globpack);
      if (gDebug > 0)
         Info("Init", " %d global package directories registered", nglb);
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
   GoParallel(-1, kFALSE);

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
   R__LOCKGUARD(gROOTMutex);
   gROOT->GetListOfSockets()->Add(this);

   AskParallel();

   return fActiveSlaves->GetSize();
}
////////////////////////////////////////////////////////////////////////////////
/// Destructor

TProofLite::~TProofLite()
{
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

   SafeDelete(fReInvalid);
   SafeDelete(fDataSetManager);
   SafeDelete(fDataSetStgRepo);

   // Cleanup the socket
   SafeDelete(fServSock);
   gSystem->Unlink(fSockPath);
}

////////////////////////////////////////////////////////////////////////////////
/// Static method to determine the number of workers giving priority to users request.
/// Otherwise use the system information, if available, or just start
/// the minimal number, i.e. 2 .

Int_t TProofLite::GetNumberOfWorkers(const char *url)
{
   Bool_t notify = kFALSE;
   if (fgWrksMax == -2) {
      // Find the max number of workers, if any
      TString sysname = "system.rootrc";
      char *s = gSystem->ConcatFileName(TROOT::GetEtcDir(), sysname);
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

   TString nw;
   Int_t nWorkers = -1;
   Bool_t urlSetting = kFALSE;
   if (url && strlen(url)) {
      nw = url;
      Int_t in = nw.Index("workers=");
      if (in != kNPOS) {
         nw.Remove(0, in + strlen("workers="));
         while (!nw.IsDigit())
            nw.Remove(nw.Length()-1);
         if (!nw.IsNull()) {
            if ((nWorkers = nw.Atoi()) <= 0) {
               ::Warning("TProofLite::GetNumberOfWorkers",
                         "number of workers specified by 'workers='"
                         " is non-positive: using default");
            } else {
               urlSetting = kFALSE;
            }
         }
      }
   }
   if (!urlSetting && fgProofEnvList) {
      // Check PROOF_NWORKERS
      TNamed *nm = (TNamed *) fgProofEnvList->FindObject("PROOF_NWORKERS");
      if (nm) {
         nw = nm->GetTitle();
         if (nw.IsDigit()) {
            if ((nWorkers = nw.Atoi()) == 0) {
               ::Warning("TProofLite::GetNumberOfWorkers",
                         "number of workers specified by 'workers='"
                         " is non-positive: using default");
            }
         }
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

////////////////////////////////////////////////////////////////////////////////
/// Start up PROOF workers.

Int_t TProofLite::SetupWorkers(Int_t opt, TList *startedWorkers)
{
   // Create server socket on the assigned UNIX sock path
   if (!fServSock) {
      if ((fServSock = new TServerSocket(fSockPath))) {
         R__LOCKGUARD(gROOTMutex);
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
         const char *o = (gProofServ) ? gProofServ->GetOrdinal() : "0";
         fullord.Form("%s.%d", o, ord);

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
         const char *o = (gProofServ) ? gProofServ->GetOrdinal() : "0";
         fullord.Form("%s.%d", o, ord + 1);
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
         if (s->Recv(msg) < 0) {
            Warning("SetupWorkers", "problems receiving message from accepted socket!");
         } else {
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
                  {  R__LOCKGUARD(gROOTMutex);
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
                  fSlaves->Add(wrk);
                  if (wrk->IsValid()) {
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
            } else {
               Warning("SetupWorkers", "received empty message from accepted socket!");
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
      SetParallel(-1, 0);
   }
   // Done
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Notify setting-up operation message

void TProofLite::NotifyStartUp(const char *action, Int_t done, Int_t tot)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Create environment files for worker 'ord'

Int_t TProofLite::SetProofServEnv(const char *ord)
{
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
   TString sandbox = fSandbox;
   if (GetSandbox(sandbox, kFALSE, "ProofServ.Sandbox") != 0)
      Warning("SetProofServEnv", "problems getting sandbox string for worker");
   fprintf(frc,"# Users sandbox\n");
   fprintf(frc, "ProofServ.Sandbox: %s\n", sandbox.Data());

   // Cache dir
   fprintf(frc,"# Users cache\n");
   fprintf(frc, "ProofServ.CacheDir: %s\n", fCacheDir.Data());

   // Package dir
   fprintf(frc,"# Users packages\n");
   fprintf(frc, "ProofServ.PackageDir: %s\n", fPackMgr->GetDir());

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
   fprintf(fenv, "export ROOTSYS=%s\n", TROOT::GetRootSys().Data());
   // Conf dir
   fprintf(fenv, "export ROOTCONFDIR=%s\n", TROOT::GetRootSys().Data());
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
   // Local files are on the local file system
   fprintf(fenv, "export LOCALDATASERVER=\"file://\"\n");
   // Set the user envs
   if (fgProofEnvList) {
      TString namelist;
      TIter nxenv(fgProofEnvList);
      TNamed *env = 0;
      while ((env = (TNamed *)nxenv())) {
         TString senv(env->GetTitle());
         ResolveKeywords(senv, ord, logfile.Data());
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

////////////////////////////////////////////////////////////////////////////////
/// Resolve some keywords in 's'
///    <logfilewrk>, <user>, <rootsys>, <cpupin>

void TProofLite::ResolveKeywords(TString &s, const char *ord,
   const char *logfile)
{
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

   // cpupin: pin to this CPU num (from 0 to ncpus-1)
   if (s.Contains("<cpupin>")) {
      TString o = ord;
      Int_t n = o.Index('.');
      if (n != kNPOS) {

         o.Remove(0, n+1);
         n = o.Atoi();  // n is ord

         TString cpuPinList;
         {
            const TList *envVars = GetEnvVars();
            TNamed *var;
            if (envVars) {
               var = dynamic_cast<TNamed *>(envVars->FindObject("PROOF_SLAVE_CPUPIN_ORDER"));
               if (var) cpuPinList = var->GetTitle();
            }
         }

         UInt_t nCpus = 1;
         {
            SysInfo_t si;
            if (gSystem->GetSysInfo(&si) == 0 && (si.fCpus > 0))
               nCpus = si.fCpus;
            else nCpus = 1;  // fallback
         }

         if (cpuPinList.IsNull() || (cpuPinList == "*")) {
            // Use processors in order
            n = n % nCpus;
         }
         else {
            // Use processors in user's order
            // n is now the ordinal, converting to idx
            n = n % (cpuPinList.CountChar('+')+1);
            TString tok;
            Ssiz_t from = 0;
            for (Int_t i=0; cpuPinList.Tokenize(tok, from, "\\+"); i++) {
               if (i == n) {
                  n = (tok.Atoi() % nCpus);
                  break;
               }
            }
         }

         o.Form("%d", n);
      }
      else {
         o = "0";  // should not happen
      }
      s.ReplaceAll("<cpupin>", o);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Create the sandbox for this session

Int_t TProofLite::CreateSandbox()
{
   // Make sure the sandbox area exist and is writable
   if (GetSandbox(fSandbox, kTRUE, "ProofLite.Sandbox") != 0) return -1;

   // Package Manager
   TString packdir = gEnv->GetValue("Proof.PackageDir", "");
   if (packdir.IsNull())
      packdir.Form("%s/%s", fSandbox.Data(), kPROOF_PackDir);
   if (AssertPath(packdir, kTRUE) != 0) return -1;
   fPackMgr = new TPackMgr(packdir);

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

   Int_t subpath = gEnv->GetValue("ProofLite.SubPath", 1);
   // Subpath for this session in the fSandbox (<sandbox>/path-to-working-dir)
   TString sessdir;
   if (subpath != 0) {
      sessdir = gSystem->WorkingDirectory();
      sessdir.ReplaceAll(gSystem->HomeDirectory(),"");
      sessdir.ReplaceAll("/","-");
      sessdir.Replace(0,1,"/",1);
      sessdir.Insert(0, fSandbox.Data());
   } else {
      // USe the sandbox
      sessdir = fSandbox;
   }

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

////////////////////////////////////////////////////////////////////////////////
/// Print status of PROOF-Lite cluster.

void TProofLite::Print(Option_t *option) const
{
   TString ord;
   if (gProofServ) ord.Form("%s ", gProofServ->GetOrdinal());
   if (IsParallel())
      Printf("*** PROOF-Lite cluster %s(parallel mode, %d workers):", ord.Data(), GetParallel());
   else
      Printf("*** PROOF-Lite cluster %s(sequential mode)", ord.Data());

   if (gProofServ) {
      TString url(gSystem->HostName());
      // Add port to URL, if defined
      Int_t port = gEnv->GetValue("ProofServ.XpdPort", 1093);
      if (port > -1) url.Form("%s:%d",gSystem->HostName(), port);
      Printf("URL:                        %s", url.Data());
   } else {
      Printf("Host name:                  %s", gSystem->HostName());
   }
   Printf("User:                       %s", GetUser());
   TString ver(gROOT->GetVersion());
   ver += TString::Format("|%s", gROOT->GetGitCommit());
   if (gSystem->Getenv("ROOTVERSIONTAG"))
      ver += TString::Format("|%s", gSystem->Getenv("ROOTVERSIONTAG"));
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

////////////////////////////////////////////////////////////////////////////////
/// Create a TProofQueryResult instance for this query.

TProofQueryResult *TProofLite::MakeQueryResult(Long64_t nent, const char *opt,
                                               Long64_t fst, TDSet *dset,
                                               const char *selec)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Set query in running state.

void TProofLite::SetQueryRunning(TProofQueryResult *pq)
{
   // Record current position in the log file at start
   fflush(fLogFileW);
   Int_t startlog = lseek(fileno(fLogFileW), (off_t) 0, SEEK_END);

   // Add some header to logs
   Printf(" ");
   Info("SetQueryRunning", "starting query: %d", pq->GetSeqNum());

   // Build the list of loaded PAR packages
   TString parlist = "";
   fPackMgr->GetEnabledPackages(parlist);

   // Set in running state
   pq->SetRunning(startlog, parlist, GetParallel());

   // Bytes and CPU at start (we will calculate the differential at end)
   AskStatistics();
   pq->SetProcessInfo(pq->GetEntries(), GetCpuTime(), GetBytesRead());
}

////////////////////////////////////////////////////////////////////////////////
/// Execute the specified drawing action on a data set (TDSet).
/// Event- or Entry-lists should be set in the data set object using
/// TDSet::SetEntryList.
/// Returns -1 in case of error or number of selected events otherwise.

Long64_t TProofLite::DrawSelect(TDSet *dset, const char *varexp,
                                const char *selection, Option_t *option,
                                Long64_t nentries, Long64_t first)
{
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

   // Fill the internal variables
   fVarExp = varexp;
   fSelection = selection;

   return Process(dset, "draw:", opt, nentries, first);
}

////////////////////////////////////////////////////////////////////////////////
/// Process a data set (TDSet) using the specified selector (.C) file.
/// Entry- or event-lists should be set in the data set object using
/// TDSet::SetEntryList.
/// The return value is -1 in case of error and TSelector::GetStatus() in
/// in case of success.

Long64_t TProofLite::Process(TDSet *dset, const char *selector, Option_t *option,
                             Long64_t nentries, Long64_t first)
{
   // For the time being cannot accept other queries if not idle, even if in async
   // mode; needs to set up an event handler to manage that

   TString opt(option), optfb, outfile;
   // Enable feedback, if required
   if (opt.Contains("fb=") || opt.Contains("feedback=")) SetFeedback(opt, optfb, 0);
   // Define output file, either from 'opt' or the default one
   if (HandleOutputOptions(opt, outfile, 0) != 0) return -1;

   // Resolve query mode
   fSync = (GetQueryMode(opt) == kSync);
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
      SetParameter("PROOF_MaxSlavesPerNode", (Long_t)0);

   Bool_t hasNoData = (!dset || (dset && dset->TestBit(TDSet::kEmpty))) ? kTRUE : kFALSE;

   // If just a name was given to identify the dataset, retrieve it from the
   // local files
   // Make sure the dataset contains the information needed
   TString emsg;
   if ((!hasNoData) && dset->GetListOfElements()->GetSize() == 0) {
      if (TProof::AssertDataSet(dset, fPlayer->GetInputList(), fDataSetManager, emsg) != 0) {
         Error("Process", "from AssertDataSet: %s", emsg.Data());
         return -1;
      }
      if (dset->GetListOfElements()->GetSize() == 0) {
         Error("Process", "no files to process!");
         return -1;
      }
   } else if (hasNoData) {
      // Check if we are required to process with TPacketizerFile a registered dataset
      TNamed *ftp = dynamic_cast<TNamed *>(fPlayer->GetInputList()->FindObject("PROOF_FilesToProcess"));
      if (ftp) {
         TString dsn(ftp->GetTitle());
         if (!dsn.Contains(":") || dsn.BeginsWith("dataset:")) {
            dsn.ReplaceAll("dataset:", "");
            // Make sure we have something in input and a dataset manager
            if (!fDataSetManager) {
               emsg.Form("dataset manager not initialized!");
            } else {
               TFileCollection *fc = 0;
               // Get the dataset
               if (!(fc = fDataSetManager->GetDataSet(dsn))) {
                  emsg.Form("requested dataset '%s' does not exists", dsn.Data());
               } else {
                  TMap *fcmap = TProofServ::GetDataSetNodeMap(fc, emsg);
                  if (fcmap) {
                     fPlayer->GetInputList()->Remove(ftp);
                     delete ftp;
                     fcmap->SetOwner(kTRUE);
                     fcmap->SetName("PROOF_FilesToProcess");
                     fPlayer->GetInputList()->Add(fcmap);
                  }
               }
            }
            if (!emsg.IsNull()) {
               Error("HandleProcess", "%s", emsg.Data());
               return -1;
            }
         }
      }
   }

   TString selec(selector), varexp, selection, objname;
   // If a draw query, extract the relevant info
   if (selec.BeginsWith("draw:")) {
      varexp = fVarExp;
      selection = fSelection;
      // Decode now the expression
      if (fPlayer->GetDrawArgs(varexp, selection, opt, selec, objname) != 0) {
         Error("Process", "draw query: error parsing arguments '%s', '%s', '%s'",
                          varexp.Data(), selection.Data(), opt.Data());
         return -1;
      }
   }

   // Create instance of query results (the data set is added after Process)
   TProofQueryResult *pq = MakeQueryResult(nentries, opt, first, 0, selec);

   // Check if queries must be saved into files
   // Automatic saving is controlled by ProofLite.AutoSaveQueries
   Bool_t savequeries =
      (!strcmp(gEnv->GetValue("ProofLite.AutoSaveQueries", "off"), "on")) ? kTRUE : kFALSE;

   // Keep queries in memory and how many (-1 = all, 0 = none, ...)
   Int_t memqueries = gEnv->GetValue("ProofLite.MaxQueriesMemory", 1);

   // If not a draw action add the query to the main list
   if (!(pq->IsDraw())) {
      if (fQMgr->Queries()) {
         if (memqueries != 0) fQMgr->Queries()->Add(pq);
         if (memqueries >= 0 && fQMgr->Queries()->GetSize() > memqueries) {
            // Remove oldest
            TObject *qfst = fQMgr->Queries()->First();
            fQMgr->Queries()->Remove(qfst);
            delete qfst;
         }
      }
      // Also save it to queries dir
      if (savequeries) fQMgr->SaveQuery(pq);
   }

   // Set the query number
   fSeqNum = pq->GetSeqNum();

   // Set in running state
   SetQueryRunning(pq);

   // Save to queries dir, if not standard draw
   if (!(pq->IsDraw())) {
      if (savequeries) fQMgr->SaveQuery(pq);
   } else {
      fQMgr->IncrementDrawQueries();
   }

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

   // Make sure we get a fresh result
   fOutputList.Clear();

   // Start the additional workers now if using fork-based startup
   TList *startedWorkers = 0;
   if (fForkStartup) {
      startedWorkers = new TList;
      startedWorkers->SetOwner(kFALSE);
      SetupWorkers(1, startedWorkers);
   }

   // This is the end of preparation
   fQuerySTW.Reset();

   Long64_t rv = 0;
   if (!(pq->IsDraw())) {
      if (selector && strlen(selector)) {
         rv = fPlayer->Process(dset, selec, opt, nentries, first);
      } else {
         rv = fPlayer->Process(dset, fSelector, opt, nentries, first);
      }
   } else {
      rv = fPlayer->DrawSelect(dset, varexp, selection, opt, nentries, first);
   }

   // This is the end of merging
   fQuerySTW.Stop();
   Float_t rt = fQuerySTW.RealTime();
   // Update the query content
   TQueryResult *qr = GetQueryResult();
   if (qr) {
      qr->SetTermTime(rt);
      // Preparation time is always null in PROOF-Lite
   }

   // Disable feedback, if required
   if (!optfb.IsNull()) SetFeedback(opt, optfb, 1);

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
      if (rv == 0 && dset && !dset->TestBit(TDSet::kEmpty) && pq->GetInputList()) {
         pq->GetInputList()->Add(dset);
         if (dset->GetEntryList())
            pq->GetInputList()->Add(dset->GetEntryList());
      }

      // Register any dataset produced during this processing, if required
      if (fDataSetManager && fPlayer->GetOutputList()) {
         TNamed *psr = (TNamed *) fPlayer->GetOutputList()->FindObject("PROOFSERV_RegisterDataSet");
         if (psr) {
            TString err;
            if (TProofServ::RegisterDataSets(fPlayer->GetInputList(),
                                             fPlayer->GetOutputList(), fDataSetManager, err) != 0)
               Warning("ProcessNext", "problems registering produced datasets: %s", err.Data());
            fPlayer->GetOutputList()->Remove(psr);
            delete psr;
         }
      }

      // Complete filling of the TQueryResult instance
      AskStatistics();
      if (!(pq->IsDraw())) {
         if (fQMgr->FinalizeQuery(pq, this, fPlayer)) {
            if (savequeries) fQMgr->SaveQuery(pq, -1);
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
         if (!(pq->IsDraw()) && memqueries >= 0) {
            if (fQMgr && fQMgr->Queries()) {
               TQueryResult *pqr = pq->CloneInfo();
               if (pqr) fQMgr->Queries()->Add(pqr);
               // Remove from the fQueries list
               fQMgr->Queries()->Remove(pq);
            }
         }
         // To get the prompt back
         TString msg;
         msg.Form("Lite-0: all output objects have been merged                                                         ");
         fprintf(stderr, "%s\n", msg.Data());
      }
      // Save the performance info, if required
      if (!fPerfTree.IsNull()) {
         if (SavePerfTree() != 0) Error("Process", "saving performance info ...");
         // Must be re-enabled each time
         SetPerfTree(0);
      }
   }
   // Finalise output file settings (opt is ignored in here)
   if (HandleOutputOptions(opt, outfile, 1) != 0) return -1;

   // Retrieve status from the output list
   if (rv >= 0) {
      TParameter<Long64_t> *sst =
        (TParameter<Long64_t> *) fOutputList.FindObject("PROOF_SelectorStatus");
      if (sst) rv = sst->GetVal();
   }


   // Done
   return rv;
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize the dataset manager from directives or from defaults
/// Return 0 on success, -1 on failure

Int_t TProofLite::InitDataSetManager()
{
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

   // Dataset manager for staging requests
   TString dsReqCfg = gEnv->GetValue("Proof.DataSetStagingRequests", "");
   if (!dsReqCfg.IsNull()) {
      TPMERegexp reReqDir("(^| )(dir:)?([^ ]+)( |$)");

      if (reReqDir.Match(dsReqCfg) == 5) {
         TString dsDirFmt;
         dsDirFmt.Form("dir:%s perms:open", reReqDir[3].Data());
         fDataSetStgRepo = new TDataSetManagerFile("_stage_", "_stage_", dsDirFmt);
         if (fDataSetStgRepo && fDataSetStgRepo->TestBit(TObject::kInvalidObject)) {
            Warning("InitDataSetManager", "failed init of dataset staging requests repository");
            SafeDelete(fDataSetStgRepo);
         }
      } else {
         Warning("InitDataSetManager", "specify, with [dir:]<path>, a valid path for staging requests");
      }
   } else if (gDebug > 0) {
      Warning("InitDataSetManager", "no repository for staging requests available");
   }

   // Done
   return (fDataSetManager ? 0 : -1);
}

////////////////////////////////////////////////////////////////////////////////
/// List contents of file cache. If all is true show all caches also on
/// slaves. If everything is ok all caches are to be the same.

void TProofLite::ShowCache(Bool_t)
{
   if (!IsValid()) return;

   Printf("*** Local file cache %s ***", fCacheDir.Data());
   gSystem->Exec(Form("%s %s", kLS, fCacheDir.Data()));
}

////////////////////////////////////////////////////////////////////////////////
/// Remove files from all file caches.

void TProofLite::ClearCache(const char *file)
{
   if (!IsValid()) return;

   fCacheLock->Lock();
   if (!file || strlen(file) <= 0) {
      gSystem->Exec(Form("%s %s/*", kRM, fCacheDir.Data()));
   } else {
      gSystem->Exec(Form("%s %s/%s", kRM, fCacheDir.Data(), file));
   }
   fCacheLock->Unlock();
}

////////////////////////////////////////////////////////////////////////////////
/// Copy the specified macro in the cache directory. The macro file is
/// uploaded if new or updated. If existing, the corresponding header
/// basename(macro).h or .hh, is also uploaded. For the other arguments
/// see TProof::Load().
/// Returns 0 in case of success and -1 in case of error.

Int_t TProofLite::Load(const char *macro, Bool_t notOnClient, Bool_t uniqueOnly,
                       TList *wrks)
{
   if (!IsValid()) return -1;

   if (!macro || !macro[0]) {
      Error("Load", "need to specify a macro name");
      return -1;
   }

   TString macs(macro), mac;
   Int_t from = 0;
   while (macs.Tokenize(mac, from, ",")) {
      if (IsIdle()) {
         if (CopyMacroToCache(mac) < 0) return -1;
      } else {
         // The name
         TString macn = gSystem->BaseName(mac);
         macn.Remove(macn.Last('.'));
         // Relevant pointers
         TList cachedFiles;
         TString cacheDir = fCacheDir;
         gSystem->ExpandPathName(cacheDir);
         void * dirp = gSystem->OpenDirectory(cacheDir);
         if (dirp) {
            const char *e = 0;
            while ((e = gSystem->GetDirEntry(dirp))) {
               if (!strncmp(e, macn.Data(), macn.Length())) {
                  TString fncache = Form("%s/%s", cacheDir.Data(), e);
                  cachedFiles.Add(new TObjString(fncache.Data()));
               }
            }
            gSystem->FreeDirectory(dirp);
         }
      }
   }

   return TProof::Load(macro, notOnClient, uniqueOnly, wrks);
}

////////////////////////////////////////////////////////////////////////////////
/// Copy a macro, and its possible associated .h[h] file,
/// to the cache directory, from where the workers can get the file.
/// If headerRequired is 1, return -1 in case the header is not found.
/// If headerRequired is 0, try to copy header too.
/// If headerRequired is -1, don't look for header, only copy macro.
/// If the selector pionter is not 0, consider the macro to be a selector
/// and try to load the selector and set it to the pointer.
/// The mask 'opt' is an or of ESendFileOpt:
///       kCpBin   (0x8)     Retrieve from the cache the binaries associated
///                          with the file
///       kCp      (0x10)    Retrieve the files from the cache
/// Return -1 in case of error, 0 otherwise.

Int_t TProofLite::CopyMacroToCache(const char *macro, Int_t headerRequired,
                                   TSelector **selector, Int_t opt, TList *)
{
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
   TString np = gSystem->GetDirName(name);
   if (!np.IsNull()) {
      np += ":";
      if (!mp.BeginsWith(np) && !mp.Contains(":"+np)) {
         Int_t ip = (mp.BeginsWith(".:")) ? 2 : 0;
         mp.Insert(ip, np);
         TROOT::SetMacroPath(mp);
         PDB(kGlobal,1)
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
      TString v, r;
      FILE *f = fopen(Form("%s/%s", cacheDir.Data(), vername.Data()), "r");
      if (f) {
         v.Gets(f);
         r.Gets(f);
         fclose(f);
      }
      if (!f || v != gROOT->GetVersion() || r != gROOT->GetGitCommit())
         useCacheBinaries = kFALSE;
   }

   // Create binary name template
   TString binname = gSystem->BaseName(name);
   dot = binname.Last('.');
   if (dot != kNPOS)
      binname.Replace(dot,1,"_");
   TString pcmname = TString::Format("%s_ACLiC_dict_rdict.pcm", binname.Data());
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
            if (!strncmp(e, binname.Data(), binname.Length()) ||
                !strncmp(e, pcmname.Data(), pcmname.Length())) {
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
         if (!strncmp(e, binname.Data(), binname.Length()) ||
             !strncmp(e, pcmname.Data(), pcmname.Length())) {
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
         fputs(Form("\n%s", gROOT->GetGitCommit()), f);
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

   cachedFiles->SetOwner();
   delete cachedFiles;

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Remove old sessions dirs keep at most 'Proof.MaxOldSessions' (default 10)

Int_t TProofLite::CleanupSandbox()
{
   Int_t maxold = gEnv->GetValue("Proof.MaxOldSessions", 1);

   if (maxold < 0) return 0;

   TSortedList *olddirs = new TSortedList(kFALSE);

   TString sandbox = gSystem->GetDirName(fWorkDir.Data());

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

////////////////////////////////////////////////////////////////////////////////
/// Get the list of queries.

TList *TProofLite::GetListOfQueries(Option_t *opt)
{
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
            if ((pqm = pqr->CloneInfo())) {
               pqm->fSeqNum = ntot;
               ql->Add(pqm);
            } else {
               Warning("GetListOfQueries", "unable to clone TProofQueryResult '%s:%s'",
                       pqr->GetName(), pqr->GetTitle());
            }
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

////////////////////////////////////////////////////////////////////////////////
/// Register the 'dataSet' on the cluster under the current
/// user, group and the given 'dataSetName'.
/// Fails if a dataset named 'dataSetName' already exists, unless 'optStr'
/// contains 'O', in which case the old dataset is overwritten.
/// If 'optStr' contains 'V' the dataset files are verified (default no
/// verification).
/// Returns kTRUE on success.

Bool_t TProofLite::RegisterDataSet(const char *uri,
                                   TFileCollection *dataSet, const char* optStr)
{
   if (!fDataSetManager) {
      Info("RegisterDataSet", "dataset manager not available");
      return kFALSE;
   }

   if (!uri || strlen(uri) <= 0) {
      Info("RegisterDataSet", "specifying a dataset name is mandatory");
      return kFALSE;
   }

   Bool_t parallelverify = kFALSE;
   TString sopt(optStr);
   if (sopt.Contains("V") && !sopt.Contains("S")) {
      // We do verification in parallel later on; just register for now
      parallelverify = kTRUE;
      sopt.ReplaceAll("V", "");
   }
   // This would screw up things remotely, make sure is not there
   sopt.ReplaceAll("S", "");

   Bool_t result = kTRUE;
   if (fDataSetManager->TestBit(TDataSetManager::kAllowRegister)) {
      // Check the list
      if (!dataSet || dataSet->GetList()->GetSize() == 0) {
         Error("RegisterDataSet", "can not save an empty list.");
         result = kFALSE;
      }
      // Register the dataset (quota checks are done inside here)
      result = (fDataSetManager->RegisterDataSet(uri, dataSet, sopt) == 0)
             ? kTRUE : kFALSE;
   } else {
      Info("RegisterDataSet", "dataset registration not allowed");
      result = kFALSE;
   }

   if (!result)
      Error("RegisterDataSet", "dataset was not saved");

   // If old server or not verifying in parallel we are done
   if (!parallelverify) return result;

   // If we are here it means that we will verify in parallel
   sopt += "V";
   if (VerifyDataSet(uri, sopt) < 0){
      Error("RegisterDataSet", "problems verifying dataset '%s'", uri);
      return kFALSE;
   }

   // Done
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Set/Change the name of the default tree. The tree name may contain
/// subdir specification in the form "subdir/name".
/// Returns 0 on success, -1 otherwise.

Int_t TProofLite::SetDataSetTreeName(const char *dataset, const char *treename)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Returns kTRUE if 'dataset' described by 'uri' exists, kFALSE otherwise

Bool_t TProofLite::ExistsDataSet(const char *uri)
{
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

////////////////////////////////////////////////////////////////////////////////
/// lists all datasets that match given uri

TMap *TProofLite::GetDataSets(const char *uri, const char *srvex)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Shows datasets in locations that match the uri
/// By default shows the user's datasets and global ones

void TProofLite::ShowDataSets(const char *uri, const char *opt)
{
   if (!fDataSetManager) {
      Info("GetDataSet", "dataset manager not available");
      return;
   }

   fDataSetManager->ShowDataSets(uri, opt);
}

////////////////////////////////////////////////////////////////////////////////
/// Get a list of TFileInfo objects describing the files of the specified
/// dataset.

TFileCollection *TProofLite::GetDataSet(const char *uri, const char *)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Remove the specified dataset from the PROOF cluster.
/// Files are not deleted.

Int_t TProofLite::RemoveDataSet(const char *uri, const char *)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Allows users to request staging of a particular dataset. Requests are
/// saved in a special dataset repository and must be honored by the endpoint.
/// This is the special PROOF-Lite re-implementation of the TProof function
/// and includes code originally implemented in TProofServ.

Bool_t TProofLite::RequestStagingDataSet(const char *dataset)
{
   if (!dataset) {
      Error("RequestStagingDataSet", "invalid dataset specified");
      return kFALSE;
   }

   if (!fDataSetStgRepo) {
      Error("RequestStagingDataSet", "no dataset staging request repository available");
      return kFALSE;
   }

   TString dsUser, dsGroup, dsName, dsTree;

   // Transform input URI in a valid dataset name
   TString validUri = dataset;
   while (fReInvalid->Substitute(validUri, "_")) {}

   // Check if dataset exists beforehand: if it does, staging has already been requested
   if (fDataSetStgRepo->ExistsDataSet(validUri.Data())) {
      Warning("RequestStagingDataSet", "staging of %s already requested", dataset);
      return kFALSE;
   }

   // Try to get dataset from current manager
   TFileCollection *fc = fDataSetManager->GetDataSet(dataset);
   if (!fc || (fc->GetNFiles() == 0)) {
      Error("RequestStagingDataSet", "empty dataset or no dataset returned");
      if (fc) delete fc;
      return kFALSE;
   }

   // Reset all staged bits and remove unnecessary URLs (all but last)
   TIter it(fc->GetList());
   TFileInfo *fi;
   while ((fi = dynamic_cast<TFileInfo *>(it.Next()))) {
      fi->ResetBit(TFileInfo::kStaged);
      Int_t nToErase = fi->GetNUrls() - 1;
      for (Int_t i=0; i<nToErase; i++)
         fi->RemoveUrlAt(0);
   }

   fc->Update();  // absolutely necessary

   // Save request
   fDataSetStgRepo->ParseUri(validUri, &dsGroup, &dsUser, &dsName);
   if (fDataSetStgRepo->WriteDataSet(dsGroup, dsUser, dsName, fc) == 0) {
      // Error, can't save dataset
      Error("RequestStagingDataSet", "can't register staging request for %s", dataset);
      delete fc;
      return kFALSE;
   }

   Info("RequestStagingDataSet", "Staging request registered for %s", dataset);
   delete fc;

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Cancels a dataset staging request. Returns kTRUE on success, kFALSE on
/// failure. Dataset not found equals to a failure. PROOF-Lite
/// re-implementation of the equivalent function in TProofServ.

Bool_t TProofLite::CancelStagingDataSet(const char *dataset)
{
   if (!dataset) {
      Error("CancelStagingDataSet", "invalid dataset specified");
      return kFALSE;
   }

   if (!fDataSetStgRepo) {
      Error("CancelStagingDataSet", "no dataset staging request repository available");
      return kFALSE;
   }

   // Transform URI in a valid dataset name
   TString validUri = dataset;
   while (fReInvalid->Substitute(validUri, "_")) {}

   if (!fDataSetStgRepo->RemoveDataSet(validUri.Data()))
      return kFALSE;

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Obtains a TFileCollection showing the staging status of the specified
/// dataset. A valid dataset manager and dataset staging requests repository
/// must be present on the endpoint. PROOF-Lite version of the equivalent
/// function from TProofServ.

TFileCollection *TProofLite::GetStagingStatusDataSet(const char *dataset)
{
   if (!dataset) {
      Error("GetStagingStatusDataSet", "invalid dataset specified");
      return 0;
   }

   if (!fDataSetStgRepo) {
      Error("GetStagingStatusDataSet", "no dataset staging request repository available");
      return 0;
   }

   // Transform URI in a valid dataset name
   TString validUri = dataset;
   while (fReInvalid->Substitute(validUri, "_")) {}

   // Get the list
   TFileCollection *fc = fDataSetStgRepo->GetDataSet(validUri.Data());
   if (!fc) {
      // No such dataset (not an error)
      Info("GetStagingStatusDataSet", "no pending staging request for %s", dataset);
      return 0;
   }

   // Dataset found: return it (must be cleaned by caller)
   return fc;
}

////////////////////////////////////////////////////////////////////////////////
/// Verify if all files in the specified dataset are available.
/// Print a list and return the number of missing files.

Int_t TProofLite::VerifyDataSet(const char *uri, const char *optStr)
{
   if (!fDataSetManager) {
      Info("VerifyDataSet", "dataset manager not available");
      return -1;
   }

   Int_t rc = -1;
   TString sopt(optStr);
   if (sopt.Contains("S")) {

      if (fDataSetManager->TestBit(TDataSetManager::kAllowVerify)) {
         rc = fDataSetManager->ScanDataSet(uri);
      } else {
         Info("VerifyDataSet", "dataset verification not allowed");
         rc = -1;
      }
      return rc;
   }

   // Done
   return VerifyDataSetParallel(uri, optStr);
}

////////////////////////////////////////////////////////////////////////////////
/// Clear the content of the dataset cache, if any (matching 'dataset', if defined).

void TProofLite::ClearDataSetCache(const char *dataset)
{
   if (fDataSetManager) fDataSetManager->ClearCache(dataset);
   // Done
   return;
}

////////////////////////////////////////////////////////////////////////////////
/// Display the content of the dataset cache, if any (matching 'dataset', if defined).

void TProofLite::ShowDataSetCache(const char *dataset)
{
   // For PROOF-Lite act locally
   if (fDataSetManager) fDataSetManager->ShowCache(dataset);
   // Done
   return;
}

////////////////////////////////////////////////////////////////////////////////
/// Make sure that the input data objects are available to the workers in a
/// dedicated file in the cache; the objects are taken from the dedicated list
/// and / or the specified file.
/// If the fInputData is empty the specified file is sent over.
/// If there is no specified file, a file named "inputdata.root" is created locally
/// with the content of fInputData and sent over to the master.
/// If both fInputData and the specified file are not empty, a copy of the file
/// is made locally and augmented with the content of fInputData.

void TProofLite::SendInputDataFile()
{
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
         if (gSystem->CopyFile(dataFile, dst) != 0)
            Warning("SendInputDataFile", "problems copying '%s' to '%s'",
                                         dataFile.Data(), dst.Data());
      }

      // Set the name in the input list so that the workers can find it
      AddInput(new TNamed("PROOF_InputDataFile", Form("%s", gSystem->BaseName(dataFile))));
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Handle remove request.

Int_t TProofLite::Remove(const char *ref, Bool_t all)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Creates a tree header (a tree with nonexisting files) object for
/// the DataSet.

TTree *TProofLite::GetTreeHeader(TDSet *dset)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Add to the fUniqueSlave list the active slaves that have a unique
/// (user) file system image. This information is used to transfer files
/// only once to nodes that share a file system (an image). Submasters
/// which are not in fUniqueSlaves are put in the fNonUniqueMasters
/// list. That list is used to trigger the transferring of files to
/// the submaster's unique slaves without the need to transfer the file
/// to the submaster.

void TProofLite::FindUniqueSlaves()
{
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

////////////////////////////////////////////////////////////////////////////////
/// List contents of the data directory in the sandbox.
/// This is the place where files produced by the client queries are kept

void TProofLite::ShowData()
{
   if (!IsValid()) return;

   // Get worker infos
   TList *wrki = GetListOfSlaveInfos();
   TSlaveInfo *wi = 0;
   TIter nxwi(wrki);
   while ((wi = (TSlaveInfo *) nxwi())) {
      ShowDataDir(wi->GetDataDir());
   }
}

////////////////////////////////////////////////////////////////////////////////
/// List contents of the data directory 'dirname'

void TProofLite::ShowDataDir(const char *dirname)
{
   if (!dirname) return;

   FileStat_t dirst;
   if (gSystem->GetPathInfo(dirname, dirst) != 0) return;
   if (!R_ISDIR(dirst.fMode)) return;

   void *dirp = gSystem->OpenDirectory(dirname);
   TString fn;
   const char *ent = 0;
   while ((ent = gSystem->GetDirEntry(dirp))) {
      fn.Form("%s/%s", dirname, ent);
      FileStat_t st;
      if (gSystem->GetPathInfo(fn.Data(), st) == 0) {
         if (R_ISREG(st.fMode)) {
            Printf("lite:0| %s", fn.Data());
         } else if (R_ISREG(st.fMode)) {
            ShowDataDir(fn.Data());
         }
      }
   }
   // Done
   return;
}

////////////////////////////////////////////////////////////////////////////////
/// Simulate dynamic addition, for test purposes.
/// Here we decide how many workers to add, we create them and set the
/// environment.
/// This call is called regularly by Collect if the opton is enabled.
/// Returns the number of new workers added, or <0 on errors.

Int_t TProofLite::PollForNewWorkers()
{
   // Max workers
   if (fDynamicStartupNMax <= 0) {
      SysInfo_t si;
      if (gSystem->GetSysInfo(&si) == 0 && si.fCpus > 2) {
         fDynamicStartupNMax = si.fCpus;
      } else {
         fDynamicStartupNMax = 2;
      }
   }
   if (fNWorkers >= fDynamicStartupNMax) {
      // Max reached: disable
      Info("PollForNewWorkers", "max reached: %d workers started", fNWorkers);
      fDynamicStartup =  kFALSE;
      return 0;
   }

   // Number of new workers
   Int_t nAdd = (fDynamicStartupStep > 0) ? fDynamicStartupStep : 1;

   // Create a monitor and add the socket to it
   TMonitor *mon = new TMonitor;
   mon->Add(fServSock);

   TList started;
   TSlave *wrk = 0;
   Int_t nWrksDone = 0, nWrksTot = -1;
   TString fullord;

   nWrksTot = fNWorkers + nAdd;
   // Now we create the worker applications which will call us back to finalize
   // the setup
   Int_t ord = fNWorkers;
   for (; ord < nWrksTot; ord++) {

      // Ordinal for this worker server
      fullord = Form("0.%d", ord);

      // Create environment files
      SetProofServEnv(fullord);

      // Create worker server and add to the list
      if ((wrk = CreateSlave("lite", fullord, 100, fImage, fWorkDir)))
         started.Add(wrk);

      PDB(kGlobal, 3)
         Info("PollForNewWorkers", "additional worker '%s' started", fullord.Data());

      // Notify
      NotifyStartUp("Opening connections to workers", ++nWrksDone, nWrksTot);

   } //end of worker loop
   fNWorkers = nWrksTot;

   // A list of TSlave objects for workers that are being added
   TList *addedWorkers = new TList();
   addedWorkers->SetOwner(kFALSE);

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
         if (s->Recv(msg) < 0) {
            Warning("PollForNewWorkers", "problems receiving message from accepted socket!");
         } else {
            if (msg) {
               *msg >> fullord;
               // Find who is calling back
               if ((wrk = (TSlave *) started.FindObject(fullord))) {
                  // Remove it from the started list
                  started.Remove(wrk);

                  // Assign tis socket the selected worker
                  wrk->SetSocket(s);
                  // Remove socket from global TROOT socket list. Only the TProof object,
                  // representing all worker sockets, will be added to this list. This will
                  // ensure the correct termination of all proof servers in case the
                  // root session terminates.
                  {  R__LOCKGUARD(gROOTMutex);
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
                  fSlaves->Add(wrk);
                  if (wrk->IsValid()) {
                     fActiveSlaves->Add(wrk);             // Is this required? Check!
                     fAllMonitor->Add(wrk->GetSocket());
                     // Record also in the list for termination
                     addedWorkers->Add(wrk);
                     // Notify startup operations
                     NotifyStartUp("Setting up added worker servers", ++nWrksDone, nWrksTot);
                  } else {
                     // Flag as bad
                     fBadSlaves->Add(wrk);
                  }
               }
            } else {
               Warning("PollForNewWorkers", "received empty message from accepted socket!");
            }
         }
      }
   }

   // Cleanup the monitor and the server socket
   mon->DeActivateAll();
   delete mon;

   Broadcast(kPROOF_GETSTATS, addedWorkers);
   Collect(addedWorkers, fCollectTimeout);

   // Update group view
   // SendGroupView();

   // By default go into parallel mode
   // SetParallel(-1, 0);
   SendCurrentState(addedWorkers);

   // Set worker processing environment
   SetupWorkersEnv(addedWorkers, kTRUE);

   // We are adding workers dynamically to an existing process, we
   // should invoke a special player's Process() to set only added workers
   // to the proper state
   if (fPlayer) {
      PDB(kGlobal, 3)
         Info("PollForNewWorkers", "Will send the PROCESS message to selected workers");
      fPlayer->JoinProcess(addedWorkers);
   }

   // Cleanup fwhat remained from startup
   Collect(addedWorkers);

   // Activate
   TIter naw(addedWorkers);
   while ((wrk = (TSlave *)naw())) {
      fActiveMonitor->Add(wrk->GetSocket());
   }
   // Cleanup
   delete addedWorkers;

   // Done
   return nWrksDone;
}
