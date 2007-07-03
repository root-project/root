// @(#)root/proof:$Name:  $:$Id: TProof.cxx,v 1.208 2007/06/22 21:59:43 ganis Exp $
// Author: Fons Rademakers   13/02/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProof                                                               //
//                                                                      //
// This class controls a Parallel ROOT Facility, PROOF, cluster.        //
// It fires the slave servers, it keeps track of how many slaves are    //
// running, it keeps track of the slaves running status, it broadcasts  //
// messages to all slaves, it collects results, etc.                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <fcntl.h>
#include <errno.h>
#ifdef WIN32
#   include <io.h>
#   include <sys/stat.h>
#   include <sys/types.h>
#else
#   include <unistd.h>
#endif
#include <vector>

#ifdef R__HAVE_CONFIG
#include "RConfigure.h"
#endif

#include "Riostream.h"
#include "Getline.h"
#include "TBrowser.h"
#include "TChain.h"
#include "TCondor.h"
#include "TDSet.h"
#include "TError.h"
#include "TEnv.h"
#include "TEventList.h"
#include "TFile.h"
#include "TFileInfo.h"
#include "TFTP.h"
#include "THashList.h"
#include "TInterpreter.h"
#include "TMap.h"
#include "TMessage.h"
#include "TMonitor.h"
#include "TMutex.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TParameter.h"
#include "TProof.h"
#include "TProofNodeInfo.h"
#include "TVirtualProofPlayer.h"
#include "TProofServ.h"
#include "TPluginManager.h"
#include "TQueryResult.h"
#include "TRandom.h"
#include "TRegexp.h"
#include "TROOT.h"
#include "TSemaphore.h"
#include "TSlave.h"
#include "TSocket.h"
#include "TSortedList.h"
#include "TSystem.h"
#include "TThread.h"
#include "TTree.h"
#include "TUrl.h"


TProof *gProof = 0;
TVirtualMutex *gProofMutex = 0;

TList   *TProof::fgProofEnvList = 0;  // List of env vars for proofserv

ClassImp(TProof)

//----- Helper classes used for parallel startup -------------------------------
//______________________________________________________________________________
TProofThreadArg::TProofThreadArg(const char *h, Int_t po, const char *o,
                                 Int_t pe, const char *i, const char *w,
                                 TList *s, TProof *prf)
  : fOrd(o), fPerf(pe), fImage(i), fWorkdir(w),
    fSlaves(s), fProof(prf), fCslave(0), fClaims(0),
    fType(TSlave::kSlave)
{
   // Constructor

   fUrl = new TUrl(Form("%s:%d",h,po));
}

//______________________________________________________________________________
TProofThreadArg::TProofThreadArg(TCondorSlave *csl, TList *clist,
                                 TList *s, TProof *prf)
  : fUrl(0), fOrd(0), fPerf(-1), fImage(0), fWorkdir(0),
    fSlaves(s), fProof(prf), fCslave(csl), fClaims(clist),
    fType(TSlave::kSlave)
{
   // Constructor

   if (csl) {
      fUrl     = new TUrl(Form("%s:%d",csl->fHostname.Data(),csl->fPort));
      fImage   = csl->fImage;
      fOrd     = csl->fOrdinal;
      fWorkdir = csl->fWorkDir;
      fPerf    = csl->fPerfIdx;
   }
}

//______________________________________________________________________________
TProofThreadArg::TProofThreadArg(const char *h, Int_t po, const char *o,
                                 const char *i, const char *w, const char *m,
                                TList *s, TProof *prf)
  : fOrd(o), fPerf(-1), fImage(i), fWorkdir(w),
    fMsd(m), fSlaves(s), fProof(prf), fCslave(0), fClaims(0),
    fType(TSlave::kSlave)
{
   // Constructor

   fUrl = new TUrl(Form("%s:%d",h,po));
}

//----- PROOF Interrupt signal handler -----------------------------------------
//______________________________________________________________________________
Bool_t TProofInterruptHandler::Notify()
{
   // TProof interrupt handler.

   Info("Notify","Processing interrupt signal ...");

   // Stop any remote processing
   fProof->StopProcess(kTRUE);

   // Handle also interrupt condition on socket(s)
   fProof->Interrupt(TProof::kLocalInterrupt);

   return kTRUE;
}

//----- Input handler for messages from TProofServ -----------------------------
//______________________________________________________________________________
TProofInputHandler::TProofInputHandler(TProof *p, TSocket *s)
                   : TFileHandler(s->GetDescriptor(),1),
                     fSocket(s), fProof(p)
{
   // Constructor
}

//______________________________________________________________________________
Bool_t TProofInputHandler::Notify()
{
   // Handle input

   fProof->CollectInputFrom(fSocket);
   return kTRUE;
}


//------------------------------------------------------------------------------

ClassImp(TSlaveInfo)

//______________________________________________________________________________
Int_t TSlaveInfo::Compare(const TObject *obj) const
{
   // Used to sort slaveinfos by ordinal.

   if (!obj) return 1;

   const TSlaveInfo *si = dynamic_cast<const TSlaveInfo*>(obj);

   if (!si) return fOrdinal.CompareTo(obj->GetName());

   const char *myord = GetOrdinal();
   const char *otherord = si->GetOrdinal();
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
void TSlaveInfo::Print(Option_t *opt) const
{
   // Print slave info. If opt = "active" print only the active
   // slaves, if opt="notactive" print only the not active slaves,
   // if opt = "bad" print only the bad slaves, else
   // print all slaves.

   TString stat = fStatus == kActive ? "active" :
                  fStatus == kBad ? "bad" :
                  "not active";
   TString msd  = fMsd.IsNull() ? "<null>" : fMsd.Data();

   if (!opt) opt = "";
   if (!strcmp(opt, "active") && fStatus != kActive)
      return;
   if (!strcmp(opt, "notactive") && fStatus != kNotActive)
      return;
   if (!strcmp(opt, "bad") && fStatus != kBad)
      return;

   cout << "Slave: "          << fOrdinal
        << "  hostname: "     << fHostName
        << "  msd: "          << msd
        << "  perf index: "   << fPerfIndex
        << "  "               << stat
        << endl;
}


//------------------------------------------------------------------------------

//______________________________________________________________________________
static char *CollapseSlashesInPath(const char *path)
{
   // Get rid of spare slashes in a path. Returned path must be deleted[]
   // by the user.

   if (path) {
      Int_t i = 1; // current index as we go along the string
      Int_t j = 0; // current end of new path in newPath
      char *newPath = new char [strlen(path) + 1];
      newPath[0] = path[0];
      while (path[i]) {
         if (path[i] != '/' || newPath[j] != '/') {
            j++;
            newPath[j] = path[i];
         }
         i++;
      }
      if (newPath[j] != '/')
         j++;
      newPath[j] = 0; // We have to terminate the new path.
      return newPath;
   }
   return 0;
}

ClassImp(TProof)

TSemaphore    *TProof::fgSemaphore = 0;

//______________________________________________________________________________
TProof::TProof(const char *masterurl, const char *conffile, const char *confdir,
               Int_t loglevel, const char *alias, TProofMgr *mgr)
       : fUrl(masterurl)
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

   // This may be needed during init
   fManager = mgr;

   // Default server type
   fServType = TProofMgr::kXProofd;

   // Default query mode
   fQueryMode = kSync;

   if (!conffile || strlen(conffile) == 0)
      conffile = kPROOF_ConfFile;
   if (!confdir  || strlen(confdir) == 0)
      confdir = kPROOF_ConfDir;

   Init(masterurl, conffile, confdir, loglevel, alias);

   // If called by a manager, make sure it stays in last position
   // for cleaning
   if (mgr) {
      R__LOCKGUARD2(gROOTMutex);
      gROOT->GetListOfSockets()->Remove(mgr);
      gROOT->GetListOfSockets()->Add(mgr);
   }

   // Old-style server type: we add this to the list and set the global pointer
   if (IsProofd() || IsMaster())
      gROOT->GetListOfProofs()->Add(this);

   // Still needed by the packetizers: needs to be changed
   gProof = this;
}

//______________________________________________________________________________
TProof::TProof() : fUrl(""), fServType(TProofMgr::kXProofd)
{
   // Protected constructor to be used by classes deriving from TProof
   // (they have to call Init themselves and override StartSlaves
   // appropriately).
   //
   // This constructor simply closes any previous gProof and sets gProof
   // to this instance.

   gROOT->GetListOfProofs()->Add(this);

   gProof = this;
}

//______________________________________________________________________________
TProof::~TProof()
{
   // Clean up PROOF environment.

   while (TChain *chain = dynamic_cast<TChain*> (fChains->First()) ) {
      // remove "chain" from list
      chain->SetProof(0);
      RemoveChain(chain);
   }

   // remove links to packages enabled on the client
   if (!IsMaster()) {
      // iterate over all packages
      TIter nextpackage(fEnabledPackagesOnClient);
      while (TObjString *package = dynamic_cast<TObjString*>(nextpackage())) {
         FileStat_t stat;
         gSystem->GetPathInfo(package->String(), stat);
         // check if symlink, if so unlink
         // NOTE: GetPathnfo() returns 1 in case of symlink that does not point to
         // existing file or to a directory, but if fIsLink is true the symlink exists
         if (stat.fIsLink)
            gSystem->Unlink(package->String());
      }
   }

   Close();
   SafeDelete(fIntHandler);
   SafeDelete(fSlaves);
   SafeDelete(fActiveSlaves);
   SafeDelete(fInactiveSlaves);
   SafeDelete(fUniqueSlaves);
   SafeDelete(fAllUniqueSlaves);
   SafeDelete(fNonUniqueMasters);
   SafeDelete(fBadSlaves);
   SafeDelete(fAllMonitor);
   SafeDelete(fActiveMonitor);
   SafeDelete(fUniqueMonitor);
   SafeDelete(fAllUniqueMonitor);
   SafeDelete(fSlaveInfo);
   SafeDelete(fChains);
   SafeDelete(fPlayer);
   SafeDelete(fFeedback);
   SafeDelete(fWaitingSlaves);
   SafeDelete(fAvailablePackages);
   SafeDelete(fEnabledPackages);
   SafeDelete(fEnabledPackagesOnClient);
   SafeDelete(fPackageLock);
   SafeDelete(fGlobalPackageDirList);

   // remove file with redirected logs
   if (!IsMaster()) {
      if (fLogFileR)
         fclose(fLogFileR);
      if (fLogFileW)
         fclose(fLogFileW);
      if (fLogFileName.Length())
         gSystem->Unlink(fLogFileName);
   }

   // For those interested in our destruction ...
   Emit("~TProof()");
}

//______________________________________________________________________________
Int_t TProof::Init(const char *masterurl, const char *conffile,
                   const char *confdir, Int_t loglevel, const char *alias)
{
   // Start the PROOF environment. Starting PROOF involves either connecting
   // to a master server, which in turn will start a set of slave servers, or
   // directly starting as master server (if master = ""). For a description
   // of the arguments see the TProof ctor. Returns the number of started
   // master or slave servers, returns 0 in case of error, in which case
   // fValid remains false.

   R__ASSERT(gSystem);

   fValid = kFALSE;

   if (strlen(fUrl.GetOptions()) > 0 && !(strncmp(fUrl.GetOptions(),"std",3))) {
      fServType = TProofMgr::kProofd;
      fUrl.SetOptions("");
   }

   if (!masterurl || !*masterurl) {
      fUrl.SetProtocol("proof");
      fUrl.SetHost("__master__");
   } else if (!(strstr(masterurl, "://"))) {
      fUrl.SetProtocol("proof");
   }
   if (fUrl.GetPort() == TUrl(" ").GetPort())
      fUrl.SetPort(TUrl("proof:// ").GetPort());

   // If in attach mode, options is filled with additiona info
   Bool_t attach = kFALSE;
   if (strlen(fUrl.GetOptions()) > 0) {
      attach = kTRUE;
      // A flag from the GUI
      TString opts = fUrl.GetOptions();
      if (opts.Contains("GUI")) {
         SetBit(TProof::kUsingSessionGui);
         opts.Remove(opts.Index("GUI"));
         fUrl.SetOptions(opts);
      }
   }

   if (strlen(fUrl.GetUser()) <= 0) {
      // Get user logon name
      UserGroup_t *pw = gSystem->GetUserInfo();
      if (pw) {
         fUrl.SetUser(pw->fUser);
         delete pw;
      }
   }
   // Make sure to store the FQDN, so to get a solid reference for
   // subsequent checks (strings corresponding to non-existing hosts
   // - like "__master__" - will not be touched by this)
   if (!strlen(fUrl.GetHost()))
      fMaster = gSystem->GetHostByName(gSystem->HostName()).GetHostName();
   else
      fMaster = gSystem->GetHostByName(fUrl.GetHost()).GetHostName();
   fConfDir        = confdir;
   fConfFile       = conffile;
   fWorkDir        = gSystem->WorkingDirectory();
   fLogLevel       = loglevel;
   fProtocol       = kPROOF_Protocol;
   fMasterServ     = (fMaster == "__master__") ? kTRUE : kFALSE;
   fSendGroupView  = kTRUE;
   fImage          = fMasterServ ? "" : "<local>";
   fIntHandler     = 0;
   fStatus         = 0;
   fSlaveInfo      = 0;
   fChains         = new TList;
   fAvailablePackages = 0;
   fEnabledPackages = 0;
   fEndMaster      = IsMaster() ? kTRUE : kFALSE;

   // Default entry point for the data pool is the master
   if (!IsMaster())
      fDataPoolUrl.Form("root://%s", fMaster.Data());
   else
      fDataPoolUrl = "";

   fProgressDialog        = 0;
   fProgressDialogStarted = kFALSE;

   // Default alias is the master name
   TString      al = (alias) ? alias : fMaster.Data();
   SetAlias(al);

   // Client logging of messages from the master and slaves
   fRedirLog = kFALSE;
   if (!IsMaster()) {
      fLogFileName    = "ProofLog_";
      if ((fLogFileW = gSystem->TempFileName(fLogFileName)) == 0)
         Error("Init", "could not create temporary logfile");
      if ((fLogFileR = fopen(fLogFileName, "r")) == 0)
         Error("Init", "could not open temp logfile for reading");
   }
   fLogToWindowOnly = kFALSE;

   // Status of cluster
   fIdle = kTRUE;

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
   MakePlayer();

   fFeedback = new TList;
   fFeedback->SetOwner();
   fFeedback->SetName("FeedbackList");
   AddInput(fFeedback);

   // sort slaves by descending performance index
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

   fPackageLock             = 0;
   fEnabledPackagesOnClient = 0;
   fGlobalPackageDirList    = 0;
   if (!IsMaster()) {
      fPackageDir = kPROOF_WorkDir;
      gSystem->ExpandPathName(fPackageDir);
      if (gSystem->AccessPathName(fPackageDir)) {
         if (gSystem->MakeDirectory(fPackageDir) == -1) {
            Error("Init", "failure creating directory %s", fPackageDir.Data());
            return 0;
         }
      }
      fPackageDir += TString("/") + kPROOF_PackDir;
      if (gSystem->AccessPathName(fPackageDir)) {
         if (gSystem->MakeDirectory(fPackageDir) == -1) {
            Error("Init", "failure creating directory %s", fPackageDir.Data());
            return 0;
         }
      }

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

      UserGroup_t *ug = gSystem->GetUserInfo();
      fPackageLock = new TProofLockPath(Form("%s%s", kPROOF_PackageLockFile, ug->fUser.Data()));
      delete ug;

      fEnabledPackagesOnClient = new TList;
      fEnabledPackagesOnClient->SetOwner();
   }

   // Master may want parallel startup
   Bool_t parallelStartup = kFALSE;
   if (!attach && IsMaster()) {
      parallelStartup = gEnv->GetValue("Proof.ParallelStartup", kFALSE);
      PDB(kGlobal,1) Info("Init", "Parallel Startup: %s",
                          parallelStartup ? "kTRUE" : "kFALSE");
      if (parallelStartup) {
         // Load thread lib, if not done already
         TString threadLib = "libThread";
         char *p;
         if ((p = gSystem->DynamicPathName(threadLib, kTRUE))) {
            delete[]p;
            if (gSystem->Load(threadLib) == -1) {
               Warning("Init",
                       "Cannot load libThread: switch to serial startup (%s)",
                       threadLib.Data());
               parallelStartup = kFALSE;
            }
         } else {
            Warning("Init",
                    "Cannot find libThread: switch to serial startup (%s)",
                    threadLib.Data());
            parallelStartup = kFALSE;
         }

         // Get no of parallel requests and set semaphore correspondingly
         Int_t parallelRequests = gEnv->GetValue("Proof.ParallelStartupRequests", 0);
         if (parallelRequests > 0) {
            PDB(kGlobal,1)
               Info("Init", "Parallel Startup Requests: %d", parallelRequests);
            fgSemaphore = new TSemaphore((UInt_t)(parallelRequests));
         }
      }
   }

   // Start slaves
   if (!StartSlaves(parallelStartup, attach))
      return 0;

   if (fgSemaphore)
      SafeDelete(fgSemaphore);

   // we are now properly initialized
   fValid = kTRUE;

   // De-activate monitor (will be activated in Collect)
   fAllMonitor->DeActivateAll();

   // By default go into parallel mode
   GoParallel(9999, attach);

   // Send relevant initial state to slaves
   if (!attach)
      SendInitialState();
   else if (!IsIdle())
      // redirect log
      fRedirLog = kTRUE;

   // Done at this point, the alias will be communicated to the coordinator, if any
   if (!IsMaster())
      SetAlias(al);

   SetActive(kFALSE);

   if (IsValid()) {

      // Activate input handler
      ActivateAsyncInput();

      R__LOCKGUARD2(gROOTMutex);
      gROOT->GetListOfSockets()->Add(this);
   }
   return fActiveSlaves->GetSize();
}

//______________________________________________________________________________
void TProof::SetManager(TProofMgr *mgr)
{
   // Set manager and schedule its destruction after this for clean
   // operations.

   fManager = mgr;

   if (mgr) {
      R__LOCKGUARD2(gROOTMutex);
      gROOT->GetListOfSockets()->Remove(mgr);
      gROOT->GetListOfSockets()->Add(mgr);
   }
}

//______________________________________________________________________________
Bool_t TProof::StartSlaves(Bool_t parallel, Bool_t attach)
{
   // Start up PROOF slaves.

   // If this is a master server, find the config file and start slave
   // servers as specified in the config file
   if (IsMaster()) {

      Int_t pc = 0;
      TList *workerList = new TList;
      // Get list of workers
      if (gProofServ->GetWorkers(workerList, pc) == TProofServ::kQueryStop) {
         Error("StartSlaves", "getting list of worker nodes");
         return kFALSE;
      }
      fImage = gProofServ->GetImage();

      // Get all workers
      UInt_t nSlaves = workerList->GetSize();
      UInt_t nSlavesDone = 0;
      Int_t ord = 0;

      // Init arrays for threads, if neeeded
      std::vector<TProofThread *> thrHandlers;
      if (parallel) {
         thrHandlers.reserve(nSlaves);
         if (thrHandlers.max_size() < nSlaves) {
            PDB(kGlobal,1)
               Info("StartSlaves","cannot reserve enough space for thread"
                    " handlers - switch to serial startup");
            parallel = kFALSE;
         }
      }

      // Loop over all workers and start them
      TListIter next(workerList);
      TObject *to;
      TProofNodeInfo *worker;
      while ((to = next())) {
         // Get the next worker from the list
         worker = (TProofNodeInfo *)to;

         // Read back worker node info
         const Char_t *image = worker->GetImage().Data();
         const Char_t *workdir = worker->GetWorkDir().Data();
         Int_t perfidx = worker->GetPerfIndex();
         Int_t sport = worker->GetPort();
         if (sport == -1)
            sport = fUrl.GetPort();

         // create slave server
         TString fullord = TString(gProofServ->GetOrdinal()) + "." + ((Long_t) ord);
         if (parallel) {
            // Prepare arguments
            TProofThreadArg *ta =
               new TProofThreadArg(worker->GetNodeName().Data(), sport,
                                   fullord, perfidx, image, workdir,
                                   fSlaves, this);
            if (ta) {
               // The type of the thread func makes it a detached thread
               TThread *th = new TThread(SlaveStartupThread, ta);
               if (!th) {
                  Info("StartSlaves","Can't create startup thread:"
                       " out of system resources");
                  SafeDelete(ta);
               } else {
                  // Save in vector
                  thrHandlers.push_back(new TProofThread(th, ta));
                  // Run the thread
                  th->Run();
                  // Notify opening of connection
                  nSlavesDone++;
                  TMessage m(kPROOF_SERVERSTARTED);
                  m << TString("Opening connections to workers") << nSlaves
                    << nSlavesDone << kTRUE;
                  gProofServ->GetSocket()->Send(m);
               }
            } // end if (ta)
            else {
               Info("StartSlaves","Can't create thread arguments object:"
                    " out of system resources");
            }
         } // end if parallel
         else {
            // create slave server
            TUrl u(Form("%s:%d",worker->GetNodeName().Data(), sport));
            // Add group info in the password firdl, if any
            if (strlen(gProofServ->GetGroup()) > 0) {
               // Set also the user, otherwise the password is not exported
               if (strlen(u.GetUser()) <= 0)
                  u.SetUser(gProofServ->GetUser());
               u.SetPasswd(gProofServ->GetGroup());
            }
            TSlave *slave = CreateSlave(u.GetUrl(), fullord, perfidx,
                                        image, workdir);

            // Add to global list (we will add to the monitor list after
            // finalizing the server startup)
            Bool_t slaveOk = kTRUE;
            if (slave->IsValid()) {
               fSlaves->Add(slave);
            } else {
               slaveOk = kFALSE;
               fBadSlaves->Add(slave);
            }

            PDB(kGlobal,3)
               Info("StartSlaves", "worker on host %s created"
                    " and added to list", worker->GetNodeName().Data());

            // Notify opening of connection
            nSlavesDone++;
            TMessage m(kPROOF_SERVERSTARTED);
            m << TString("Opening connections to workers") << nSlaves
              << nSlavesDone << slaveOk;
            gProofServ->GetSocket()->Send(m);
         }
         ord++;
      } //end of worker loop

      // Cleanup
      SafeDelete(workerList);

      nSlavesDone = 0;
      if (parallel) {

         // Wait completion of startup operations
         std::vector<TProofThread *>::iterator i;
         for (i = thrHandlers.begin(); i != thrHandlers.end(); ++i) {
            TProofThread *pt = *i;

            // Wait on this condition
            if (pt && pt->fThread->GetState() == TThread::kRunningState) {
               PDB(kGlobal,3)
                  Info("Init",
                       "parallel startup: waiting for worker %s (%s:%d)",
                        pt->fArgs->fOrd.Data(), pt->fArgs->fUrl->GetHost(),
                        pt->fArgs->fUrl->GetPort());
               pt->fThread->Join();
            }

            // Notify end of startup operations
            nSlavesDone++;
            TMessage m(kPROOF_SERVERSTARTED);
            m << TString("Setting up worker servers") << nSlaves
              << nSlavesDone << kTRUE;
            gProofServ->GetSocket()->Send(m);
         }

         TIter next(fSlaves);
         TSlave *sl = 0;
         while ((sl = (TSlave *)next())) {
            if (sl->IsValid())
               fAllMonitor->Add(sl->GetSocket());
            else
               fBadSlaves->Add(sl);
         }

         // We can cleanup now
         while (!thrHandlers.empty()) {
            i = thrHandlers.end()-1;
            if (*i) {
               SafeDelete(*i);
               thrHandlers.erase(i);
            }
         }

      } else {

         // Here we finalize the server startup: in this way the bulk
         // of remote operations are almost parallelized
         TIter nxsl(fSlaves);
         TSlave *sl = 0;
         while ((sl = (TSlave *) nxsl())) {

            // Finalize setup of the server
            if (sl->IsValid())
               sl->SetupServ(TSlave::kSlave, 0);

            // Monitor good slaves
            Bool_t slaveOk = kTRUE;
            if (sl->IsValid()) {
               fAllMonitor->Add(sl->GetSocket());
            } else {
               slaveOk = kFALSE;
               fBadSlaves->Add(sl);
            }

            // Notify end of startup operations
            nSlavesDone++;
            TMessage m(kPROOF_SERVERSTARTED);
            m << TString("Setting up worker servers") << nSlaves
              << nSlavesDone << slaveOk;
            gProofServ->GetSocket()->Send(m);
         }
      }

   } else {

      // create master server
      fprintf(stderr,"Starting master: opening connection ... \n");
      TSlave *slave = CreateSubmaster(fUrl.GetUrl(), "0", "master", 0);

      if (slave->IsValid()) {

         // Notify
         fprintf(stderr,"Starting master:"
                        " connection open: setting up server ...             \r");
         StartupMessage("Connection to master opened", kTRUE, 1, 1);

         if (!attach) {

            // Set worker interrupt handler
            slave->SetInterruptHandler(kTRUE);

            // Finalize setup of the server
            slave->SetupServ(TSlave::kMaster, fConfFile);

            if (slave->IsValid()) {

               // Notify
               fprintf(stderr,"Starting master: OK                                     \n");
               StartupMessage("Master started", kTRUE, 1, 1);

               // check protocol compatibility
               // protocol 1 is not supported anymore
               if (fProtocol == 1) {
                  Error("StartSlaves",
                        "client and remote protocols not compatible (%d and %d)",
                        kPROOF_Protocol, fProtocol);
                  slave->Close("S");
                  delete slave;
                  return kFALSE;
               }

               fSlaves->Add(slave);
               fAllMonitor->Add(slave->GetSocket());

               // Unset worker interrupt handler
               slave->SetInterruptHandler(kFALSE);

               // Set interrupt PROOF handler from now on
               fIntHandler = new TProofInterruptHandler(this);

               Collect(slave);
               Int_t slStatus = slave->GetStatus();
               if (slStatus == -99 || slStatus == -98) {
                  fSlaves->Remove(slave);
                  fAllMonitor->Remove(slave->GetSocket());
                  if (slStatus == -99)
                     Error("StartSlaves", "not allowed to connect to PROOF master server");
                  else if (slStatus == -98)
                     Error("StartSlaves", "could not setup output redirection on master");
                  else
                     Error("StartSlaves", "setting up master");
                  slave->Close("S");
                  delete slave;
                  return 0;
               }

               if (!slave->IsValid()) {
                  fSlaves->Remove(slave);
                  fAllMonitor->Remove(slave->GetSocket());
                  slave->Close("S");
                  delete slave;
                  Error("StartSlaves",
                        "failed to setup connection with PROOF master server");
                  return kFALSE;
               }

               if (!gROOT->IsBatch()) {
                  if ((fProgressDialog =
                     gROOT->GetPluginManager()->FindHandler("TProofProgressDialog")))
                     if (fProgressDialog->LoadPlugin() == -1)
                        fProgressDialog = 0;
               }
            } else {
               // Notify
               fprintf(stderr,"Starting master: failure\n");
            }
         } else {

            // Notify
            if (attach) {
               fprintf(stderr,"Starting master: OK                                     \n");
               StartupMessage("Master attached", kTRUE, 1, 1);

               if (!gROOT->IsBatch()) {
                  if ((fProgressDialog =
                     gROOT->GetPluginManager()->FindHandler("TProofProgressDialog")))
                     if (fProgressDialog->LoadPlugin() == -1)
                        fProgressDialog = 0;
               }
            } else {
               fprintf(stderr,"Starting manager: OK                                    \n");
               StartupMessage("Manager started", kTRUE, 1, 1);
            }

            fSlaves->Add(slave);
            fAllMonitor->Add(slave->GetSocket());

            fIntHandler = new TProofInterruptHandler(this);
         }

      } else {
         delete slave;
         Error("StartSlaves", "failed to connect to a PROOF master server");
         return kFALSE;
      }
   }

   return kTRUE;
}

//______________________________________________________________________________
void TProof::Close(Option_t *opt)
{
   // Close all open slave servers.
   // Client can decide to shutdown the remote session by passing option is 'S'
   // or 's'. Default for clients is detach, if supported. Masters always
   // shutdown the remote counterpart.

   if (fSlaves) {
      if (fIntHandler)
         fIntHandler->Remove();

      TIter nxs(fSlaves);
      TSlave *sl = 0;
      while ((sl = (TSlave *)nxs()))
         sl->Close(opt);

      fActiveSlaves->Clear("nodelete");
      fUniqueSlaves->Clear("nodelete");
      fAllUniqueSlaves->Clear("nodelete");
      fNonUniqueMasters->Clear("nodelete");
      fBadSlaves->Clear("nodelete");
      fSlaves->Delete();
   }

   {
      R__LOCKGUARD2(gROOTMutex);
      gROOT->GetListOfSockets()->Remove(this);

      if (IsProofd()) {

         gROOT->GetListOfProofs()->Remove(this);
         if (gProof && gProof == this) {
            // Set previous proofd-related as default
            TIter pvp(gROOT->GetListOfProofs(), kIterBackward);
            while ((gProof = (TProof *)pvp())) {
               if (gProof->IsProofd())
                  break;
            }
         }
      }
   }
}

//______________________________________________________________________________
TSlave *TProof::CreateSlave(const char *url, const char *ord,
                            Int_t perf, const char *image, const char *workdir)
{
   // Create a new TSlave of type TSlave::kSlave.
   // Note: creation of TSlave is private with TProof as a friend.
   // Derived classes must use this function to create slaves.

   TSlave* sl = TSlave::Create(url, ord, perf, image,
                               this, TSlave::kSlave, workdir, 0);

   if (sl->IsValid()) {
      sl->SetInputHandler(new TProofInputHandler(this, sl->GetSocket()));
      // must set fParallel to 1 for slaves since they do not
      // report their fParallel with a LOG_DONE message
      sl->fParallel = 1;
   }

   return sl;
}

//______________________________________________________________________________
TSlave *TProof::CreateSubmaster(const char *url, const char *ord,
                                const char *image, const char *msd)
{
   // Create a new TSlave of type TSlave::kMaster.
   // Note: creation of TSlave is private with TProof as a friend.
   // Derived classes must use this function to create slaves.

   TSlave *sl = TSlave::Create(url, ord, 100, image, this,
                               TSlave::kMaster, 0, msd);

   if (sl->IsValid()) {
      sl->SetInputHandler(new TProofInputHandler(this, sl->GetSocket()));
   }

   return sl;
}

//______________________________________________________________________________
TSlave *TProof::FindSlave(TSocket *s) const
{
   // Find slave that has TSocket s. Returns 0 in case slave is not found.

   TSlave *sl;
   TIter   next(fSlaves);

   while ((sl = (TSlave *)next())) {
      if (sl->IsValid() && sl->GetSocket() == s)
         return sl;
   }
   return 0;
}

//______________________________________________________________________________
void TProof::FindUniqueSlaves()
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

   TIter next(fActiveSlaves);

   while (TSlave *sl = dynamic_cast<TSlave*>(next())) {
      if (fImage == sl->fImage) {
         if (sl->GetSlaveType() == TSlave::kMaster) {
            fNonUniqueMasters->Add(sl);
            fAllUniqueSlaves->Add(sl);
            fAllUniqueMonitor->Add(sl->GetSocket());
         }
         continue;
      }

      TIter next2(fUniqueSlaves);
      TSlave *replace_slave = 0;
      Bool_t add = kTRUE;
      while (TSlave *sl2 = dynamic_cast<TSlave*>(next2())) {
         if (sl->fImage == sl2->fImage) {
            add = kFALSE;
            if (sl->GetSlaveType() == TSlave::kMaster) {
               if (sl2->GetSlaveType() == TSlave::kSlave) {
                  // give preference to master
                  replace_slave = sl2;
                  add = kTRUE;
               } else if (sl2->GetSlaveType() == TSlave::kMaster) {
                  fNonUniqueMasters->Add(sl);
                  fAllUniqueSlaves->Add(sl);
                  fAllUniqueMonitor->Add(sl->GetSocket());
               } else {
                  Error("FindUniqueSlaves", "TSlave is neither Master nor Slave");
                  R__ASSERT(0);
               }
            }
            break;
         }
      }

      if (add) {
         fUniqueSlaves->Add(sl);
         fAllUniqueSlaves->Add(sl);
         fUniqueMonitor->Add(sl->GetSocket());
         fAllUniqueMonitor->Add(sl->GetSocket());
         if (replace_slave) {
            fUniqueSlaves->Remove(replace_slave);
            fAllUniqueSlaves->Remove(replace_slave);
            fUniqueMonitor->Remove(replace_slave->GetSocket());
            fAllUniqueMonitor->Remove(replace_slave->GetSocket());
         }
      }
   }

   // will be actiavted in Collect()
   fUniqueMonitor->DeActivateAll();
   fAllUniqueMonitor->DeActivateAll();
}

//______________________________________________________________________________
Int_t TProof::GetNumberOfSlaves() const
{
   // Return number of slaves as described in the config file.

   return fSlaves->GetSize();
}

//______________________________________________________________________________
Int_t TProof::GetNumberOfActiveSlaves() const
{
   // Return number of active slaves, i.e. slaves that are valid and in
   // the current computing group.

   return fActiveSlaves->GetSize();
}

//______________________________________________________________________________
Int_t TProof::GetNumberOfInactiveSlaves() const
{
   // Return number of inactive slaves, i.e. slaves that are valid but not in
   // the current computing group.

   return fInactiveSlaves->GetSize();
}

//______________________________________________________________________________
Int_t TProof::GetNumberOfUniqueSlaves() const
{
   // Return number of unique slaves, i.e. active slaves that have each a
   // unique different user files system.

   return fUniqueSlaves->GetSize();
}

//______________________________________________________________________________
Int_t TProof::GetNumberOfBadSlaves() const
{
   // Return number of bad slaves. This are slaves that we in the config
   // file, but refused to startup or that died during the PROOF session.

   return fBadSlaves->GetSize();
}

//______________________________________________________________________________
void TProof::AskStatistics()
{
   // Ask the for the statistics of the slaves.

   if (!IsValid()) return;

   Broadcast(kPROOF_GETSTATS, kActive);
   Collect(kActive);
}

//______________________________________________________________________________
void TProof::AskParallel()
{
   // Ask the for the number of parallel slaves.

   if (!IsValid()) return;

   Broadcast(kPROOF_GETPARALLEL, kActive);
   Collect(kActive);
}

//______________________________________________________________________________
TList *TProof::GetListOfQueries(Option_t *opt)
{
   // Ask the master for the list of queries.

   if (!IsValid() || IsMaster()) return (TList *)0;

   Bool_t all = ((strchr(opt,'A') || strchr(opt,'a'))) ? kTRUE : kFALSE;
   TMessage m(kPROOF_QUERYLIST);
   m << all;
   Broadcast(m, kActive);
   Collect(kActive);

   // This should have been filled by now
   return fQueries;
}

//______________________________________________________________________________
Int_t TProof::GetNumberOfQueries()
{
   // Number of queries processed by this session

   if (fQueries)
      return fQueries->GetSize() - fOtherQueries;
   return 0;
}

//______________________________________________________________________________
void TProof::SetMaxDrawQueries(Int_t max)
{
   // Set max number of draw queries whose results are saved

   if (max > 0) {
      if (fPlayer)
         fPlayer->SetMaxDrawQueries(max);
      fMaxDrawQueries = max;
   }
}

//______________________________________________________________________________
void TProof::GetMaxQueries()
{
   // Get max number of queries whose full results are kept in the
   // remote sandbox

   TMessage m(kPROOF_MAXQUERIES);
   m << kFALSE;
   Broadcast(m, kActive);
   Collect(kActive);
}

//______________________________________________________________________________
TList *TProof::GetQueryResults()
{
   // Return pointer to the list of query results in the player

   return fPlayer->GetListOfResults();
}

//______________________________________________________________________________
TQueryResult *TProof::GetQueryResult(const char *ref)
{
   // Return pointer to the full TQueryResult instance owned by the player
   // and referenced by 'ref'. If ref = 0 or "", return the last query result.

   return fPlayer->GetQueryResult(ref);
}

//______________________________________________________________________________
void TProof::ShowQueries(Option_t *opt)
{
   // Ask the master for the list of queries.
   // Options:
   //           "A"     show information about all the queries known to the
   //                   server, i.e. even those processed by other sessions
   //           "L"     show only information about queries locally available
   //                   i.e. already retrieved. If "L" is specified, "A" is
   //                   ignored.
   //           "F"     show all details available about queries
   //           "H"     print help menu
   // Default ""

   Bool_t help = ((strchr(opt,'H') || strchr(opt,'h'))) ? kTRUE : kFALSE;
   if (help) {

      // Help

      Printf("+++");
      Printf("+++ Options: \"A\" show all queries known to server");
      Printf("+++          \"L\" show retrieved queries");
      Printf("+++          \"F\" full listing of query info");
      Printf("+++          \"H\" print this menu");
      Printf("+++");
      Printf("+++ (case insensitive)");
      Printf("+++");
      Printf("+++ Use Retrieve(<#>) to retrieve the full"
             " query results from the master");
      Printf("+++     e.g. Retrieve(8)");

      Printf("+++");

      return;
   }

   if (!IsValid()) return;

   Bool_t local = ((strchr(opt,'L') || strchr(opt,'l'))) ? kTRUE : kFALSE;

   TObject *pq = 0;
   if (!local) {
      GetListOfQueries(opt);

      if (!fQueries) return;

      TIter nxq(fQueries);

      // Queries processed by other sessions
      if (fOtherQueries > 0) {
         Printf("+++");
         Printf("+++ Queries processed during other sessions: %d", fOtherQueries);
         Int_t nq = 0;
         while (nq++ < fOtherQueries && (pq = nxq()))
            pq->Print(opt);
      }

      // Queries processed by this session
      Printf("+++");
      Printf("+++ Queries processed during this session: selector: %d, draw: %d",
              GetNumberOfQueries(), fDrawQueries);
      while ((pq = nxq()))
         pq->Print(opt);

   } else {

      // Queries processed by this session
      Printf("+++");
      Printf("+++ Queries processed during this session: selector: %d, draw: %d",
              GetNumberOfQueries(), fDrawQueries);

      // Queries available locally
      TList *listlocal = fPlayer->GetListOfResults();
      if (listlocal) {
         Printf("+++");
         Printf("+++ Queries available locally: %d", listlocal->GetSize());
         TIter nxlq(listlocal);
         while ((pq = nxlq()))
            pq->Print(opt);
      }
   }
   Printf("+++");
}

//______________________________________________________________________________
Bool_t TProof::IsDataReady(Long64_t &totalbytes, Long64_t &bytesready)
{
   // See if the data is ready to be analyzed.

   if (!IsValid()) return kFALSE;

   TList submasters;
   TIter nextSlave(GetListOfActiveSlaves());
   while (TSlave *sl = dynamic_cast<TSlave*>(nextSlave())) {
      if (sl->GetSlaveType() == TSlave::kMaster) {
         submasters.Add(sl);
      }
   }

   fDataReady = kTRUE; //see if any submasters set it to false
   fBytesReady = 0;
   fTotalBytes = 0;
   //loop over submasters and see if data is ready
   if (submasters.GetSize() > 0) {
      Broadcast(kPROOF_DATA_READY, &submasters);
      Collect(&submasters);
   }

   bytesready = fBytesReady;
   totalbytes = fTotalBytes;

   EmitVA("IsDataReady(Long64_t,Long64_t)", 2, totalbytes, bytesready);

   //PDB(kGlobal,2)
   Info("IsDataReady", "%lld / %lld (%s)",
        bytesready, totalbytes, fDataReady?"READY":"NOT READY");

   return fDataReady;
}

//______________________________________________________________________________
void TProof::Interrupt(EUrgent type, ESlaves list)
{
   // Send interrupt to master or slave servers.

   if (!IsValid()) return;

   TList *slaves = 0;
   if (list == kAll)       slaves = fSlaves;
   if (list == kActive)    slaves = fActiveSlaves;
   if (list == kUnique)    slaves = fUniqueSlaves;
   if (list == kAllUnique) slaves = fAllUniqueSlaves;

   if (slaves->GetSize() == 0) return;

   TSlave *sl;
   TIter   next(slaves);

   while ((sl = (TSlave *)next())) {
      if (sl->IsValid()) {

         // Ask slave to progate the interrupt request
         sl->Interrupt((Int_t)type);
      }
   }
}

//______________________________________________________________________________
Int_t TProof::GetParallel() const
{
   // Returns number of slaves active in parallel mode. Returns 0 in case
   // there are no active slaves. Returns -1 in case of error.

   if (!IsValid()) return -1;

   // iterate over active slaves and return total number of slaves
   TIter nextSlave(GetListOfActiveSlaves());
   Int_t nparallel = 0;
   while (TSlave* sl = dynamic_cast<TSlave*>(nextSlave()))
      if (sl->GetParallel() >= 0)
         nparallel += sl->GetParallel();

   return nparallel;
}

//______________________________________________________________________________
TList *TProof::GetListOfSlaveInfos()
{
   // Returns list of TSlaveInfo's. In case of error return 0.

   if (!IsValid()) return 0;

   if (fSlaveInfo == 0) {
      fSlaveInfo = new TSortedList(kSortDescending);
      fSlaveInfo->SetOwner();
   } else {
      fSlaveInfo->Delete();
   }

   TList masters;
   TIter next(GetListOfSlaves());
   TSlave *slave;

   while ((slave = (TSlave *) next()) != 0) {
      if (slave->GetSlaveType() == TSlave::kSlave) {
         TSlaveInfo *slaveinfo = new TSlaveInfo(slave->GetOrdinal(),
                                                slave->GetName(),
                                                slave->GetPerfIdx());
         fSlaveInfo->Add(slaveinfo);

         TIter nextactive(GetListOfActiveSlaves());
         TSlave *activeslave;
         while ((activeslave = (TSlave *) nextactive())) {
            if (TString(slaveinfo->GetOrdinal()) == activeslave->GetOrdinal()) {
               slaveinfo->SetStatus(TSlaveInfo::kActive);
               break;
            }
         }

         TIter nextbad(GetListOfBadSlaves());
         TSlave *badslave;
         while ((badslave = (TSlave *) nextbad())) {
            if (TString(slaveinfo->GetOrdinal()) == badslave->GetOrdinal()) {
               slaveinfo->SetStatus(TSlaveInfo::kBad);
               break;
            }
         }

      } else if (slave->GetSlaveType() == TSlave::kMaster) {
         if (slave->IsValid()) {
            if (slave->GetSocket()->Send(kPROOF_GETSLAVEINFO) == -1)
               MarkBad(slave);
            else
               masters.Add(slave);
         }
      } else {
         Error("GetSlaveInfo", "TSlave is neither Master nor Slave");
         R__ASSERT(0);
      }
   }
   if (masters.GetSize() > 0) Collect(&masters);

   return fSlaveInfo;
}

//______________________________________________________________________________
void TProof::Activate(TList *slaves)
{
   // Activate slave server list.

   TMonitor *mon = fAllMonitor;
   mon->DeActivateAll();

   slaves = !slaves ? fActiveSlaves : slaves;

   TIter next(slaves);
   TSlave *sl;
   while ((sl = (TSlave*) next())) {
      if (sl->IsValid())
         mon->Activate(sl->GetSocket());
   }
}

//______________________________________________________________________________
Int_t TProof::BroadcastGroupPriority(const char *grp, Int_t priority, TList *workers)
{
   // Broadcast the group priority to all workers in the specified list. Returns
   // the number of workers the message was successfully sent to.
   // Returns -1 in case of error.

   if (!IsValid()) return -1;

   if (workers->GetSize() == 0) return 0;

   int   nsent = 0;
   TIter next(workers);

   TSlave *wrk;
   while ((wrk = (TSlave *)next())) {
      if (wrk->IsValid()) {
         if (wrk->SendGroupPriority(grp, priority) == -1)
            MarkBad(wrk);
         else
            nsent++;
      }
   }

   return nsent;
}

//______________________________________________________________________________
Int_t TProof::BroadcastGroupPriority(const char *grp, Int_t priority, ESlaves list)
{
   // Broadcast the group priority to all workers in the specified list. Returns
   // the number of workers the message was successfully sent to.
   // Returns -1 in case of error.

   TList *workers = 0;
   if (list == kAll)       workers = fSlaves;
   if (list == kActive)    workers = fActiveSlaves;
   if (list == kUnique)    workers = fUniqueSlaves;
   if (list == kAllUnique) workers = fAllUniqueSlaves;

   return BroadcastGroupPriority(grp, priority, workers);
}

//______________________________________________________________________________
Int_t TProof::Broadcast(const TMessage &mess, TList *slaves)
{
   // Broadcast a message to all slaves in the specified list. Returns
   // the number of slaves the message was successfully sent to.
   // Returns -1 in case of error.

   if (!IsValid()) return -1;

   if (slaves->GetSize() == 0) return 0;

   int   nsent = 0;
   TIter next(slaves);

   TSlave *sl;
   while ((sl = (TSlave *)next())) {
      if (sl->IsValid()) {
         if (sl->GetSocket()->Send(mess) == -1)
            MarkBad(sl);
         else
            nsent++;
      }
   }

   return nsent;
}

//______________________________________________________________________________
Int_t TProof::Broadcast(const TMessage &mess, ESlaves list)
{
   // Broadcast a message to all slaves in the specified list (either
   // all slaves or only the active slaves). Returns the number of slaves
   // the message was successfully sent to. Returns -1 in case of error.

   TList *slaves = 0;
   if (list == kAll)       slaves = fSlaves;
   if (list == kActive)    slaves = fActiveSlaves;
   if (list == kUnique)    slaves = fUniqueSlaves;
   if (list == kAllUnique) slaves = fAllUniqueSlaves;

   return Broadcast(mess, slaves);
}

//______________________________________________________________________________
Int_t TProof::Broadcast(const char *str, Int_t kind, TList *slaves)
{
   // Broadcast a character string buffer to all slaves in the specified
   // list. Use kind to set the TMessage what field. Returns the number of
   // slaves the message was sent to. Returns -1 in case of error.

   TMessage mess(kind);
   if (str) mess.WriteString(str);
   return Broadcast(mess, slaves);
}

//______________________________________________________________________________
Int_t TProof::Broadcast(const char *str, Int_t kind, ESlaves list)
{
   // Broadcast a character string buffer to all slaves in the specified
   // list (either all slaves or only the active slaves). Use kind to
   // set the TMessage what field. Returns the number of slaves the message
   // was sent to. Returns -1 in case of error.

   TMessage mess(kind);
   if (str) mess.WriteString(str);
   return Broadcast(mess, list);
}

//______________________________________________________________________________
Int_t TProof::BroadcastObject(const TObject *obj, Int_t kind, TList *slaves)
{
   // Broadcast an object to all slaves in the specified list. Use kind to
   // set the TMEssage what field. Returns the number of slaves the message
   // was sent to. Returns -1 in case of error.

   TMessage mess(kind);
   mess.WriteObject(obj);
   return Broadcast(mess, slaves);
}

//______________________________________________________________________________
Int_t TProof::BroadcastObject(const TObject *obj, Int_t kind, ESlaves list)
{
   // Broadcast an object to all slaves in the specified list. Use kind to
   // set the TMEssage what field. Returns the number of slaves the message
   // was sent to. Returns -1 in case of error.

   TMessage mess(kind);
   mess.WriteObject(obj);
   return Broadcast(mess, list);
}

//______________________________________________________________________________
Int_t TProof::BroadcastRaw(const void *buffer, Int_t length, TList *slaves)
{
   // Broadcast a raw buffer of specified length to all slaves in the
   // specified list. Returns the number of slaves the buffer was sent to.
   // Returns -1 in case of error.

   if (!IsValid()) return -1;

   if (slaves->GetSize() == 0) return 0;

   int   nsent = 0;
   TIter next(slaves);

   TSlave *sl;
   while ((sl = (TSlave *)next())) {
      if (sl->IsValid()) {
         if (sl->GetSocket()->SendRaw(buffer, length) == -1)
            MarkBad(sl);
         else
            nsent++;
      }
   }

   return nsent;
}

//______________________________________________________________________________
Int_t TProof::BroadcastRaw(const void *buffer, Int_t length, ESlaves list)
{
   // Broadcast a raw buffer of specified length to all slaves in the
   // specified list. Returns the number of slaves the buffer was sent to.
   // Returns -1 in case of error.

   TList *slaves = 0;
   if (list == kAll)       slaves = fSlaves;
   if (list == kActive)    slaves = fActiveSlaves;
   if (list == kUnique)    slaves = fUniqueSlaves;
   if (list == kAllUnique) slaves = fAllUniqueSlaves;

   return BroadcastRaw(buffer, length, slaves);
}

//______________________________________________________________________________
Int_t TProof::Collect(const TSlave *sl, Long_t timeout)
{
   // Collect responses from slave sl. Returns the number of slaves that
   // responded (=1).
   // If timeout >= 0, wait at most timeout seconds (timeout = -1 by default,
   // which means wait forever).

   if (!sl->IsValid()) return 0;

   TMonitor *mon = fAllMonitor;

   mon->DeActivateAll();
   mon->Activate(sl->GetSocket());

   return Collect(mon, timeout);
}

//______________________________________________________________________________
Int_t TProof::Collect(TList *slaves, Long_t timeout)
{
   // Collect responses from the slave servers. Returns the number of slaves
   // that responded.
   // If timeout >= 0, wait at most timeout seconds (timeout = -1 by default,
   // which means wait forever).

   TMonitor *mon = fAllMonitor;
   mon->DeActivateAll();

   TIter next(slaves);
   TSlave *sl;
   while ((sl = (TSlave*) next())) {
      if (sl->IsValid())
         mon->Activate(sl->GetSocket());
   }

   return Collect(mon, timeout);
}

//______________________________________________________________________________
Int_t TProof::Collect(ESlaves list, Long_t timeout)
{
   // Collect responses from the slave servers. Returns the number of slaves
   // that responded.
   // If timeout >= 0, wait at most timeout seconds (timeout = -1 by default,
   // which means wait forever).

   TMonitor *mon = 0;
   if (list == kAll)       mon = fAllMonitor;
   if (list == kActive)    mon = fActiveMonitor;
   if (list == kUnique)    mon = fUniqueMonitor;
   if (list == kAllUnique) mon = fAllUniqueMonitor;

   mon->ActivateAll();

   return Collect(mon, timeout);
}

//______________________________________________________________________________
Int_t TProof::Collect(TMonitor *mon, Long_t timeout)
{
   // Collect responses from the slave servers. Returns the number of messages
   // received. Can be 0 if there are no active slaves.
   // If timeout >= 0, wait at most timeout seconds (timeout = -1 by default,
   // which means wait forever).

   fStatus = 0;
   if (!mon->GetActive()) return 0;

   DeActivateAsyncInput();

   // Used by external code to know what we are monitoring
   fCurrentMonitor = mon;

   // We want messages on the main window during synchronous collection,
   // but we save the present status to restore it at the end
   Bool_t saveRedirLog = fRedirLog;
   if (!IsIdle() && !IsSync())
      fRedirLog = kFALSE;

   int cnt = 0, rc = 0;

   fBytesRead = 0;
   fRealTime  = 0.0;
   fCpuTime   = 0.0;

   // Timeout counter
   Long_t nto = timeout;
   if (gDebug > 2)
      Info("Collect","active: %d", mon->GetActive());

   // On clients, handle Ctrl-C during collection
   if (fIntHandler)
      fIntHandler->Add();

   while (mon->GetActive() && (nto < 0 || nto > 0)) {

      // Wait for a ready socket
      TSocket *s = mon->Select(1000);

      if (s && s != (TSocket *)(-1)) {
         // Get and analyse the info it did receive
         if ((rc = CollectInputFrom(s)) == 1) {
            // Deactivate it if we are done with it
            mon->DeActivate(s);
            if (gDebug > 2)
               Info("Collect","deactivating %p (active: %d, %p)",
                              s, mon->GetActive(),
                              mon->GetListOfActives()->First());
         }

         // Update counter (if no error occured)
         if (rc >= 0)
            cnt++;
      } else {
         // If not timed-out, exit if not stopped or not aborted
         // (player exits status is finished in such a case); otherwise,
         // we still need to collect the partial output info
         if (!s)
            if (fPlayer && (fPlayer->GetExitStatus() == TVirtualProofPlayer::kFinished))
               mon->DeActivateAll();
         // Decrease the timeout counter if requested
         if (s == (TSocket *)(-1) && nto > 0)
            nto--;
      }
   }

   // If timed-out, deactivate the remaining sockets
   if (nto == 0)
      mon->DeActivateAll();

   // Deactivate Ctrl-C special handler
   if (fIntHandler)
      fIntHandler->Remove();

   // make sure group view is up to date
   SendGroupView();

   // Restore redirection setting
   fRedirLog = saveRedirLog;

   // To avoid useless loops in external code
   fCurrentMonitor = 0;

   ActivateAsyncInput();

   return cnt;
}

//______________________________________________________________________________
R__HIDDEN void TProof::CleanGDirectory(TList *ol)
{
   // Remove links to objects in list 'ol' from gDirectory

   if (ol) {
      TIter nxo(ol);
      TObject *o = 0;
      while ((o = nxo()))
         gDirectory->RecursiveRemove(o);
   }
}

//______________________________________________________________________________
Int_t TProof::CollectInputFrom(TSocket *s)
{
   // Collect and analyze available input from socket s.
   // Returns 0 on success, -1 if any failure occurs.

   TMessage *mess;
   Int_t rc = 0;

   char      str[512];
   TSlave   *sl;
   TObject  *obj;
   Int_t     what;
   Bool_t    delete_mess = kTRUE;

   if (s->Recv(mess) < 0) {
      MarkBad(s);
      return -1;
   }
   if (!mess) {
      // we get here in case the remote server died
      MarkBad(s);
      return -1;
   }

   what = mess->What();

   PDB(kGlobal,3) {
      sl = FindSlave(s);
      Info("CollectInputFrom","got %d from %s", what, (sl ? sl->GetOrdinal() : "undef"));
   }

   switch (what) {

      case kMESS_OBJECT:
         fPlayer->HandleRecvHisto(mess);
         break;

      case kPROOF_FATAL:
         MarkBad(s);
         break;

      case kPROOF_GETOBJECT:
         // send slave object it asks for
         mess->ReadString(str, sizeof(str));
         obj = gDirectory->Get(str);
         if (obj)
            s->SendObject(obj);
         else
            s->Send(kMESS_NOTOK);
         break;

      case kPROOF_GETPACKET:
         {
            TDSetElement *elem = 0;
            sl = FindSlave(s);
            elem = fPlayer->GetNextPacket(sl, mess);

            if (elem != (TDSetElement*) -1) {
               TMessage answ(kPROOF_GETPACKET);
               answ << elem;
               s->Send(answ);

               while (fWaitingSlaves != 0 && fWaitingSlaves->GetSize()) {
                  TPair *p = (TPair*) fWaitingSlaves->First();
                  s = (TSocket*) p->Key();
                  sl = FindSlave(s);
                  TMessage *m = (TMessage*) p->Value();

                  elem = fPlayer->GetNextPacket(sl, m);
                  if (elem != (TDSetElement*) -1) {
                     TMessage a(kPROOF_GETPACKET);
                     a << elem;
                     s->Send(a);
                     // remove has to happen via Links because TPair does not have
                     // a Compare() function and therefore RemoveFirst() and
                     // Remove(TObject*) do not work
                     fWaitingSlaves->Remove(fWaitingSlaves->FirstLink());
                     delete p;
                     delete m;
                  } else {
                     break;
                  }
               }
            } else {
               if (fWaitingSlaves == 0) fWaitingSlaves = new TList;
               fWaitingSlaves->Add(new TPair(s, mess));
               delete_mess = kFALSE;
            }
         }
         break;

      case kPROOF_LOGFILE:
         {
            Int_t size;
            (*mess) >> size;
            RecvLogFile(s, size);
         }
         break;

      case kPROOF_LOGDONE:
         sl = FindSlave(s);
         (*mess) >> sl->fStatus >> sl->fParallel;
         PDB(kGlobal,2)
            Info("CollectInputFrom","kPROOF_LOGDONE:%s: status %d  parallel %d",
                 sl->GetOrdinal(), sl->fStatus, sl->fParallel);
         if (sl->fStatus != 0) fStatus = sl->fStatus; //return last nonzero status
         rc = 1;
         break;

      case kPROOF_GETSTATS:
         sl = FindSlave(s);
         (*mess) >> sl->fBytesRead >> sl->fRealTime >> sl->fCpuTime
                 >> sl->fWorkDir >> sl->fProofWorkDir;
         fBytesRead += sl->fBytesRead;
         fRealTime  += sl->fRealTime;
         fCpuTime   += sl->fCpuTime;
         rc = 1;
         break;

      case kPROOF_GETPARALLEL:
         sl = FindSlave(s);
         (*mess) >> sl->fParallel;
         rc = 1;
         break;

      case kPROOF_PACKAGE_LIST:
         {
            PDB(kGlobal,2) Info("CollectInputFrom","kPROOF_PACKAGE_LIST: enter");
            Int_t type = 0;
            (*mess) >> type;
            switch (type) {
            case TProof::kListEnabledPackages:
               SafeDelete(fEnabledPackages);
               fEnabledPackages = (TList *) mess->ReadObject(TList::Class());
               fEnabledPackages->SetOwner();
               break;
            case TProof::kListPackages:
               SafeDelete(fAvailablePackages);
               fAvailablePackages = (TList *) mess->ReadObject(TList::Class());
               fAvailablePackages->SetOwner();
               break;
            default:
               Info("CollectInputFrom","kPROOF_PACKAGE_LIST: unknown type: %d", type);
            }
         }
         break;

      case kPROOF_OUTPUTOBJECT:
         {
            PDB(kGlobal,2) Info("CollectInputFrom","kPROOF_OUTPUTOBJECT: enter");
            Int_t type = 0;
            (*mess) >> type;
            // If a query result header, add it to the player list
            if (type == 0) {
               // Retrieve query result instance (output list not filled)
               TQueryResult *pq =
                  (TQueryResult *) mess->ReadObject(TQueryResult::Class());
               if (pq) {
                  // Add query to the result list in TProofPlayer
                  fPlayer->AddQueryResult(pq);
                  fPlayer->SetCurrentQuery(pq);
                  // Add the unique query tag as TNamed object to the input list
                  // so that it is available in TSelectors for monitoring
                  fPlayer->AddInput(new TNamed("PROOF_QueryTag",
                                    Form("%s:%s",pq->GetTitle(),pq->GetName())));
               } else {
                  Warning("CollectInputFrom","kPROOF_OUTPUTOBJECT: query result missing");
               }
            } else if (type > 0) {
               // Read object
               TObject *obj = mess->ReadObject(TObject::Class());
               // Add or merge it
               if ((fPlayer->AddOutputObject(obj) == 1))
                  // Remove the object if it has been merged
                  SafeDelete(obj);
               if (type > 1 && !IsMaster()) {
                  TQueryResult *pq = fPlayer->GetCurrentQuery();
                  pq->SetOutputList(fPlayer->GetOutputList(), kFALSE);
                  pq->SetInputList(fPlayer->GetInputList(), kFALSE);
                  // If the last object, notify the GUI that the result arrived
                  QueryResultReady(Form("%s:%s", pq->GetTitle(), pq->GetName()));
               }
            }
         }
         break;

      case kPROOF_OUTPUTLIST:
         {
            PDB(kGlobal,2) Info("CollectInputFrom","kPROOF_OUTPUTLIST: enter");
            TList *out = 0;
            if (IsMaster() || fProtocol < 7) {
               out = (TList *) mess->ReadObject(TList::Class());
            } else {
               TQueryResult *pq =
                  (TQueryResult *) mess->ReadObject(TQueryResult::Class());
               if (pq) {
                  // Add query to the result list in TProofPlayer
                  fPlayer->AddQueryResult(pq);
                  fPlayer->SetCurrentQuery(pq);
                  // To avoid accidental cleanups from anywhere else
                  // remove objects from gDirectory and clone the list
                  out = pq->GetOutputList();
                  CleanGDirectory(out);
                  out = (TList *) out->Clone();
                  // Notify the GUI that the result arrived
                  QueryResultReady(Form("%s:%s", pq->GetTitle(), pq->GetName()));
               } else {
                  PDB(kGlobal,2)
                     Info("CollectInputFrom","kPROOF_OUTPUTLIST: query result missing");
               }
            }
            if (out) {
               out->SetOwner();
               fPlayer->AddOutput(out); // Incorporate the list
               SafeDelete(out);
            } else {
               PDB(kGlobal,2) Info("CollectInputFrom","kPROOF_OUTPUTLIST: ouputlist is empty");
            }

            // On clients at this point processing is over
            if (!IsMaster()) {

               // Handle abort ...
               if (fPlayer->GetExitStatus() == TVirtualProofPlayer::kAborted) {
                  if (fSync)
                     Info("CollectInputFrom",
                          "processing was aborted - %lld events processed",
                          fPlayer->GetEventsProcessed());

                  if (GetRemoteProtocol() > 11) {
                     // New format
                     Progress(-1, fPlayer->GetEventsProcessed(), -1, -1., -1., -1., -1.);
                  } else {
                     Progress(-1, fPlayer->GetEventsProcessed());
                  }
                  Emit("StopProcess(Bool_t)", kTRUE);
               }

               // Handle stop ...
               if (fPlayer->GetExitStatus() == TVirtualProofPlayer::kStopped) {
                  if (fSync)
                     Info("CollectInputFrom",
                          "processing was stopped - %lld events processed",
                          fPlayer->GetEventsProcessed());

                  if (GetRemoteProtocol() > 11) {
                     // New format
                     Progress(-1, fPlayer->GetEventsProcessed(), -1, -1., -1., -1., -1.);
                  } else {
                     Progress(-1, fPlayer->GetEventsProcessed());
                  }
                  Emit("StopProcess(Bool_t)", kFALSE);
               }

               // Final update of the dialog box
               if (GetRemoteProtocol() > 11) {
                  // New format
                  EmitVA("Progress(Long64_t,Long64_t,Long64_t,Float_t,Float_t,Float_t,Float_t,)",
                          7, (Long64_t)(-1), (Long64_t)(-1), (Long64_t)(-1),
                             (Float_t)(-1.),(Float_t)(-1.),(Float_t)(-1.),(Float_t)(-1.));
               } else {
                  EmitVA("Progress(Long64_t,Long64_t)", 2, (Long64_t)(-1), (Long64_t)(-1));
               }
            }
         }
         break;

      case kPROOF_QUERYLIST:
         {
            PDB(kGlobal,2) Info("CollectInputFrom","kPROOF_QUERYLIST: enter");
            (*mess) >> fOtherQueries >> fDrawQueries;
            if (fQueries) {
               fQueries->Delete();
               delete fQueries;
               fQueries = 0;
            }
            fQueries = (TList *) mess->ReadObject(TList::Class());
         }
         break;

      case kPROOF_RETRIEVE:
         {
            PDB(kGlobal,2) Info("CollectInputFrom","kPROOF_RETRIEVE: enter");
            TQueryResult *pq =
               (TQueryResult *) mess->ReadObject(TQueryResult::Class());
            if (pq) {
               fPlayer->AddQueryResult(pq);
               // Notify the GUI that the result arrived
               QueryResultReady(Form("%s:%s", pq->GetTitle(), pq->GetName()));
            } else {
               PDB(kGlobal,2)
                  Info("CollectInputFrom","kPROOF_RETRIEVE: query result missing");
            }
         }
         break;

      case kPROOF_MAXQUERIES:
         {
            PDB(kGlobal,2) Info("CollectInputFrom","kPROOF_MAXQUERIES: enter");
            Int_t max = 0;

            (*mess) >> max;
            Printf("Number of queries fully kept remotely: %d", max);
         }
         break;

      case kPROOF_SERVERSTARTED:
         {
            PDB(kGlobal,2) Info("CollectInputFrom","kPROOF_SERVERSTARTED: enter");

            UInt_t tot = 0, done = 0;
            TString action;
            Bool_t st = kTRUE;

            (*mess) >> action >> tot >> done >> st;

            if (!IsMaster()) {
               if (tot) {
                  TString type = (action.Contains("submas")) ? "submasters"
                                                             : "workers";
                  Int_t frac = (Int_t) (done*100.)/tot;
                  char msg[512] = {0};
                  if (frac >= 100) {
                     sprintf(msg,"%s: OK (%d %s)                 \n",
                             action.Data(),tot, type.Data());
                  } else {
                     sprintf(msg,"%s: %d out of %d (%d %%)\r",
                             action.Data(), done, tot, frac);
                  }
                  if (fSync)
                     fprintf(stderr,"%s", msg);
                  else
                     NotifyLogMsg(msg, 0);
               }
               // Notify GUIs
               StartupMessage(action.Data(), st, (Int_t)done, (Int_t)tot);
            } else {

               // Just send the message one level up
               TMessage m(kPROOF_SERVERSTARTED);
               m << action << tot << done << st;
               gProofServ->GetSocket()->Send(m);
            }
         }
         break;

      case kPROOF_DATASET_STATUS:
         {
            PDB(kGlobal,2) Info("CollectInputFrom","kPROOF_DATASET_STATUS: enter");

            UInt_t tot = 0, done = 0;
            TString action;
            Bool_t st = kTRUE;

            (*mess) >> action >> tot >> done >> st;

            if (!IsMaster()) {
               if (tot) {
                  TString type = "files";
                  Int_t frac = (Int_t) (done*100.)/tot;
                  char msg[512] = {0};
                  if (frac >= 100) {
                     sprintf(msg,"%s: OK (%d %s)                 \n",
                             action.Data(),tot, type.Data());
                  } else {
                     sprintf(msg,"%s: %d out of %d (%d %%)\r",
                             action.Data(), done, tot, frac);
                  }
                  if (fSync)
                     fprintf(stderr,"%s", msg);
                  else
                     NotifyLogMsg(msg, 0);
               }
               // Notify GUIs
               DataSetStatus(action.Data(), st, (Int_t)done, (Int_t)tot);
            } else {

               // Just send the message one level up
               TMessage m(kPROOF_DATASET_STATUS);
               m << action << tot << done << st;
               gProofServ->GetSocket()->Send(m);
            }
         }
         break;

      case kPROOF_STARTPROCESS:
         {
            PDB(kGlobal,2) Info("CollectInputFrom","kPROOF_STARTPROCESS: enter");

            fIdle = kFALSE;

            TString selec;
            Int_t dsz = -1;
            Long64_t first = -1, nent = -1;
            (*mess) >> selec >> dsz >> first >> nent;

            // Start or reset the progress dialog
            if (fProgressDialog && !TestBit(kUsingSessionGui)) {
               if (!fProgressDialogStarted) {
                  fProgressDialog->ExecPlugin(5, this,
                                              selec.Data(), dsz, first, nent);
                  fProgressDialogStarted = kTRUE;
               } else {
                  ResetProgressDialog(selec, dsz, first, nent);
               }
            }
            ResetBit(kUsingSessionGui);
         }
         break;

      case kPROOF_ENDINIT:
         {
            PDB(kGlobal,2) Info("CollectInputFrom","kPROOF_ENDINIT: enter");

            if (IsMaster()) {
               if (fPlayer)
                  fPlayer->SetInitTime();
            }
         }
         break;

      case kPROOF_SETIDLE:
         {
            PDB(kGlobal,2) Info("CollectInputFrom","kPROOF_SETIDLE: enter");

            // The session is idle
            fIdle = kTRUE;
         }
         break;

      case kPROOF_QUERYSUBMITTED:
         {
            PDB(kGlobal,2) Info("CollectInputFrom","kPROOF_QUERYSUBMITTED: enter");

            // We have received the sequential number
            (*mess) >> fSeqNum;

            rc = 1;
         }
         break;

      case kPROOF_SESSIONTAG:
         {
            PDB(kGlobal,2) Info("CollectInputFrom","kPROOF_SESSIONTAG: enter");

            // We have received the unique tag and save it as name of this object
            TString stag;
            (*mess) >> stag;
            SetName(stag);
         }
         break;

      case kPROOF_FEEDBACK:
         {
            PDB(kGlobal,2) Info("CollectInputFrom","kPROOF_FEEDBACK: enter");
            TList *out = (TList *) mess->ReadObject(TList::Class());
            out->SetOwner();
            sl = FindSlave(s);
            if (fPlayer)
               fPlayer->StoreFeedback(sl, out); // Adopts the list
            else
               // Not yet ready: stop collect asap
               rc = 1;
         }
         break;

      case kPROOF_AUTOBIN:
         {
            PDB(kGlobal,2) Info("CollectInputFrom","kPROOF_AUTOBIN: enter");

            TString name;
            Double_t xmin, xmax, ymin, ymax, zmin, zmax;

            (*mess) >> name >> xmin >> xmax >> ymin >> ymax >> zmin >> zmax;

            fPlayer->UpdateAutoBin(name,xmin,xmax,ymin,ymax,zmin,zmax);

            TMessage answ(kPROOF_AUTOBIN);

            answ << name << xmin << xmax << ymin << ymax << zmin << zmax;

            s->Send(answ);
         }
         break;

      case kPROOF_PROGRESS:
         {
            PDB(kGlobal,2) Info("CollectInputFrom","kPROOF_PROGRESS: enter");

            sl = FindSlave(s);

            if (GetRemoteProtocol() > 11) {
               // New format
               Long64_t total, processed, bytesread;
               Float_t initTime, procTime, evtrti, mbrti;
               (*mess) >> total >> processed >> bytesread
                       >> initTime >> procTime
                       >> evtrti >> mbrti;
               fPlayer->Progress(total, processed, bytesread,
                                 initTime, procTime, evtrti, mbrti);

            } else {
               // Old format
               Long64_t total, processed;
               (*mess) >> total >> processed;
               fPlayer->Progress(sl, total, processed);
            }
         }
         break;

      case kPROOF_STOPPROCESS:
         {
            // answer contains number of processed events;
            PDB(kGlobal,2) Info("CollectInputFrom","kPROOF_STOPPROCESS: enter");

            Long64_t events;
            Bool_t abort = kFALSE;

            if ((mess->BufferSize() > mess->Length()) && (fProtocol > 8))
               (*mess) >> events >> abort;
            else
               (*mess) >> events;
            if (!abort) {
               fPlayer->AddEventsProcessed(events);
            } else if (IsMaster()) {
               fPlayer->StopProcess(kTRUE);
            }
            if (!IsMaster())
               Emit("StopProcess(Bool_t)", abort);
            break;
         }

      case kPROOF_GETSLAVEINFO:
         {
            PDB(kGlobal,2) Info("CollectInputFrom","kPROOF_GETSLAVEINFO: enter");

            sl = FindSlave(s);
            Bool_t active = (GetListOfActiveSlaves()->FindObject(sl) != 0);
            Bool_t bad = (GetListOfBadSlaves()->FindObject(sl) != 0);
            TList* tmpinfo = 0;
            (*mess) >> tmpinfo;
            tmpinfo->SetOwner(kFALSE);
            Int_t nentries = tmpinfo->GetSize();
            for (Int_t i=0; i<nentries; i++) {
               TSlaveInfo* slinfo =
                  dynamic_cast<TSlaveInfo*>(tmpinfo->At(i));
               if (slinfo) {
                  fSlaveInfo->Add(slinfo);
                  if (slinfo->fStatus != TSlaveInfo::kBad) {
                     if (!active) slinfo->SetStatus(TSlaveInfo::kNotActive);
                     if (bad) slinfo->SetStatus(TSlaveInfo::kBad);
                  }
                  if (!sl->GetMsd().IsNull()) slinfo->fMsd = sl->GetMsd();
               }
            }
            delete tmpinfo;
            rc = 1;
         }
         break;

      case kPROOF_VALIDATE_DSET:
         {
            PDB(kGlobal,2) Info("CollectInputFrom","kPROOF_VALIDATE_DSET: enter");
            TDSet* dset = 0;
            (*mess) >> dset;
            if (!fDSet)
               Error("CollectInputFrom","kPROOF_VALIDATE_DSET: fDSet not set");
            else
               fDSet->Validate(dset);
            delete dset;
         }
         break;

      case kPROOF_DATA_READY:
         {
            PDB(kGlobal,2) Info("CollectInputFrom","kPROOF_DATA_READY: enter");
            Bool_t dataready = kFALSE;
            Long64_t totalbytes, bytesready;
            (*mess) >> dataready >> totalbytes >> bytesready;
            fTotalBytes += totalbytes;
            fBytesReady += bytesready;
            if (dataready == kFALSE) fDataReady = dataready;
         }
         break;

      case kPROOF_PING:
         // do nothing (ping is already acknowledged)
         break;

      case kPROOF_MESSAGE:
         {
            PDB(kGlobal,2) Info("CollectInputFrom","kPROOF_MESSAGE: enter");

            // We have received the unique tag and save it as name of this object
            TString msg;
            (*mess) >> msg;
            Bool_t lfeed = kTRUE;
            if ((mess->BufferSize() > mess->Length()))
               (*mess) >> lfeed;

            if (!IsMaster()) {

               if (fSync) {
                  // Notify locally
                  fprintf(stderr,"%s%c", msg.Data(), (lfeed ? '\n' : '\r'));
               } else {
                  // Notify locally taking care of redirection, windows logs, ...
                  NotifyLogMsg(msg, (lfeed ? "\n" : "\r"));
               }
            } else {

               // The message is logged for debugging purposes.
               fprintf(stderr,"%s%c", msg.Data(), (lfeed ? '\n' : '\r'));
               if (gProofServ) {
                  // We hide it during normal operations
                  gProofServ->FlushLogFile();

                  // And send the message one level up
                  gProofServ->SendAsynMessage(msg, lfeed);
               }
            }
         }
         break;

      case kPROOF_VERSARCHCOMP:
         {
            TString vac;
            (*mess) >> vac;
            PDB(kGlobal,2) Info("CollectInputFrom","kPROOF_VERSARCHCOMP: %s", vac.Data());
            Int_t from = 0;
            TString vers, archcomp;
            if (vac.Tokenize(vers, from, "|"))
               vac.Tokenize(archcomp, from, "|");
            if ((sl = FindSlave(s))) {
               sl->SetArchCompiler(archcomp);
               sl->SetROOTVersion(vers);
            }
         }
         break;

      default:
         Error("Collect", "unknown command received from slave (what = %d)", what);
         break;
   }

   // Cleanup
   if (delete_mess)
      delete mess;

   // We are done successfully
   return rc;
}

//______________________________________________________________________________
void TProof::ActivateAsyncInput()
{
   // Activate the a-sync input handler.

   TIter next(fSlaves);
   TSlave *sl;

   while ((sl = (TSlave*) next()))
      if (sl->GetInputHandler())
         sl->GetInputHandler()->Add();
}

//______________________________________________________________________________
void TProof::DeActivateAsyncInput()
{
   // De-actiate a-sync input handler.

   TIter next(fSlaves);
   TSlave *sl;

   while ((sl = (TSlave*) next()))
      if (sl->GetInputHandler())
         sl->GetInputHandler()->Remove();
}

//______________________________________________________________________________
void TProof::HandleAsyncInput(TSocket *sl)
{
   // Handle input coming from the master server (when this is a client)
   // or from a slave server (when this is a master server). This is mainly
   // for a-synchronous communication. Normally when PROOF issues a command
   // the (slave) server messages are directly handle by Collect().

   TMessage *mess;
   Int_t     what;

   if (sl->Recv(mess) <= 0)
      return;                // do something more intelligent here

   what = mess->What();

   switch (what) {

      case kPROOF_PING:
         // do nothing (ping is already acknowledged)
         break;

      default:
         Error("HandleAsyncInput", "unknown command (what = %d)", what);
         break;
   }

   delete mess;
}

//______________________________________________________________________________
void TProof::MarkBad(TSlave *sl)
{
   // Add a bad slave server to the bad slave list and remove it from
   // the active list and from the two monitor objects.

   fActiveSlaves->Remove(sl);
   FindUniqueSlaves();
   fBadSlaves->Add(sl);

   fAllMonitor->Remove(sl->GetSocket());
   fActiveMonitor->Remove(sl->GetSocket());

   sl->Close();

   fSendGroupView = kTRUE;

   // Update session workers files
   SaveWorkerInfo();
}

//______________________________________________________________________________
void TProof::MarkBad(TSocket *s)
{
   // Add slave with socket s to the bad slave list and remove if from
   // the active list and from the two monitor objects.

   TSlave *sl = FindSlave(s);
   MarkBad(sl);
}

//______________________________________________________________________________
Int_t TProof::Ping()
{
   // Ping PROOF. Returns 1 if master server responded.

   return Ping(kActive);
}

//______________________________________________________________________________
Int_t TProof::Ping(ESlaves list)
{
   // Ping PROOF slaves. Returns the number of slaves that responded.

   TList *slaves = 0;
   if (list == kAll)       slaves = fSlaves;
   if (list == kActive)    slaves = fActiveSlaves;
   if (list == kUnique)    slaves = fUniqueSlaves;
   if (list == kAllUnique) slaves = fAllUniqueSlaves;

   if (slaves->GetSize() == 0) return 0;

   int   nsent = 0;
   TIter next(slaves);

   TSlave *sl;
   while ((sl = (TSlave *)next())) {
      if (sl->IsValid()) {
         if (sl->Ping() == -1)
            MarkBad(sl);
         else
            nsent++;
      }
   }

   return nsent;
}

//______________________________________________________________________________
void TProof::Print(Option_t *option) const
{
   // Print status of PROOF cluster.

   TString secCont;

   if (!IsMaster()) {
      Printf("Connected to:             %s (%s)", GetMaster(),
                                             IsValid() ? "valid" : "invalid");
      Printf("Port number:              %d", GetPort());
      Printf("User:                     %s", GetUser());
      Printf("ROOT version:             %s", gROOT->GetVersion());
      Printf("Architecture-Compiler:    %s-%s", gSystem->GetBuildArch(),
                                                gSystem->GetBuildCompilerVersion());
      TSlave *sl = (TSlave *)fActiveSlaves->First();
      if (sl) {
         TString sc;
         if (sl->GetSocket()->GetSecContext())
            Printf("Security context:         %s",
                                      sl->GetSocket()->GetSecContext()->AsString(sc));
         Printf("Proofd protocol version:  %d", sl->GetSocket()->GetRemoteProtocol());
      } else {
         Printf("Security context:         Error - No connection");
         Printf("Proofd protocol version:  Error - No connection");
      }
      Printf("Client protocol version:  %d", GetClientProtocol());
      Printf("Remote protocol version:  %d", GetRemoteProtocol());
      Printf("Log level:                %d", GetLogLevel());
      Printf("Session unique tag:       %s", IsValid() ? GetSessionTag() : "");
      Printf("Default data pool:        %s", IsValid() ? GetDataPoolUrl() : "");
      if (IsValid())
         const_cast<TProof*>(this)->SendPrint(option);
   } else {
      const_cast<TProof*>(this)->AskStatistics();
      if (IsParallel())
         Printf("*** Master server %s (parallel mode, %d slaves):",
                gProofServ->GetOrdinal(), GetParallel());
      else
         Printf("*** Master server %s (sequential mode):",
                gProofServ->GetOrdinal());

      Printf("Master host name:           %s", gSystem->HostName());
      Printf("Port number:                %d", GetPort());
      if (strlen(gProofServ->GetGroup()) > 0) {
         Printf("User/Group:                 %s/%s", GetUser(), gProofServ->GetGroup());
      } else {
         Printf("User:                       %s", GetUser());
      }
      if (gSystem->Getenv("ROOTVERSIONTAG"))
         Printf("ROOT version:               %s-%s", gROOT->GetVersion(),
                                                     gSystem->Getenv("ROOTVERSIONTAG"));
      else
         Printf("ROOT version:               %s", gROOT->GetVersion());
      Printf("Architecture-Compiler:      %s-%s", gSystem->GetBuildArch(),
                                                  gSystem->GetBuildCompilerVersion());
      Printf("Protocol version:           %d", GetClientProtocol());
      Printf("Image name:                 %s", GetImage());
      Printf("Working directory:          %s", gSystem->WorkingDirectory());
      Printf("Config directory:           %s", GetConfDir());
      Printf("Config file:                %s", GetConfFile());
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
         TList masters;
         TIter nextslave(fSlaves);
         while (TSlave* sl = dynamic_cast<TSlave*>(nextslave())) {
            if (!sl->IsValid()) continue;

            if (sl->GetSlaveType() == TSlave::kSlave) {
               sl->Print(option);
            } else if (sl->GetSlaveType() == TSlave::kMaster) {
               TMessage mess(kPROOF_PRINT);
               mess.WriteString(option);
               if (sl->GetSocket()->Send(mess) == -1)
                  const_cast<TProof*>(this)->MarkBad(sl);
               else
                  masters.Add(sl);
            } else {
               Error("Print", "TSlave is neither Master nor Worker");
               R__ASSERT(0);
            }
         }
         const_cast<TProof*>(this)->Collect(&masters);
      }
   }
}

//______________________________________________________________________________
Long64_t TProof::Process(TDSet *dset, const char *selector, Option_t *option,
                         Long64_t nentries, Long64_t first, TEventList *evl)
{
   // Process a data set (TDSet) using the specified selector (.C) file.
   // The return value is -1 in case of error and TSelector::GetStatus() in
   // in case of success.

   if (!IsValid()) return -1;

   // Resolve query mode
   fSync = (GetQueryMode(option) == kSync);

   if (fSync && !IsIdle()) {
      Info("Process","not idle, cannot submit synchronous query");
      return -1;
   }

   // deactivate the default application interrupt handler
   // ctrl-c's will be forwarded to PROOF to stop the processing
   TSignalHandler *sh = 0;
   if (fSync) {
      if (gApplication)
         sh = gSystem->RemoveSignalHandler(gApplication->GetSignalHandler());
   }

   Long64_t rv = fPlayer->Process(dset, selector, option, nentries, first, evl);

//   // Clear input list
//   fPlayer->ClearInput();

   if (fSync) {
      // reactivate the default application interrupt handler
      if (sh)
         gSystem->AddSignalHandler(sh);
   }

   return rv;
}

//______________________________________________________________________________
Long64_t TProof::Process(const char *dsetname, const char *selector,
                         Option_t *option, Long64_t nentries,
                         Long64_t first, TEventList *evl)
{
   // Process a dataset which is stored on the master with name 'dsetname'.
   // The syntax for dsetname is name[#[dir/]objname], e.g.
   //   "mydset"       analysis of the first tree in the top dir of the dataset
   //                  named "mydset"
   //   "mydset#T"     analysis tree "T" in the top dir of the dataset
   //                  named "mydset"
   //   "mydset#adir/T" analysis tree "T" in the dir "adir" of the dataset
   //                  named "mydset"
   //   "mydset#adir/" analysis of the first tree in the dir "adir" of the
   //                  dataset named "mydset"
   // The return value is -1 in case of error and TSelector::GetStatus() in
   // in case of success.

   if (fProtocol < 13) {
      Info("Process", "processing 'by name' not supported by the server");
      return -1;
   }

   TString name(dsetname);
   TString obj;
   TString dir = "/";
   Int_t idxc = name.Index("#");
   if (idxc != kNPOS) {
      Int_t idxs = name.Index("/", 1, idxc, TString::kExact);
      if (idxs != kNPOS && idxc != kNPOS) {
         obj = name(idxs+1, name.Length());
         dir = name(idxc+1, name.Length());
         dir.Remove(dir.Index("/") + 1);
         name.Remove(idxc);
      } else if (idxc != kNPOS && idxs == kNPOS) {
         obj = name(idxc+1, name.Length());
         name.Remove(idxc);
      } else if (idxs != kNPOS && idxc == kNPOS) {
         Error("Process", "bad name syntax (%s): specification of additional"
                          " attributes needs a '#' after the dataset name", dsetname);
         return -1;
      }
   } else if (name.Index(":") != kNPOS && name.Index("://") == kNPOS) {
      // protection against using ':' instead of '#'
      Error("Process", "bad name syntax (%s): please use"
                       " a '#' after the dataset name", dsetname);
      return -1;
   }

   TDSet *dset = new TDSet(name, obj, dir);
   Long64_t retval = Process(dset, selector, option, nentries, first, evl);
   delete dset;
   return retval;
}

//______________________________________________________________________________
Int_t TProof::GetQueryReference(Int_t qry, TString &ref)
{
   // Get reference for the qry-th query in fQueries (as
   // displayed by ShowQueries).

   ref = "";
   if (qry > 0) {
      if (!fQueries)
         GetListOfQueries();
      if (fQueries) {
         TIter nxq(fQueries);
         TQueryResult *qr = 0;
         while ((qr = (TQueryResult *) nxq()))
            if (qr->GetSeqNum() == qry) {
               ref = Form("%s:%s", qr->GetTitle(), qr->GetName());
               return 0;
            }
      }
   }
   return -1;
}

//______________________________________________________________________________
Long64_t TProof::Finalize(Int_t qry, Bool_t force)
{
   // Finalize the qry-th query in fQueries.
   // If force, force retrieval if the query is found in the local list
   // but has already been finalized (default kFALSE).
   // If query < 0, finalize current query.
   // Return 0 on success, -1 on error

   if (fPlayer) {
      if (qry > 0) {
         TString ref;
         if (GetQueryReference(qry, ref) == 0) {
            return Finalize(ref, force);
         } else {
            Info("Finalize", "query #%d not found", qry);
         }
      } else {
         // The last query
         return fPlayer->Finalize(force);
      }
   }
   return -1;
}

//______________________________________________________________________________
Long64_t TProof::Finalize(const char *ref, Bool_t force)
{
   // Finalize query with reference ref.
   // If force, force retrieval if the query is found in the local list
   // but has already been finalized (default kFALSE).
   // If ref = 0, finalize current query.
   // Return 0 on success, -1 on error

   if (fPlayer) {
      if (ref) {
         // Get the pointer to the query
         TQueryResult *qr = fPlayer->GetQueryResult(ref);
         // If not found, try retrieving it
         Bool_t retrieve = kFALSE;
         if (!qr) {
            retrieve = kTRUE;
         } else {
            if (qr->IsFinalized()) {
               if (force) {
                  retrieve = kTRUE;
               } else {
                  Info("Finalize","query already finalized:"
                       " use Finalize(<qry>,kTRUE) to force new retrieval");
                  qr = 0;
               }
            }
         }
         if (retrieve) {
            Retrieve(ref);
            qr = fPlayer->GetQueryResult(ref);
         }
         if (qr)
            return fPlayer->Finalize(qr);
      }
   }
   return -1;
}

//______________________________________________________________________________
Int_t TProof::Retrieve(Int_t qry, const char *path)
{
   // Send retrieve request for the qry-th query in fQueries.
   // If path is defined save it to path.

   if (qry > 0) {
      TString ref;
      if (GetQueryReference(qry, ref) == 0)
         return Retrieve(ref, path);
      else
         Info("Retrieve", "query #%d not found", qry);
   } else {
      Info("Retrieve","positive argument required - do nothing");
   }
   return -1;
}

//______________________________________________________________________________
Int_t TProof::Retrieve(const char *ref, const char *path)
{
   // Send retrieve request for the query specified by ref.
   // If path is defined save it to path.
   // Generic method working for all queries known by the server.

   if (ref) {
      TMessage m(kPROOF_RETRIEVE);
      m << TString(ref);
      Broadcast(m, kActive);
      Collect(kActive);

      // Archive ir locally, if required
      if (path) {

         // Get pointer to query
         TQueryResult *qr = fPlayer ? fPlayer->GetQueryResult(ref) : 0;

         if (qr) {

            TFile *farc = TFile::Open(path,"UPDATE");
            if (!(farc->IsOpen())) {
               Info("Retrieve", "archive file cannot be open (%s)", path);
               return 0;
            }
            farc->cd();

            // Update query status
            qr->SetArchived(path);

            // Write to file
            qr->Write();

            farc->Close();
            SafeDelete(farc);

         } else {
            Info("Retrieve", "query not found after retrieve");
            return -1;
         }
      }

      return 0;
   }
   return -1;
}

//______________________________________________________________________________
Int_t TProof::Remove(Int_t qry, Bool_t all)
{
   // Send remove request for the qry-th query in fQueries.

   if (qry > 0) {
      TString ref;
      if (GetQueryReference(qry, ref) == 0)
         return Remove(ref, all);
      else
         Info("Remove", "query #%d not found", qry);
   } else {
      Info("Remove","positive argument required - do nothing");
   }
   return -1;
}

//______________________________________________________________________________
Int_t TProof::Remove(const char *ref, Bool_t all)
{
   // Send remove request for the query specified by ref.
   // If all = TRUE remove also local copies of the query, if any.
   // Generic method working for all queries known by the server.
   // This method can be also used to reset the list of queries
   // waiting to be processed: for that purpose use ref == "cleanupqueue".

   if (all) {
      // Remove also local copies, if any
      if (fPlayer)
         fPlayer->RemoveQueryResult(ref);
   }

   if (ref) {
      TMessage m(kPROOF_REMOVE);
      m << TString(ref);
      Broadcast(m, kActive);
      Collect(kActive);
      return 0;
   }
   return -1;
}

//______________________________________________________________________________
Int_t TProof::Archive(Int_t qry, const char *path)
{
   // Send archive request for the qry-th query in fQueries.

   if (qry > 0) {
      TString ref;
      if (GetQueryReference(qry, ref) == 0)
         return Archive(ref, path);
      else
         Info("Archive", "query #%d not found", qry);
   } else {
      Info("Archive","positive argument required - do nothing");
   }
   return -1;
}

//______________________________________________________________________________
Int_t TProof::Archive(const char *ref, const char *path)
{
   // Send archive request for the query specified by ref.
   // Generic method working for all queries known by the server.
   // If ref == "Default", path is understood as a default path for
   // archiving.

   if (ref) {
      TMessage m(kPROOF_ARCHIVE);
      m << TString(ref) << TString(path);
      Broadcast(m, kActive);
      Collect(kActive);
      return 0;
   }
   return -1;
}

//______________________________________________________________________________
Int_t TProof::CleanupSession(const char *sessiontag)
{
   // Send cleanup request for the session specified by tag.

   if (sessiontag) {
      TMessage m(kPROOF_CLEANUPSESSION);
      m << TString(sessiontag);
      Broadcast(m, kActive);
      Collect(kActive);
      return 0;
   }
   return -1;
}

//_____________________________________________________________________________
void TProof::SetQueryMode(EQueryMode mode)
{
   // Change query running mode to the one specified by 'mode'.

   fQueryMode = mode;

   if (gDebug > 0)
      Info("SetQueryMode","query mode is set to: %s", fQueryMode == kSync ?
           "Sync" : "Async");
}

//______________________________________________________________________________
TProof::EQueryMode TProof::GetQueryMode(Option_t *mode) const
{
   // Find out the query mode based on the current setting and 'mode'.

   EQueryMode qmode = fQueryMode;

   if (mode && (strlen(mode) > 0)) {
      TString m(mode);
      m.ToUpper();
      if (m.Contains("ASYN")) {
         qmode = kAsync;
      } else if (m.Contains("SYNC")) {
         qmode = kSync;
      }
   }

   if (gDebug > 0)
      Info("GetQueryMode","query mode is set to: %s", qmode == kSync ?
           "Sync" : "Async");

   return qmode;
}

//______________________________________________________________________________
Long64_t TProof::DrawSelect(TDSet *dset, const char *varexp,
                            const char *selection, Option_t *option,
                            Long64_t nentries, Long64_t first)
{
   // Process a data set (TDSet) using the specified selector (.C) file.
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

   return fPlayer->DrawSelect(dset, varexp, selection, opt, nentries, first);
}

//______________________________________________________________________________
void TProof::StopProcess(Bool_t abort, Int_t timeout)
{
   // Send STOPPROCESS message to master and workers.

   PDB(kGlobal,2)
      Info("StopProcess","enter %d", abort);

   if (!IsValid())
      return;

   if (fPlayer)
      fPlayer->StopProcess(abort, timeout);

   // Stop any blocking 'Collect' request
   if (!IsMaster())
      InterruptCurrentMonitor();

   if (fSlaves->GetSize() == 0)
      return;

   // Notify the remote counterpart
   TSlave *sl;
   TIter   next(fSlaves);
   while ((sl = (TSlave *)next()))
      if (sl->IsValid())
         // Ask slave to progate the stop/abort request
         sl->StopProcess(abort, timeout);
}

//______________________________________________________________________________
void TProof::RecvLogFile(TSocket *s, Int_t size)
{
   // Receive the log file of the slave with socket s.

   const Int_t kMAXBUF = 16384;  //32768  //16384  //65536;
   char buf[kMAXBUF];

   // Append messages to active logging unit
   Int_t fdout = -1;
   if (!fLogToWindowOnly) {
      fdout = (fRedirLog) ? fileno(fLogFileW) : fileno(stdout);
      if (fdout < 0) {
         Warning("RecvLogFile", "file descriptor for outputs undefined (%d):"
                 " will not log msgs", fdout);
         return;
      }
      lseek(fdout, (off_t) 0, SEEK_END);
   }

   Int_t  left, rec, r;
   Long_t filesize = 0;

   while (filesize < size) {
      left = Int_t(size - filesize);
      if (left > kMAXBUF)
         left = kMAXBUF;
      rec = s->RecvRaw(&buf, left);
      filesize = (rec > 0) ? (filesize + rec) : filesize;
      if (!fLogToWindowOnly) {
         if (rec > 0) {

            char *p = buf;
            r = rec;
            while (r) {
               Int_t w;

               w = write(fdout, p, r);

               if (w < 0) {
                  SysError("RecvLogFile", "error writing to unit: %d", fdout);
                  break;
               }
               r -= w;
               p += w;
            }
         } else if (rec < 0) {
            Error("RecvLogFile", "error during receiving log file");
            break;
         }
      }
      if (rec > 0) {
         buf[rec] = 0;
         EmitVA("LogMessage(const char*,Bool_t)", 2, buf, kFALSE);
      }
   }

   // If idle restore logs to main session window
   if (fRedirLog && IsIdle())
      fRedirLog = kFALSE;
}

//______________________________________________________________________________
void TProof::NotifyLogMsg(const char *msg, const char *sfx)
{
   // Notify locally 'msg' to the appropriate units (file, stdout, window)
   // If defined, 'sfx' is added after 'msg' (typically a line-feed);

   // Must have somenthing to notify
   Int_t len = 0;
   if (!msg || (len = strlen(msg)) <= 0)
      return;

   // Get suffix length if any
   Int_t lsfx = (sfx) ? strlen(sfx) : 0;

   // Append messages to active logging unit
   Int_t fdout = -1;
   if (!fLogToWindowOnly) {
      fdout = (fRedirLog) ? fileno(fLogFileW) : fileno(stdout);
      if (fdout < 0) {
         Warning("NotifyLogMsg", "file descriptor for outputs undefined (%d):"
                 " will not notify msgs", fdout);
         return;
      }
      lseek(fdout, (off_t) 0, SEEK_END);
   }

   if (!fLogToWindowOnly) {
      // Write to output unit (stdout or a log file)
      if (len > 0) {
         char *p = (char *)msg;
         Int_t r = len;
         while (r) {
            Int_t w = write(fdout, p, r);
            if (w < 0) {
               SysError("NotifyLogMsg", "error writing to unit: %d", fdout);
               break;
            }
            r -= w;
            p += w;
         }
         // Add a suffix, if requested
         if (lsfx > 0)
            write(fdout, sfx, lsfx);
      }
   }
   if (len > 0) {
      // Publish the message to the separate window (if the latter is missing
      // the message will just get lost)
      EmitVA("LogMessage(const char*,Bool_t)", 2, msg, kFALSE);
   }

   // If idle restore logs to main session window
   if (fRedirLog && IsIdle())
      fRedirLog = kFALSE;
}

//______________________________________________________________________________
void TProof::LogMessage(const char *msg, Bool_t all)
{
   // Log a message into the appropriate window by emitting a signal.

   PDB(kGlobal,1)
      Info("LogMessage","Enter ... %s, 'all: %s", msg ? msg : "",
           all ? "true" : "false");

   if (gROOT->IsBatch()) {
      PDB(kGlobal,1) Info("LogMessage","GUI not started - use TProof::ShowLog()");
      return;
   }

   if (msg)
      EmitVA("LogMessage(const char*,Bool_t)", 2, msg, all);

   // Re-position at the beginning of the file, if requested.
   // This is used by the dialog when it re-opens the log window to
   // provide all the session messages
   if (all)
      lseek(fileno(fLogFileR), (off_t) 0, SEEK_SET);

   const Int_t kMAXBUF = 32768;
   char buf[kMAXBUF];
   Int_t len;
   do {
      while ((len = read(fileno(fLogFileR), buf, kMAXBUF-1)) < 0 &&
             TSystem::GetErrno() == EINTR)
         TSystem::ResetErrno();

      if (len < 0) {
         Error("LogMessage", "error reading log file");
         break;
      }

      if (len > 0) {
         buf[len] = 0;
         EmitVA("LogMessage(const char*,Bool_t)", 2, buf, kFALSE);
      }

   } while (len > 0);
}

//______________________________________________________________________________
Int_t TProof::SendGroupView()
{
   // Send to all active slaves servers the current slave group size
   // and their unique id. Returns number of active slaves.
   // Returns -1 in case of error.

   if (!IsValid()) return -1;
   if (!IsMaster()) return 0;
   if (!fSendGroupView) return 0;
   fSendGroupView = kFALSE;

   TIter   next(fActiveSlaves);
   TSlave *sl;

   int  bad = 0, cnt = 0, size = GetNumberOfActiveSlaves();
   char str[32];

   while ((sl = (TSlave *)next())) {
      sprintf(str, "%d %d", cnt, size);
      if (sl->GetSocket()->Send(str, kPROOF_GROUPVIEW) == -1) {
         MarkBad(sl);
         bad++;
      } else
         cnt++;
   }

   // Send the group view again in case there was a change in the
   // group size due to a bad slave

   if (bad) SendGroupView();

   return GetNumberOfActiveSlaves();
}

//______________________________________________________________________________
Int_t TProof::Exec(const char *cmd, Bool_t plusMaster)
{
   // Send command to be executed on the PROOF master and/or slaves.
   // If plusMaster is kTRUE then exeucte on slaves and master too.
   // Command can be any legal command line command. Commands like
   // ".x file.C" or ".L file.C" will cause the file file.C to be send
   // to the PROOF cluster. Returns -1 in case of error, >=0 in case of
   // succes.

   return Exec(cmd, kActive, plusMaster);
}

//______________________________________________________________________________
R__HIDDEN Int_t TProof::Exec(const char *cmd, ESlaves list, Bool_t plusMaster)
{
   // Send command to be executed on the PROOF master and/or slaves.
   // Command can be any legal command line command. Commands like
   // ".x file.C" or ".L file.C" will cause the file file.C to be send
   // to the PROOF cluster. Returns -1 in case of error, >=0 in case of
   // succes.

   if (!IsValid()) return -1;

   TString s = cmd;
   s = s.Strip(TString::kBoth);

   if (!s.Length()) return 0;

   // check for macro file and make sure the file is available on all slaves
   if (s.BeginsWith(".L") || s.BeginsWith(".x") || s.BeginsWith(".X")) {
      TString file = s(2, s.Length());
      TString acm, arg, io;
      TString filename = gSystem->SplitAclicMode(file, acm, arg, io);
      char *fn = gSystem->Which(TROOT::GetMacroPath(), filename, kReadPermission);
      if (fn) {
         if (GetNumberOfUniqueSlaves() > 0) {
            if (SendFile(fn, kAscii | kForward) < 0) {
               Error("Exec", "file %s could not be transfered", fn);
               delete [] fn;
               return -1;
            }
         } else {
            TString scmd = s(0,3) + fn;
            Int_t n = SendCommand(scmd, list);
            delete [] fn;
            return n;
         }
      } else {
         Error("Exec", "macro %s not found", file.Data());
         return -1;
      }
      delete [] fn;
   }

   if (plusMaster) {
      Int_t n = GetParallel();
      SetParallelSilent(0);
      Int_t res = SendCommand(cmd, list);
      SetParallelSilent(n);
      if (res < 0)
         return res;
   }
   return SendCommand(cmd, list);
}

//______________________________________________________________________________
Int_t TProof::SendCommand(const char *cmd, ESlaves list)
{
   // Send command to be executed on the PROOF master and/or slaves.
   // Command can be any legal command line command, however commands
   // like ".x file.C" or ".L file.C" will not cause the file.C to be
   // transfered to the PROOF cluster. In that case use TProof::Exec().
   // Returns the status send by the remote server as part of the
   // kPROOF_LOGDONE message. Typically this is the return code of the
   // command on the remote side. Returns -1 in case of error.

   if (!IsValid()) return -1;

   Broadcast(cmd, kMESS_CINT, list);
   Collect(list);

   return fStatus;
}

//______________________________________________________________________________
Int_t TProof::SendCurrentState(ESlaves list)
{
   // Transfer the current state of the master to the active slave servers.
   // The current state includes: the current working directory, etc.
   // Returns the number of active slaves. Returns -1 in case of error.

   if (!IsValid()) return -1;

   // Go to the new directory, reset the interpreter environment and
   // tell slave to delete all objects from its new current directory.
   Broadcast(gDirectory->GetPath(), kPROOF_RESET, list);

   return GetParallel();
}

//______________________________________________________________________________
Int_t TProof::SendInitialState()
{
   // Transfer the initial (i.e. current) state of the master to all
   // slave servers. Currently the initial state includes: log level.
   // Returns the number of active slaves. Returns -1 in case of error.

   if (!IsValid()) return -1;

   SetLogLevel(fLogLevel, gProofDebugMask);

   return GetNumberOfActiveSlaves();
}

//______________________________________________________________________________
Bool_t TProof::CheckFile(const char *file, TSlave *slave, Long_t modtime)
{
   // Check if a file needs to be send to the slave. Use the following
   // algorithm:
   //   - check if file appears in file map
   //     - if yes, get file's modtime and check against time in map,
   //       if modtime not same get md5 and compare against md5 in map,
   //       if not same return kTRUE.
   //     - if no, get file's md5 and modtime and store in file map, ask
   //       slave if file exists with specific md5, if yes return kFALSE,
   //       if no return kTRUE.
   // Returns kTRUE in case file needs to be send, returns kFALSE in case
   // file is already on remote node.

   Bool_t sendto = kFALSE;

   // create slave based filename
   TString sn = slave->GetName();
   sn += ":";
   sn += slave->GetOrdinal();
   sn += ":";
   sn += gSystem->BaseName(file);

   // check if file is in map
   FileMap_t::const_iterator it;
   if ((it = fFileMap.find(sn)) != fFileMap.end()) {
      // file in map
      MD5Mod_t md = (*it).second;
      if (md.fModtime != modtime) {
         TMD5 *md5 = TMD5::FileChecksum(file);
         if (md5) {
            if ((*md5) != md.fMD5) {
               sendto       = kTRUE;
               md.fMD5      = *md5;
               md.fModtime  = modtime;
               fFileMap[sn] = md;
               // When on the master, the master and/or slaves may share
               // their file systems and cache. Therefore always make a
               // check for the file. If the file already exists with the
               // expected md5 the kPROOF_CHECKFILE command will cause the
               // file to be copied from cache to slave sandbox.
               if (IsMaster()) {
                  sendto = kFALSE;
                  TMessage mess(kPROOF_CHECKFILE);
                  mess << TString(gSystem->BaseName(file)) << md.fMD5;
                  slave->GetSocket()->Send(mess);

                  TMessage *reply;
                  slave->GetSocket()->Recv(reply);
                  if (reply->What() != kPROOF_CHECKFILE)
                     sendto = kTRUE;
                  delete reply;
               }
            }
            delete md5;
         } else {
            Error("CheckFile", "could not calculate local MD5 check sum - dont send");
            return kFALSE;
         }
      }
   } else {
      // file not in map
      TMD5 *md5 = TMD5::FileChecksum(file);
      MD5Mod_t md;
      if (md5) {
         md.fMD5      = *md5;
         md.fModtime  = modtime;
         fFileMap[sn] = md;
         delete md5;
      } else {
         Error("CheckFile", "could not calculate local MD5 check sum - dont send");
         return kFALSE;
      }
      TMessage mess(kPROOF_CHECKFILE);
      mess << TString(gSystem->BaseName(file)) << md.fMD5;
      slave->GetSocket()->Send(mess);

      TMessage *reply;
      slave->GetSocket()->Recv(reply);
      sendto = (!reply || reply->What() != kPROOF_CHECKFILE) ? kTRUE : kFALSE;
      if (reply)
         delete reply;
      else
         Error("CheckFile", "received empty message from worker: %s", slave->GetName());
   }

   return sendto;
}

//______________________________________________________________________________
Int_t TProof::SendFile(const char *file, Int_t opt, const char *rfile, TSlave *wrk)
{
   // Send a file to master or slave servers. Returns number of slaves
   // the file was sent to, maybe 0 in case master and slaves have the same
   // file system image, -1 in case of error.
   // If defined, send to worker 'wrk' only.
   // If defined, the full path of the remote path will be rfile.
   // The mask 'opt' is an or of ESendFileOpt:
   //
   //       kAscii  (0x0)      if set true ascii file transfer is used
   //       kBinary (0x1)      if set true binary file transfer is used
   //       kForce  (0x2)      if not set an attempt is done to find out
   //                          whether the file really needs to be downloaded
   //                          (a valid copy may already exist in the cache
   //                          from a previous run); the bit is set by
   //                          UploadPackage, since the check is done elsewhere.
   //       kForward (0x4)     if set, ask server to forward the file to slave
   //                          or submaster (meaningless for slave servers).
   //

   if (!IsValid()) return -1;

   // Use the active slaves list ...
   TList *slaves = fActiveSlaves;
   // ... or the specified slave, if any
   if (wrk) {
      slaves = new TList();
      slaves->Add(wrk);
   }

   if (slaves->GetSize() == 0) return 0;

#ifndef R__WIN32
   Int_t fd = open(file, O_RDONLY);
#else
   Int_t fd = open(file, O_RDONLY | O_BINARY);
#endif
   if (fd < 0) {
      SysError("SendFile", "cannot open file %s", file);
      return -1;
   }

   // Get info about the file
   Long64_t size;
   Long_t id, flags, modtime;
   if (gSystem->GetPathInfo(file, &id, &size, &flags, &modtime) == 1) {
      Error("SendFile", "cannot stat file %s", file);
      return -1;
   }
   if (size == 0) {
      Error("SendFile", "empty file %s", file);
      return -1;
   }

   // Decode options
   Bool_t bin   = (opt & kBinary)  ? kTRUE : kFALSE;
   Bool_t force = (opt & kForce)   ? kTRUE : kFALSE;
   Bool_t fw    = (opt & kForward) ? kTRUE : kFALSE;

   const Int_t kMAXBUF = 32768;  //16384  //65536;
   char buf[kMAXBUF];
   Int_t nsl = 0;

   TIter next(slaves);
   TSlave *sl;
   const char *fnam = (rfile) ? rfile : gSystem->BaseName(file);
   while ((sl = (TSlave *)next())) {
      if (!sl->IsValid())
         continue;

      Bool_t sendto = force ? kTRUE : CheckFile(file, sl, modtime);
      // Don't send the kPROOF_SENDFILE command to real slaves when sendto
      // is false. Masters might still need to send the file to newly added
      // slaves.
      if (sl->fSlaveType == TSlave::kSlave && !sendto)
         continue;
      // The value of 'size' is used as flag remotely, so we need to
      // reset it to 0 if we are not going to send the file
      size = sendto ? size : 0;

      PDB(kPackage,2)
         if (size > 0) {
            if (!nsl)
               Info("SendFile", "sending file %s to:", file);
            printf("   slave = %s:%s\n", sl->GetName(), sl->GetOrdinal());
         }

      sprintf(buf, "%s %d %lld %d", fnam, bin, size, fw);
      if (sl->GetSocket()->Send(buf, kPROOF_SENDFILE) == -1) {
         MarkBad(sl);
         continue;
      }

      if (!sendto)
         continue;

      lseek(fd, 0, SEEK_SET);

      Int_t len;
      do {
         while ((len = read(fd, buf, kMAXBUF)) < 0 && TSystem::GetErrno() == EINTR)
            TSystem::ResetErrno();

         if (len < 0) {
            SysError("SendFile", "error reading from file %s", file);
            Interrupt(kSoftInterrupt, kActive);
            close(fd);
            return -1;
         }

         if (len > 0 && sl->GetSocket()->SendRaw(buf, len) == -1) {
            SysError("SendFile", "error writing to slave %s:%s (now offline)",
                     sl->GetName(), sl->GetOrdinal());
            MarkBad(sl);
            break;
         }

      } while (len > 0);

      nsl++;
   }

   close(fd);

   // Cleanup temporary list, if any
   if (slaves != fActiveSlaves)
      SafeDelete(slaves);

   return nsl;
}

//______________________________________________________________________________
Int_t TProof::SendObject(const TObject *obj, ESlaves list)
{
   // Send object to master or slave servers. Returns number of slaves object
   // was sent to, -1 in case of error.

   if (!IsValid() || !obj) return -1;

   TMessage mess(kMESS_OBJECT);

   mess.WriteObject(obj);
   return Broadcast(mess, list);
}

//______________________________________________________________________________
Int_t TProof::SendPrint(Option_t *option)
{
   // Send print command to master server. Returns number of slaves message
   // was sent to. Returns -1 in case of error.

   if (!IsValid()) return -1;

   Broadcast(option, kPROOF_PRINT, kActive);
   return Collect(kActive);
}

//______________________________________________________________________________
void TProof::SetLogLevel(Int_t level, UInt_t mask)
{
   // Set server logging level.

   char str[32];
   fLogLevel        = level;
   gProofDebugLevel = level;
   gProofDebugMask  = (TProofDebug::EProofDebugMask) mask;
   sprintf(str, "%d %u", level, mask);
   Broadcast(str, kPROOF_LOGLEVEL, kAll);
}

//______________________________________________________________________________
void TProof::SetRealTimeLog(Bool_t on)
{
   // Switch ON/OFF the real-time logging facility. When this option is
   // ON, log messages from processing are sent back as they come, instead of
   // being sent back at the end in one go. This may help debugging or monitoring
   // in some cases, but, depending on the amount of log, it may have significant
   // consequencies on the load over the network, so it must be used with care.

   if (IsValid()) {
      TMessage mess(kPROOF_REALTIMELOG);
      mess << on;
      Broadcast(mess);
   } else {
      Warning("SetRealTimeLog","session is invalid - do nothing");
   }
}

//______________________________________________________________________________
Int_t TProof::SetParallelSilent(Int_t nodes, Bool_t random)
{
   // Tell RPOOF how many slaves to use in parallel. If random is TRUE a random
   // selection is done (if nodes is less than the available nodes).
   // Returns the number of parallel slaves. Returns -1 in case of error.

   if (!IsValid()) return -1;

   if (IsMaster()) {
      GoParallel(nodes, kFALSE, random);
      return SendCurrentState();
   } else {
      PDB(kGlobal,1) Info("SetParallelSilent", "request %d node%s", nodes,
          nodes == 1 ? "" : "s");
      TMessage mess(kPROOF_PARALLEL);
      mess << nodes << random;
      Broadcast(mess);
      Collect();
      Int_t n = GetParallel();
      PDB(kGlobal,1) Info("SetParallelSilent", "got %d node%s", n, n == 1 ? "" : "s");
      return n;
   }
}

//______________________________________________________________________________
Int_t TProof::SetParallel(Int_t nodes, Bool_t random)
{
   // Tell PROOF how many slaves to use in parallel. Returns the number of
   // parallel slaves. Returns -1 in case of error.

   Int_t n = SetParallelSilent(nodes, random);
   if (!IsMaster()) {
      if (n < 1) {
         Printf("PROOF set to sequential mode");
      } else {
         TString subfix = (n == 1) ? "" : "s";
         if (random)
            subfix += ", randomly selected";
         Printf("PROOF set to parallel mode (%d worker%s)", n, subfix.Data());
      }
   }
   return n;
}

//______________________________________________________________________________
Int_t TProof::GoParallel(Int_t nodes, Bool_t attach, Bool_t random)
{
   // Go in parallel mode with at most "nodes" slaves. Since the fSlaves
   // list is sorted by slave performace the active list will contain first
   // the most performant nodes. Returns the number of active slaves.
   // If random is TRUE, and nodes is less than the number of available workers,
   // a random selection is done.
   // Returns -1 in case of error.

   if (!IsValid()) return -1;

   if (nodes < 0) nodes = 0;

   fActiveSlaves->Clear();
   fActiveMonitor->RemoveAll();

   // Prepare the list of candidates first.
   // Algorithm depends on random option.
   TSlave *sl = 0;
   TList *wlst = new TList;
   TIter nxt(fSlaves);
   fInactiveSlaves->Clear();
   while ((sl = (TSlave *)nxt())) {
      if (sl->IsValid() && !fBadSlaves->FindObject(sl)) {
         if (strcmp("IGNORE", sl->GetImage()) == 0) continue;
         if ((sl->GetSlaveType() != TSlave::kSlave) &&
             (sl->GetSlaveType() != TSlave::kMaster)) {
            Error("GoParallel", "TSlave is neither Master nor Slave");
            R__ASSERT(0);
         }
         // Good candidate
         wlst->Add(sl);
         // Set it inactive
         fInactiveSlaves->Add(sl);
         sl->SetStatus(TSlave::kInactive);
      }
   }
   Int_t nwrks = (nodes > wlst->GetSize()) ? wlst->GetSize() : nodes;
   int cnt = 0;
   fEndMaster = IsMaster() ? kTRUE : kFALSE;
   while (cnt < nwrks) {
      // Random choice, if requested
      if (random) {
         Int_t iwrk = (Int_t) (gRandom->Rndm() * wlst->GetSize());
         sl = (TSlave *) wlst->At(iwrk);
      } else {
         // The first available
         sl = (TSlave *) wlst->First();
      }
      if (!sl) {
         Error("GoParallel", "attaching to candidate!");
         break;
      }
      Int_t slavenodes = 0;
      if (sl->GetSlaveType() == TSlave::kSlave) {
         sl->SetStatus(TSlave::kActive);
         fActiveSlaves->Add(sl);
         fInactiveSlaves->Remove(sl);
         fActiveMonitor->Add(sl->GetSocket());
         slavenodes = 1;
      } else if (sl->GetSlaveType() == TSlave::kMaster) {
         fEndMaster = kFALSE;
         TMessage mess(kPROOF_PARALLEL);
         if (!attach) {
            mess << nodes-cnt;
         } else {
            // To get the number of slaves
            mess.SetWhat(kPROOF_LOGFILE);
            mess << -1 << -1;
         }
         if (sl->GetSocket()->Send(mess) == -1) {
            MarkBad(sl);
            slavenodes = 0;
         } else {
            Collect(sl);
            sl->SetStatus(TSlave::kActive);
            fActiveSlaves->Add(sl);
            fInactiveSlaves->Remove(sl);
            fActiveMonitor->Add(sl->GetSocket());
            if (sl->GetParallel() > 0) {
               slavenodes = sl->GetParallel();
            } else {
               slavenodes = 0;
            }
         }
      }
      // Remove from the list
      wlst->Remove(sl);
      cnt += slavenodes;
   }

   // Cleanup list
   wlst->SetOwner(0);
   SafeDelete(wlst);

   // Get slave status (will set the slaves fWorkDir correctly)
   AskStatistics();

   // Find active slaves with unique image
   FindUniqueSlaves();

   // Send new group-view to slaves
   if (!attach)
      SendGroupView();

   Int_t n = GetParallel();

   if (!IsMaster()) {
      if (n < 1)
         printf("PROOF set to sequential mode\n");
      else
         printf("PROOF set to parallel mode (%d worker%s)\n",
                n, n == 1 ? "" : "s");
   }

   PDB(kGlobal,1) Info("GoParallel", "got %d node%s", n, n == 1 ? "" : "s");
   return n;
}

//______________________________________________________________________________
void TProof::ShowCache(Bool_t all)
{
   // List contents of file cache. If all is true show all caches also on
   // slaves. If everything is ok all caches are to be the same.

   if (!IsValid()) return;

   TMessage mess(kPROOF_CACHE);
   mess << Int_t(kShowCache) << all;
   Broadcast(mess, kUnique);

   if (all) {
      TMessage mess2(kPROOF_CACHE);
      mess2 << Int_t(kShowSubCache) << all;
      Broadcast(mess2, fNonUniqueMasters);

      Collect(kAllUnique);
   } else {
      Collect(kUnique);
   }
}

//______________________________________________________________________________
void TProof::ClearCache()
{
    // Remove files from all file caches.

   if (!IsValid()) return;

   TMessage mess(kPROOF_CACHE);
   mess << Int_t(kClearCache);
   Broadcast(mess, kUnique);

   TMessage mess2(kPROOF_CACHE);
   mess2 << Int_t(kClearSubCache);
   Broadcast(mess2, fNonUniqueMasters);

   Collect(kAllUnique);

   // clear file map so files get send again to remote nodes
   fFileMap.clear();
}

//______________________________________________________________________________
void TProof::ShowPackages(Bool_t all)
{
   // List contents of package directory. If all is true show all package
   // directries also on slaves. If everything is ok all package directories
   // should be the same.

   if (!IsValid()) return;

   if (!IsMaster()) {
      if (fGlobalPackageDirList && fGlobalPackageDirList->GetSize() > 0) {
         // Scan the list of global packages dirs
         TIter nxd(fGlobalPackageDirList);
         TNamed *nm = 0;
         while ((nm = (TNamed *)nxd())) {
            printf("*** Global Package cache %s client:%s ***\n",
                   nm->GetName(), nm->GetTitle());
            fflush(stdout);
            gSystem->Exec(Form("%s %s", kLS, nm->GetTitle()));
            printf("\n");
            fflush(stdout);
         }
      }
      printf("*** Package cache client:%s ***\n", fPackageDir.Data());
      fflush(stdout);
      gSystem->Exec(Form("%s %s", kLS, fPackageDir.Data()));
   }

   TMessage mess(kPROOF_CACHE);
   mess << Int_t(kShowPackages) << all;
   Broadcast(mess, kUnique);

   if (all) {
      TMessage mess2(kPROOF_CACHE);
      mess2 << Int_t(kShowSubPackages) << all;
      Broadcast(mess2, fNonUniqueMasters);

      Collect(kAllUnique);
   } else {
      Collect(kUnique);
   }
}

//______________________________________________________________________________
void TProof::ShowEnabledPackages(Bool_t all)
{
   // List which packages are enabled. If all is true show enabled packages
   // for all active slaves. If everything is ok all active slaves should
   // have the same packages enabled.

   if (!IsValid()) return;

   if (!IsMaster()) {
      printf("*** Enabled packages on client on %s\n", gSystem->HostName());
      TIter next(fEnabledPackagesOnClient);
      while (TObjString *str = (TObjString*) next())
         printf("%s\n", str->GetName());
   }

   TMessage mess(kPROOF_CACHE);
   mess << Int_t(kShowEnabledPackages) << all;
   Broadcast(mess);
   Collect();
}

//______________________________________________________________________________
Int_t TProof::ClearPackages()
{
   // Remove all packages.
   // Returns 0 in case of success and -1 in case of error.

   if (!IsValid()) return -1;

   if (UnloadPackages() == -1)
      return -1;

   if (DisablePackages() == -1)
      return -1;

   return fStatus;
}

//______________________________________________________________________________
Int_t TProof::ClearPackage(const char *package)
{
   // Remove a specific package.
   // Returns 0 in case of success and -1 in case of error.

   if (!IsValid()) return -1;

   if (!package || !strlen(package)) {
      Error("ClearPackage", "need to specify a package name");
      return -1;
   }

   // if name, erroneously, is a par pathname strip off .par and path
   TString pac = package;
   if (pac.EndsWith(".par"))
      pac.Remove(pac.Length()-4);
   pac = gSystem->BaseName(pac);

   if (UnloadPackage(pac) == -1)
      return -1;

   if (DisablePackage(pac) == -1)
      return -1;

   return fStatus;
}

//______________________________________________________________________________
R__HIDDEN Int_t TProof::DisablePackage(const char *package)
{
   // Remove a specific package.
   // Returns 0 in case of success and -1 in case of error.

   if (!IsValid()) return -1;

   if (!package || !strlen(package)) {
      Error("DisablePackage", "need to specify a package name");
      return -1;
   }

   // if name, erroneously, is a par pathname strip off .par and path
   TString pac = package;
   if (pac.EndsWith(".par"))
      pac.Remove(pac.Length()-4);
   pac = gSystem->BaseName(pac);

   if (DisablePackageOnClient(pac) == -1)
      return -1;

   TMessage mess(kPROOF_CACHE);
   mess << Int_t(kDisablePackage) << pac;
   Broadcast(mess, kUnique);

   TMessage mess2(kPROOF_CACHE);
   mess2 << Int_t(kDisableSubPackage) << pac;
   Broadcast(mess2, fNonUniqueMasters);

   Collect(kAllUnique);

   return fStatus;
}

//______________________________________________________________________________
R__HIDDEN Int_t TProof::DisablePackageOnClient(const char *package)
{
   // Remove a specific package from the client.
   // Returns 0 in case of success and -1 in case of error.

   if (!IsMaster()) {
      // remove package directory and par file
      fPackageLock->Lock();
      gSystem->Exec(Form("%s %s/%s", kRM, fPackageDir.Data(), package));
      gSystem->Exec(Form("%s %s/%s.par", kRM, fPackageDir.Data(), package));
      fPackageLock->Unlock();
   }

   return 0;
}

//______________________________________________________________________________
R__HIDDEN Int_t TProof::DisablePackages()
{
   // Remove all packages.
   // Returns 0 in case of success and -1 in case of error.

   if (!IsValid()) return -1;

   // remove all packages on client
   if (!IsMaster()) {
      fPackageLock->Lock();
      gSystem->Exec(Form("%s %s/*", kRM, fPackageDir.Data()));
      fPackageLock->Unlock();
   }

   TMessage mess(kPROOF_CACHE);
   mess << Int_t(kDisablePackages);
   Broadcast(mess, kUnique);

   TMessage mess2(kPROOF_CACHE);
   mess2 << Int_t(kDisableSubPackages);
   Broadcast(mess2, fNonUniqueMasters);

   Collect(kAllUnique);

   return fStatus;
}

//______________________________________________________________________________
Int_t TProof::BuildPackage(const char *package, EBuildPackageOpt opt)
{
   // Build specified package. Executes the PROOF-INF/BUILD.sh
   // script if it exists on all unique nodes. If opt is kBuildOnSlavesNoWait
   // then submit build command to slaves, but don't wait
   // for results. If opt is kCollectBuildResults then collect result
   // from slaves. To be used on the master.
   // If opt = kBuildAll (default) then submit and wait for results
   // (to be used on the client).
   // Returns 0 in case of success and -1 in case of error.

   if (!IsValid()) return -1;

   if (!package || !strlen(package)) {
      Error("BuildPackage", "need to specify a package name");
      return -1;
   }

   // if name, erroneously, is a par pathname strip off .par and path
   TString pac = package;
   if (pac.EndsWith(".par"))
      pac.Remove(pac.Length()-4);
   pac = gSystem->BaseName(pac);

   Bool_t buildOnClient = kTRUE;
   if (opt == kDontBuildOnClient) {
      buildOnClient = kFALSE;
      opt = kBuildAll;
   }

   if (opt <= kBuildAll) {
      TMessage mess(kPROOF_CACHE);
      mess << Int_t(kBuildPackage) << pac;
      Broadcast(mess, kUnique);

      TMessage mess2(kPROOF_CACHE);
      mess2 << Int_t(kBuildSubPackage) << pac;
      Broadcast(mess2, fNonUniqueMasters);
   }

   if (opt >= kBuildAll) {
      // by first forwarding the build commands to the master and slaves
      // and only then building locally we build in parallel
      Int_t st = 0;
      if (buildOnClient)
         st = BuildPackageOnClient(pac);

      Collect(kAllUnique);

      if (fStatus < 0 || st < 0)
         return -1;
   }

   return 0;
}

//______________________________________________________________________________
Int_t TProof::BuildPackageOnClient(const TString &package)
{
   // Build specified package on the client. Executes the PROOF-INF/BUILD.sh
   // script if it exists on the client.
   // Returns 0 in case of success and -1 in case of error.
   // The code is equivalent to the one in TProofServ.cxx (TProof::kBuildPackage
   // case). Keep in sync in case of changes.

   if (!IsMaster()) {
      Int_t status = 0;
      TString pdir, ocwd;

      // Package path
      pdir = fPackageDir + "/" + package;
      if (gSystem->AccessPathName(pdir, kReadPermission) ||
         gSystem->AccessPathName(pdir + "/PROOF-INF", kReadPermission)) {
         // Is there a global package with this name?
         if (fGlobalPackageDirList && fGlobalPackageDirList->GetSize() > 0) {
            // Scan the list of global packages dirs
            TIter nxd(fGlobalPackageDirList);
            TNamed *nm = 0;
            while ((nm = (TNamed *)nxd())) {
               pdir = Form("%s/%s", nm->GetTitle(), package.Data());
               if (!gSystem->AccessPathName(pdir, kReadPermission) &&
                   !gSystem->AccessPathName(pdir + "/PROOF-INF", kReadPermission)) {
                  // Package found, stop searching
                  break;
               }
               pdir = "";
            }
            if (pdir.Length() <= 0) {
               // Package not found
               Error("BuildPackageOnClient", "failure locating %s ...", package.Data());
               return -1;
            } else {
               // Package is in the global dirs
               if (gDebug > 0)
                  Info("BuildPackageOnClient", "found global package: %s", pdir.Data());
               return 0;
            }
         }
      }
      PDB(kPackage, 1)
         Info("BuildPackageOnCLient",
              "package %s exists and has PROOF-INF directory", package.Data());

      fPackageLock->Lock();

      ocwd = gSystem->WorkingDirectory();
      gSystem->ChangeDirectory(pdir);

      // check for BUILD.sh and execute
      if (!gSystem->AccessPathName("PROOF-INF/BUILD.sh")) {

         // read version from file proofvers.txt, and if current version is
         // not the same do a "BUILD.sh clean"
         FILE *f = fopen("PROOF-INF/proofvers.txt", "r");
         if (f) {
            TString v;
            v.Gets(f);
            fclose(f);
            if (v != gROOT->GetVersion()) {
               if (gSystem->Exec("PROOF-INF/BUILD.sh clean")) {
                  Error("BuildPackageOnClient", "cleaning package %s on the client failed", package.Data());
                  status = -1;
               }
            }
         }

         if (gSystem->Exec("PROOF-INF/BUILD.sh")) {
            Error("BuildPackageOnClient", "building package %s on the client failed", package.Data());
            status = -1;
         }

         f = fopen("PROOF-INF/proofvers.txt", "w");
         if (f) {
            fputs(gROOT->GetVersion(), f);
            fclose(f);
         }
      } else {
         PDB(kPackage, 1)
            Info("BuildPackageOnCLient",
                 "package %s exists but has no PROOF-INF/BUILD.sh script", package.Data());
      }

      gSystem->ChangeDirectory(ocwd);

      fPackageLock->Unlock();

      return status;
   }
   return 0;
}

//______________________________________________________________________________
Int_t TProof::LoadPackage(const char *package, Bool_t notOnClient)
{
   // Load specified package. Executes the PROOF-INF/SETUP.C script
   // on all active nodes. If notOnClient = true, don't load package
   // on the client. The default is to load the package also on the client.
   // Returns 0 in case of success and -1 in case of error.

   if (!IsValid()) return -1;

   if (!package || !strlen(package)) {
      Error("LoadPackage", "need to specify a package name");
      return -1;
   }

   // if name, erroneously, is a par pathname strip off .par and path
   TString pac = package;
   if (pac.EndsWith(".par"))
      pac.Remove(pac.Length()-4);
   pac = gSystem->BaseName(pac);

   if (!notOnClient)
      if (LoadPackageOnClient(pac) == -1)
         return -1;

   TMessage mess(kPROOF_CACHE);
   mess << Int_t(kLoadPackage) << pac;
   Broadcast(mess);
   Collect();

   return fStatus;
}

//______________________________________________________________________________
R__HIDDEN Int_t TProof::LoadPackageOnClient(const TString &package)
{
   // Load specified package in the client. Executes the PROOF-INF/SETUP.C
   // script on the client. Returns 0 in case of success and -1 in case of error.
   // The code is equivalent to the one in TProofServ.cxx (TProof::kLoadPackage
   // case). Keep in sync in case of changes.

   if (!IsMaster()) {
      Int_t status = 0;
      TString pdir, ocwd;
      // If already loaded don't do it again
      if (fEnabledPackagesOnClient->FindObject(package)) {
         Info("LoadPackageOnClient",
              "package %s already loaded", package.Data());
         return 0;
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
               pdir = Form("%s/%s", nm->GetTitle(), package.Data());
               if (!gSystem->AccessPathName(pdir, kReadPermission)) {
                  // Package found, stop searching
                  break;
               }
               pdir = "";
            }
            if (pdir.Length() <= 0) {
               // Package not found
               Error("LoadPackageOnClient", "failure locating %s ...", package.Data());
               return -1;
            }
         }
      }

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
      } else {
         PDB(kPackage, 1)
            Info("LoadPackageOnCLient",
                 "package %s exists but has no PROOF-INF/SETUP.C script", package.Data());
      }

      gSystem->ChangeDirectory(ocwd);

      if (!status) {
         // create link to package in working directory

         fPackageLock->Lock();

         FileStat_t stat;
         Int_t st = gSystem->GetPathInfo(package, stat);
         // check if symlink, if so unlink, if not give error
         // NOTE: GetPathnfo() returns 1 in case of symlink that does not point to
         // existing file or to a directory, but if fIsLink is true the symlink exists
         if (stat.fIsLink)
            gSystem->Unlink(package);
         else if (st == 0) {
            Error("LoadPackageOnClient", "cannot create symlink %s in %s on client, "
                  "another item with same name already exists", package.Data(), ocwd.Data());
            fPackageLock->Unlock();
            return -1;
         }
         gSystem->Symlink(pdir, package);

         fPackageLock->Unlock();

         // add package to list of include directories to be searched by ACliC
         gSystem->AddIncludePath(TString("-I") + package);

         // add package to list of include directories to be searched by CINT
         gROOT->ProcessLine(TString(".include ") + package);

         fEnabledPackagesOnClient->Add(new TObjString(package));
         PDB(kPackage, 1)
            Info("LoadPackageOnClient",
                 "package %s successfully loaded", package.Data());
      } else
         Error("LoadPackageOnClient", "loading package %s on client failed", package.Data());

      return status;
   }
   return 0;
}

//______________________________________________________________________________
R__HIDDEN Int_t TProof::UnloadPackage(const char *package)
{
   // Unload specified package.
   // Returns 0 in case of success and -1 in case of error.

   if (!IsValid()) return -1;

   if (!package || !strlen(package)) {
      Error("UnloadPackage", "need to specify a package name");
      return -1;
   }

   // if name, erroneously, is a par pathname strip off .par and path
   TString pac = package;
   if (pac.EndsWith(".par"))
      pac.Remove(pac.Length()-4);
   pac = gSystem->BaseName(pac);

   if (UnloadPackageOnClient(pac) == -1)
      return -1;

   TMessage mess(kPROOF_CACHE);
   mess << Int_t(kUnloadPackage) << pac;
   Broadcast(mess);
   Collect();

   return fStatus;
}

//______________________________________________________________________________
R__HIDDEN Int_t TProof::UnloadPackageOnClient(const char *package)
{
   // Unload a specific package on the client.
   // Returns 0 in case of success and -1 in case of error.
   // The code is equivalent to the one in TProofServ.cxx (TProof::UnloadPackage
   // case). Keep in sync in case of changes.

   if (!IsMaster()) {
      TObjString *pack = (TObjString *) fEnabledPackagesOnClient->FindObject(package);
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
         delete fEnabledPackagesOnClient->Remove(pack);
      }

      // cleanup the link
      if (!gSystem->AccessPathName(package))
         if (gSystem->Unlink(package) != 0)
            Warning("UnloadPackageOnClient", "unable to remove symlink to %s", package);
   }
   return 0;
}

//______________________________________________________________________________
R__HIDDEN Int_t TProof::UnloadPackages()
{
   // Unload all packages.
   // Returns 0 in case of success and -1 in case of error.

   if (!IsValid()) return -1;

   if (!IsMaster()) {
      // Iterate over packages on the client and remove each package
      TIter nextpackage(fEnabledPackagesOnClient);
      while (TObjString *objstr = dynamic_cast<TObjString*>(nextpackage()))
         if (UnloadPackageOnClient(objstr->String()) == -1 )
            return -1;
   }

   TMessage mess(kPROOF_CACHE);
   mess << Int_t(kUnloadPackages);
   Broadcast(mess);
   Collect();

   return fStatus;
}

//______________________________________________________________________________
Int_t TProof::EnablePackage(const char *package, Bool_t notOnClient)
{
   // Enable specified package. Executes the PROOF-INF/BUILD.sh
   // script if it exists followed by the PROOF-INF/SETUP.C script.
   // In case notOnClient = true, don't enable the package on the client.
   // The default is to enable packages also on the client.
   // Returns 0 in case of success and -1 in case of error.

   if (!IsValid()) return -1;

   if (!package || !strlen(package)) {
      Error("EnablePackage", "need to specify a package name");
      return -1;
   }

   // if name, erroneously, is a par pathname strip off .par and path
   TString pac = package;
   if (pac.EndsWith(".par"))
      pac.Remove(pac.Length()-4);
   pac = gSystem->BaseName(pac);

   EBuildPackageOpt opt = kBuildAll;
   if (notOnClient)
      opt = kDontBuildOnClient;

   if (BuildPackage(pac, opt) == -1)
      return -1;

   if (LoadPackage(pac, notOnClient) == -1)
      return -1;

   return 0;
}

//______________________________________________________________________________
Int_t TProof::UploadPackage(const char *pack, EUploadPackageOpt opt)
{
   // Upload a PROOF archive (PAR file). A PAR file is a compressed
   // tar file with one special additional directory, PROOF-INF
   // (blatantly copied from Java's jar format). It must have the extension
   // .par. A PAR file can be directly a binary or a source with a build
   // procedure. In the PROOF-INF directory there can be a build script:
   // BUILD.sh to be called to build the package, in case of a binary PAR
   // file don't specify a build script or make it a no-op. Then there is
   // SETUP.C which sets the right environment variables to use the package,
   // like LD_LIBRARY_PATH, etc.
   // The 'opt' allows to specify whether the .PAR should be just unpacked
   // in the existing dir (opt = kUntar, default) or a remove of the existing
   // directory should be executed (opt = kRemoveOld), so triggering a full
   // re-build. The option if effective only for PROOF protocol > 8 .
   // Returns 0 in case of success and -1 in case of error.

   if (!IsValid()) return -1;

   TString par = pack;
   if (!par.EndsWith(".par"))
      // The client specified only the name: add the extension
      par += ".par";

   // Default location is the local working dir; then the package dir
   gSystem->ExpandPathName(par);
   if (gSystem->AccessPathName(par, kReadPermission)) {
      TString tried = par;
      // Try the package dir
      par = Form("%s/%s", fPackageDir.Data(), gSystem->BaseName(par));
      if (gSystem->AccessPathName(par, kReadPermission)) {
         // Is the package a global one
         if (fGlobalPackageDirList && fGlobalPackageDirList->GetSize() > 0) {
            // Scan the list of global packages dirs
            TIter nxd(fGlobalPackageDirList);
            TNamed *nm = 0;
            TString pdir;
            while ((nm = (TNamed *)nxd())) {
               pdir = Form("%s/%s", nm->GetTitle(), pack);
               if (!gSystem->AccessPathName(pdir, kReadPermission)) {
                  // Package found, stop searching
                  break;
               }
               pdir = "";
            }
            if (pdir.Length() > 0) {
               // Package is in the global dirs
               if (gDebug > 0)
                  Info("UploadPackage", "global package found (%s): no upload needed",
                                        pdir.Data());
               return 0;
            }
         }
         Error("UploadPackage", "PAR file '%s' not found; paths tried: %s, %s",
                                gSystem->BaseName(par), tried.Data(), par.Data());
         return -1;
      }
   }

   // Strategy:
   // On the client:
   // get md5 of package and check if it is different
   // from the one stored in the local package directory. If it is lock
   // the package directory and copy the package, unlock the directory.
   // On the masters:
   // get md5 of package and check if it is different from the
   // one stored on the remote node. If it is different lock the remote
   // package directory and use TFTP or SendFile to ftp the package to the
   // remote node, unlock the directory.

   TMD5 *md5 = TMD5::FileChecksum(par);

   if (UploadPackageOnClient(par, opt, md5) == -1) {
      delete md5;
      return -1;
   }

   TMessage mess(kPROOF_CHECKFILE);
   mess << TString("+")+TString(gSystem->BaseName(par)) << (*md5);
   TMessage mess2(kPROOF_CHECKFILE);
   mess2 << TString("-")+TString(gSystem->BaseName(par)) << (*md5);
   TMessage mess3(kPROOF_CHECKFILE);
   mess3 << TString("=")+TString(gSystem->BaseName(par)) << (*md5);
   delete md5;

   if (fProtocol > 8) {
      // Send also the option
      mess << (UInt_t) opt;
      mess2 << (UInt_t) opt;
      mess3 << (UInt_t) opt;
   }

   // loop over all selected nodes
   TIter next(fUniqueSlaves);
   TSlave *sl = 0;
   while ((sl = (TSlave *) next())) {
      if (!sl->IsValid())
         continue;

      sl->GetSocket()->Send(mess);

      TMessage *reply;
      sl->GetSocket()->Recv(reply);
      if (reply->What() != kPROOF_CHECKFILE) {

         if (fProtocol > 5) {
            // remote directory is locked, upload file over the open channel
            if (SendFile(par, (kBinary | kForce), Form("%s/%s/%s",
                         sl->GetProofWorkDir(), kPROOF_PackDir,
                         gSystem->BaseName(par)), sl) < 0) {
               Error("UploadPackage", "problems uploading file %s", par.Data());
               SafeDelete(reply);
               return -1;
            }
         } else {
            // old servers receive it via TFTP
            TFTP ftp(TString("root://")+sl->GetName(), 1);
            if (!ftp.IsZombie()) {
               ftp.cd(Form("%s/%s", sl->GetProofWorkDir(), kPROOF_PackDir));
               ftp.put(par, gSystem->BaseName(par));
            }
         }

         // install package and unlock dir
         sl->GetSocket()->Send(mess2);
         SafeDelete(reply);
         sl->GetSocket()->Recv(reply);
         if (!reply || reply->What() != kPROOF_CHECKFILE) {
            Error("UploadPackage", "unpacking of package %s failed", par.Data());
            SafeDelete(reply);
            return -1;
         }
      }
      SafeDelete(reply);
   }

   // loop over all other master nodes
   TIter nextmaster(fNonUniqueMasters);
   TSlave *ma;
   while ((ma = (TSlave *) nextmaster())) {
      if (!ma->IsValid())
         continue;

      ma->GetSocket()->Send(mess3);

      TMessage *reply = 0;
      ma->GetSocket()->Recv(reply);
      if (!reply || reply->What() != kPROOF_CHECKFILE) {
         // error -> package should have been found
         Error("UploadPackage", "package %s did not exist on submaster %s",
               par.Data(), ma->GetOrdinal());
         SafeDelete(reply);
         return -1;
      }
      SafeDelete(reply);
   }

   return 0;
}

//______________________________________________________________________________
Int_t TProof::UploadPackageOnClient(const TString &par, EUploadPackageOpt opt, TMD5 *md5)
{
   // Upload a package on the client in ~/proof/packages.
   // The 'opt' allows to specify whether the .PAR should be just unpacked
   // in the existing dir (opt = kUntar, default) or a remove of the existing
   // directory should be executed (opt = kRemoveOld), thereby triggering a full
   // re-build. The option if effective only for PROOF protocol > 8 .
   // Returns 0 in case of success and -1 in case of error.

   // Strategy:
   // get md5 of package and check if it is different
   // from the one stored in the local package directory. If it is lock
   // the package directory and copy the package, unlock the directory.

   Int_t status = 0;

   if (!IsMaster()) {
      // the fPackageDir directory exists (has been created in Init())

      // create symlink to the par file in the fPackageDir (needed by
      // master in case we run on the localhost)
      fPackageLock->Lock();

      TString lpar = fPackageDir + "/" + gSystem->BaseName(par);
      FileStat_t stat;
      Int_t st = gSystem->GetPathInfo(lpar, stat);
      // check if symlink, if so unlink, if not give error
      // NOTE: GetPathInfo() returns 1 in case of symlink that does not point to
      // existing file, but if fIsLink is true the symlink exists
      if (stat.fIsLink)
         gSystem->Unlink(lpar);
      else if (st == 0) {
         Error("UploadPackageOnClient", "cannot create symlink %s on client, "
               "another item with same name already exists",
               lpar.Data());
         fPackageLock->Unlock();
         return -1;
      }
      if (!gSystem->IsAbsoluteFileName(par)) {
         TString fpar = par;
         gSystem->Symlink(gSystem->PrependPathName(gSystem->WorkingDirectory(), fpar), lpar);
      } else
         gSystem->Symlink(par, lpar);
      // TODO: On Windows need to copy instead of symlink

      // compare md5
      TString packnam = par(0, par.Length() - 4);  // strip off ".par"
      packnam = gSystem->BaseName(packnam);        // strip off path
      TString md5f = fPackageDir + "/" + packnam + "/PROOF-INF/md5.txt";
      TMD5 *md5local = TMD5::ReadChecksum(md5f);
      if (!md5local || (*md5) != (*md5local)) {
         // if not, unzip and untar package in package directory
         Int_t st = 0;
         if ((opt & TProof::kRemoveOld)) {
            // remove any previous package directory with same name
            st = gSystem->Exec(Form("%s %s/%s", kRM, fPackageDir.Data(),
                               packnam.Data()));
            if (st)
               Error("UploadPackageOnClient", "failure executing: %s %s/%s",
                     kRM, fPackageDir.Data(), packnam.Data());
         }
         // find gunzip
         char *gunzip = gSystem->Which(gSystem->Getenv("PATH"), kGUNZIP,
                                       kExecutePermission);
         if (gunzip) {
            // untar package
            st = gSystem->Exec(Form(kUNTAR2, gunzip, par.Data(), fPackageDir.Data()));
            if (st)
               Error("Uploadpackage", "failure executing: %s",
                     Form(kUNTAR2, gunzip, par.Data(), fPackageDir.Data()));
            delete [] gunzip;
         } else
            Error("UploadPackageOnClient", "%s not found", kGUNZIP);

         // check that fPackageDir/packnam now exists
         if (gSystem->AccessPathName(fPackageDir + "/" + packnam, kWritePermission)) {
            // par file did not unpack itself in the expected directory, failure
            Error("UploadPackageOnClient",
                  "package %s did not unpack into %s/%s", par.Data(), fPackageDir.Data(),
                  packnam.Data());
            status = -1;
         } else {
            // store md5 in package/PROOF-INF/md5.txt
            TMD5::WriteChecksum(md5f, md5);
         }
      }
      fPackageLock->Unlock();
      delete md5local;
   }
   return status;
}

//______________________________________________________________________________
Int_t TProof::Load(const char *macro, Bool_t notOnClient)
{
   // Load the specified macro on master, workers and, if notOnClient is
   // kFALSE, on the client. The macro file is uploaded if new or updated.
   // If existing, the corresponding header basename(macro).h or .hh, is also
   // uploaded. The default is to load the macro also on the client.
   // Returns 0 in case of success and -1 in case of error.

   if (!IsValid()) return -1;

   if (!macro || !strlen(macro)) {
      Error("Load", "need to specify a macro name");
      return -1;
   }

   if (!IsMaster()) {

      // Extract the file implementation name first
      TString implname = macro;
      TString acmode, args, io;
      implname = gSystem->SplitAclicMode(implname, acmode, args, io);

      // Macro names must have a standard format
      Int_t dot = implname.Last('.');
      if (dot == kNPOS) {
         Info("Load", "macro '%s' does not contain a '.': do nothing", macro);
         return -1;
      }

      // Is there any associated header file
      Bool_t hasHeader = kTRUE;
      TString headname = implname;
      headname.Remove(dot);
      headname += ".h";
      if (gSystem->AccessPathName(headname, kReadPermission)) {
         TString h = headname;
         headname.Remove(dot);
         headname += ".hh";
         if (gSystem->AccessPathName(headname, kReadPermission)) {
            hasHeader = kFALSE;
            if (gDebug > 0)
               Info("Load", "no associated header file found: tried: %s %s",
                            h.Data(), headname.Data());
         }
      }

      // Send files now; the md5 check is run here; see SendFile for more
      // details.
      if (SendFile(implname) == -1) {
         Info("Load", "problems sending implementation file %s", implname.Data());
         return -1;
      }
      if (hasHeader)
         if (SendFile(headname) == -1) {
            Info("Load", "problems sending header file %s", headname.Data());
            return -1;
         }

      // The files are now on the workers: now we send the loading request
      TString basemacro = gSystem->BaseName(macro);
      TMessage mess(kPROOF_CACHE);
      mess << Int_t(kLoadMacro) << basemacro;
      Broadcast(mess, kUnique);

      // Load locally, if required
      if (!notOnClient)
         // by first forwarding the load command to the master and workers
         // and only then loading locally we load/build in parallel
         gROOT->ProcessLine(Form(".L %s", macro));

      // Wait for master and workers to be done
      Collect(kAllUnique);

   } else {
      // On master

      // The files are now on the workers: now we send the loading request
      // On the master we do not wait here for the results, but after the local
      // load
      TString basemacro = gSystem->BaseName(macro);
      TMessage mess(kPROOF_CACHE);
      mess << Int_t(kLoadMacro) << basemacro;
      Broadcast(mess, kUnique);
   }

   // Done
   return 0;
}

//______________________________________________________________________________
Int_t TProof::AddDynamicPath(const char *libpath)
{
   // Add 'libpath' to the lib path search.
   // Multiple paths can be specified at once separating them with a comma or
   // a blank.
   // Return 0 on success, -1 otherwise

   if ((!libpath || !strlen(libpath))) {
      if (gDebug > 0)
         Info("AddDynamicPath", "list is empty - nothing to do");
      return 0;
   }

   TMessage m(kPROOF_LIB_INC_PATH);
   m << TString("lib") << (Bool_t)kTRUE;

   // Add paths
   if (libpath && strlen(libpath))
      m << TString(libpath);
   else
      m << TString("-");

   // Forward the request
   Broadcast(m);
   Collect();

   return 0;
}

//______________________________________________________________________________
Int_t TProof::AddIncludePath(const char *incpath)
{
   // Add 'incpath' to the inc path search.
   // Multiple paths can be specified at once separating them with a comma or
   // a blank.
   // Return 0 on success, -1 otherwise

   if ((!incpath || !strlen(incpath))) {
      if (gDebug > 0)
         Info("AddIncludePath", "list is empty - nothing to do");
      return 0;
   }

   TMessage m(kPROOF_LIB_INC_PATH);
   m << TString("inc") << (Bool_t)kTRUE;

   // Add paths
   if (incpath && strlen(incpath))
      m << TString(incpath);
   else
      m << TString("-");

   // Forward the request
   Broadcast(m);
   Collect();

   return 0;
}

//______________________________________________________________________________
Int_t TProof::RemoveDynamicPath(const char *libpath)
{
   // Remove 'libpath' from the lib path search.
   // Multiple paths can be specified at once separating them with a comma or
   // a blank.
   // Return 0 on success, -1 otherwise

   if ((!libpath || !strlen(libpath))) {
      if (gDebug > 0)
         Info("AddDynamicPath", "list is empty - nothing to do");
      return 0;
   }

   TMessage m(kPROOF_LIB_INC_PATH);
   m << TString("lib") <<(Bool_t)kFALSE;

   // Add paths
   if (libpath && strlen(libpath))
      m << TString(libpath);
   else
      m << TString("-");

   // Forward the request
   Broadcast(m);
   Collect();

   return 0;
}

//______________________________________________________________________________
Int_t TProof::RemoveIncludePath(const char *incpath)
{
   // Remove 'incpath' from the inc path search.
   // Multiple paths can be specified at once separating them with a comma or
   // a blank.
   // Return 0 on success, -1 otherwise

   if ((!incpath || !strlen(incpath))) {
      if (gDebug > 0)
         Info("RemoveIncludePath", "list is empty - nothing to do");
      return 0;
   }

   TMessage m(kPROOF_LIB_INC_PATH);
   m << TString("inc") << (Bool_t)kFALSE;

   // Add paths
   if (incpath && strlen(incpath))
      m << TString(incpath);
   else
      m << TString("-");

   // Forward the request
   Broadcast(m);
   Collect();

   return 0;
}

//______________________________________________________________________________
TList *TProof::GetListOfPackages()
{
   // Get from the master the list of names of the packages available.

   if (!IsValid())
      return (TList *)0;

   TMessage mess(kPROOF_CACHE);
   mess << Int_t(kListPackages);
   Broadcast(mess);
   Collect();

   return fAvailablePackages;
}

//______________________________________________________________________________
TList *TProof::GetListOfEnabledPackages()
{
   // Get from the master the list of names of the packages enabled.

   if (!IsValid())
      return (TList *)0;

   TMessage mess(kPROOF_CACHE);
   mess << Int_t(kListEnabledPackages);
   Broadcast(mess);
   Collect();

   return fEnabledPackages;
}

//______________________________________________________________________________
void TProof::PrintProgress(Long64_t total, Long64_t processed, Float_t procTime)
{
   // Print a progress bar on stderr. Used in batch mode.

   fprintf(stderr, "[TProof::Progress] Total %lld events\t|", total);

   for (int l = 0; l < 20; l++) {
      if (total > 0) {
         if (l < 20*processed/total)
            fprintf(stderr, "=");
         else if (l == 20*processed/total)
            fprintf(stderr, ">");
         else if (l > 20*processed/total)
            fprintf(stderr, ".");
      } else
         fprintf(stderr, "=");
   }
   Float_t evtrti = (procTime > 0. && processed > 0) ? processed / procTime : -1.;
   if (evtrti > 0.)
      fprintf(stderr, "| %.02f %% [%.1f evts/s]\r",
              (total ? ((100.0*processed)/total) : 100.0), evtrti);
   else
      fprintf(stderr, "| %.02f %%\r",
              (total ? ((100.0*processed)/total) : 100.0));
   if (processed >= total)
      fprintf(stderr, "\n");
}

//______________________________________________________________________________
void TProof::Progress(Long64_t total, Long64_t processed)
{
   // Get query progress information. Connect a slot to this signal
   // to track progress.

   PDB(kGlobal,1)
      Info("Progress","%2f (%lld/%lld)", 100.*processed/total, processed, total);

   if (gROOT->IsBatch()) {
      // Simple progress bar
      PrintProgress(total, processed);
   } else {
      EmitVA("Progress(Long64_t,Long64_t)", 2, total, processed);
   }
}

//______________________________________________________________________________
void TProof::Progress(Long64_t total, Long64_t processed, Long64_t bytesread,
                      Float_t initTime, Float_t procTime,
                      Float_t evtrti, Float_t mbrti)
{
   // Get query progress information. Connect a slot to this signal
   // to track progress.

   PDB(kGlobal,1)
      Info("Progress","%lld %lld %lld %f %f %f %f", total, processed, bytesread,
                                initTime, procTime, evtrti, mbrti);

   if (gROOT->IsBatch()) {
      // Simple progress bar
      PrintProgress(total, processed, procTime);
   } else {
      EmitVA("Progress(Long64_t,Long64_t,Long64_t,Float_t,Float_t,Float_t,Float_t)",
             7, total, processed, bytesread, initTime, procTime, evtrti, mbrti);
   }
}

//______________________________________________________________________________
void TProof::Feedback(TList *objs)
{
   // Get list of feedback objects. Connect a slot to this signal
   // to monitor the feedback object.

   PDB(kGlobal,1)
      Info("Feedback","%d objects", objs->GetSize());
   PDB(kFeedback,1) {
      Info("Feedback","%d objects", objs->GetSize());
      objs->ls();
   }

   Emit("Feedback(TList *objs)", (Long_t) objs);
}

//______________________________________________________________________________
void TProof::CloseProgressDialog()
{
   // Close progress dialog.

   PDB(kGlobal,1)
      Info("CloseProgressDialog",
           "called: have progress dialog: %d", fProgressDialogStarted);

   // Nothing to do if not there
   if (!fProgressDialogStarted)
      return;

   Emit("CloseProgressDialog()");
}

//______________________________________________________________________________
void TProof::ResetProgressDialog(const char *sel, Int_t sz, Long64_t fst,
                                 Long64_t ent)
{
   // Reset progress dialog.

   PDB(kGlobal,1)
      Info("ResetProgressDialog","(%s,%d,%lld,%lld)", sel, sz, fst, ent);

   EmitVA("ResetProgressDialog(const char*,Int_t,Long64_t,Long64_t)",
          4, sel, sz, fst, ent);
}

//______________________________________________________________________________
void TProof::StartupMessage(const char *msg, Bool_t st, Int_t done, Int_t total)
{
   // Send startup message.

   PDB(kGlobal,1)
      Info("StartupMessage","(%s,%d,%d,%d)", msg, st, done, total);

   EmitVA("StartupMessage(const char*,Bool_t,Int_t,Int_t)",
          4, msg, st, done, total);
}

//______________________________________________________________________________
void TProof::DataSetStatus(const char *msg, Bool_t st, Int_t done, Int_t total)
{
   // Send dataset preparation status.

   PDB(kGlobal,1)
      Info("DataSetStatus","(%s,%d,%d,%d)", msg, st, done, total);

   EmitVA("DataSetStatus(const char*,Bool_t,Int_t,Int_t)",
          4, msg, st, done, total);
}

//______________________________________________________________________________
void TProof::SendDataSetStatus(const char *msg, UInt_t n,
                                 UInt_t tot, Bool_t st)
{
   // Send data set status

   if (IsMaster()) {
      TMessage mess(kPROOF_DATASET_STATUS);
      mess << TString(msg) << tot << n << st;
      gProofServ->GetSocket()->Send(mess);
   }
}

//______________________________________________________________________________
void TProof::QueryResultReady(const char *ref)
{
   // Notify availability of a query result.

   PDB(kGlobal,1)
      Info("QueryResultReady","ref: %s", ref);

   Emit("QueryResultReady(const char*)",ref);
}

//______________________________________________________________________________
void TProof::ValidateDSet(TDSet *dset)
{
   // Validate a TDSet.

   if (dset->ElementsValid()) return;

   TList nodes;
   nodes.SetOwner();

   TList slholder;
   slholder.SetOwner();
   TList elemholder;
   elemholder.SetOwner();

   // build nodelist with slaves and elements
   TIter nextSlave(GetListOfActiveSlaves());
   while (TSlave *sl = dynamic_cast<TSlave*>(nextSlave())) {
      TList *sllist = 0;
      TPair *p = dynamic_cast<TPair*>(nodes.FindObject(sl->GetName()));
      if (!p) {
         sllist = new TList;
         sllist->SetName(sl->GetName());
         slholder.Add(sllist);
         TList *elemlist = new TList;
         elemlist->SetName(TString(sl->GetName())+"_elem");
         elemholder.Add(elemlist);
         nodes.Add(new TPair(sllist, elemlist));
      } else {
         sllist = dynamic_cast<TList*>(p->Key());
      }
      sllist->Add(sl);
   }

   // add local elements to nodes
   TList nonLocal; // list of nonlocal elements
   // make two iterations - first add local elements - then distribute nonlocals
   for (Int_t i = 0; i < 2; i++) {
      Bool_t local = i>0?kFALSE:kTRUE;
      TIter nextElem(local ? dset->GetListOfElements() : &nonLocal);
      while (TDSetElement *elem = dynamic_cast<TDSetElement*>(nextElem())) {
         if (elem->GetValid()) continue;
         TPair *p = dynamic_cast<TPair*>(local?nodes.FindObject(TUrl(elem->GetFileName()).GetHost()):nodes.At(0));
         if (p) {
            TList *eli = dynamic_cast<TList*>(p->Value());
            TList *sli = dynamic_cast<TList*>(p->Key());
            eli->Add(elem);

            // order list by elements/slave
            TPair *p2 = p;
            Bool_t stop = kFALSE;
            while (!stop) {
               TPair *p3 = dynamic_cast<TPair*>(nodes.After(p2->Key()));
               if (p3) {
                  Int_t nelem = dynamic_cast<TList*>(p3->Value())->GetSize();
                  Int_t nsl = dynamic_cast<TList*>(p3->Key())->GetSize();
                  if (nelem*sli->GetSize() < eli->GetSize()*nsl) p2 = p3;
                  else stop = kTRUE;
               } else {
                  stop = kTRUE;
               }
            }

            if (p2!=p) {
               nodes.Remove(p->Key());
               nodes.AddAfter(p2->Key(), p);
            }

         } else {
            if (local) {
               nonLocal.Add(elem);
            } else {
               Error("ValidateDSet", "No Node to allocate TDSetElement to");
               R__ASSERT(0);
            }
         }
      }
   }

   // send to slaves
   TList usedslaves;
   TIter nextNode(&nodes);
   SetDSet(dset); // set dset to be validated in Collect()
   while (TPair *node = dynamic_cast<TPair*>(nextNode())) {
      TList *slaves = dynamic_cast<TList*>(node->Key());
      TList *setelements = dynamic_cast<TList*>(node->Value());

      // distribute elements over the slaves
      Int_t nslaves = slaves->GetSize();
      Int_t nelements = setelements->GetSize();
      for (Int_t i=0; i<nslaves; i++) {

         TDSet copyset(dset->GetType(), dset->GetObjName(),
                       dset->GetDirectory());
         for (Int_t j = (i*nelements)/nslaves;
                    j < ((i+1)*nelements)/nslaves;
                    j++) {
            TDSetElement *elem =
               dynamic_cast<TDSetElement*>(setelements->At(j));
            copyset.Add(elem->GetFileName(), elem->GetObjName(),
                        elem->GetDirectory(), elem->GetFirst(),
                        elem->GetNum(), elem->GetMsd());
         }

         if (copyset.GetListOfElements()->GetSize()>0) {
            TMessage mesg(kPROOF_VALIDATE_DSET);
            mesg << &copyset;

            TSlave *sl = dynamic_cast<TSlave*>(slaves->At(i));
            PDB(kGlobal,1) Info("ValidateDSet",
                                "Sending TDSet with %d elements to slave %s"
                                " to be validated",
                                copyset.GetListOfElements()->GetSize(),
                                sl->GetOrdinal());
            sl->GetSocket()->Send(mesg);
            usedslaves.Add(sl);
         }
      }
   }

   PDB(kGlobal,1) Info("ValidateDSet","Calling Collect");
   Collect(&usedslaves);
   SetDSet(0);
}

//______________________________________________________________________________
void TProof::AddInput(TObject *obj)
{
   // Add objects that might be needed during the processing of
   // the selector (see Process()).

   fPlayer->AddInput(obj);
}

//______________________________________________________________________________
void TProof::ClearInput()
{
   // Clear input object list.

   fPlayer->ClearInput();

   // the system feedback list is always in the input list
   AddInput(fFeedback);
}

//______________________________________________________________________________
TObject *TProof::GetOutput(const char *name)
{
   // Get specified object that has been produced during the processing
   // (see Process()).

   return fPlayer->GetOutput(name);
}

//______________________________________________________________________________
TList *TProof::GetOutputList()
{
   // Get list with all object created during processing (see Process()).

   return fPlayer->GetOutputList();
}

//______________________________________________________________________________
void TProof::SetParameter(const char *par, const char *value)
{
   // Set input list parameter. If the parameter is already
   // set it will be set to the new value.

   TList *il = fPlayer->GetInputList();
   TObject *item = il->FindObject(par);
   if (item) {
      il->Remove(item);
      delete item;
   }
   il->Add(new TNamed(par, value));
}

//______________________________________________________________________________
void TProof::SetParameter(const char *par, Long_t value)
{
   // Set an input list parameter.

   TList *il = fPlayer->GetInputList();
   TObject *item = il->FindObject(par);
   if (item) {
      il->Remove(item);
      delete item;
   }
   il->Add(new TParameter<Long_t>(par, value));
}

//______________________________________________________________________________
void TProof::SetParameter(const char *par, Long64_t value)
{
   // Set an input list parameter.

   TList *il = fPlayer->GetInputList();
   TObject *item = il->FindObject(par);
   if (item) {
      il->Remove(item);
      delete item;
   }
   il->Add(new TParameter<Long64_t>(par, value));
}

//______________________________________________________________________________
void TProof::SetParameter(const char *par, Double_t value)
{
   // Set an input list parameter.

   TList *il = fPlayer->GetInputList();
   TObject *item = il->FindObject(par);
   if (item) {
      il->Remove(item);
      delete item;
   }
   il->Add(new TParameter<Double_t>(par, value));
}

//______________________________________________________________________________
TObject *TProof::GetParameter(const char *par) const
{
   // Get specified parameter. A parameter set via SetParameter() is either
   // a TParameter or a TNamed or 0 in case par is not defined.

   TList *il = fPlayer->GetInputList();
   return il->FindObject(par);
}

//______________________________________________________________________________
void TProof::DeleteParameters(const char *wildcard)
{
   // Delete the input list parameters specified by a wildcard (e.g. PROOF_*)
   // or exact name (e.g. PROOF_MaxSlavesPerNode).

   if (!wildcard) wildcard = "";
   TRegexp re(wildcard, kTRUE);
   Int_t nch = strlen(wildcard);

   TList *il = fPlayer->GetInputList();
   TObject *p;
   TIter next(il);
   while ((p = next())) {
      TString s = p->GetName();
      if (nch && s != wildcard && s.Index(re) == kNPOS) continue;
      il->Remove(p);
      delete p;
   }
}

//______________________________________________________________________________
void TProof::ShowParameters(const char *wildcard) const
{
   // Show the input list parameters specified by the wildcard.
   // Default is the special PROOF control parameters (PROOF_*).

   if (!wildcard) wildcard = "";
   TRegexp re(wildcard, kTRUE);
   Int_t nch = strlen(wildcard);

   TList *il = fPlayer->GetInputList();
   TObject *p;
   TIter next(il);
   while ((p = next())) {
      TString s = p->GetName();
      if (nch && s != wildcard && s.Index(re) == kNPOS) continue;
      if (p->IsA() == TNamed::Class()) {
         Printf("%s\t\t\t%s", s.Data(), p->GetTitle());
      } else if (p->IsA() == TParameter<Long_t>::Class()) {
         Printf("%s\t\t\t%ld", s.Data(), dynamic_cast<TParameter<Long_t>*>(p)->GetVal());
      } else if (p->IsA() == TParameter<Long64_t>::Class()) {
         Printf("%s\t\t\t%lld", s.Data(), dynamic_cast<TParameter<Long64_t>*>(p)->GetVal());
      } else if (p->IsA() == TParameter<Double_t>::Class()) {
         Printf("%s\t\t\t%f", s.Data(), dynamic_cast<TParameter<Double_t>*>(p)->GetVal());
      } else {
         Printf("%s\t\t\t%s", s.Data(), p->GetTitle());
      }
   }
}

//______________________________________________________________________________
void TProof::AddFeedback(const char *name)
{
   // Add object to feedback list.

   PDB(kFeedback, 3)
      Info("AddFeedback", "Adding object \"%s\" to feedback", name);
   if (fFeedback->FindObject(name) == 0)
      fFeedback->Add(new TObjString(name));
}

//______________________________________________________________________________
void TProof::RemoveFeedback(const char *name)
{
   // Remove object from feedback list.

   TObject *obj = fFeedback->FindObject(name);
   if (obj != 0) {
      fFeedback->Remove(obj);
      delete obj;
   }
}

//______________________________________________________________________________
void TProof::ClearFeedback()
{
   // Clear feedback list.

   fFeedback->Delete();
}

//______________________________________________________________________________
void TProof::ShowFeedback() const
{
   // Show items in feedback list.

   if (fFeedback->GetSize() == 0) {
      Info("","no feedback requested");
      return;
   }

   fFeedback->Print();
}

//______________________________________________________________________________
TList *TProof::GetFeedbackList() const
{
   // Return feedback list.

   return fFeedback;
}

//______________________________________________________________________________
TTree *TProof::GetTreeHeader(TDSet *dset)
{
   // Creates a tree header (a tree with nonexisting files) object for
   // the DataSet.

   TList *l = GetListOfActiveSlaves();
   TSlave *sl = (TSlave*) l->First();
   if (sl == 0) {
      Error("GetTreeHeader", "No connection");
      return 0;
   }

   TSocket *soc = sl->GetSocket();
   TMessage msg(kPROOF_GETTREEHEADER);

   msg << dset;

   soc->Send(msg);

   TMessage *reply;
   Int_t d = soc->Recv(reply);
   if (reply <= 0) {
      Error("GetTreeHeader", "Error getting a replay from the master.Result %d", (int) d);
      return 0;
   }

   TString s1;
   TTree * t;
   (*reply) >> s1;
   (*reply) >> t;

   PDB(kGlobal, 1)
      if (t)
         Info("GetTreeHeader", Form("%s, message size: %d, entries: %d\n",
             s1.Data(), reply->BufferSize(), (int) t->GetMaxEntryLoop()));
      else
         Info("GetTreeHeader", Form("%s, message size: %d\n", s1.Data(), reply->BufferSize()));

   delete reply;

   return t;
}

//______________________________________________________________________________
TDrawFeedback *TProof::CreateDrawFeedback()
{
   // Draw feedback creation proxy. When accessed via TProof avoids
   // link dependency on libProofPlayer.

   return fPlayer->CreateDrawFeedback(this);
}

//______________________________________________________________________________
void TProof::SetDrawFeedbackOption(TDrawFeedback *f, Option_t *opt)
{
   // Set draw feedback option.

   fPlayer->SetDrawFeedbackOption(f, opt);
}

//______________________________________________________________________________
void TProof::DeleteDrawFeedback(TDrawFeedback *f)
{
   // Delete draw feedback object.

   fPlayer->DeleteDrawFeedback(f);
}

//______________________________________________________________________________
TList *TProof::GetOutputNames()
{
   //   FIXME: to be written

   return 0;
/*
   TMessage msg(kPROOF_GETOUTPUTLIST);
   TList* slaves = fActiveSlaves;
   Broadcast(msg, slaves);
   TMonitor mon;
   TList* outputList = new TList();

   TIter    si(slaves);
   TSlave   *slave;
   while ((slave = (TSlave*)si.Next()) != 0) {
      PDB(kGlobal,4) Info("GetOutputNames","Socket added to monitor: %p (%s)",
          slave->GetSocket(), slave->GetName());
      mon.Add(slave->GetSocket());
   }
   mon.ActivateAll();
   ((TProof*)gProof)->DeActivateAsyncInput();
   ((TProof*)gProof)->fCurrentMonitor = &mon;

   while (mon.GetActive() != 0) {
      TSocket *sock = mon.Select();
      if (!sock) {
         Error("GetOutputList","TMonitor::.Select failed!");
         break;
      }
      mon.DeActivate(sock);
      TMessage *reply;
      if (sock->Recv(reply) <= 0) {
         MarkBad(slave);
//         Error("GetOutputList","Recv failed! for slave-%d (%s)",
//               slave->GetOrdinal(), slave->GetName());
         continue;
      }
      if (reply->What() != kPROOF_GETOUTPUTNAMES ) {
//         Error("GetOutputList","unexpected message %d from slawe-%d (%s)",  reply->What(),
//               slave->GetOrdinal(), slave->GetName());
         MarkBad(slave);
         continue;
      }
      TList* l;

      (*reply) >> l;
      TIter next(l);
      TNamed *n;
      while ( (n = dynamic_cast<TNamed*> (next())) ) {
         if (!outputList->FindObject(n->GetName()))
            outputList->Add(n);
      }
      delete reply;
   }
   ((TProof*)gProof)->fCurrentMonitor = 0;

   return outputList;
*/
}

//______________________________________________________________________________
void TProof::Browse(TBrowser *b)
{
   // Build the PROOF's structure in the browser.

   b->Add(fActiveSlaves, fActiveSlaves->Class(), "fActiveSlaves");
   b->Add(&fMaster, fMaster.Class(), "fMaster");
   b->Add(fFeedback, fFeedback->Class(), "fFeedback");
   b->Add(fChains, fChains->Class(), "fChains");

   b->Add(fPlayer->GetInputList(), fPlayer->GetInputList()->Class(), "InputList");
   if (fPlayer->GetOutputList())
      b->Add(fPlayer->GetOutputList(), fPlayer->GetOutputList()->Class(), "OutputList");
   if (fPlayer->GetListOfResults())
      b->Add(fPlayer->GetListOfResults(),
             fPlayer->GetListOfResults()->Class(), "ListOfResults");
}

//______________________________________________________________________________
void TProof::SetPlayer(TVirtualProofPlayer *player)
{
   // Set a new PROOF player.

   if (fPlayer)
      delete fPlayer;
   fPlayer = player;
};

//______________________________________________________________________________
TVirtualProofPlayer *TProof::MakePlayer(const char *player, TSocket *s)
{
   // Construct a TProofPlayer object. The player string specifies which
   // player should be created: remote, slave, sm (supermaster) or base.
   // Default is remote. Socket is needed in case a slave player is created.

   if (!player)
      player = "remote";

   SetPlayer(TVirtualProofPlayer::Create(player, this, s));
   return GetPlayer();
}

//______________________________________________________________________________
void TProof::AddChain(TChain *chain)
{
   // Add chain to data set

   fChains->Add(chain);
}

//______________________________________________________________________________
void TProof::RemoveChain(TChain *chain)
{
   // Remove chain from data set

   fChains->Remove(chain);
}

//_____________________________________________________________________________
void *TProof::SlaveStartupThread(void *arg)
{
   // Function executed in the slave startup thread.

   if (fgSemaphore) fgSemaphore->Wait();

   TProofThreadArg *ta = (TProofThreadArg *)arg;

   PDB(kGlobal,1)
      ::Info("TProof::SlaveStartupThread",
             "Starting slave %s on host %s", ta->fOrd.Data(), ta->fUrl->GetHost());

   TSlave *sl = 0;
   if (ta->fType == TSlave::kSlave) {
      // Open the connection
      sl = ta->fProof->CreateSlave(ta->fUrl->GetUrl(), ta->fOrd,
                                   ta->fPerf, ta->fImage, ta->fWorkdir);
      // Finalize setup of the server
      if (sl && sl->IsValid())
         sl->SetupServ(TSlave::kSlave, 0);
   } else {
      // Open the connection
      sl = ta->fProof->CreateSubmaster(ta->fUrl->GetUrl(), ta->fOrd,
                                       ta->fImage, ta->fMsd);
      // Finalize setup of the server
      if (sl && sl->IsValid())
         sl->SetupServ(TSlave::kMaster, ta->fWorkdir);
   }

   if (sl && sl->IsValid()) {

      {  R__LOCKGUARD2(gProofMutex);

         // Add to the started slaves list
         ta->fSlaves->Add(sl);

         if (ta->fClaims) { // Condor slave
            // Remove from the pending claims list
            TCondorSlave *c = ta->fCslave;
            ta->fClaims->Remove(c);
         }
      }

      // Notify we are done
      PDB(kGlobal,1)
         ::Info("TProof::SlaveStartupThread",
                "slave %s on host %s created and added to list",
                ta->fOrd.Data(), ta->fUrl->GetHost());
   } else {
      // Failure
      SafeDelete(sl);
      ::Error("TProof::SlaveStartupThread",
              "slave %s on host %s could not be created",
              ta->fOrd.Data(), ta->fUrl->GetHost());
   }

   if (fgSemaphore) fgSemaphore->Post();

   return 0;
}

//______________________________________________________________________________
void TProof::GetLog(Int_t start, Int_t end)
{
   // Ask for remote logs in the range [start, end]. If start == -1 all the
   // messages not yet received are sent back.

   if (!IsValid() || IsMaster()) return;

   TMessage msg(kPROOF_LOGFILE);

   msg << start << end;

   Broadcast(msg, kActive);
   Collect(kActive);
}

//______________________________________________________________________________
void TProof::PutLog(TQueryResult *pq)
{
   // Display log of query pq into the log window frame

   if (!pq) return;

   TList *lines = pq->GetLogFile()->GetListOfLines();
   if (lines) {
      TIter nxl(lines);
      TObjString *l = 0;
      while ((l = (TObjString *)nxl()))
         EmitVA("LogMessage(const char*,Bool_t)", 2, l->GetName(), kFALSE);
   }
}

//______________________________________________________________________________
void TProof::ShowLog(const char *queryref)
{
   // Display on screen the content of the temporary log file for query
   // in reference

   // Make sure we have all info (GetListOfQueries retrieves the
   // head info only)
   Retrieve(queryref);

   if (fPlayer) {
      if (queryref) {
         if (fPlayer->GetListOfResults()) {
            TIter nxq(fPlayer->GetListOfResults());
            TQueryResult *qr = 0;
            while ((qr = (TQueryResult *) nxq()))
               if (strstr(queryref, qr->GetTitle()) &&
                   strstr(queryref, qr->GetName()))
                  break;
            if (qr) {
               PutLog(qr);
               return;
            }

         }
      }
   }
}

//______________________________________________________________________________
void TProof::ShowLog(Int_t qry)
{
   // Display on screen the content of the temporary log file.
   // If qry == -2 show messages from the last (current) query.
   // If qry == -1 all the messages not yet displayed are shown (default).
   // If qry == 0, all the messages in the file are shown.
   // If qry  > 0, only the messages related to query 'qry' are shown.
   // For qry != -1 the original file offset is restored at the end

   // Save present offset
   Int_t nowlog = lseek(fileno(fLogFileR), (off_t) 0, SEEK_CUR);

   // Get extremes
   Int_t startlog = nowlog;
   Int_t endlog = lseek(fileno(fLogFileR), (off_t) 0, SEEK_END);

   lseek(fileno(fLogFileR), (off_t) nowlog, SEEK_SET);
   if (qry == 0) {
      startlog = 0;
      lseek(fileno(fLogFileR), (off_t) 0, SEEK_SET);
   } else if (qry != -1) {

      TQueryResult *pq = 0;
      if (qry == -2) {
         // Pickup the last one
         pq = (GetQueryResults()) ? ((TQueryResult *)(GetQueryResults()->Last())) : 0;
         if (!pq) {
            GetListOfQueries();
            if (fQueries)
               pq = (TQueryResult *)(fQueries->Last());
         }
      } else if (qry > 0) {
         TList *queries = GetQueryResults();
         if (queries) {
            TIter nxq(queries);
            while ((pq = (TQueryResult *)nxq()))
               if (qry == pq->GetSeqNum())
                  break;
         }
         if (!pq) {
            queries = GetListOfQueries();
            TIter nxq(queries);
            while ((pq = (TQueryResult *)nxq()))
               if (qry == pq->GetSeqNum())
                  break;
         }
      }
      if (pq) {
         PutLog(pq);
         return;
      } else {
         if (gDebug > 0)
            Info("ShowLog","query %d not found in list", qry);
         qry = -1;
      }
   }

   // Number of bytes to log
   UInt_t tolog = (UInt_t)(endlog - startlog);

   // Perhaps nothing
   if (tolog <= 0)

   // Set starting point
   lseek(fileno(fLogFileR), (off_t) startlog, SEEK_SET);

   // Now we go
   Int_t np = 0;
   char line[2048];
   Int_t wanted = (tolog > sizeof(line)) ? sizeof(line) : tolog;
   while (fgets(line, wanted, fLogFileR)) {

      Int_t r = strlen(line);
      if (!SendingLogToWindow()) {
         if (line[r-1] != '\n') line[r-1] = '\n';
         if (r > 0) {
            char *p = line;
            while (r) {
               Int_t w = write(fileno(stdout), p, r);
               if (w < 0) {
                  SysError("ShowLogFile", "error writing to stdout");
                  break;
               }
               r -= w;
               p += w;
            }
         }
         tolog -= strlen(line);
         np++;

         // Ask if more is wanted
         if (!(np%10)) {
            char *opt = Getline("More (y/n)? [y]");
            if (opt[0] == 'n')
               break;
         }

         // We may be over
         if (tolog <= 0)
            break;

         // Update wanted bytes
         wanted = (tolog > sizeof(line)) ? sizeof(line) : tolog;
      } else {
         // Log to window
         if (line[r-1] == '\n') line[r-1] = 0;
         LogMessage(line, kFALSE);
      }
   }
   if (!SendingLogToWindow()) {
      // Avoid screwing up the prompt
      write(fileno(stdout), "\n", 1);
   }

   // Restore original pointer
   if (qry > -1)
      lseek(fileno(fLogFileR), (off_t) nowlog, SEEK_SET);
}

//______________________________________________________________________________
void TProof::cd(Int_t id)
{
   // Set session with 'id' the default one. If 'id' is not found in the list,
   // the current session is set as default

   if (GetManager()) {
      TProofDesc *d = GetManager()->GetProofDesc(id);
      if (d) {
         if (d->GetProof()) {
            gProof = d->GetProof();
            return;
         }
      }

      // Id not found or undefined: set as default this session
      gProof = this;
   }

   return;
}

//______________________________________________________________________________
void TProof::Detach(Option_t *opt)
{
   // Detach this instance to its proofserv.
   // If opt is 'S' or 's' the remote server is shutdown

   // Nothing to do if not in contact with proofserv
   if (!IsValid()) return;

   // Get worker and socket instances
   TSlave *sl = (TSlave *) fActiveSlaves->First();
   TSocket *s = sl->GetSocket();
   if (!sl || !(sl->IsValid()) || !s) {
      Error("Detach","corrupted worker instance: wrk:%p, sock:%p", sl, s);
      return;
   }

   Bool_t shutdown = (strchr(opt,'s') || strchr(opt,'S')) ? kTRUE : kFALSE;

   // If processing, try to stop processing first
   if (shutdown && !IsIdle()) {
      // Remove pending requests
      Remove("cleanupqueue");
      // Do not wait for ever, but al least 20 seconds
      Long_t timeout = gEnv->GetValue("Proof.ShutdownTimeout", 60);
      timeout = (timeout > 20) ? timeout : 20;
      // Send stop signal
      StopProcess(kFALSE, (Long_t) (timeout / 2));
      // Receive results
      Collect(kActive, timeout);
   }

   // Avoid spurious messages: deactivate new inputs ...
   DeActivateAsyncInput();

   // ... and discard existing ones
   sl->FlushSocket();

   // Close session (we always close the connection)
   Close(opt);

   // Close the progress dialog, if any
   if (fProgressDialogStarted)
      CloseProgressDialog();

   // Update info in the table of our manager, if any
   if (GetManager() && GetManager()->QuerySessions("L")) {
      TIter nxd(GetManager()->QuerySessions("L"));
      TProofDesc *d = 0;
      while ((d = (TProofDesc *)nxd())) {
         if (d->GetProof() == this) {
            d->SetProof(0);
            GetManager()->QuerySessions("L")->Remove(d);
            break;
         }
      }
   }

   // Delete this instance
   if (!fProgressDialogStarted)
      delete this;
   else
      // ~TProgressDialog will delete this
      fValid = kFALSE;

   return;
}

//______________________________________________________________________________
void TProof::SetAlias(const char *alias)
{
   // Set an alias for this session. If reconnection is supported, the alias
   // will be communicated to the remote coordinator so that it can be recovered
   // when reconnecting

   // Set it locally
   TNamed::SetTitle(alias);
   if (IsMaster())
      // Set the name at the same value
      TNamed::SetName(alias);

   // Nothing to do if not in contact with coordinator
   if (!IsValid()) return;

   if (!IsProofd() && !IsMaster()) {
      TSlave *sl = (TSlave *) fActiveSlaves->First();
      if (sl)
         sl->SetAlias(alias);
   }

   return;
}

//______________________________________________________________________________
Int_t TProof::UploadDataSet(const char *dataSetName,
                            TList *files,
                            const char *desiredDest,
                            Int_t opt,
                            TList *skippedFiles)
{
   // Upload a set of files and save the list of files by name dataSetName.
   // The 'files' argument is a list of TFileInfo objects describing the files
   // as first url.
   // The mask 'opt' is a combination of EUploadOpt:
   //   kAppend             (0x1)   if set true files will be appended to
   //                               the dataset existing by given name
   //   kOverwriteDataSet   (0x2)   if dataset with given name exited it
   //                               would be overwritten
   //   kNoOverwriteDataSet (0x4)   do not overwirte if the dataset exists
   //   kOverwriteAllFiles  (0x8)   overwrite all files that may exist
   //   kOverwriteNoFiles   (0x10)  overwrite none
   //   kAskUser            (0x0)   ask user before overwriteng dataset/files
   // The default value is kAskUser.
   // The user will be asked to confirm overwriting dataset or files unless
   // specified opt provides the answer!
   // If kOverwriteNoFiles is set, then a pointer to TList must be passed as
   // skippedFiles argument. The function will add to this list TFileInfo
   // objects describing all files that existed on the cluster and were
   // not uploaded.
   //
   // Communication Summary
   // Client                             Master
   //    |------------>DataSetName----------->|
   //    |<-------kMESS_OK/kMESS_NOTOK<-------| (Name OK/file exist)
   // (*)|-------> call CreateDataSet ------->|
   // (*) - optional

   // check if  dataSetName is not excluded
   if (strchr(dataSetName, '/')) {
      if (strstr(dataSetName, "public") != dataSetName) {
         Error("UploadDataSet",
               "Name of public dataset should start with public/");
         return kError;
      }
   }
   if (opt & kOverwriteAllFiles && opt & kOverwriteNoFiles
       || opt & kNoOverwriteDataSet && opt & kAppend
       || opt & kOverwriteDataSet && opt & kAppend
       || opt & kNoOverwriteDataSet && opt & kOverwriteDataSet
       || opt & kAskUser && opt & (kOverwriteDataSet |
                                   kNoOverwriteDataSet |
                                   kAppend |
                                   kOverwriteAllFiles |
                                   kOverwriteNoFiles)) {
      Error("UploadDataSet", "you specified contradicting options.");
      return kError;
   }

   // Decode options
   Int_t overwriteAll = (opt & kOverwriteAllFiles) ? kTRUE : kFALSE;
   Int_t overwriteNone = (opt & kOverwriteNoFiles) ? kTRUE : kFALSE;
   Int_t goodName = (opt & (kOverwriteDataSet | kAppend)) ? 1 : -1;
   Int_t appendToDataSet = (opt & kAppend) ? kTRUE : kFALSE;
   Int_t overwriteNoDataSet = (opt & kNoOverwriteDataSet) ? kTRUE : kFALSE;


   //If skippedFiles is not provided we can not return list of skipped files.
   if ((!skippedFiles || !&skippedFiles) && overwriteNone) {
      Error("UploadDataSet",
            "Provide pointer to TList object as skippedFiles argument when using kOverwriteNoFiles option.");
      return kError;
   }
   //If skippedFiles is provided but did not point to a TList the have to STOP
   if (skippedFiles && &skippedFiles)

      if (skippedFiles->Class() != TList::Class()) {
         Error("UploadDataSet",
               "Provided skippedFiles argument does not point to a TList object.");
         return kError;
      }

   TSocket *master;
   if (fActiveSlaves->GetSize())
      master = ((TSlave*)(fActiveSlaves->First()))->GetSocket();
   else {
      Error("UploadDataSet", "No connection to the master!");
      return kError;
   }

   Int_t fileCount = 0; // return value
   TMessage *retMess;
   if (goodName == -1) { // -1 for undefined
      // First check whether this dataset already exists unless
      // kAppend or kOverWriteDataSet
      TMessage nameMess(kPROOF_DATASETS);
      nameMess << Int_t(kCheckDataSetName);
      nameMess << TString(dataSetName);
      Broadcast(nameMess);
      master->Recv(retMess);
      Collect(); //after each call to HandleDataSets
      if (retMess->What() == kMESS_NOTOK) {
         //We ask user to agree on overwriting the dataset name
         while (goodName == -1 && !overwriteNoDataSet) {
            Printf("Dataset %s already exist. ",
                   dataSetName);
            Printf("Do you want to overwrite it[Yes/No/Append]?");
            TString answer;
            answer.ReadToken(cin);
            if (!strncasecmp(answer.Data(), "y", 1)) {
               goodName = 1;
            } else if (!strncasecmp(answer.Data(), "n", 1)) {
               goodName = 0;
            } else if (!strncasecmp(answer.Data(), "a", 1)) {
               goodName = 1;
               appendToDataSet = kTRUE;
            }
         }
      }
      else if (retMess->What() == kMESS_OK)
         goodName = 1;
      else
         Error("UploadDataSet", "unrecongnized message type: %d!",
            retMess->What());
      delete retMess;
   } // if (goodName == -1)
   if (goodName == 1) {  //must be == 1 as -1 was used for a bad name!
      //Code for enforcing writing in user "home dir" only
      char *relativeDestDir = Form("%s/%s/",
                                   gSystem->GetUserInfo()->fUser.Data(),
                                   desiredDest?desiredDest:"");
                                   //Consider adding dataSetName to the path

      relativeDestDir = CollapseSlashesInPath(relativeDestDir);
      TString dest = Form("%s/%s", GetDataPoolUrl(), relativeDestDir);

      delete[] relativeDestDir;

      // Now we will actually copy files and create the TList object
      TList *fileList = new TList();
      TIter next(files);
      while (TFileInfo *fileInfo = ((TFileInfo*)next())) {
         TUrl *fileUrl = fileInfo->GetFirstUrl();
         if (gSystem->AccessPathName(fileUrl->GetUrl()) == kFALSE) {
            //matching dir entry
            //getting the file name from the path represented by fileUrl
            const char *ent = gSystem->BaseName(fileUrl->GetFile());

            Int_t goodFileName = 1;
            if (!overwriteAll &&
               gSystem->AccessPathName(Form("%s/%s", dest.Data(), ent), kFileExists)
                  == kFALSE) {  //Destination file exists
               goodFileName = -1;
               while (goodFileName == -1 && !overwriteAll && !overwriteNone) {
                  Printf("File %s already exists. ", Form("%s/%s", dest.Data(), ent));
                  Printf("Do you want to overwrite it [Yes/No/all/none]?");
                  TString answer;
                  answer.ReadToken(cin);
                  if (!strncasecmp(answer.Data(), "y", 1))
                     goodFileName = 1;
                  else if (!strncasecmp(answer.Data(), "all", 3))
                     overwriteAll = kTRUE;
                  else if (!strncasecmp(answer.Data(), "none", 4))
                     overwriteNone = kTRUE;
                  else if (!strncasecmp(answer.Data(), "n", 1))
                     goodFileName = 0;
               }
            } //if file exists

            // Copy the file to the redirector indicated
            if (goodFileName == 1 || overwriteAll) {
               //must be == 1 as -1 was meant for bad name!
               Printf("Uploading %s to %s/%s",
                      fileUrl->GetUrl(), dest.Data(), ent);
               if (TFile::Cp(fileUrl->GetUrl(), Form("%s/%s", dest.Data(), ent))) {
                  fileList->Add(new TFileInfo(Form("%s/%s", dest.Data(), ent)));
               } else
                  Error("UploadDataSet", "file %s was not copied", fileUrl->GetUrl());
            } else {  // don't overwrite, but file exist and must be included
               fileList->Add(new TFileInfo(Form("%s/%s", dest.Data(), ent)));
               if (skippedFiles && &skippedFiles) {
                  // user specified the TList *skippedFiles argument so we create
                  // the list of skipped files
                  skippedFiles->Add(new TFileInfo(fileUrl->GetUrl()));
               }
            }
         } //if matching dir entry
      } //while

      if ((fileCount = fileList->GetSize()) == 0) {
         Printf("No files were copied. The dataset will not be saved");
      } else {
         if (CreateDataSet(dataSetName, fileList,
                     appendToDataSet?kAppend:kOverwriteDataSet) <= 0) {
            Error("UploadDataSet", "Error while saving dataset!");
            fileCount = kError;
         }
      }
      fileList->SetOwner();
      delete fileList;
   } else if (overwriteNoDataSet) {
      Printf("Dataset %s already exists", dataSetName);
      return kDataSetExists;
   } //if(goodName == 1)

   return fileCount;
}

//______________________________________________________________________________
Int_t TProof::UploadDataSet(const char *dataSetName,
                            const char *files,
                            const char *desiredDest,
                            Int_t opt,
                            TList *skippedFiles)
{
   // Upload a set of files and save the list of files by name dataSetName.
   // The mask 'opt' is a combination of EUploadOpt:
   //   kAppend             (0x1)   if set true files will be appended to
   //                               the dataset existing by given name
   //   kOverwriteDataSet   (0x2)   if dataset with given name exited it
   //                               would be overwritten
   //   kNoOverwriteDataSet (0x4)   do not overwirte if the dataset exists
   //   kOverwriteAllFiles  (0x8)   overwrite all files that may exist
   //   kOverwriteNoFiles   (0x10)  overwrite none
   //   kAskUser            (0x0)   ask user before overwriteng dataset/files
   // The default value is kAskUser.
   // The user will be asked to confirm overwriting dataset or files unless
   // specified opt provides the answer!
   // If kOverwriteNoFiles is set, then a pointer to TList must be passed as
   // skippedFiles argument. The function will add to this list TFileInfo
   // objects describing all files that existed on the cluster and were
   // not uploaded.
   //

   TList *fileList = new TList();
   void *dataSetDir = gSystem->OpenDirectory(gSystem->DirName(files));
   const char* ent;
   TString filesExp(gSystem->BaseName(files));
   filesExp.ReplaceAll("*",".*");
   TRegexp rg(filesExp);
   while ((ent = gSystem->GetDirEntry(dataSetDir))) {
      TString entryString(ent);
      if (entryString.Index(rg) != kNPOS) {
         //matching dir entry

         // Creating the intermediate TUrl with kTRUE flag to make sure
         // file:// is added for a local file
         TUrl *url = new TUrl(Form("%s/%s",
                                   gSystem->DirName(files), ent), kTRUE);
         if (gSystem->AccessPathName(url->GetUrl(), kReadPermission) == kFALSE)
            fileList->Add(new TFileInfo(url->GetUrl()));
         delete url;
      } //if matching dir entry
   } //while
   Int_t fileCount;
   if ((fileCount = fileList->GetSize()) == 0)
      Printf("No files match your selection. The dataset will not be saved");
   else
      fileCount = UploadDataSet(dataSetName, fileList, desiredDest,
                                opt, skippedFiles);
   fileList->SetOwner();
   delete fileList;
   return fileCount;
}

//______________________________________________________________________________
Int_t TProof::UploadDataSetFromFile(const char *dataset, const char *file,
                                    const char *dest, Int_t opt)
{
   // Upload files listed in "file" to PROOF cluster.
   // Where file = name of file containing list of files and
   // dataset = dataset name and opt is a combination of EUploadOpt bits.
   // Each file description (line) can include wildcards.

   //TODO: This method should use UploadDataSet(char *dataset, TList *l, ...)
   Int_t fileCount = 0;
   ifstream f;
   f.open(gSystem->ExpandPathName(file), ifstream::out);
   if (f.is_open()) {
      while (f.good()) {
         TString line;
         line.ReadToDelim(f);
         if (fileCount == 0) {
            // when uploading the first file user may have to decide
            fileCount += UploadDataSet(dataset, line.Data(), dest, opt);
         } else // later - just append
            fileCount += UploadDataSet(dataset, line.Data(), dest,
                                       opt | kAppend);
      }
      f.close();
   } else {
      Error("UploadDataSetFromFile", "unable to open the specified file");
      return -1;
   }
   return fileCount;
}

//______________________________________________________________________________
Int_t TProof::CreateDataSet(const char *dataSetName,
                              TList *files,
                              Int_t opt)
{
   // Create a dataSet from files existing on the cluster (listed in files)
   // and save it as dataSetName.
   // No files are uploaded nor verified to exist on the cluster
   // The 'files' argument is a list of TFileInfo objects describing the files
   // as first url.
   // The mask 'opt' is a combination of EUploadOpt:
   //   kAppend             (0x1)   if set true files will be appended to
   //                               the dataset existing by given name
   //   kOverwriteDataSet   (0x2)   if dataset with given name exited it
   //                               would be overwritten
   //   kNoOverwriteDataSet (0x4)   do not overwirte if the dataset exists
   //   kAskUser            (0x0)   ask user before overwriteng dataset/files
   // The default value is kAskUser.
   // The user will be asked to confirm overwriting dataset or files unless
   // specified opt provides the answer!
   //
   // Communication Summary
   //   Client                              Master
   //     |------------>DataSetName----------->|
   //     |<-------kMESS_OK/kMESS_NOTOK<-------| (Name OK/file exist)
   //  (*)|------->TList of TFileInfo -------->| (dataset to save)
   //  (*)|<-------kMESS_OK/kMESS_NOTOK<-------| (transaction complete?)
   //  (*) - optional


   // check if  dataSetName is not excluded
   if (strchr(dataSetName, '/')) {
      if (strstr(dataSetName, "public") != dataSetName) {
         Error("CreateDataSet",
               "Name of public dataset should start with public/");
         return kError;
      }
   }
   if (opt & kOverwriteDataSet && opt & kAppend
       || opt & kNoOverwriteDataSet && opt & kAppend
       || opt & kNoOverwriteDataSet && opt & kOverwriteDataSet
       || opt & kAskUser && opt & (kOverwriteDataSet |
                                   kNoOverwriteDataSet |
                                   kAppend)) {
      Error("CreateDataSet", "you specified contradicting options.");
      return kError;
   }

   if (opt & kOverwriteAllFiles || opt & kOverwriteNoFiles) {
      Error("CreateDataSet", "you specified unsupported options.");
      return kError;
   }

   // Decode options
   Int_t goodName = (opt & (kOverwriteDataSet | kAppend)) ? 1 : -1;
   Int_t appendToDataSet = (opt & kAppend) ? kTRUE : kFALSE;
   Int_t overwriteNoDataSet = (opt & kNoOverwriteDataSet) ? kTRUE : kFALSE;

   TSocket *master;
   if (fActiveSlaves->GetSize())
      master = ((TSlave*)(fActiveSlaves->First()))->GetSocket();
   else {
      Error("CreateDataSet", "No connection to the master!");
      return kError;
   }

   Int_t fileCount = 0; // return value
   //TODO Below if statement is a copy from UploadDataSet
   TMessage *retMess;
   if (goodName == -1) { // -1 for undefined
      // First check whether this dataset already exist unless
      // kAppend or kOverWriteDataSet
      TMessage nameMess(kPROOF_DATASETS);
      nameMess << Int_t(kCheckDataSetName);
      nameMess << TString(dataSetName);
      Broadcast(nameMess);
      master->Recv(retMess);
      Collect(); //after each call to HandleDataSets
      if (retMess->What() == kMESS_NOTOK) {
         //We ask user to agree on overwriting the dataset name
         while (goodName == -1 && !overwriteNoDataSet) {
            Printf("Dataset %s already exists. ",
                   dataSetName);
            Printf("Do you want to overwrite it[Yes/No/Append]?");
            TString answer;
            answer.ReadToken(cin);
            if (!strncasecmp(answer.Data(), "y", 1)) {
               goodName = 1;
            } else if (!strncasecmp(answer.Data(), "n", 1)) {
               goodName = 0;
            } else if (!strncasecmp(answer.Data(), "a", 1)) {
               goodName = 1;
               appendToDataSet = kTRUE;
            }
         }
      }
      else if (retMess->What() == kMESS_OK)
         goodName = 1;
      else
         Error("CreateDataSet", "unrecongnized message type: %d!",
            retMess->What());
      delete retMess;
   } // if (goodName == -1)
   if (goodName == 1) {
      if ((fileCount = files->GetSize()) == 0) {
         Printf("No files specified!");
      } else {
         TMessage mess(kPROOF_DATASETS);
         if (appendToDataSet)
            mess << Int_t(kAppendDataSet);
         else
            mess << Int_t(kCreateDataSet);
         mess << TString(dataSetName);
         mess.WriteObject(files);
         Broadcast(mess);
         //Reusing the retMess.
         if (master->Recv(retMess) <= 0) {
            Error("CreateDataSet", "No response form the master");
            fileCount = -1;
         } else {
            if (retMess->What() == kMESS_NOTOK) {
               Printf("Dataset was not saved.");
               fileCount = -1;
            } else if (retMess->What() != kMESS_OK)
               Error("CreateDataSet",
                     "Unexpected message type: %d", retMess->What());
            delete retMess;
         }
      }
   } else if (overwriteNoDataSet) {
      Printf("Dataset %s already exists", dataSetName);
      Collect();
      return kDataSetExists;
   } //if(goodName == 1)

   Collect();
   return fileCount;
}

//______________________________________________________________________________
TList *TProof::GetDataSets(const char *dir)
{
   // Get TList of TObjStrings with all datasets available on master:
   // * with dir undifined - just ls contents of ~/proof/datasets,
   // * with dir == "public" - ls ~/proof/datasets/public
   // * with dir == "~username/public" - ls ~/username/datasets/public

   TSocket *master;
   if (fActiveSlaves->GetSize())
      master = ((TSlave*)(fActiveSlaves->First()))->GetSocket();
   else {
      Error("GetDataSets", "No connection to the master!");
      return 0;
   }

   if (dir) {
      // check if dir is correct; this check is not exhaustive
      if (strstr(dir, "public") != dir && strchr(dir, '~') != dir) {
         // dir does not start with "public" nor with '~'
         Error("GetDataSets",
               "directory should be of form '[~userName/]public'");
         return 0;
      }
   }

   TMessage mess(kPROOF_DATASETS);
   mess << Int_t(kGetDataSets);
   mess << TString(dir?dir:"");
   Broadcast(mess);
   TMessage *retMess;
   master->Recv(retMess);
   TList *dataSetList = 0;
   if (retMess->What() == kMESS_OBJECT) {
      dataSetList = (TList*)(retMess->ReadObject(TList::Class()));
      if (!dataSetList)
         Error("GetDataSets", "Error receiving list of datasets");
   } else
      Printf("The dataset directory could not be open");
   Collect();
   delete retMess;
   return dataSetList;
}

//______________________________________________________________________________
void TProof::ShowDataSets(const char *dir)
{
   // Show all datasets uploaded to the cluster (just ls contents of
   // ~/proof/datasets or user/proof/datasets/public if 'dir' is defined).
   // * with dir undifined - just ls contents of ~/proof/datasets,
   // * with dir == "public" - ls ~/proof/datasets/public
   // * with dir == "~username/public" - ls ~/username/datasets/public

   TList *dataSetList;
   if ((dataSetList = GetDataSets(dir))) {
      if (dir)
         Printf("DataSets in %s :", dir);
      else
         Printf("Existing DataSets:");
      TIter next(dataSetList);
      while (TObjString *obj = (TObjString*)next())
         Printf("%s", obj->GetString().Data());
      dataSetList->SetOwner();
      delete dataSetList;
   } else
      Printf("Error getting a list of datasets");
}

//______________________________________________________________________________
TList *TProof::GetDataSet(const char *dataset)
{
   // Get a list of TFileInfo objects describing the files of the specified
   // dataset.

   TSocket *master;
   if (fActiveSlaves->GetSize())
      master = ((TSlave*)(fActiveSlaves->First()))->GetSocket();
   else {
      Error("GetDataSet", "No connection to the master!");
      return 0;
   }
   TMessage nameMess(kPROOF_DATASETS);
   nameMess << Int_t(kGetDataSet);
   nameMess << TString(dataset);
   if (Broadcast(nameMess) < 0)
      Error("GetDataSet", "Sending request failed");
   TMessage *retMess;
   master->Recv(retMess);
   TList *fileList = 0;
   if (retMess->What() == kMESS_OK) {
      if (!(fileList = (TList*)(retMess->ReadObject(TList::Class()))))
         Error("GetDataSet", "Error reading list of files");
   } else if (retMess->What() != kMESS_NOTOK)
      Error("GetDataSet", "Wrong message type %d", retMess->What());
   Collect();
   delete retMess;
   return fileList;
}

//______________________________________________________________________________
void TProof::ShowDataSet(const char *dataset)
{
   //Show content of specific dataset (cat ~/proof/datasets/dataset).

   TList *fileList;
   if ((fileList = GetDataSet(dataset))) {
      if (fileList->GetSize()) {
         //printing sorted list
         Printf("Files in %s:", dataset);
         TIter next(fileList);
         while (TFileInfo *obj = (TFileInfo*)next())
            Printf("%s", obj->GetFirstUrl()->GetUrl());
      } else
         Printf("There are no files in %s", dataset);
      delete fileList;
   }
   else
      Printf("No such dataset: %s", dataset);
}

//______________________________________________________________________________
Int_t TProof::RemoveDataSet(const char *dataSet)
{
   // Remove the specified dataset from the PROOF cluster.
   // Files are not deleted.

   // check if  dataSetName is not excluded
//   if (strchr(dataSet, '/')) {
//      Error("RemoveDataSet", "Dataset name shall not include '/'");
//      return kError;
//   }

   TSocket *master;
   if (fActiveSlaves->GetSize())
      master = ((TSlave*)(fActiveSlaves->First()))->GetSocket();
   else {
      Error("RemoveDataSet", "No connection to the master!");
      return kError;
   }
   TMessage nameMess(kPROOF_DATASETS);
   nameMess << Int_t(kRemoveDataSet);
   nameMess << TString(dataSet);
   if (Broadcast(nameMess) < 0)
      Error("RemoveDataSet", "Sending request failed");
   TMessage *mess;
   TString errorMess;
   master->Recv(mess);
   Collect();
   if (mess->What() != kMESS_OK) {
      if (mess->What() != kMESS_NOTOK)
         Error("RemoveDataSet", "unrecongnized message type: %d!",
               mess->What());
      delete mess;
      return -1;
   } else {
      delete mess;
      return 0;
   }
}

//______________________________________________________________________________
Int_t TProof::VerifyDataSet(const char *dataSet)
{
   // Verify if all files in the specified dataset are available.
   // Print a list and return the number of missing files.

   Int_t nMissingFiles = 0;
   TSocket *master;
   if (fActiveSlaves->GetSize())
      master = ((TSlave*)(fActiveSlaves->First()))->GetSocket();
   else {
      Error("VerifyDataSet", "No connection to the master!");
      return kError;
   }
   TMessage nameMess(kPROOF_DATASETS);
   nameMess << Int_t(kVerifyDataSet);
   nameMess << TString(dataSet);
   if (Broadcast(nameMess) < 0)
      Error("VerifyDataSet", "Sending request failed");
   TMessage *mess;
   master->Recv(mess);
   Collect();
   if (mess->What() == kMESS_OK) {
      TList *missingFiles;
      missingFiles = (TList*)(mess->ReadObject(TList::Class()));
      nMissingFiles = missingFiles->GetSize();
      if (nMissingFiles == 0)
         Printf("The files from %s dataset are all present on the cluster",
                dataSet);
      else {
         Printf("The following files are missing from dataset %s ", dataSet);
         Printf("at the moment:");
         TIter next(missingFiles);
         TFileInfo* fileInfo;
         while ((fileInfo = (TFileInfo*)next())) {
            Printf("\t%s", fileInfo->GetFirstUrl()->GetUrl());
         }
      }
      missingFiles->SetOwner();
      delete missingFiles;
   } else if (mess->What() == kMESS_NOTOK) {
      Printf("ValidateDataSet: no such dataset %s", dataSet);
      delete mess;
      return  -1;
   } else
      Fatal("ValidateDataSet", "unknown message type %d", mess->What());
   delete mess;
   return nMissingFiles;
}

//_____________________________________________________________________________
void TProof::InterruptCurrentMonitor()
{
   // If in active in a monitor set ready state
   if (fCurrentMonitor)
      fCurrentMonitor->Interrupt();
}

//_____________________________________________________________________________
void TProof::ActivateWorker(const char *ord)
{
   // Make sure that the worker identified by the ordinal number 'ord' is
   // in the active list. The request will be forwarded to the master
   // in direct contact with the worker. If needed, this master will move
   // the worker from the inactive to the active list and rebuild the list
   // of unique workers.
   // Use ord = "*" to activate all inactive workers.

   ModifyWorkerLists(ord, kTRUE);
}

//_____________________________________________________________________________
void TProof::DeactivateWorker(const char *ord)
{
   // Remove the worker identified by the ordinal number 'ord' from the
   // the active list. The request will be forwarded to the master
   // in direct contact with the worker. If needed, this master will move
   // the worker from the active to the inactive list and rebuild the list
   // of unique workers.
   // Use ord = "*" to deactivate all active workers.

   ModifyWorkerLists(ord, kFALSE);
}

//_____________________________________________________________________________
void TProof::ModifyWorkerLists(const char *ord, Bool_t add)
{
   // Modify the worker active/inactive list by making the worker identified by
   // the ordinal number 'ord' active (add == TRUE) or inactive (add == FALSE).
   // If needed, the request will be forwarded to the master in direct contact
   // with the worker. The end-master will move the worker from one list to the
   // other active and rebuild the list of unique active workers.
   // Use ord = "*" to deactivate all active workers.

   // Make sure the input make sense
   if (!ord || strlen(ord) <= 0) {
      Info("ModifyWorkerLists",
           "An ordinal number - e.g. \"0.4\" or \"*\" for all - is required as input");
      return;
   }

   Bool_t fw = kTRUE;    // Whether to forward one step down
   Bool_t rs = kFALSE;   // Whether to rescan for unique workers

   // Appropriate list pointing
   TList *in = (add) ? fInactiveSlaves : fActiveSlaves;
   TList *out = (add) ? fActiveSlaves : fInactiveSlaves;

   if (IsMaster()) {
      fw = IsEndMaster() ? kFALSE : kTRUE;
      // Look for the worker in the inactive list
      if (in->GetSize() > 0) {
         TIter nxw(in);
         TSlave *wrk = 0;
         while ((wrk = (TSlave *) nxw())) {
            if (ord[0] == '*' || !strncmp(wrk->GetOrdinal(), ord, strlen(ord))) {
               // Add it to the inactive list
               if (!out->FindObject(wrk)) {
                  out->Add(wrk);
                  if (add)
                     fActiveMonitor->Add(wrk->GetSocket());
               }
               // Remove it from the active list
               in->Remove(wrk);
               if (!add) {
                  fActiveMonitor->Remove(wrk->GetSocket());
                  wrk->SetStatus(TSlave::kInactive);
               } else
                  wrk->SetStatus(TSlave::kActive);

               // Nothing to forward (ord is unique)
               fw = kFALSE;
               // Rescan for unique workers (active list modified)
               rs = kTRUE;
               // We are done, if not option 'all'
               if (ord[0] != '*')
                  break;
            }
         }
      }
   }

   // Rescan for unique workers
   if (rs)
      FindUniqueSlaves();

   // Forward the request one step down, if needed
   Int_t action = (add) ? (Int_t) kActivateWorker : (Int_t) kDeactivateWorker;
   if (fw) {
      TMessage mess(kPROOF_WORKERLISTS);
      mess << action << TString(ord);
      Broadcast(mess);
      Collect();
   }
}

//_____________________________________________________________________________
TProof *TProof::Open(const char *cluster, const char *conffile,
                                   const char *confdir, Int_t loglevel)
{
   // Start a PROOF session on a specific cluster. If cluster is 0 (the
   // default) then the PROOF Session Viewer GUI pops up and 0 is returned.
   // If cluster is "" (empty string) then we connect to a PROOF session
   // on the localhost ("proof://localhost"). Via conffile a specific
   // PROOF config file in the confir directory can be specified.
   // Use loglevel to set the default loging level for debugging.
   // The appropriate instance of TProofMgr is created, if not
   // yet existing. The instantiated TProof object is returned.
   // Use TProof::cd() to switch between PROOF sessions.
   // For more info on PROOF see the TProof ctor.

   const char *pn = "TProof::Open";

   // Make sure libProof and dependents are loaded and TProof can be created,
   // dependents are loaded via the information in the [system].rootmap file
   if (!cluster) {

      TPluginManager *pm = gROOT->GetPluginManager();
      if (!pm) {
         ::Error(pn, "plugin manager not found");
         return 0;
      }

      if (gROOT->IsBatch()) {
         ::Error(pn, "we are in batch mode, cannot show PROOF Session Viewer");
         return 0;
      }
      // start PROOF Session Viewer
      TPluginHandler *sv = pm->FindHandler("TSessionViewer", "");
      if (!sv) {
         ::Error(pn, "no plugin found for TSessionViewer");
         return 0;
      }
      if (sv->LoadPlugin() == -1) {
         ::Error(pn, "plugin for TSessionViewer could not be loaded");
         return 0;
      }
      sv->ExecPlugin(0);
      return 0;

   } else {

      // Parse input URL
      TUrl u(cluster);

      // Find out if we are required to attach to a specific session
      TString o(u.GetOptions());
      Int_t locid = -1;
      Bool_t create = kFALSE;
      if (o.Length() > 0) {
         if (o.BeginsWith("N",TString::kIgnoreCase)) {
            create = kTRUE;
         } else if (o.IsDigit()) {
            locid = o.Atoi();
         }
         u.SetOptions("");
      }

      // Attach-to or create the appropriate manager
      TProofMgr *mgr = TProofMgr::Create(u.GetUrl());

      TProof *proof = 0;
      if (mgr && mgr->IsValid()) {

         // If XProofd we always attempt an attach first (unless
         // explicitely not requested).
         Bool_t attach = (create || mgr->IsProofd()) ? kFALSE : kTRUE;
         if (attach) {
            TProofDesc *d = 0;
            if (locid < 0)
               // Get the list of sessions
               d = (TProofDesc *) mgr->QuerySessions("")->First();
            else
               d = (TProofDesc *) mgr->GetProofDesc(locid);
            if (d) {
               proof = (TProof*) mgr->AttachSession(d->GetLocalId());
               if (!proof || !proof->IsValid()) {
                  if (locid)
                     ::Error(pn, "new session could not be attached");
                  SafeDelete(proof);
               }
            }
         }

         // start the PROOF session
         if (!proof) {
            proof = (TProof*) mgr->CreateSession(conffile, confdir, loglevel);
            if (!proof || !proof->IsValid()) {
               ::Error(pn, "new session could not be created");
               SafeDelete(proof);
            }
         }
      }
      return proof;
   }
}

//_____________________________________________________________________________
TProofMgr *TProof::Mgr(const char *url)
{
   // Get instance of the effective manager for 'url'
   // Return 0 on failure.

   if (!url)
      return (TProofMgr *)0;

   // Attach or create the relevant instance
   return TProofMgr::Create(url);
}

//_____________________________________________________________________________
void TProof::Reset(const char *url)
{
   // Wrapper around TProofMgr::Reset().

   if (url) {
      TProofMgr *mgr = TProof::Mgr(url);
      if (mgr && mgr->IsValid())
         mgr->Reset();
      else
         ::Error("TProof::Reset",
                 "unable to initialize a valid manager instance");
   }
}

//_____________________________________________________________________________
const TList *TProof::GetEnvVars()
{
   // Get environemnt variables.

   return fgProofEnvList;
}

//_____________________________________________________________________________
void TProof::AddEnvVar(const char *name, const char *value)
{
   // Add an variable to the list of environment variables passed to proofserv
   // on the master and slaves

   if (gDebug > 0) ::Info("TProof::AddEnvVar","%s=%s", name, value);

   if (fgProofEnvList == 0) {
      // initialize the list if needed
      fgProofEnvList = new TList;
      fgProofEnvList->SetOwner();
   } else {
      // replace old entries with the same name
      TObject *o = fgProofEnvList->FindObject(name);
      if (o != 0) {
         fgProofEnvList->Remove(o);
      }
   }
   fgProofEnvList->Add(new TNamed(name, value));
}

//_____________________________________________________________________________
void TProof::DelEnvVar(const char *name)
{
   // Remove an variable from the list of environment variables passed to proofserv
   // on the master and slaves

   if (fgProofEnvList == 0) return;

   TObject *o = fgProofEnvList->FindObject(name);
   if (o != 0) {
      fgProofEnvList->Remove(o);
   }
}

//_____________________________________________________________________________
void TProof::ResetEnvVars()
{
   // Clear the list of environment variables passed to proofserv
   // on the master and slaves

   if (fgProofEnvList == 0) return;

   SafeDelete(fgProofEnvList);
}

//______________________________________________________________________________
void TProof::SaveWorkerInfo()
{
   // Save informations about the worker set in the file .workers in the working
   // dir. Called each time there is a change in the worker setup, e.g. by
   // TProof::MarkBad().

   // We must be masters
   if (!IsMaster())
      return;

   // We must have a server defined
   if (!gProofServ) {
      Error("SaveWorkerInfo","gProofServ undefined");
      return;
   }

   // Update info
   const_cast<TProof*>(this)->AskStatistics();

   // The relevant lists must be defined
   if (!fSlaves && !fBadSlaves) {
      Warning("SaveWorkerInfo","all relevant worker lists is undefined");
      return;
   }

   // Create or truncate the file first
   TString fnwrk = Form("%s/.workers",
                        gSystem->DirName(gProofServ->GetSessionDir()));
   FILE *fwrk = fopen(fnwrk.Data(),"w");
   if (!fwrk) {
      Error("SaveWorkerInfo",
            "cannot open %s for writing (errno: %d)", fnwrk.Data(), errno);
      return;
   }

   // Loop over the list of workers (active is any worker not flagged as bad)
   TIter nxa(fSlaves);
   TSlave *wrk = 0;
   while ((wrk = (TSlave *) nxa())) {
      Int_t status = (fBadSlaves && fBadSlaves->FindObject(wrk)) ? 0 : 1;
      // Write out record for this worker
      fprintf(fwrk,"%s@%s:%d %d %s %s.log\n",
                   wrk->GetUser(), wrk->GetName(), wrk->GetPort(), status,
                   wrk->GetOrdinal(), wrk->GetWorkDir());
   }

   // Close file
   fclose(fwrk);

   // We are done
   return;
}

//______________________________________________________________________________
Int_t TProof::GetParameter(TCollection *c, const char *par, TString &value)
{
   // Get the value from the specified parameter from the specified collection.
   // Returns -1 in case of error (i.e. list is 0, parameter does not exist
   // or value type does not match), 0 otherwise.

   TObject *obj = c->FindObject(par);
   if (obj) {
      TNamed *par = dynamic_cast<TNamed*>(obj);
      if (par) {
         value = par->GetTitle();
         return 0;
      }
   }
   return -1;

}

//______________________________________________________________________________
Int_t TProof::GetParameter(TCollection *c, const char *par, Long_t &value)
{
   // Get the value from the specified parameter from the specified collection.
   // Returns -1 in case of error (i.e. list is 0, parameter does not exist
   // or value type does not match), 0 otherwise.

   TObject *obj = c->FindObject(par);
   if (obj) {
      TParameter<Long_t> *par = dynamic_cast<TParameter<Long_t>*>(obj);
      if (par) {
         value = par->GetVal();
         return 0;
      }
   }
   return -1;
}

//______________________________________________________________________________
Int_t TProof::GetParameter(TCollection *c, const char *par, Long64_t &value)
{
   // Get the value from the specified parameter from the specified collection.
   // Returns -1 in case of error (i.e. list is 0, parameter does not exist
   // or value type does not match), 0 otherwise.

   TObject *obj = c->FindObject(par);
   if (obj) {
      TParameter<Long64_t> *par = dynamic_cast<TParameter<Long64_t>*>(obj);
      if (par) {
         value = par->GetVal();
         return 0;
      }
   }
   return -1;
}

//______________________________________________________________________________
Int_t TProof::GetParameter(TCollection *c, const char *par, Double_t &value)
{
   // Get the value from the specified parameter from the specified collection.
   // Returns -1 in case of error (i.e. list is 0, parameter does not exist
   // or value type does not match), 0 otherwise.

   TObject *obj = c->FindObject(par);
   if (obj) {
      TParameter<Double_t> *par = dynamic_cast<TParameter<Double_t>*>(obj);
      if (par) {
         value = par->GetVal();
         return 0;
      }
   }
   return -1;
}
