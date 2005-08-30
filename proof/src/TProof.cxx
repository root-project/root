// @(#)root/proof:$Name:  $:$Id: TProof.cxx,v 1.100 2005/08/30 10:25:29 rdm Exp $
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

#include <vector>

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

#include "TProof.h"
#include "TSortedList.h"
#include "TSlave.h"
#include "TMonitor.h"
#include "TMessage.h"
#include "TSystem.h"
#include "TError.h"
#include "TUrl.h"
#include "TFTP.h"
#include "TROOT.h"
#include "TH1.h"
#include "TProofPlayer.h"
#include "TProofQuery.h"
#include "TDSet.h"
#include "TEnv.h"
#include "TPluginManager.h"
#include "TCondor.h"
#include "Riostream.h"
#include "TTree.h"
#include "TDrawFeedback.h"
#include "TEventList.h"
#include "TMonitor.h"
#include "TBrowser.h"
#include "TChain.h"
#include "TProofServ.h"
#include "TMap.h"
#include "TTimer.h"
#include "TThread.h"
#include "TSemaphore.h"
#include "TMutex.h"
#include "TObjString.h"
#include "Getline.h"


TVirtualMutex *gProofMutex = 0;

// Helper classes used for parallel startup
//______________________________________________________________________________
TProofThreadArg::TProofThreadArg(const char *h, Int_t po, const char *o,
                                 Int_t pe, const char *i, const char *w,
                                 TList *s, TProof *prf)
  : host(h ? StrDup(h) : 0), port(po), ord(o ? StrDup(o) : 0),
    perf(pe), image(i ? StrDup(i) : 0), workdir(w ? StrDup(w) : 0),
    msd(0), slaves(s), proof(prf), cslave(0), claims(0)
{
}

//______________________________________________________________________________
TProofThreadArg::TProofThreadArg(TCondorSlave *csl, TList *clist,
                                 TList *s, TProof *prf)
  : host(0), port(-1), ord(0), perf(-1), image(0), workdir(0),
    msd(0), slaves(s), proof(prf), cslave(csl), claims(clist)
{
   if(csl){
      host = StrDup(csl->fHostname.Data());
      image = StrDup(csl->fImage.Data());
      ord = StrDup(csl->fOrdinal.Data());
      workdir = StrDup(csl->fWorkDir.Data());
      port = csl->fPort;
      perf = csl->fPerfIdx;
   }
}

//______________________________________________________________________________
TProofThreadArg::TProofThreadArg(const char *h, Int_t po, const char *o,
                                 const char *i, const char *w, const char *m,
                                TList *s, TProof *prf)
  : host(h ? StrDup(h) : 0), port(po), ord(o ? StrDup(o) : 0),
    perf(-1), image(i ? StrDup(i) : 0), workdir(w ? StrDup(w) : 0),
    msd(m ? StrDup(m) : 0), slaves(s), proof(prf), cslave(0), claims(0)
{
}

//______________________________________________________________________________
TProofThreadArg::~TProofThreadArg()
{
   if (host)    delete [] (char*)host;
   if (ord)     delete [] (char*)ord;
   if (image)   delete [] (char*)image;
   if (msd)     delete [] (char*)msd;
   if (workdir) delete [] (char*)workdir;
}

//----- PROOF Interrupt signal handler -----------------------------------------
//______________________________________________________________________________
Bool_t TProofInterruptHandler::Notify()
{
   // TProof interrupt handler.

   fProof->StopProcess(kTRUE);
   return kTRUE;
}

//----- Input handler for messages from TProofServ -----------------------------
//______________________________________________________________________________
Bool_t TProofInputHandler::Notify()
{
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

   if (!opt) opt = "";
   if (!strcmp(opt, "active") && fStatus != kActive)
      return;
   if (!strcmp(opt, "notactive") && fStatus != kNotActive)
      return;
   if (!strcmp(opt, "bad") && fStatus != kBad)
      return;

   cout << "Slave: "          << fOrdinal
        << "  hostname: "     << fHostName
        << "  msd: "          << fMsd
        << "  perf index: "   << fPerfIndex
        << "  "               << stat
        << endl;
}


//------------------------------------------------------------------------------

ClassImp(TProof)

TSemaphore    *TProof::fgSemaphore = 0;

//______________________________________________________________________________
TProof::TProof(const char *masterurl, const char *conffile,
               const char *confdir, Int_t loglevel)
{
   // Create a PROOF environment. Starting PROOF involves either connecting
   // to a master server, which in turn will start a set of slave servers, or
   // directly starting as master server (if master = ""). Masterurl is of
   // the form: proof://host[:port] or proofs://host[:port]. Conffile is
   // the name of the config file describing the remote PROOF cluster
   // (this argument alows you to describe different cluster configurations).
   // The default is proof.conf. Confdir is the directory where the config
   // file and other PROOF related files are (like motd and noproof files).
   // Loglevel is the log level (default = 1).

   if (!conffile || strlen(conffile) == 0)
      conffile = kPROOF_ConfFile;
   if (!confdir  || strlen(confdir) == 0)
      confdir = kPROOF_ConfDir;

   // Can have only one PROOF session open at a time (for the time being).
   if (gProof) {
      Warning("TProof", "closing currently open PROOF session");
      gProof->Close();
   }

   Init(masterurl, conffile, confdir, loglevel);

   gProof = this;
}

//______________________________________________________________________________
TProof::TProof()
{
   // Protected constructor to be used by classes deriving from TProof
   // (they have to call Init themselves and override StartSlaves
   // appropriately).
   //
   // This constructor simply closes any previous gProof and sets gProof
   // to this instance.

   // Can have only one PROOF session open at a time (for the time being).
   if (gProof) {
      Warning("TProof", "closing currently open PROOF session");
      gProof->Close();
   }

   gProof = this;
}

//______________________________________________________________________________
TProof::~TProof()
{
   // Clean up PROOF environment.

   while (TChain *chain = dynamic_cast<TChain*> (fChains->First()) ) {
      // remove "chain" from list
      chain->SetProof(0);
   }

   Close();
   SafeDelete(fIntHandler);
   SafeDelete(fSlaves);
   SafeDelete(fActiveSlaves);
   SafeDelete(fUniqueSlaves);
   SafeDelete(fNonUniqueMasters);
   SafeDelete(fBadSlaves);
   SafeDelete(fAllMonitor);
   SafeDelete(fActiveMonitor);
   SafeDelete(fUniqueMonitor);
   SafeDelete(fSlaveInfo);
   SafeDelete(fChains);
   SafeDelete(fPlayer);
   SafeDelete(fFeedback);

   // remove file with redirected logs
   fclose(fLogFileR);
   fclose(fLogFileW);
   gSystem->Unlink(fLogFileName);

   {
      R__LOCKGUARD2(gROOTMutex);
      gROOT->GetListOfSockets()->Remove(this);
   }

   if (gProof == this) {
      gProof = 0;
   }
}

//______________________________________________________________________________
Int_t TProof::Init(const char *masterurl, const char *conffile,
                   const char *confdir, Int_t loglevel)
{
   // Start the PROOF environment. Starting PROOF involves either connecting
   // to a master server, which in turn will start a set of slave servers, or
   // directly starting as master server (if master = ""). For a description
   // of the arguments see the TProof ctor. Returns the number of started
   // master or slave servers, returns 0 in case of error, in which case
   // fValid remains false.

   Assert(gSystem);

   fValid = kFALSE;

   TUrl *u;
   if (!masterurl || !*masterurl)
      u = new TUrl("proof://__master__");
   else if (strstr(masterurl, "://"))
      u = new TUrl(masterurl);
   else
      u = new TUrl(Form("proof://%s", masterurl));

   fUser           = u->GetUser();
   fMaster         = u->GetHost();
   fPort           = u->GetPort();
   fConfDir        = confdir;
   fConfFile       = conffile;
   fWorkDir        = gSystem->WorkingDirectory();
   fLogLevel       = loglevel;
   fProtocol       = kPROOF_Protocol;
   fMasterServ     = fMaster == "__master__" ? kTRUE : kFALSE;
   fSendGroupView  = kTRUE;
   fImage          = fMasterServ ? "" : "<local>";
   fIntHandler     = 0;
   fStatus         = 0;
   fSlaveInfo      = 0;
   fChains         = new TList;
   fUrlProtocol    = u->GetProtocol();
   delete u;

   fProgressDialog        = 0;
   fProgressDialogStarted = kFALSE;

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

   fPlayer   = MakePlayer();
   fFeedback = new TList;
   fFeedback->SetOwner();
   fFeedback->SetName("FeedbackList");
   AddInput(fFeedback);

   // sort slaves by descending performance index
   fSlaves           = new TSortedList(kSortDescending);
   fActiveSlaves     = new TList;
   fUniqueSlaves     = new TList;
   fNonUniqueMasters = new TList;
   fBadSlaves        = new TList;
   fAllMonitor       = new TMonitor;
   fActiveMonitor    = new TMonitor;
   fUniqueMonitor    = new TMonitor;

   // Master may want parallel startup
   Bool_t parallelStartup = kFALSE;
   if (IsMaster()) {
      parallelStartup = gEnv->GetValue("Proof.ParallelStartup", kFALSE);
      PDB(kGlobal,1) Info("Init", "Parallel Startup: %s",
                          parallelStartup ? "kTRUE" : "kFALSE");
      if (parallelStartup) {
         // Load thread lib, if not done already
#ifdef ROOTLIBDIR
         TString threadLib = TString(ROOTLIBDIR) + "/libThread";
#else
         TString threadLib = TString(gRootDir) + "/lib/libThread";
#endif
         char *p;
         if ((p = gSystem->DynamicPathName(threadLib, kTRUE))) {
            delete[]p;
            if (gSystem->Load(threadLib) == -1) {
               Warning("Init",
                       "Cannot load libThread: switch to serial startup");
               parallelStartup = kFALSE;
            }
         } else {
            Warning("Init",
                    "Cannot find libThread: switch to serial startup");
            parallelStartup = kFALSE;
         }
      }

      // Get no of parallel requests and set semaphore correspondingly
      Int_t parallelRequests = gEnv->GetValue("Proof.ParallelStartupRequests", 0);
      if (parallelRequests > 0) {
         PDB(kGlobal,1)
            Info("Init", "Parallel Startup Requests: %d", parallelRequests);
         fgSemaphore = new TSemaphore((UInt_t)(parallelRequests));
      }
   }

   // Start slaves
   if (!StartSlaves(parallelStartup)) return 0;

   if (fgSemaphore) {
     SafeDelete(fgSemaphore);
   }

   // we are now properly initialized
   fValid = kTRUE;

   // De-activate monitor (will be activated in Collect)
   fAllMonitor->DeActivateAll();

   // By default go into parallel mode
   GoParallel(9999);

   // Send relevant initial state to slaves
   SendInitialState();

   SetActive(kFALSE);

   if (IsValid()) {
      R__LOCKGUARD2(gROOTMutex);
      gROOT->GetListOfSockets()->Add(this);
   }
   return fActiveSlaves->GetSize();
}

//______________________________________________________________________________
Bool_t TProof::StartSlaves(Bool_t parallel)
{
   // Start up PROOF slaves.

   // If this is a master server, find the config file and start slave
   // servers as specified in the config file
   if (IsMaster()) {

      TString fconf;
      fconf.Form("%s/.%s", gSystem->Getenv("HOME"), fConfFile.Data());
      PDB(kGlobal,2)
         Info("StartSlaves", "checking PROOF config file %s", fconf.Data());
      if (gSystem->AccessPathName(fconf, kReadPermission)) {
         fconf.Form("%s/proof/etc/%s", fConfDir.Data(), fConfFile.Data());
         PDB(kGlobal,2)
            Info("StartSlaves", "checking PROOF config file %s", fconf.Data());
         if (gSystem->AccessPathName(fconf, kReadPermission)) {
            Error("StartSlaves", "no PROOF config file found");
            return kFALSE;
         }
      }

      PDB(kGlobal,1)
         Info("StartSlaves", "using PROOF config file: %s", fconf.Data());

      UInt_t nSlaves = 0;

      FILE *pconf;
      if ((pconf = fopen(fconf, "r"))) {

         fConfFile = fconf;

         // read the config file
         char line[1024];
         TString host = gSystem->GetHostByName(gSystem->HostName()).GetHostName();

         int  ord = 0;
         // check for valid master line
         while (fgets(line, sizeof(line), pconf)) {
            char word[12][128];
            if (line[0] == '#') continue;   // skip comment lines
            int nword = sscanf(line, "%s %s %s %s %s %s %s %s %s %s %s %s",
                word[0], word[1],
                word[2], word[3], word[4], word[5], word[6],
                word[7], word[8], word[9], word[10], word[11]);

            // see if master may run on this node, accept both old "node"
            // and new "master" lines
            if (nword >= 2 &&
                (!strcmp(word[0], "node") || !strcmp(word[0], "master")) &&
                !fImage.Length()) {
               TInetAddress a = gSystem->GetHostByName(word[1]);
               if (!host.CompareTo(a.GetHostName()) ||
                   !strcmp(word[1], "localhost")) {
                  const char *image = word[1];
                  const char *workdir = kPROOF_WorkDir;
                  for (int i = 2; i < nword; i++) {

                     if (!strncmp(word[i], "image=", 6))
                        image = word[i]+6;
                     if (!strncmp(word[i], "workdir=", 8))
                        workdir = word[i]+8;

                  }
                  const char* expworkdir = gSystem->ExpandPathName(workdir);
                  if (!strcmp(expworkdir,gProofServ->GetWorkDir())) {
                     fImage = image;
                  }
                  delete [] (char *) expworkdir;
               }
            } else if (nword >= 2 &&
               (!strcmp(word[0], "slave") || !strcmp(word[0], "worker"))) {
               nSlaves++;
            }
         }

         if (fImage.Length() == 0) {
            fclose(pconf);
            Error("StartSlaves",
                  "no appropriate master line found in %s", fconf.Data());
            return kFALSE;
         }

         // Init arrays for threads, if neeeded
         std::vector<TProofThread *> thrHandlers;
         if (parallel) {
            thrHandlers.reserve(nSlaves);
            if (thrHandlers.max_size() < nSlaves) {
               PDB(kGlobal,1)
                  Info("StartSlaves","cannot reserve enough space thread"
                       " handlers - switch to serial startup");
               parallel = kFALSE;
            }
         }

         // check for valid slave lines and start slaves
         rewind(pconf);
         while (fgets(line, sizeof(line), pconf)) {
            char word[12][128];
            if (line[0] == '#') continue;   // skip comment lines
            int nword = sscanf(line, "%s %s %s %s %s %s %s %s %s %s %s %s",
                word[0], word[1],
                word[2], word[3], word[4], word[5], word[6],
                word[7], word[8], word[9], word[10], word[11]);

            // find all slave servers, accept both "slave" and "worker" lines
            if (nword >= 2 &&
                (!strcmp(word[0], "slave") || !strcmp(word[0], "worker"))) {
               int perfidx  = 100;
               int sport    = fPort;

               const char *image = word[1];
               const char *workdir = 0;
               for (int i = 2; i < nword; i++) {

                  if (!strncmp(word[i], "perf=", 5))
                     perfidx = atoi(word[i]+5);
                  if (!strncmp(word[i], "image=", 6))
                     image = word[i]+6;
                  if (!strncmp(word[i], "port=", 5))
                     sport = atoi(word[i]+5);
                  if (!strncmp(word[i], "workdir=", 8))
                     workdir = word[i]+8;

               }

               // Get slave FQDN ...
               TString SlaveFqdn;
               TInetAddress SlaveAddr =
                  gSystem->GetHostByName((const char *)word[1]);
               if (SlaveAddr.IsValid()) {
                  SlaveFqdn = SlaveAddr.GetHostName();
                  if (SlaveFqdn == "UnNamedHost")
                  SlaveFqdn = SlaveAddr.GetHostAddress();
               }

               // create slave server
               TString fullord = TString(gProofServ->GetOrdinal()) +
                                 "." + ((Long_t) ord);
               if (parallel) {

                  // Prepare arguments
                  TProofThreadArg *ta =
                      new TProofThreadArg((const char *)(word[1]), sport,
                                         fullord, perfidx, image, workdir,
                                         fSlaves, this);
                  if (ta) {
                     // The type of the thread func makes it a detached thread
                     TThread *th = new TThread(SlaveStartupThread,ta);
                     if (!th) {
                        Info("StartSlaves","Can't create startup thread:"
                                      " out of system resources");
                        SafeDelete(ta);
                     } else {
                        // Save in vector
                        thrHandlers.push_back(new TProofThread(th,ta));
                        // Run the thread
                        th->Run();
                     }
                  } else {
                     Info("StartSlaves","Can't create thread arguments object:"
                                      " out of system resources");
                  }

               } else {
                  // create slave server
                  TSlave *slave =
                      CreateSlave(word[1], sport, fullord, perfidx,
                                  image, workdir);

                  fSlaves->Add(slave);
                  if (slave->IsValid()) {
                     fAllMonitor->Add(slave->GetSocket());
                  } else {
                     fBadSlaves->Add(slave);
                  }
                  PDB(kGlobal,3)
                     Info("StartSlaves", "slave on host %s created"
                                         " and added to list", word[1]);

               }
               ord++;
            }
         }

         if (parallel) {

            // Wait completion of startup operations
            std::vector<TProofThread *>::iterator i;
            for (i = thrHandlers.begin(); i != thrHandlers.end(); ++i) {
               TProofThread *pt = *i;

               // Wait on this condition
               if (pt && pt->thread->GetState() == TThread::kRunningState) {
                  PDB(kGlobal,3)
                     Info("Init",
                          "parallel startup: waiting for slave %s (%s:%d)",
                           pt->args->ord, pt->args->host, pt->args->port);
                  pt->thread->Join();
               }
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

         }
      }
      fclose(pconf);

   } else {

      // create master server
      TSlave *slave = CreateSubmaster(fMaster, fPort, "0", "master", fConfFile, 0);

      if (slave->IsValid()) {
         // check protocol compatability
         // protocol 1 is not supported anymore
         if (fProtocol == 1) {
            Error("StartSlaves",
                  "client and remote protocols not compatible (%d and %d)",
                  kPROOF_Protocol, fProtocol);
            delete slave;
            return kFALSE;
         }

         fSlaves->Add(slave);
         fAllMonitor->Add(slave->GetSocket());
         Collect(slave);
         if (slave->GetStatus() == -99) {
            Error("StartSlaves", "not allowed to connect to PROOF master server");
            return 0;
         }

         if (!slave->IsValid()) {
            delete slave;
            Error("StartSlaves",
                  "failed to setup connection with PROOF master server");
            return kFALSE;
         }

         fIntHandler = new TProofInterruptHandler(this);
         fIntHandler->Add();

         if (!gROOT->IsBatch()) {
            if ((fProgressDialog =
                 gROOT->GetPluginManager()->FindHandler("TProofProgressDialog")))
               if (fProgressDialog->LoadPlugin() == -1)
                  fProgressDialog = 0;
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
void TProof::Close(Option_t *)
{
   // Close all open slave servers.

   if (fSlaves) {
      if (fIntHandler) fIntHandler->Remove();

      // If local client ...
      if (!IsMaster()) {
         // ... tell master and slaves to stop
         Interrupt(kShutdownInterrupt, kAll);
      }

      fActiveSlaves->Clear("nodelete");
      fUniqueSlaves->Clear("nodelete");
      fNonUniqueMasters->Clear("nodelete");
      fBadSlaves->Clear("nodelete");
      fSlaves->Delete();
   }
}

//______________________________________________________________________________
TSlave *TProof::CreateSlave(const char *host, Int_t port, const char *ord,
                            Int_t perf, const char *image, const char *workdir)
{
   // Create a new TSlave of type TSlave::kSlave.
   // Note: creation of TSlave is private with TProof as a friend.
   // Derived classes must use this function to create slaves.

   TSlave* sl = TSlave::Create(host, port, ord, perf, image,
                               this, TSlave::kSlave, workdir, 0, 0);

   if (sl->IsValid()) {
      sl->SetInputHandler(new TProofInputHandler(this, sl->GetSocket()));
      // must set fParallel to 1 for slaves since they do not
      // report their fParallel with a LOG_DONE message
      sl->fParallel = 1;
   }

   return sl;
}

//______________________________________________________________________________
TSlave *TProof::CreateSubmaster(const char *host, Int_t port, const char *ord,
                                const char *image, const char *conffile,
                                const char *msd)
{
   // Create a new TSlave of type TSlave::kMaster.
   // Note: creation of TSlave is private with TProof as a friend.
   // Derived classes must use this function to create slaves.

   TSlave *sl = TSlave::Create(host, port, ord, 100, image, this,
                               TSlave::kMaster, 0, conffile, msd);

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
   fNonUniqueMasters->Clear();

   TIter next(fActiveSlaves);

   while (TSlave *sl = dynamic_cast<TSlave*>(next())) {
      if (fImage == sl->fImage) {
         if (sl->GetSlaveType() == TSlave::kMaster)
            fNonUniqueMasters->Add(sl);
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
               } else {
                  Error("FindUniqueSlaves", "TSlave is neither Master nor Slave");
                  Assert(0);
               }
            }
            break;
         }
      }

      if (add) {
         fUniqueSlaves->Add(sl);
         fUniqueMonitor->Add(sl->GetSocket());
         if (replace_slave) {
            fUniqueSlaves->Remove(replace_slave);
            fUniqueMonitor->Remove(replace_slave->GetSocket());
         }
      }
   }

   // will be actiavted in Collect()
   fUniqueMonitor->DeActivateAll();
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
void TProof::GetListOfQueries()
{
   // Ask the master for the list of queries.

   if (!IsValid() || IsMaster()) return;

   Broadcast(kPROOF_QUERYLIST, kActive);
   Collect(kActive);
}

//______________________________________________________________________________
void TProof::ShowQueries(Option_t *opt)
{
   // Ask the master for the list of queries.

   if (!IsValid()) return;

   GetListOfQueries();

   if (!fQueries) return;

   Printf("+++");
   Printf("+++ Processed queries: %d", fQueries->GetSize());
   TIter nxq(fQueries);
   TProofQuery *pq = 0;
   while ((pq = (TProofQuery *)nxq()))
      pq->Print(opt);

   // Advise
   if (strncasecmp(opt,"F",1)) {
      Printf("+++");
      Printf("+++ NB: use ShowQueries(\"F\") to get the full listing");
   }

   Printf("+++");
}

//______________________________________________________________________________
Bool_t TProof::IsDataReady(Long64_t &totalbytes, Long64_t &bytesready)
{
   // See if the data is ready to be analyzed.

   if (!IsValid()) return kFALSE;

   TList submasters;
   TIter NextSlave(GetListOfActiveSlaves());
   while (TSlave *sl = dynamic_cast<TSlave*>(NextSlave())) {
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
   // Send interrupt OOB byte to master or slave servers.

   if (!IsValid()) return;

   TList *slaves = 0;
   if (list == kAll)    slaves = fSlaves;
   if (list == kActive) slaves = fActiveSlaves;
   if (list == kUnique) slaves = fUniqueSlaves;

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
   TIter NextSlave(GetListOfActiveSlaves());
   Int_t nparallel = 0;
   while (TSlave* sl = dynamic_cast<TSlave*>(NextSlave()))
      if (sl->GetParallel() >= 0)
         nparallel += sl->GetParallel();

   return nparallel;
}

//______________________________________________________________________________
TList *TProof::GetSlaveInfo()
{
   // Returns number of slaves active in parallel mode. Returns 0 in case
   // there are no active slaves. Returns -1 in case of error.

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

   while((slave = (TSlave *) next()) != 0) {
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
         Assert(0);
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
   if (list == kAll)    slaves = fSlaves;
   if (list == kActive) slaves = fActiveSlaves;
   if (list == kUnique) slaves = fUniqueSlaves;

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
   if (list == kAll)    slaves = fSlaves;
   if (list == kActive) slaves = fActiveSlaves;
   if (list == kUnique) slaves = fUniqueSlaves;

   return BroadcastRaw(buffer, length, slaves);
}

//______________________________________________________________________________
Int_t TProof::Collect(const TSlave *sl)
{
   // Collect responses from slave sl. Returns the number of slaves that
   // responded (=1).

   if (!sl->IsValid()) return 0;

   TMonitor *mon = fAllMonitor;

   mon->DeActivateAll();
   mon->Activate(sl->GetSocket());

   return Collect(mon);
}

//______________________________________________________________________________
Int_t TProof::Collect(TList *slaves)
{
   // Collect responses from the slave servers. Returns the number of slaves
   // that responded.

   TMonitor *mon = fAllMonitor;
   mon->DeActivateAll();

   TIter next(slaves);
   TSlave *sl;
   while ((sl = (TSlave*) next())) {
      if (sl->IsValid())
         mon->Activate(sl->GetSocket());
   }

   return Collect(mon);
}

//______________________________________________________________________________
Int_t TProof::Collect(ESlaves list)
{
   // Collect responses from the slave servers. Returns the number of slaves
   // that responded.

   TMonitor *mon = 0;
   if (list == kAll)    mon = fAllMonitor;
   if (list == kActive) mon = fActiveMonitor;
   if (list == kUnique) mon = fUniqueMonitor;

   mon->ActivateAll();

   return Collect(mon);
}

//______________________________________________________________________________
Int_t TProof::Collect(TMonitor *mon)
{
   // Collect responses from the slave servers. Returns the number of messages
   // received. Can be 0 if there are no active slaves.

   fStatus = 0;
   if (!mon->GetActive()) return 0;

   DeActivateAsyncInput();

   // We want messages on the main window during synchronous collection,
   // but we save the present status to restore it at the end
   Bool_t saveRedirLog = fRedirLog;
   if (!IsIdle() && !IsSync())
      fRedirLog = kFALSE;

   int cnt = 0, loop = 1, rc = 0;

   fBytesRead = 0;
   fRealTime  = 0.0;
   fCpuTime   = 0.0;

   while (loop) {

      // Wait for a ready socket
      TSocket  *s = mon->Select();

      // Get and analyse the info it did receive
      if ((rc = CollectInputFrom(s)) == 1)
         // Deactivate it if we are done with it
         mon->DeActivate(s);

      // Check if we are done
      if (!mon->GetActive()) loop = 0;

      // Update counter (if no error occured)
      if (rc >= 0) cnt++;
   }

   // make sure group view is up to date
   SendGroupView();

   // Restore redirection setting
   fRedirLog = saveRedirLog;

   ActivateAsyncInput();

   return cnt;
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

   switch (what) {

      case kMESS_OBJECT:
         obj = mess->ReadObject(mess->GetClass());
         if (obj->InheritsFrom(TH1::Class())) {
            TH1 *h = (TH1*)obj;
            h->SetDirectory(0);
            TH1 *horg = (TH1*)gDirectory->GetList()->FindObject(h->GetName());
            if (horg)
               horg->Add(h);
            else
               h->SetDirectory(gDirectory);
         }
         break;

      case kPROOF_FATAL:
         MarkBad(s);
         break;

      case kPROOF_GETOBJECT:
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

            TMessage answ(kPROOF_GETPACKET);
            answ << elem;
            s->Send(answ);
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
         PDB(kGlobal,2) Info("Collect:kPROOF_LOGDONE","status %d  parallel %d",
            sl->fStatus, sl->fParallel);
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

      case kPROOF_OUTPUTLIST:
         {
            PDB(kGlobal,2) Info("Collect:kPROOF_OUTPUTLIST","Enter");
            TList *out = (TList *) mess->ReadObject(TList::Class());
            if (out) {
               out->SetOwner();
               fPlayer->StoreOutput(out); // Adopts the list
            } else {
               PDB(kGlobal,2) Info("Collect:kPROOF_OUTPUTLIST","ouputlist is empty");
            }
            // On clients at this point processing is over
            if (!IsMaster()) {

               // Set idle ...
               fIdle = kTRUE;

               // Handle abort ...
               if (fPlayer->GetExitStatus() == TProofPlayer::kAborted) {
                  if (!fSync) RedirectLog(kTRUE);
                  Info("CollectInputFrom",
                       "the processing was aborted - %lld events processed",
                       fPlayer->GetEventsProcessed());
                  if (!fSync) RedirectLog(kFALSE);

                  gProof->Progress(-1, fPlayer->GetEventsProcessed());
                  Emit("StopProcess(Bool_t)", kTRUE);
               }

               // Handle stop ...
               if (fPlayer->GetExitStatus() == TProofPlayer::kStopped) {
                  if (!fSync) RedirectLog(kTRUE);
                  Info("CollectInputFrom",
                       "the processing was stopped - %lld events processed",
                       fPlayer->GetEventsProcessed());
                  if (!fSync) RedirectLog(kFALSE);

                  gProof->Progress(-1, fPlayer->GetEventsProcessed());
                  Emit("StopProcess(Bool_t)", kFALSE);
               }
            }
         }
         break;

      case kPROOF_QUERYLIST:
         {
            PDB(kGlobal,2) Info("Collect:kPROOF_QUERYLIST","Enter");
            if (fQueries) {
               fQueries->Delete();
               delete fQueries;
               fQueries = 0;
            }
            fQueries = (TList *) mess->ReadObject(TList::Class());
         }
         break;

      case kPROOF_FEEDBACK:
         {
            PDB(kGlobal,2) Info("Collect:kPROOF_FEEDBACK","Enter");
            TList *out = (TList *) mess->ReadObject(TList::Class());
            out->SetOwner();
            sl = FindSlave(s);
            fPlayer->StoreFeedback(sl, out); // Adopts the list
         }
         break;

      case kPROOF_AUTOBIN:
         {
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
            PDB(kGlobal,2) Info("Collect:kPROOF_PROGRESS","Enter");

            sl = FindSlave(s);

            Long64_t total, processed;

            (*mess) >> total >> processed;

            fPlayer->Progress(sl, total, processed);
         }
         break;

      case kPROOF_STOPPROCESS:
         {
            // answer contains number of processed events;
            Long64_t events;
            (*mess) >> events;
            fPlayer->AddEventsProcessed(events);
            break;
         }

      case kPROOF_GETSLAVEINFO:
         {
            PDB(kGlobal,2) Info("Collect:kPROOF_GETSLAVEINFO","Enter");

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
            PDB(kGlobal,2) Info("Collect:kPROOF_VALIDATE_DSET","Enter");
            TDSet* dset = 0;
            (*mess) >> dset;
            if (!fDSet)
               Error("Collect:kPROOF_VALIDATE_DSET", "fDSet not set");
            else
               fDSet->Validate(dset);
            delete dset;
         }
         break;

      case kPROOF_DATA_READY:
         {
            PDB(kGlobal,2) Info("Collect:kPROOF_DATA_READY","Enter");
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

      default:
         Error("Collect", "unknown command received from slave (%d)", what);
         break;
   }

   // Cleanup
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
         Error("HandleAsyncInput", "unknown command %d", what);
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
   if (list == kAll)    slaves = fSlaves;
   if (list == kActive) slaves = fActiveSlaves;
   if (list == kUnique) slaves = fUniqueSlaves;

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
      TSlave *sl = (TSlave *)fActiveSlaves->First();
      if (sl) {
         TString sc;
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

      Printf("Master host name:         %s", gSystem->HostName());
      Printf("Port number:              %d", GetPort());
      Printf("User:                     %s", GetUser());
      Printf("Protocol version:         %d", GetClientProtocol());
      Printf("Image name:               %s", GetImage());
      Printf("Working directory:        %s", gSystem->WorkingDirectory());
      Printf("Config directory:         %s", GetConfDir());
      Printf("Config file:              %s", GetConfFile());
      Printf("Log level:                %d", GetLogLevel());
      Printf("Number of slaves:         %d", GetNumberOfSlaves());
      Printf("Number of active slaves:  %d", GetNumberOfActiveSlaves());
      Printf("Number of unique slaves:  %d", GetNumberOfUniqueSlaves());
      Printf("Number of bad slaves:     %d", GetNumberOfBadSlaves());
      Printf("Total MB's processed:     %.2f", float(GetBytesRead())/(1024*1024));
      Printf("Total real time used (s): %.3f", GetRealTime());
      Printf("Total CPU time used (s):  %.3f", GetCpuTime());
      if (TString(option).Contains("a") && GetNumberOfSlaves()) {
         Printf("List of slaves:");
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
               Error("Print", "TSlave is neither Master nor Slave");
               Assert(0);
            }
         }
         const_cast<TProof*>(this)->Collect(&masters);
      }
   }
}

//______________________________________________________________________________
Int_t TProof::Process(TDSet *dset, const char *selector, Option_t *option,
                      Long64_t nentries, Long64_t first, TEventList *evl)
{
   // Process a data set (TDSet) using the specified selector (.C) file.
   // Returns -1 in case of error, 0 otherwise.

   if (!IsValid()) return -1;

   if (!IsIdle()) {
      Info("Process","not idle, cannot submit query");
      return -1;
   }

   // Set non idle
   fIdle = kFALSE;

   // Start or reset the progress dialog
   if (fProgressDialog) {
      Int_t dsz = dset->GetListOfElements()->GetSize();
      if (!fProgressDialogStarted) {
         fProgressDialog->ExecPlugin(5, this, selector, dsz, first, nentries);
         fProgressDialogStarted = kTRUE;
      } else {
         ResetProgressDialog(selector, dsz, first, nentries);
      }
   }

   // Resolve query mode
   fSync = (GetQueryMode(option) == kSync);

   // deactivate the default application interrupt handler
   // ctrl-c's will be forwarded to PROOF to stop the processing
   TSignalHandler *sh = 0;
   if (fSync) {
      if (gApplication)
         sh = gSystem->RemoveSignalHandler(gApplication->GetSignalHandler());
   }

   Long64_t rv = fPlayer->Process(dset, selector, option, nentries, first, evl);

   if (fSync) {
      // reactivate the default application interrupt handler
      if (sh)
         gSystem->AddSignalHandler(sh);
   }

   return rv;
}

//______________________________________________________________________________
Int_t TProof::Finalize()
{
   // Finalize current query.
   // Return 0 on success, -1 on error

   return (fPlayer ? fPlayer->Finalize() : -1);
}

//______________________________________________________________________________
Int_t TProof::Retrieve(Int_t)
{
   MayNotUse("Retrieve");
   return -1;
}

//______________________________________________________________________________
Int_t TProof::Archive(Int_t, const char *)
{
   MayNotUse("Archive");
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

//_____________________________________________________________________________
TProof::EQueryMode TProof::GetQueryMode() const
{
   // Get query running mode.

   if (gDebug > 0)
      Info("GetQueryMode","query mode is set to: %s", fQueryMode == kSync ?
           "Sync" : "Async");

   return fQueryMode;
}

//______________________________________________________________________________
TProof::EQueryMode TProof::GetQueryMode(Option_t *mode) const
{
   // Find out the query mode based on the current setting and 'mode'.

   EQueryMode qmode = fQueryMode;

   if (mode) {
      if (strchr(mode, 'A'))
         qmode = kAsync;
      if (strchr(mode, 'S'))
         qmode = kSync;
   }
   return qmode;
}

//______________________________________________________________________________
Int_t TProof::DrawSelect(TDSet *dset, const char *varexp, const char *selection, Option_t *option,
                         Long64_t nentries, Long64_t first)
{
   // Process a data set (TDSet) using the specified selector (.C) file.
   // Returns -1 in case of error, 0 otherwise.

   if (!IsValid()) return -1;

   return fPlayer->DrawSelect(dset, varexp, selection, option, nentries, first);
}

//______________________________________________________________________________
void TProof::StopProcess(Bool_t abort)
{
   // Send STOPPROCESS message to master and workers.

   PDB(kGlobal,2)
      Info("StopProcess","enter %d", abort);

   if (!IsValid()) return;

   fPlayer->StopProcess(abort);

   if (fSlaves->GetSize() == 0) return;

   TSlave *sl;
   TIter   next(fSlaves);

   while ((sl = (TSlave *)next())) {
      if (sl->IsValid()) {
         TSocket *s = sl->GetSocket();
         TMessage msg(kPROOF_STOPPROCESS);
         msg << abort;
         s->Send(msg);
      }
   }
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
                  SysError("RecvLogFile", "error writing to stdout");
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
void TProof::LogMessage(const char *msg, Bool_t all)
{
   // Log a message into the appropriate window by emitting a signal.

   PDB(kGlobal,1)
      Info("LogMessage","Enter ... %s, 'all: %s", msg ? msg : "",
           all ? "true" : "false");

   if (!fProgressDialogStarted) {
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
Int_t TProof::Exec(const char *cmd)
{
   // Send command to be executed on the PROOF master and/or slaves.
   // Command can be any legal command line command. Commands like
   // ".x file.C" or ".L file.C" will cause the file file.C to be send
   // to the PROOF cluster. Returns -1 in case of error, >=0 in case of
   // succes.

   return Exec(cmd, kActive);
}

//______________________________________________________________________________
Int_t TProof::Exec(const char *cmd, ESlaves list)
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
               mess << TString(file) << md.fMD5;
               slave->GetSocket()->Send(mess);

               TMessage *reply;
               slave->GetSocket()->Recv(reply);
               if (reply->What() != kPROOF_CHECKFILE)
                  sendto = kTRUE;
               delete reply;
            }
         }
         delete md5;
      }
   } else {
      // file not in map
      TMD5 *md5 = TMD5::FileChecksum(file);
      MD5Mod_t md;
      md.fMD5      = *md5;
      md.fModtime  = modtime;
      fFileMap[sn] = md;
      delete md5;
      TMessage mess(kPROOF_CHECKFILE);
      mess << TString(file) << md.fMD5;
      slave->GetSocket()->Send(mess);

      TMessage *reply;
      slave->GetSocket()->Recv(reply);
      if (reply->What() != kPROOF_CHECKFILE)
         sendto = kTRUE;
      delete reply;
   }

   return sendto;
}

//______________________________________________________________________________
Int_t TProof::SendFile(const char *file, Int_t opt, const char *rfile)
{
   // Send a file to master or slave servers. Returns number of slaves
   // the file was sent to, maybe 0 in case master and slaves have the same
   // file system image, -1 in case of error.
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

   TList *slaves = fActiveSlaves;

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

      Bool_t sendto = force ? kTRUE : CheckFile(fnam, sl, modtime);
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

         if (sl->GetSocket()->SendRaw(buf, len) == -1) {
            SysError("SendFile", "error writing to slave %s:%s (now offline)",
                     sl->GetName(), sl->GetOrdinal());
            MarkBad(sl);
            break;
         }

      } while (len > 0);

      nsl++;
   }

   close(fd);

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
Int_t TProof::SetParallel(Int_t nodes)
{
   // Tell RPOOF how many slaves to use in parallel. Returns the number of
   // parallel slaves. Returns -1 in case of error.

   if (!IsValid()) return -1;

   if (IsMaster()) {
      GoParallel(nodes);
      return SendCurrentState();
   } else {
      PDB(kGlobal,1) Info("SetParallel", "request %d node%s", nodes,
         nodes == 1 ? "" : "s");
      TMessage mess(kPROOF_PARALLEL);
      mess << nodes;
      Broadcast(mess);
      Collect();
      Int_t parallel = GetParallel();
      PDB(kGlobal,1) Info("SetParallel", "got %d node%s", parallel,
         parallel == 1 ? "" : "s");
      if (parallel > 0) printf("PROOF set to parallel mode (%d slaves)\n", parallel);
      return parallel;
   }
}

//______________________________________________________________________________
Int_t TProof::GoParallel(Int_t nodes)
{
   // Go in parallel mode with at most "nodes" slaves. Since the fSlaves
   // list is sorted by slave performace the active list will contain first
   // the most performant nodes. Returns the number of active slaves.
   // Returns -1 in case of error.

   if (!IsValid()) return -1;

   if (nodes < 0) nodes = 0;

   fActiveSlaves->Clear();
   fActiveMonitor->RemoveAll();

   TIter next(fSlaves);
   //Simple algorithm for going parallel - fill up first nodes
   int cnt = 0;
   TSlave *sl;
   while (cnt < nodes && (sl = (TSlave *)next())) {
      if (sl->IsValid()) {
         if ( strcmp("IGNORE", sl->GetImage()) == 0 ) continue;
         Int_t slavenodes = 0;
         if (sl->GetSlaveType() == TSlave::kSlave) {
            fActiveSlaves->Add(sl);
            fActiveMonitor->Add(sl->GetSocket());
            slavenodes = 1;
         } else if (sl->GetSlaveType() == TSlave::kMaster) {
            TMessage mess(kPROOF_PARALLEL);
            mess << nodes-cnt;
            if (sl->GetSocket()->Send(mess) == -1) {
               MarkBad(sl);
               slavenodes = 0;
            } else {
               Collect(sl);
               fActiveSlaves->Add(sl);
               fActiveMonitor->Add(sl->GetSocket());
               if (sl->GetParallel()>0) {
                  slavenodes = sl->GetParallel();
               } else {
                  slavenodes = 0;
               }
            }
         } else {
            Error("GoParallel", "TSlave is neither Master nor Slave");
            Assert(0);
         }
         cnt += slavenodes;
      }
   }

   // Get slave status (will set the slaves fWorkDir correctly)
   AskStatistics();

   // Find active slaves with unique image
   FindUniqueSlaves();

   // Send new group-view to slaves
   SendGroupView();

   Int_t n = GetParallel();
   if (IsMaster()) {
      if (n < 1)
         printf("PROOF set to sequential mode\n");
   } else {
      printf("PROOF set to parallel mode (%d slaves)\n", n);
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

      // make list of unique slaves (which will include
      // unique slave on submasters)
      TList allunique;
      Int_t i;
      for (i = 0; i < fUniqueSlaves->GetSize(); i++) {
         TSlave* sl = dynamic_cast<TSlave*>(fUniqueSlaves->At(i));
         if (sl) allunique.Add(sl);
      }
      for (i = 0; i < fNonUniqueMasters->GetSize(); i++) {
         TSlave* sl = dynamic_cast<TSlave*>(fNonUniqueMasters->At(i));
         if (sl) allunique.Add(sl);
      }
      Collect(&allunique);
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

   // make list of unique slaves (which will include
   // unique slave on submasters
   TList allunique;
   Int_t i;
   for (i = 0; i<fUniqueSlaves->GetSize(); i++) {
      TSlave* sl =
         dynamic_cast<TSlave*>(fUniqueSlaves->At(i));
      if (sl) allunique.Add(sl);
   }
   for (i = 0; i<fNonUniqueMasters->GetSize(); i++) {
      TSlave* sl =
         dynamic_cast<TSlave*>(fNonUniqueMasters->At(i));
      if (sl) allunique.Add(sl);
   }
   Collect(&allunique);

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

   TMessage mess(kPROOF_CACHE);
   mess << Int_t(kShowPackages) << all;
   Broadcast(mess, kUnique);

   if (all) {
      TMessage mess2(kPROOF_CACHE);
      mess2 << Int_t(kShowSubPackages) << all;
      Broadcast(mess2, fNonUniqueMasters);

      // make list of unique slaves (which will include
      // unique slave on submasters
      TList allunique;
      Int_t i;
      for (i = 0; i < fUniqueSlaves->GetSize(); i++) {
         TSlave* sl = dynamic_cast<TSlave*>(fUniqueSlaves->At(i));
         if (sl) allunique.Add(sl);
      }
      for (i = 0; i < fNonUniqueMasters->GetSize(); i++) {
         TSlave* sl = dynamic_cast<TSlave*>(fNonUniqueMasters->At(i));
         if (sl) allunique.Add(sl);
      }
      Collect(&allunique);
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

   TMessage mess(kPROOF_CACHE);
   mess << Int_t(kShowEnabledPackages) << all;
   Broadcast(mess);
   Collect();
}

//______________________________________________________________________________
Int_t TProof::ClearPackages()
{
   // Remove all packages.

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
Int_t TProof::DisablePackage(const char *package)
{
   // Remove a specific package.

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

   TMessage mess(kPROOF_CACHE);
   mess << Int_t(kDisablePackage) << pac;
   Broadcast(mess, kUnique);

   TMessage mess2(kPROOF_CACHE);
   mess2 << Int_t(kDisableSubPackage) << pac;
   Broadcast(mess2, fNonUniqueMasters);

   // make list of unique slaves (which will include
   // unique slave on submasters)
   TList allunique;
   Int_t i;
   for (i = 0; i < fUniqueSlaves->GetSize(); i++) {
      TSlave* sl = dynamic_cast<TSlave*>(fUniqueSlaves->At(i));
      if (sl) allunique.Add(sl);
   }
   for (i = 0; i < fNonUniqueMasters->GetSize(); i++) {
      TSlave* sl = dynamic_cast<TSlave*>(fNonUniqueMasters->At(i));
      if (sl) allunique.Add(sl);
   }
   Collect(&allunique);

   return fStatus;
}

//______________________________________________________________________________
Int_t TProof::DisablePackages()
{
   // Remove all packages.

   if (!IsValid()) return -1;

   TMessage mess(kPROOF_CACHE);
   mess << Int_t(kDisablePackages);
   Broadcast(mess, kUnique);

   TMessage mess2(kPROOF_CACHE);
   mess2 << Int_t(kDisableSubPackages);
   Broadcast(mess2, fNonUniqueMasters);

   // make list of unique slaves (which will include
   // unique slave on submasters)
   TList allunique;
   Int_t i;
   for (i = 0; i < fUniqueSlaves->GetSize(); i++) {
      TSlave* sl = dynamic_cast<TSlave*>(fUniqueSlaves->At(i));
      if (sl) allunique.Add(sl);
   }
   for (i = 0; i < fNonUniqueMasters->GetSize(); i++) {
      TSlave* sl = dynamic_cast<TSlave*>(fNonUniqueMasters->At(i));
      if (sl) allunique.Add(sl);
   }
   Collect(&allunique);

   return fStatus;
}

//______________________________________________________________________________
Int_t TProof::BuildPackage(const char *package)
{
   // Build specified package. Executes the PROOF-INF/BUILD.sh
   // script if it exists on all unique nodes.
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

   TMessage mess(kPROOF_CACHE);
   mess << Int_t(kBuildPackage) << pac;
   Broadcast(mess, kUnique);

   TMessage mess2(kPROOF_CACHE);
   mess2 << Int_t(kBuildSubPackage) << pac;
   Broadcast(mess2, fNonUniqueMasters);

      // make list of unique slaves (which will include
   // unique slave on submasters)
   TList allunique;
   Int_t i;
   for (i = 0; i < fUniqueSlaves->GetSize(); i++) {
      TSlave* sl = dynamic_cast<TSlave*>(fUniqueSlaves->At(i));
      if (sl) allunique.Add(sl);
   }
   for (i = 0; i < fNonUniqueMasters->GetSize(); i++) {
      TSlave* sl = dynamic_cast<TSlave*>(fNonUniqueMasters->At(i));
      if (sl) allunique.Add(sl);
   }
   Collect(&allunique);

   return fStatus;
}

//______________________________________________________________________________
Int_t TProof::LoadPackage(const char *package)
{
   // Load specified package. Executes the PROOF-INF/SETUP.C script
   // on all active nodes.
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

   TMessage mess(kPROOF_CACHE);
   mess << Int_t(kLoadPackage) << pac;
   Broadcast(mess);
   Collect();

   return fStatus;
}

//______________________________________________________________________________
Int_t TProof::UnloadPackage(const char *package)
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

   TMessage mess(kPROOF_CACHE);
   mess << Int_t(kUnloadPackage) << pac;
   Broadcast(mess);
   Collect();

   return fStatus;
}

//______________________________________________________________________________
Int_t TProof::UnloadPackages()
{
   // Unload all packages.
   // Returns 0 in case of success and -1 in case of error.

   if (!IsValid()) return -1;

   TMessage mess(kPROOF_CACHE);
   mess << Int_t(kUnloadPackages);
   Broadcast(mess);
   Collect();

   return fStatus;
}

//______________________________________________________________________________
Int_t TProof::EnablePackage(const char *package)
{
   // Enable specified package. Executes the PROOF-INF/BUILD.sh
   // script if it exists followed by the PROOF-INF/SETUP.C script.
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

   if (BuildPackage(pac) == -1)
      return -1;

   if (LoadPackage(pac) == -1)
      return -1;

   return 0;
}

//______________________________________________________________________________
Int_t TProof::UploadPackage(const char *tpar)
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
   // Returns 0 in case of success and -1 in case of error.

   if (!IsValid()) return -1;

   TString par = tpar;
   if (!par.EndsWith(".par")) {
      Error("UploadPackage", "package %s must have extension .par", tpar);
      return -1;
   }

   gSystem->ExpandPathName(par);

   if (gSystem->AccessPathName(par, kReadPermission)) {
      Error("UploadPackage", "package %s does not exist", par.Data());
      return -1;
   }

   // Strategy: get md5 of package and check if it is different from the
   // one stored on the remote node. If it is different lock the remote
   // package directory and use TFTP to ftp the package to the remote node,
   // unlock the directory.

   TMD5 *md5 = TMD5::FileChecksum(par);
   TMessage mess(kPROOF_CHECKFILE);
   mess << TString("+")+TString(gSystem->BaseName(par)) << (*md5);
   TMessage mess2(kPROOF_CHECKFILE);
   mess2 << TString("-")+TString(gSystem->BaseName(par)) << (*md5);
   TMessage mess3(kPROOF_CHECKFILE);
   mess3 << TString("=")+TString(gSystem->BaseName(par)) << (*md5);
   delete md5;

   // loop over all unique nodes
   TIter next(fUniqueSlaves);
   TSlave *sl;
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
                         gSystem->BaseName(par))) < 0)
               Warning("UploadPackage", "problems uploading file %s", par.Data());
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
         delete reply;
         sl->GetSocket()->Recv(reply);
         if (reply->What() != kPROOF_CHECKFILE) {
            Error("UploadPackage", "unpacking of package %s failed", par.Data());
            delete reply;
            return -1;
         }
      }
      delete reply;
   }

   // loop over all other master nodes
   TIter nextmaster(fNonUniqueMasters);
   TSlave *ma;
   while ((ma = (TSlave *) nextmaster())) {
      if (!ma->IsValid())
         continue;

      ma->GetSocket()->Send(mess3);

      TMessage *reply;
      ma->GetSocket()->Recv(reply);
      if (reply->What() != kPROOF_CHECKFILE) {
         // error -> package should have been found
         Error("UploadPackage", "package %s did not exist on submaster %s",
               par.Data(), ma->GetOrdinal());
         delete reply;
         return -1;
      }
      delete reply;
   }

   return 0;
}

//______________________________________________________________________________
void TProof::Progress(Long64_t total, Long64_t processed)
{
   // Get query progress information. Connect a slot to this signal
   // to track progress.

   PDB(kGlobal,1)
      Info("Progress","%2f (%lld/%lld)", 100.*processed/total, processed, total);

   EmitVA("Progress(Long64_t,Long64_t)", 2, total, processed);
}

//______________________________________________________________________________
void TProof::Feedback(TList *objs)
{
   // Get list of feedback objects. Connect a slot to this signal
   // to monitor the feedback object.

   PDB(kGlobal,1) Info("Feedback","%d Objects", objs->GetSize());
   PDB(kFeedback,1) {
      Info("Feedback","%d objects", objs->GetSize());
      objs->ls();
   }

   Emit("Feedback(TList *objs)", (Long_t) objs);
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
   TIter NextSlave(GetListOfActiveSlaves());
   while (TSlave *sl = dynamic_cast<TSlave*>(NextSlave())) {
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
   TList NonLocal; // list of nonlocal elements
   // make two iterations - first add local elements - then distribute nonlocals
   for (Int_t i = 0; i < 2; i++) {
      Bool_t local = i>0?kFALSE:kTRUE;
      TIter NextElem(local?dset->GetListOfElements():&NonLocal);
      while (TDSetElement *elem = dynamic_cast<TDSetElement*>(NextElem())) {
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
               NonLocal.Add(elem);
            } else {
               Error("ValidateDSet", "No Node to allocate TDSetElement to");
               Assert(0);
            }
         }
      }
   }

   // send to slaves
   TList usedslaves;
   TIter NextNode(&nodes);
   SetDSet(dset); // set dset to be validated in Collect()
   while (TPair *node = dynamic_cast<TPair*>(NextNode())) {
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
            mesg << &dset;

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
   // Draw feedback creation proxy. When accessed via TVirtualProof avoids
   // link dependency on libProof.

   return new TDrawFeedback(this);
}

//______________________________________________________________________________
void TProof::SetDrawFeedbackOption(TDrawFeedback *f, Option_t *opt)
{
   // Set draw feedback option.

   if (f)
      f->SetOption(opt);
}

//______________________________________________________________________________
void TProof::DeleteDrawFeedback(TDrawFeedback *f)
{
   // Delete draw feedback object.

   if (f)
      delete f;
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
}

//______________________________________________________________________________
TProofPlayer *TProof::MakePlayer()
{
   // Construct a TProofPlayer object.

   SetPlayer(new TProofPlayerRemote(this));
   return GetPlayer();
}

//______________________________________________________________________________
void TProof::AddChain(TChain *chain)
{
   fChains->Add(chain);
}

//______________________________________________________________________________
void TProof::RemoveChain(TChain *chain)
{
   fChains->Remove(chain);
}

//_____________________________________________________________________________
void *TProof::SlaveStartupThread(void * arg)
{
   // Function executed in the slave startup thread

   if (fgSemaphore) fgSemaphore->Wait();

   TProofThreadArg *ta = (TProofThreadArg *)arg;

   PDB(kGlobal,1)
      ::Info("TProof::SlaveStartupThread",
             "Starting slave %s on host %s",ta->ord,ta->host);

   TSlave *sl = 0;

   if(ta->msd)
      sl = ta->proof->CreateSubmaster(ta->host, ta->port, ta->ord,
                                      ta->image, ta->workdir, ta->msd);
   else
      sl = ta->proof->CreateSlave(ta->host, ta->port, ta->ord,
                                  ta->perf, ta->image, ta->workdir);

   {
      R__LOCKGUARD2(gProofMutex);


      // Add to the started slaves list
      ta->slaves->Add(sl);

      if (ta->claims) { // Condor slave
         // Remove from the pending claims list
         TCondorSlave *c = ta->cslave;
         ta->claims->Remove(c);
      }

   }

   // Notify we are done
   PDB(kGlobal,1)
      ::Info("TProof::SlaveStartupThread",
             "slave %s on host %s created and added to list",
             ta->ord, ta->host);

   if (fgSemaphore) fgSemaphore->Post();

   return 0;
}

//______________________________________________________________________________
void TProof::RedirectLog(Bool_t on)
{
   // Redirect stderr and stdout messages to log file.

   static FILE *tmpout = 0, *stdout_svd = 0, *stderr_svd = 0;
   static Int_t tmperr_fno = -1, stdout_fno = -1, stderr_fno = -1;

   if (on) {
      //
      // redirect stdout
      stdout_fno = fileno(stdout);
      if (!(stdout_svd = fdopen(dup(fileno(stdout)),"w"))) {
         Error("RedirectLog", "stdout could not been duplicated, do not redirect");
      } else {
         if ((tmpout = freopen(fLogFileName.Data(), "a", stdout)) == 0)
            Error("RedirectLog", "could not freopen stdout");
      }
      //
      // redirect stderr
      stderr_fno = fileno(stderr);
      if (!(stderr_svd = fdopen(dup(fileno(stderr)),"w"))) {
         Error("RedirectLog", "stderr could not been duplicated, do not redirect");
      } else {
         if ((tmperr_fno = dup2(fileno(stdout), fileno(stderr))) < 0)
            Error("RedirectLog", "could not redirect stderr");
      }

   } else {
      //
      // Restore stdout
      if (stdout_svd) {
         fflush(stdout);
         stdout = stdout_svd;
         dup2(fileno(stdout), stdout_fno);
      }
      //
      // Restore stderr
      if (stderr_svd) {
         stderr = stderr_svd;
         dup2(fileno(stderr), stderr_fno);
      }
   }
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
      // Get update the list of queries
      GetListOfQueries();
      if (fQueries) {
         TProofQuery *pq = 0;
         if (qry > 0) {
            TIter nxq(fQueries);
            while ((pq = (TProofQuery *)nxq()))
               if (qry == pq->GetSeqNum())
                  break;
         } else if (qry == -2) {
            // Pickup the last one
            pq = (TProofQuery *)(fQueries->Last());
         } else
            qry = -1;

         if (pq) {
            startlog = pq->GetStartLog();
            endlog = pq->GetEndLog();
            GetLog(startlog, endlog);
            return;
         } else {
            Info("ShowLog","query %d not found in list", qry);
            qry = -1;
         }

      } else
         qry = -1;
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
   }
   // Avoid screwing up the prompt
   write(fileno(stdout), "\n", 1);

   // Restore original pointer
   if (qry > -1)
      lseek(fileno(fLogFileR), (off_t) nowlog, SEEK_SET);
}
