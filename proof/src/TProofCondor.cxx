// @(#)root/proof:$Name:  $:$Id: TProofCondor.cxx,v 1.7 2006/04/20 14:36:48 rdm Exp $
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

#include "TProofCondor.h"
#include "TString.h"
#include "TTimer.h"
#include "TList.h"
#include "TSlave.h"
#include "TCondor.h"
#include "TMap.h"
#include "TProofServ.h"
#include "TSocket.h"
#include "TMonitor.h"
#include "TProofResourcesStatic.h"
#include "TProofNodeInfo.h"

ClassImp(TProofCondor)

//______________________________________________________________________________
TProofCondor::TProofCondor(const char *masterurl, const char *conffile,
                           const char *confdir, Int_t loglevel)
  : fCondor(0), fTimer(0)
{
   // Start proof using condor

   fUrl = TUrl(masterurl);

   if (!conffile || strlen(conffile) == 0) {
      conffile = kPROOF_ConfFile;
   } else if (!strncasecmp(conffile, "condor:", 7)) {
      conffile+=7;
   }

   if (!confdir  || strlen(confdir) == 0) {
      confdir = kPROOF_ConfDir;
   }

   Init(masterurl, conffile, confdir, loglevel);
}

//______________________________________________________________________________
TProofCondor::~TProofCondor()
{
   // Clean up Condor PROOF environment.

   SafeDelete(fCondor);
   SafeDelete(fTimer);
}

//______________________________________________________________________________
Bool_t TProofCondor::StartSlaves(Bool_t parallel, Bool_t)
{
   // non static config

   fCondor = new TCondor;
   TString jobad = GetJobAd();

   fImage = fCondor->GetImage(gSystem->HostName());
   if (fImage.Length() == 0) {
      Error("StartSlaves", "Empty Condor image found for system %s",
            gSystem->HostName());
      return kFALSE;
   }

   TList claims;
   if (fConfFile.IsNull()) {
      // startup all slaves if no config file given
      TList *condorclaims = fCondor->Claim(9999, jobad);
      TIter nextclaim(condorclaims);
      while (TObject *o = nextclaim()) claims.Add(o);
   } else {
      // parse config file
      TProofResourcesStatic *resources = new TProofResourcesStatic(fConfDir, fConfFile);
      fConfFile = resources->GetFileName(); // Update the global file name (with path)
      PDB(kGlobal,1) Info("StartSlaves", "using PROOF config file: %s", fConfFile.Data());

      // Get all workers
      TList *workerList = resources->GetWorkers();
      if (workerList->GetSize() == 0) {
         Error("StartSlaves", "Found no condorworkers in %s", fConfFile.Data());
         return kFALSE;
      }

      // check for valid slave lines and claim condor nodes
      Int_t ord = 0;

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

         gSystem->Sleep(100);
         TCondorSlave* csl = fCondor->Claim(worker->GetNodeName().Data(), jobad);
         if (csl) {
            csl->fPerfIdx = perfidx;
            csl->fImage = image;
            csl->fWorkDir = gSystem->ExpandPathName(workdir);
            TString fullord = TString(gProofServ->GetOrdinal()) + "." + ((Long_t) ord);
            csl->fOrdinal = fullord.Data();
            claims.Add(csl);
            ord++;
         }
      } // end while (worker loop)

      // Cleanup
      delete resources;
      resources = 0;
   } // end else (parse config file)

   Long_t delay = 500; // timer delay 0.5s
   Int_t ntries = 20; // allow 20 tries (must be > 1 for algorithm to work)
   Int_t trial = 1;
   Int_t idx = 0;

   // Init stuff for parallel start-up, if required
   std::vector<TProofThread *> thrHandlers;
   TIter *nextsl = 0, *nextclaim = 0;
   TList *startedsl = 0;
   UInt_t nSlaves = 0;
   TTimer *ptimer = 0;
   if (parallel) {
      nSlaves = claims.GetSize();
      thrHandlers.reserve(nSlaves);
      if (thrHandlers.max_size() >= nSlaves) {
         startedsl = new TList();
         nextsl = new TIter(startedsl);
         nextclaim = new TIter(&claims);
      } else {
         PDB(kGlobal,1)
            Info("StartSlaves","cannot reserve enough space thread"
                 " handlers - switch to serial startup");
         parallel = kFALSE;
      }
   }

   while (claims.GetSize() > 0) {
      TCondorSlave* c = 0;

      if (parallel) {
         // Parallel startup in separate threads
         startedsl->RemoveAll();
         nextclaim->Reset();

         while ((c = (TCondorSlave *)(*nextclaim)())) {
            // Prepare arguments
            TProofThreadArg *ta = new TProofThreadArg(c, &claims, startedsl, this);
            if (ta) {
               // The type of the thread func makes it a detached thread
               TThread *th = new TThread(SlaveStartupThread,ta);
               if (!th) {
                  Info("StartSlaves","can't create startup thread:"
                       " out of system resources");
                  SafeDelete(ta);
               } else {
                  // Add to the vector
                  thrHandlers.push_back(new TProofThread(th, ta));
                  // Run the thread
                  th->Run();
               }
            } else {
               Info("StartSlaves","can't create thread arguments object:"
                    " out of system resources");
            }
         }

         // Start or reset timer
         if (trial == 1) {
            ptimer = new TTimer(delay);
         } else {
            ptimer->Reset();
         }

         // Wait completion of startup operations
         std::vector<TProofThread *>::iterator i;
         for (i = thrHandlers.begin(); i != thrHandlers.end(); ++i) {
            TProofThread *pt = *i;
            // Wait on this condition
            if (pt && pt->fThread->GetState() == TThread::kRunningState) {
               Info("Init",
                    "parallel startup: waiting for slave %s (%s:%d)",
                    pt->fArgs->fOrd.Data(), pt->fArgs->fUrl->GetHost(),
                    pt->fArgs->fUrl->GetPort());
               pt->fThread->Join();
            }
         }

         // Add the good slaves to the lists
         nextsl->Reset();
         TSlave *sl = 0;
         while ((sl = (TSlave *)(*nextsl)())) {
            if (sl->IsValid()) {
               fSlaves->Add(sl);
               fAllMonitor->Add(sl->GetSocket());
               startedsl->Remove(sl);
            }
         }

         // In case of failures, and if we still have some bonus trial,
         // try again when a 'delay' interval is elapsed;
         // otherwise flag the bad guys and go on
         if (claims.GetSize() > 0) {
            if (trial < ntries) {
               // Delete bad slaves
               nextsl->Reset();
               while ((sl = (TSlave *)(*nextsl)())) {
                  SafeDelete(sl);
               }
            } else {
               // Flag bad slaves
               nextsl->Reset();
               while ((sl = (TSlave *)(*nextsl)())) {
                  fBadSlaves->Add(sl);
               }
               // Empty the list of claims
               claims.RemoveAll();
            }
         }

         // Count trial
         trial++;

         // Thread vector cleanup
         while (!thrHandlers.empty()) {
            std::vector<TProofThread *>::iterator i = thrHandlers.end()-1;
            if (*i) {
               SafeDelete(*i);
               thrHandlers.erase(i);
            }
         }
      } else { // Serial startup
         // Get Condor Slave
         if (trial == 1) {
            c = dynamic_cast<TCondorSlave*>(claims.At(idx));
         } else {
            TPair *p = dynamic_cast<TPair*>(claims.At(idx));
            TTimer *t = dynamic_cast<TTimer*>(p->Value());
            // wait remaining time
            Long_t wait = (Long_t) (t->GetAbsTime()-gSystem->Now());
            if (wait>0) gSystem->Sleep(wait);
            c = dynamic_cast<TCondorSlave*>(p->Key());
         }

         // create slave
         TSlave *slave = CreateSlave(Form("%s:d",c->fHostname.Data(), c->fPort), c->fOrdinal,
                                     c->fPerfIdx, c->fImage, c->fWorkDir);

         // add slave to appropriate list
         if (trial<ntries) {
            // Finalize server startup (to be optmized)
            if (slave->IsValid()) {
               slave->SetupServ(TSlave::kSlave,0);
            }

            if (slave->IsValid()) {
               fSlaves->Add(slave);
               fAllMonitor->Add(slave->GetSocket());
               if (trial == 1) {
                  claims.Remove(c);
               } else {
                  TPair *p = dynamic_cast<TPair*>(claims.Remove(c));
                  delete dynamic_cast<TTimer*>(p->Value());
                  delete p;
               }
            } else {
               if (trial == 1) {
                  TTimer* timer = new TTimer(delay);
                  TPair *p = new TPair(c, timer);
                  claims.RemoveAt(idx);
                  claims.AddAt(p, idx);
               } else {
                  TPair *p = dynamic_cast<TPair*>(claims.At(idx));
                  dynamic_cast<TTimer*>(p->Value())->Reset();
               }
               delete slave;
               idx++;
            }
         } else {
            // Finalize server startup (to be optmized)
            if (slave->IsValid()) {
               slave->SetupServ(TSlave::kSlave,0);
            }

            fSlaves->Add(slave);
            if (slave->IsValid()) {
               fAllMonitor->Add(slave->GetSocket());
            } else {
               fBadSlaves->Add(slave);
            }
            TPair *p = dynamic_cast<TPair*>(claims.Remove(c));
            delete dynamic_cast<TTimer*>(p->Value());
            delete p;
         }

         if (idx>=claims.GetSize()) {
            trial++;
            idx = 0;
         }
      }
   }

   if (parallel) {
      SafeDelete(startedsl);
      SafeDelete(nextsl);
      SafeDelete(nextclaim);
   }

   return kTRUE;
}

//______________________________________________________________________________
void TProofCondor::SetActive(Bool_t active)
{
   // Suspend or resume PROOF via Condor.

   if (fTimer == 0) {
      fTimer = new TTimer();
   }
   if (active) {
      PDB(kCondor,1) Info("SetActive","-- Condor Resume --");
      fTimer->Stop();
      if (fCondor->GetState() == TCondor::kSuspended)
         fCondor->Resume();
   } else {
return; // don't suspend for the moment
      Int_t delay = 60000; // milli seconds
      PDB(kCondor,1) Info("SetActive","-- Delayed Condor Suspend (%d msec / to %ld) --",
                          delay, delay + long(gSystem->Now()));
      fTimer->Connect("Timeout()", "TCondor", fCondor, "Suspend()");
      fTimer->Start(10000, kTRUE); // single shot
   }
}

//______________________________________________________________________________
TString TProofCondor::GetJobAd()
{
   // Get job Ad

   TString ad;

   ad = "JobUniverse = 5\n"; // vanilla
   ad += Form("Cmd = \"%s/bin/proofd\"\n", GetConfDir());
   ad += "Iwd = \"/tmp\"\n";
   ad += "In = \"/dev/null\"\n";
   ad += "Out = \"/tmp/proofd.out.$(Port)\"\n";
   ad += "Err = \"/tmp/proofd.err.$(Port)\"\n";
   ad += Form("Args = \"-f -p $(Port) -d %d %s\"\n", GetLogLevel(), GetConfDir());

   return ad;
}
