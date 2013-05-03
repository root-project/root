// @(#)root/proof:$Id$
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

#include "TCondor.h"
#include "TList.h"
#include "TMap.h"
#include "TMessage.h"
#include "TMonitor.h"
#include "TProofNodeInfo.h"
#include "TProofResourcesStatic.h"
#include "TProofServ.h"
#include "TSlave.h"
#include "TSocket.h"
#include "TString.h"
#include "TTimer.h"

ClassImp(TProofCondor)

//______________________________________________________________________________
TProofCondor::TProofCondor(const char *masterurl, const char *conffile,
                           const char *confdir, Int_t loglevel,
                           const char *, TProofMgr *mgr)
  : fCondor(0), fTimer(0)
{
   // Start proof using condor

   // Default initializations
   InitMembers();

   // This may be needed during init
   fManager = mgr;

   fUrl = TUrl(masterurl);

   if (!conffile || !conffile[0]) {
      conffile = kPROOF_ConfFile;
   } else if (!strncasecmp(conffile, "condor:", 7)) {
      conffile+=7;
   }

   if (!confdir  || !confdir[0]) {
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
Bool_t TProofCondor::StartSlaves(Bool_t)
{
   // Setup Condor workers using dynamic information

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
      int nSlavesDone = 0;
      while ((to = next())) {
         // Get the next worker from the list
         worker = (TProofNodeInfo *)to;

         // Read back worker node info
         const Char_t *image = worker->GetImage().Data();
         const Char_t *workdir = worker->GetWorkDir().Data();
         Int_t perfidx = worker->GetPerfIndex();

         gSystem->Sleep(10 /* ms */);
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

         // Notify claim creation
         nSlavesDone++;
         TMessage m(kPROOF_SERVERSTARTED);
         m << TString("Creating COD Claim") << workerList->GetSize()
         << nSlavesDone << (csl != 0);
         gProofServ->GetSocket()->Send(m);

      } // end while (worker loop)

      // Cleanup
      delete resources;
      resources = 0;
   } // end else (parse config file)

   Long_t delay = 500; // timer delay 0.5s
   Int_t ntries = 20; // allow 20 tries (must be > 1 for algorithm to work)
   Int_t trial = 1;
   Int_t idx = 0;

   int nClaims = claims.GetSize();
   int nClaimsDone = 0;
   while (claims.GetSize() > 0) {
      TCondorSlave* c = 0;

      // Get Condor Slave
      if (trial == 1) {
         c = dynamic_cast<TCondorSlave*>(claims.At(idx));
      } else {
         TPair *p = dynamic_cast<TPair*>(claims.At(idx));
         if (p) {
            TTimer *t = dynamic_cast<TTimer*>(p->Value());
            if (t) {
               // wait remaining time
               Long64_t wait = t->GetAbsTime()-gSystem->Now();
               if (wait > 0) gSystem->Sleep((UInt_t)wait);
               c = dynamic_cast<TCondorSlave*>(p->Key());
            }
         }
      }

      // create slave
      TSlave *slave = 0;
      if (c) slave = CreateSlave(Form("%s:%d", c->fHostname.Data(), c->fPort), c->fOrdinal,
                                               c->fPerfIdx, c->fImage, c->fWorkDir);

      // add slave to appropriate list
      if (trial < ntries) {
         if (slave && slave->IsValid()) {
            fSlaves->Add(slave);
            if (trial == 1) {
               claims.Remove(c);
            } else {
               TPair *p = dynamic_cast<TPair*>(claims.Remove(c));
               if (p) {
                  TTimer *xt = dynamic_cast<TTimer*>(p->Value());
                  if (xt) delete xt;
                  delete p;
               }
            }
            nClaimsDone++;
            TMessage m(kPROOF_SERVERSTARTED);
            m << TString("Opening connections to workers") << nClaims
               << nClaimsDone << kTRUE;
            gProofServ->GetSocket()->Send(m);
         } else if (slave) {
            if (trial == 1) {
               TTimer* timer = new TTimer(delay);
               TPair *p = new TPair(c, timer);
               claims.RemoveAt(idx);
               claims.AddAt(p, idx);
            } else {
               TPair *p = dynamic_cast<TPair*>(claims.At(idx));
               if (p && p->Value()) {
                  TTimer *xt = dynamic_cast<TTimer*>(p->Value());
                  if (xt) xt->Reset();
               }
            }
            delete slave;
            idx++;
         } else {
            Warning("StartSlaves", "could not create TSlave object!");
         }
      } else {
         if (slave) {
            fSlaves->Add(slave);
            TPair *p = dynamic_cast<TPair*>(claims.Remove(c));
            if (p && p->Value()) {
               TTimer *xt = dynamic_cast<TTimer*>(p->Value());
               delete xt;
            }
            if (p) delete p;

            nClaimsDone++;
            TMessage m(kPROOF_SERVERSTARTED);
            m << TString("Opening connections to workers") << nClaims
               << nClaimsDone << slave->IsValid();
            gProofServ->GetSocket()->Send(m);
         } else {
            Warning("StartSlaves", "could not create TSlave object!");
         }
      }

      if (idx>=claims.GetSize()) {
         trial++;
         idx = 0;
      }
   }

   // Here we finalize the server startup: in this way the bulk
   // of remote operations are almost parallelized
   TIter nxsl(fSlaves);
   TSlave *sl = 0;
   int nSlavesDone = 0, nSlavesTotal = fSlaves->GetSize();
   while ((sl = (TSlave *) nxsl())) {

      // Finalize setup of the server
      if (sl->IsValid()) {
         sl->SetupServ(TSlave::kSlave, 0);
      }

      if (sl->IsValid()) {
         fAllMonitor->Add(sl->GetSocket());
      } else {
         fBadSlaves->Add(sl);
      }

      // Notify end of startup operations
      nSlavesDone++;
      TMessage m(kPROOF_SERVERSTARTED);
      Bool_t wrkvalid = sl->IsValid() ? kTRUE : kFALSE;
      m << TString("Setting up worker servers") << nSlavesTotal
         << nSlavesDone << wrkvalid;
      gProofServ->GetSocket()->Send(m);
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
#if 1
      return; // don't suspend for the moment
#else
      Int_t delay = 60000; // milli seconds
      PDB(kCondor,1) Info("SetActive","-- Delayed Condor Suspend (%d msec / to %lld) --",
                          delay, delay + Long64_t(gSystem->Now()));
      fTimer->Connect("Timeout()", "TCondor", fCondor, "Suspend()");
      fTimer->Start(10000, kTRUE); // single shot
#endif
   }
}

//______________________________________________________________________________
TString TProofCondor::GetJobAd()
{
   // Get job Ad

   TString ad;

   ad = "JobUniverse = 5\n"; // vanilla
   ad += Form("Cmd = \"%s/bin/proofd\"\n", GetConfDir());
   ad += Form("Iwd = \"%s\"\n", gSystem->TempDirectory());
   ad += "In = \"/dev/null\"\n";
   ad += Form("Out = \"%s/proofd.out.$(Port)\"\n", gSystem->TempDirectory());
   ad += Form("Err = \"%s/proofd.err.$(Port)\"\n", gSystem->TempDirectory());
   ad += Form("Args = \"-f -p $(Port) -d %d %s\"\n", GetLogLevel(), GetConfDir());

   return ad;
}
