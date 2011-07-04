// @(#)root/proof:$Id$
// Author: Sangsu Ryu 22/06/2010

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofBenchRunCPU                                                    //
//                                                                      //
// CPU-intensive PROOF benchmark test generates events and fill 1, 2,   //
// or 3-D histograms. No I/O activity is involved.                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "RConfigure.h"

#include "TProofBenchRunCPU.h"
#include "TProofNodes.h"
#include "TFileCollection.h"
#include "TFileInfo.h"
#include "TProof.h"
#include "TString.h"
#include "Riostream.h"
#include "TMap.h"
#include "TEnv.h"
#include "TTree.h"
#include "TLeaf.h"
#include "TCanvas.h"
#include "TROOT.h"
#include "TH2.h"
#include "TF1.h"
#include "TProfile.h"
#include "TLegend.h"
#include "TKey.h"
#include "TRegexp.h"
#include "TPerfStats.h"
#include "TQueryResult.h"
#include "TMath.h"
#include "TStyle.h"
#include "TProofNodes.h"
#include "TGraphErrors.h"

ClassImp(TProofBenchRunCPU)

//______________________________________________________________________________
TProofBenchRunCPU::TProofBenchRunCPU(TPBHistType *histtype, Int_t nhists,
                                     TDirectory* dirproofbench, TProof* proof,
                                     TProofNodes* nodes, Long64_t nevents, Int_t ntries,
                                     Int_t start, Int_t stop, Int_t step, Int_t draw,
                                     Int_t debug)
                  : TProofBenchRun(proof, kPROOF_BenchSelCPUDef),
                    fHistType(histtype), fNHists(nhists),
                    fNEvents(nevents), fNTries(ntries), fStart(start), fStop(stop),
                    fStep(step), fDraw(draw), fDebug(debug), fDirProofBench(dirproofbench),
                    fNodes(nodes), fListPerfPlots(0),
                    fCanvas(0), fProfile_perfstat_event(0), fHist_perfstat_event(0),
                    fProfile_queryresult_event(0), fNorm_queryresult_event(0), fName(0)
{
   // Default constructor

   if (TestBit(kInvalidObject)) {
      Error("TProofBenchRunCPU", "problems validating PROOF session or enabling selector PAR");
      return;
   }

   fName = "CPU";

   if (!fNodes) fNodes = new TProofNodes(fProof);

   if (stop == -1) fStop = fNodes->GetNWorkersCluster();

   fListPerfPlots = new TList;

   gEnv->SetValue("Proof.StatsTrace",1);
   gStyle->SetOptStat(0);
}

//______________________________________________________________________________
TProofBenchRunCPU::~TProofBenchRunCPU()
{
   // Destructor
   fProof=0;
   fDirProofBench=0;
   SafeDelete(fListPerfPlots);
   SafeDelete(fCanvas);
   SafeDelete(fNodes);
}

//______________________________________________________________________________
void TProofBenchRunCPU::BuildHistos(Int_t start, Int_t stop, Int_t step, Bool_t nx)
{
   // Build histograms, profiles and graphs needed for this run

   TObject *o = 0;
   Int_t quotient = (stop - start) / step;
   Int_t ndiv = quotient + 1;
   Double_t ns_min = start - step/2.;
   Double_t ns_max = quotient*step + start + step/2.;

   TString axtitle("Active Workers"), namelab(GetName()), sellab(GetSelName());
   if (nx) {
      axtitle = "Active Workers/Node";
      namelab.Form("x_%s", GetName());
   }
   if (fSelName == kPROOF_BenchSelCPUDef)
      sellab.Form("%s_%s", GetSelName(), GetNameStem().Data());

   TString name, title;
   // Book perfstat profile
   name.Form("Prof_%s_PS_Evts_%s", namelab.Data(), sellab.Data());
   title.Form("Profile %s PerfStat Event - %s", namelab.Data(), sellab.Data());
   fProfile_perfstat_event = new TProfile(name, title, ndiv, ns_min, ns_max);
   fProfile_perfstat_event->SetDirectory(fDirProofBench);
   fProfile_perfstat_event->GetYaxis()->SetTitle("Events/sec");
   fProfile_perfstat_event->GetXaxis()->SetTitle(axtitle);
   fProfile_perfstat_event->SetMarkerStyle(21);
   if ((o = fListPerfPlots->FindObject(name))) {
      fListPerfPlots->Remove(o);
      delete o;
   }
   fListPerfPlots->Add(fProfile_perfstat_event);

   // Book perfstat histogram
   name.Form("Hist_%s_PS_Evts_%s", namelab.Data(), sellab.Data());
   title.Form("Histogram %s PerfStat Event - %s", namelab.Data(), sellab.Data());
   fHist_perfstat_event = new TH2D(name, title, ndiv, ns_min, ns_max, 100, 0, 0);
   fHist_perfstat_event->SetDirectory(fDirProofBench);
   fHist_perfstat_event->GetYaxis()->SetTitle("Events/sec");
   fHist_perfstat_event->GetXaxis()->SetTitle(axtitle);
   fHist_perfstat_event->SetMarkerStyle(7);
   if ((o = fListPerfPlots->FindObject(name))) {
      fListPerfPlots->Remove(o);
      delete o;
   }
   fListPerfPlots->Add(fHist_perfstat_event);

   // Book queryresult profile
   name.Form("Prof_%s_QR_Evts_%s", namelab.Data(), sellab.Data());
   title.Form("Profile %s QueryResult Event - %s", namelab.Data(), sellab.Data());
   fProfile_queryresult_event = new TProfile(name, title, ndiv, ns_min, ns_max);
   fProfile_queryresult_event->SetDirectory(fDirProofBench);
   fProfile_queryresult_event->GetYaxis()->SetTitle("Events/sec");
   fProfile_queryresult_event->GetXaxis()->SetTitle(axtitle);
   fProfile_queryresult_event->SetMarkerStyle(22);
   if ((o = fListPerfPlots->FindObject(name))) {
      fListPerfPlots->Remove(o);
      delete o;
   }
   fListPerfPlots->Add(fProfile_queryresult_event);

   // Book normalized queryresult profile
   name.Form("Norm_%s_QR_Evts_%s", namelab.Data(), sellab.Data());
   title.Form("Profile %s Normalized QueryResult Event - %s", namelab.Data(), sellab.Data());
   fNorm_queryresult_event = new TProfile(name, title, ndiv, ns_min, ns_max);
   fNorm_queryresult_event->SetDirectory(fDirProofBench);
   fNorm_queryresult_event->GetYaxis()->SetTitle("Events/sec");
   fNorm_queryresult_event->GetXaxis()->SetTitle(axtitle);
   fNorm_queryresult_event->SetMarkerStyle(22);
   if ((o = fListPerfPlots->FindObject(name))) {
      fListPerfPlots->Remove(o);
      delete o;
   }
   fListPerfPlots->Add(fNorm_queryresult_event);
}

//______________________________________________________________________________
void TProofBenchRunCPU::Run(Long64_t nevents, Int_t start, Int_t stop,
                            Int_t step, Int_t ntries, Int_t debug, Int_t draw)
{
   // Run benchmark
   // Input parameters
   //   nevents:   Number of events to run per file. When it is -1, use data member fNEvents.
   //   start: Start scan with 'start' workers. When it is -1, use data member fStart.
   //          When 0, the same number of workers are activated on all nodes.
   //   stop: Stop scan at 'stop' workers. When it is -1 , use data member fStop.
   //   step: Scan every 'step' workers. When it is -1, use data member fStep.
   //   ntries: Number of repetitions.  When it is -1, use data member fNTries.
   //   debug: debug switch. When it is -1, use data member fDebug.
   //   draw: draw switch. When it is -1, use data member fDraw.
   // Returns
   //    Nothing

   if (!fProof){
      Error("Run", "Proof not set");
      return;
   }

   nevents = (nevents == -1) ? fNEvents : nevents;
   start = (start == -1) ? fStart : start;
   stop = (stop == -1) ? fStop : stop;
   step = (step == -1) ? fStep : step;
   ntries = (ntries == -1) ? fNTries : ntries;
   debug = (debug == -1) ? fDebug : debug;
   draw = (draw == -1) ? fDraw : draw;

   Bool_t nx = kFALSE;
   if (step == -2){
      nx = kTRUE;
      start = fStart;
      step = 1;
   }

   if (nx){
      Int_t minnworkersanode = fNodes->GetMinWrksPerNode();
      if (stop > minnworkersanode) stop = minnworkersanode;
   }

   // Load the selector, if needed
   if (!TClass::GetClass(fSelName)) {
      // Is it the default selector?
      if (fSelName == kPROOF_BenchSelCPUDef) {
         // Load the parfile
#ifdef R__HAVE_CONFIG
         TString par = TString::Format("%s/%s%s.par", ROOTETCDIR, kPROOF_BenchParDir, kPROOF_BenchCPUSelPar);
#else
         TString par = TString::Format("$ROOTSYS/etc/%s%s.par", kPROOF_BenchParDir, kPROOF_BenchCPUSelPar);
#endif
         Info("Run", "Uploading '%s' ...", par.Data());
         if (fProof->UploadPackage(par) != 0) {
            Error("Run", "problems uploading '%s' - cannot continue", par.Data());
            return;
         }
         Info("Run", "Enabling '%s' ...", kPROOF_BenchCPUSelPar);
         if (fProof->EnablePackage(kPROOF_BenchCPUSelPar) != 0) {
            Error("Run", "problems enabling '%s' - cannot continue", kPROOF_BenchCPUSelPar);
            return;
         }
      } else {
         if (fParList.IsNull()) {
            Error("Run", "you should load the class '%s' before running the benchmark", fSelName.Data());
            return;
         } else {
            TString par;
            Int_t from = 0;
            while (fParList.Tokenize(par, from, ",")) {
               Info("Run", "Uploading '%s' ...", par.Data());
               if (fProof->UploadPackage(par) != 0) {
                  Error("Run", "problems uploading '%s' - cannot continue", par.Data());
                  return;
               }
               Info("Run", "Enabling '%s' ...", par.Data());
               if (fProof->EnablePackage(par) != 0) {
                  Error("Run", "problems enabling '%s' - cannot continue", par.Data());
                  return;
               }
            }
         }
      }
      // Check
      if (!TClass::GetClass(fSelName)) {
         Error("Run", "failed to load '%s'", fSelName.Data());
         return;
      }
   }

   // Build histograms, profiles and graphs needed for this run
   BuildHistos(start, stop, step, nx);

   // Get pad
   if (!fCanvas) fCanvas = new TCanvas("Canvas");
   // Cleanup up the canvas
   fCanvas->Clear();

   // Divide the canvas as many as the number of profiles in the list
   fCanvas->Divide(2,1);

   TString perfstats_name = "PROOF_PerfStats";

   SetParameters();

   if (nx){
      Info("Run", "Running CPU-bound tests; %d ~ %d active worker(s)/node,"
                  " every %d worker(s)/node.", start, stop, step);
   } else {
      Info("Run", "Running CPU-bound tests; %d ~ %d active worker(s),"
                  " every %d worker(s).", start, stop, step);
   }

   Int_t npad = 1; //pad number

   Int_t nnodes = fNodes->GetNNodes(); // Number of machines

   Double_t ymi = -1., ymx = -1.;
   for (Int_t nactive = start; nactive <= stop; nactive += step) {

      // Actvate the wanted workers
      Int_t nw = -1;
      if (nx) {
         TString workers;
         workers.Form("%dx", nactive);
         nw = fNodes->ActivateWorkers(workers);
      } else {
         nw = fNodes->ActivateWorkers(nactive);
      }
      if (nw < 0){
         Error("Run", "could not activate the requested number of"
                      " workers/node on the cluster; skipping the test point"
                      " (%d workers/node)", nactive);
         continue;
      }

      for (Int_t j = 0; j < ntries; j++) {

         if (nx){
            Info("Run", "Running CPU-bound tests with %d active worker(s)/node;"
                        " trial %d/%d", nactive, j + 1, ntries);
         } else {
            Info("Run", "Running CPU-bound tests with %d active worker(s);"
                        " trial %d/%d", nactive, j + 1, ntries);
         }

         Int_t nevents_all=0;
         if (nx){
            nevents_all=nevents*nactive*nnodes;
         } else {
            nevents_all=nevents*nactive;
         }

         // Process
         fProof->Process(fSelName, nevents_all);

         TList *l = fProof->GetOutputList();

         // Save perfstats
         TTree *t = dynamic_cast<TTree*>(l->FindObject(perfstats_name.Data()));
         if (t) {

            //FillPerfStatPerfPlots(t, profile_perfstat_event, nactive);
            FillPerfStatPerfPlots(t, nactive);

            t->SetDirectory(fDirProofBench);
            t->SetDirectory(gDirectory);

            // Build up new name
            TString newname = TString::Format("%s_%s_%dwrks%dthtry", t->GetName(), GetName(), nactive, j);
            t->SetName(newname);

            if (debug && fDirProofBench->IsWritable()){
               TDirectory *curdir = gDirectory;
               TString dirn = nx ? "RunCPUx" : "RunCPU";
               if (!fDirProofBench->GetDirectory(dirn))
                  fDirProofBench->mkdir(dirn, "RunCPU results");
               if (fDirProofBench->cd(dirn)) {
                  t->Write();
               } else {
                  Warning("Run", "cannot cd to subdirectory '%s' to store the results!", dirn.Data());
               }
               curdir->cd();
            }
         } else {
            Error("RunBenchmark", "tree %s not found", perfstats_name.Data());
         }

         // Performance measures from TQueryResult

         TQueryResult *queryresult = fProof->GetQueryResult();
         TDatime qr_start = queryresult->GetStartTime();
         TDatime qr_end = queryresult->GetEndTime();
         Float_t qr_proc = queryresult->GetProcTime();

         Long64_t qr_entries = queryresult->GetEntries();

         // Calculate event rate
         Double_t qr_eventrate = qr_entries / Double_t(qr_proc);

         // Fill and draw
         fProfile_queryresult_event->Fill(nactive, qr_eventrate);
         fCanvas->cd(npad);
         fProfile_queryresult_event->Draw();
         gPad->Update();
         // The normalised histo
         Double_t nert = nx ? qr_eventrate/nactive/nnodes : qr_eventrate/nactive;
         fNorm_queryresult_event->Fill(nactive, nert);
         // Use the first bin to set the Y range for the histo
         Double_t y1 = fNorm_queryresult_event->GetBinContent(1);
         Double_t e1 = fNorm_queryresult_event->GetBinError(1);
         Double_t dy = 5 * e1;
         if (dy / y1 < 0.2) dy = y1 * 0.2;
         if (dy > y1) dy = y1*.999999;
         if (ymi < 0.) ymi = y1 - dy;
         if (fNorm_queryresult_event->GetBinContent(nactive) < ymi)
            ymi = fNorm_queryresult_event->GetBinContent(nactive) / 2.;
         if (ymx < 0.) ymx = y1 + dy;
         if (fNorm_queryresult_event->GetBinContent(nactive) > ymx)
            ymx = fNorm_queryresult_event->GetBinContent(nactive) * 1.5;
         fNorm_queryresult_event->SetMaximum(ymx);
         fNorm_queryresult_event->SetMinimum(ymi);
         fCanvas->cd(npad+1);
         fNorm_queryresult_event->Draw();
         gPad->Update();

      } // for iterations
   } // for number of workers

   // Make the result persistent
   fCanvas->cd(npad);
   fProfile_queryresult_event->DrawCopy();
   fCanvas->cd(npad + 1);
   fNorm_queryresult_event->DrawCopy();
   gPad->Update();

   //save performance profiles to file
   if (fDirProofBench && fDirProofBench->IsWritable()){
      TDirectory *curdir = gDirectory;
      TString dirn = nx ? "RunCPUx" : "RunCPU";
      if (!fDirProofBench->GetDirectory(dirn))
         fDirProofBench->mkdir(dirn, "RunCPU results");
      if (fDirProofBench->cd(dirn)) {
         fListPerfPlots->Write(0, kOverwrite);
         fListPerfPlots->SetOwner(kFALSE);
         fListPerfPlots->Clear();
      } else {
         Warning("Run", "cannot cd to subdirectory '%s' to store the results!", dirn.Data());
      }
      curdir->cd();
   }
}

//______________________________________________________________________________
void TProofBenchRunCPU::FillPerfStatPerfPlots(TTree* t, Int_t nactive)
{

   // Fill performance profiles using tree 't'(PROOF_PerfStats).
   // Input parameters
   //    t: Proof output tree (PROOF_PerfStat) containing performance statistics.
   //    profile: Profile to be filled up with information from tree 't'.
   //    nactive: Number of active workers processed the query.
   // Return
   //    Nothing

   // find perfstat profile
   if (!fProfile_perfstat_event){
      Error("FillPerfStatPerfPlots", "no perfstat profile found");
      return;
   }

   // find perfstat histogram
   if (!fHist_perfstat_event){
      Error("FillPerfStatPerfPlots", "no perfstat histogram found");
      return;
   }

   // extract timing information
   TPerfEvent pe;
   TPerfEvent* pep = &pe;
   t->SetBranchAddress("PerfEvents",&pep);
   Long64_t entries = t->GetEntries();

   Double_t event_rate_packet = 0;

   for (Long64_t k=0; k<entries; k++) {

      t->GetEntry(k);

      // Skip information from workers
      if (pe.fEvtNode.Contains(".")) continue;

      if (pe.fType == TVirtualPerfStats::kPacket){
         if (pe.fProcTime != 0.0) {
            event_rate_packet = pe.fEventsProcessed / pe.fProcTime;
            fHist_perfstat_event->Fill(Double_t(nactive), event_rate_packet);
         }
      }
   }

   return;
}

//______________________________________________________________________________
void TProofBenchRunCPU::Print(Option_t* option) const
{
   // Show settings

   Printf("+++ TProofBenchRunCPU +++++++++++++++++++++++++++++++++++++++++");
   Printf("Name      = %s", fName.Data());
   if (fProof) fProof->Print(option);
   Printf("fHistType = k%s", GetNameStem().Data());
   Printf("fNHists   = %d", fNHists);
   Printf("fNEvents  = %lld", fNEvents);
   Printf("fNTries   = %d", fNTries);
   Printf("fStart    = %d", fStart);
   Printf("fStop     = %d", fStop);
   Printf("fStep     = %d", fStep);
   Printf("fDraw     = %d", fDraw);
   Printf("fDebug    = %d", fDebug);
   if (fDirProofBench)
      Printf("fDirProofBench = %s", fDirProofBench->GetPath());
   if (fNodes) fNodes->Print(option);
   if (fListPerfPlots) fListPerfPlots->Print(option);
   if (fCanvas)
      Printf("Performance Canvas: Name = %s Title = %s",
              fCanvas->GetName(), fCanvas->GetTitle());
   Printf("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
}

//______________________________________________________________________________
void TProofBenchRunCPU::DrawPerfPlots()
{
   // Draw Performance plots

   // Get canvas
   if (!fCanvas) fCanvas = new TCanvas("Canvas");

   fCanvas->Clear();

   // Divide the canvas as many as the number of profiles in the list
   Int_t nprofiles = fListPerfPlots->GetSize();
   if (nprofiles <= 2){
      fCanvas->Divide(1,nprofiles);
   } else {
      Int_t nside = (Int_t)TMath::Sqrt((Float_t)nprofiles);
      nside = (nside*nside<nprofiles)?nside+1:nside;
      fCanvas->Divide(nside,nside);
   }

   Int_t npad=1;
   TIter nxt(fListPerfPlots);
   TProfile* profile=0;
   while ((profile=(TProfile*)(nxt()))){
      fCanvas->cd(npad++);
      profile->Draw();
      gPad->Update();
   }
   return;
}

//______________________________________________________________________________
void TProofBenchRunCPU::SetHistType(TPBHistType *histtype)
{
   // Set histogram type

   fHistType = histtype;
   fName.Form("%sCPU", GetNameStem().Data());
}

//______________________________________________________________________________
TString TProofBenchRunCPU::GetNameStem() const
{
   // Get name for this run

   TString namestem("+++undef+++");
   if (fHistType) {
      switch (fHistType->GetType()) {
         case TPBHistType::kHist1D:
            namestem = "Hist1D";
            break;
         case TPBHistType::kHist2D:
            namestem = "Hist2D";
            break;
         case TPBHistType::kHist3D:
            namestem = "Hist3D";
            break;
         case TPBHistType::kHistAll:
            namestem = "HistAll";
            break;
         default:
            break;
      }
   }
   return namestem;
}

//______________________________________________________________________________
Int_t TProofBenchRunCPU::SetParameters()
{
   // Set parameters

   if (!fProof) {
      Error("SetParameters", "proof not set; Doing nothing");
      return 1;
   }

   if (!fHistType) fHistType = new TPBHistType(TPBHistType::kHist1D);
   fProof->AddInput(fHistType);
   fProof->SetParameter("PROOF_BenchmarkNHists", fNHists);
   fProof->SetParameter("PROOF_BenchmarkDraw", Int_t(fDraw));
   return 0;
}

//______________________________________________________________________________
Int_t TProofBenchRunCPU::DeleteParameters()
{
   // Delete parameters set for this run
   if (!fProof){
      Error("DeleteParameters", "proof not set; Doing nothing");
      return 1;
   }
   if (fProof->GetInputList()) {
      TObject *type = fProof->GetInputList()->FindObject("PROOF_Benchmark_HistType");
      if (type) fProof->GetInputList()->Remove(type);
   }
   fProof->DeleteParameters("PROOF_BenchmarkNHists");
   fProof->DeleteParameters("PROOF_BenchmarkDraw");
   return 0;
}

