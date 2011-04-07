// @(#)root/proof:$Id$
// Author: G.Ganis, S.Ryu Feb 2011

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofBench                                                          //
//                                                                      //
// Steering class for PROOF benchmarks                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TProofBench.h"
#include "Getline.h"
#include "TProofBenchRunCPU.h"
#include "TProofBenchRunDataRead.h"
#include "TProofBenchDataSet.h"
#include "TProofNodes.h"
#include "TClass.h"
#include "TFile.h"
#include "TFileCollection.h"
#include "TFileInfo.h"
#include "THashList.h"
#include "TKey.h"
#include "TObjString.h"
#include "TProof.h"
#include "TROOT.h"
#include "TUrl.h"

#include "TCanvas.h"
#include "TGraphErrors.h"
#include "TH1F.h"
#include "TMath.h"
#include "TProfile.h"
#include "TStyle.h"

ClassImp(TProofBench)

//______________________________________________________________________________
TProofBench::TProofBench(const char *url, const char *outfile, const char *proofopt)
            : fUnlinkOutfile(kFALSE), fProofDS(0), fOutFile(0),
              fNtries(4), fHistType(0), fNHist(16), fReadType(0),
              fDataSet("BenchDataSet"), fDataGenSel(kPROOF_BenchSelDataGenDef),
              fRunCPU(0), fRunDS(0), fDS(0)
{
   // Constructor: check PROOF and load selectors PAR
   
   SetBit(kInvalidObject);
   if (!url) {
      Error("TProofBench", "specifying a PROOF master url is mandatory - cannot continue");
      return;
   }
   if (!(fProof = TProof::Open(url, proofopt)) || (fProof && !fProof->IsValid())) {
      Error("TProofBench", "could not open a valid PROOF session - cannot continue");
      return;
   }
   // By default we use the same instance for dataset actions
   fProofDS = fProof;
   // The object is now valid
   ResetBit(kInvalidObject);

   // Set output file
   if (SetOutFile(outfile, kFALSE) != 0)
      Warning("TProofBench", "problems opening '%s' - ignoring: use SetOutFile to try"
                                   " again or with another file", outfile);
} 

//______________________________________________________________________________
TProofBench::~TProofBench()
{
   // Destructor
   
   CloseOutFile();
   if (fUnlinkOutfile) gSystem->Unlink(fOutFileName);
   SafeDelete(fReadType);
   SafeDelete(fRunCPU);
   SafeDelete(fRunDS);
}

//______________________________________________________________________________
Int_t TProofBench::OpenOutFile(Bool_t wrt, Bool_t verbose)
{
   // Set the otuput file
   // Return 0 on success, -1 on error

   // Remove any bad file
   if (fOutFile && fOutFile->IsZombie()) SafeDelete(fOutFile);

   Int_t rc = 0;
   if (!fOutFile && fOutFileName.Length() > 0) {
      const char *mode = 0;
      if (wrt)
         mode = gSystem->AccessPathName(fOutFileName) ? "RECREATE" : "UPDATE";
      else
         mode = "READ";
      if (!(fOutFile = TFile::Open(fOutFileName, mode)) || (fOutFile && fOutFile->IsZombie())) {
         if (verbose)
            Warning("OpenOutFile", "problems opening '%s' - ignoring: use SetOutFile to try"
                                   " again or with another file", fOutFileName.Data());
         rc = -1;
      }
      if (fOutFile) gROOT->GetListOfFiles()->Remove(fOutFile);
   }
   return rc;
}


//______________________________________________________________________________
Int_t TProofBench::SetOutFile(const char *outfile, Bool_t verbose)
{
   // Set the output file
   // Return 0 on success, -1 on error

   Int_t rc = 0;
   // Close existing file, if any
   if (fOutFile) {
      if (!fOutFile->IsZombie()) fOutFile->Close();
      SafeDelete(fOutFile);
   }
   
   fOutFileName = outfile;
   if (fOutFileName == "<default>") {
      // Default output file: proofbench-<master>-<DayMonthYear-hhmm>.root
      TDatime dat;
      const char *lite = (fProof->IsLite()) ? "-lite" : "";
      fOutFileName.Form("proofbench-%s%s-%dw-%d-%.2d%.2d.root",
                        fProof->GetMaster(), lite, fProof->GetParallel(),
                        dat.GetDate(), dat.GetHour(), dat.GetMinute());
      Info("SetOutFile", "using default output file: '%s'", fOutFileName.Data());
      fUnlinkOutfile = kTRUE;
   }
   if (!fOutFileName.IsNull()) {
      if ((rc = OpenOutFile(kTRUE, kFALSE)) != 0 && verbose)
         Warning("SetOutFile", "problems opening '%s' - ignoring: use SetOutFile to try"
                               " again or with another file", outfile);
   }
   return rc;
}

//______________________________________________________________________________
Int_t TProofBench::RunCPU(Long64_t nevents, Int_t start, Int_t stop, Int_t step)
{
   // Perform the CPU run
   // Return 0 on success, -1 on error

   // Open the file for the results
   if (OpenOutFile(kTRUE) != 0) {
      Error("RunCPU", "problems opening '%s' to save the result", fOutFileName.Data());
      return -1;
   }
   fUnlinkOutfile = kFALSE;

   SafeDelete(fRunCPU);
   TPBHistType htype(TPBHistType::kHist1D);
   fRunCPU = new TProofBenchRunCPU(&htype, fNHist, fOutFile);
   if (!fCPUSel.IsNull()) fRunCPU->SetSelName(fCPUSel);
   if (!fCPUPar.IsNull()) fRunCPU->SetParList(fCPUPar);
   fRunCPU->Run(nevents, start, stop, step, fNtries, -1, -1);

   // Close the file
   if (SetOutFile(0) != 0)
      Warning("RunCPU", "problems closing '%s'", fOutFileName.Data());

   // Done
   return 0;
}

//______________________________________________________________________________
Int_t TProofBench::RunCPUx(Long64_t nevents, Int_t start, Int_t stop)
{
   // Perform the CPU run scanning over the number of workers per node
   // Return 0 on success, -1 on error

   // Open the file for the results
   if (OpenOutFile(kTRUE) != 0) {
      Error("RunCPUx", "problems opening '%s' to save the result", fOutFileName.Data());
      return -1;
   }
   fUnlinkOutfile = kFALSE;

   SafeDelete(fRunCPU);
   TPBHistType htype(TPBHistType::kHist1D);
   fRunCPU = new TProofBenchRunCPU(&htype, fNHist, fOutFile);
   if (!fCPUSel.IsNull()) fRunCPU->SetSelName(fCPUSel);
   if (!fCPUPar.IsNull()) fRunCPU->SetParList(fCPUPar);
   fRunCPU->Run(nevents, start, stop, -2, fNtries, -1, -1);

   // Close the file
   if (SetOutFile(0) != 0)
      Warning("RunCPUx", "problems closing '%s'", fOutFileName.Data());

   // Done
   return 0;
}

//______________________________________________________________________________
void TProofBench::DrawCPU(const char *outfile, const char *opt)
{
   // Draw the CPU speedup plot.
   //  opt =    'std:'      draw standard evt/s plot
   //           'stdx:'     draw standard evt/s plot, 1 worker per node
   //           'norm:'     draw normalized plot
   //           'normx:'    draw normalized plot, 1 worker per node
   //

   // Get the TProfile an create the graphs
   TFile *fout = TFile::Open(outfile, "READ");
   if (!fout || (fout && fout->IsZombie())) {
      ::Error("DrawCPU", "could not open file '%s' ...", outfile);
      return;
   }

   TDirectory *d = (TDirectory *) fout->Get("RunCPU");
   if (!d) {
      ::Error("DrawCPU", "could not find directory 'RunCPU' ...");
      fout->Close();
      delete fout;
      return;
   }
   d->cd();
   TString hprofn;
   if (!strcmp(opt, "std:")) {
      hprofn = "Prof_CPU_QR_Evts";
   } else if (!strcmp(opt, "stdx:")) {
      hprofn = "Prof_x_CPU_QR_Evts";
   } else if (!strcmp(opt, "norm:")) {
      hprofn = "Norm_CPU_QR_Evts";
   } else if (!strcmp(opt, "normx:")) {
      hprofn = "Norm_x_CPU_QR_Evts";
   } else {
      ::Error("DrawCPU", "unknown option '%s'", opt);
      fout->Close();
      delete fout;
      return;
   }
   TProfile *pf = 0;
   TList *keylist = d->GetListOfKeys();
   TKey *key = 0;
   TIter nxk(keylist);
   while ((key = (TKey *) nxk())) {
      if (TString(key->GetName()).BeginsWith(hprofn)) {
         pf = (TProfile *) d->Get(key->GetName());
         break;
      }
   }
   if (!pf) {
      ::Error("DrawCPU", "could not find '%s' ...", hprofn.Data());
      fout->Close();
      delete fout;
      return;
   }

   Int_t nbins = pf->GetNbinsX();
   TGraphErrors *gr = new TGraphErrors(nbins);
   Double_t xx, ex, yy, ey, ymi = pf->GetBinContent(1), ymx = ymi;
   Int_t k =1;
   for (;k <= nbins; k++) {
      xx = pf->GetBinCenter(k);
      ex = pf->GetBinWidth(k) * .001;
      yy = pf->GetBinContent(k);
      ey = pf->GetBinError(k);
      if (k == 1) {
         ymi = yy;
         ymx = yy;
      } else {
         if (yy < ymi) ymi = yy; 
         if (yy > ymx) ymx = yy;
      }
      gr->SetPoint(k-1, xx, yy);
      gr->SetPointError(k-1, ex, ey);
   }

   // Create the canvas
   TCanvas *cpu = new TCanvas("cpu", "Rate vs wrks",204,69,1050,502);
   cpu->Range(-3.106332,0.7490716,28.1362,1.249867);

   gStyle->SetOptTitle(0);
   gr->SetFillColor(1);
   gr->SetLineColor(13);
   gr->SetMarkerStyle(21);
   gr->SetMarkerSize(1.2);

   TH1F *hgr = new TH1F("Graph-CPU"," CPU speed-up", nbins*4,0,nbins+1);
   hgr->SetMaximum(ymx + (ymx-ymi)*0.2);
   hgr->SetMinimum(0);
   hgr->SetDirectory(0);
   hgr->SetStats(0);
//   hgr->CenterTitle(true);
   hgr->GetXaxis()->SetTitle(pf->GetXaxis()->GetTitle());
   hgr->GetXaxis()->CenterTitle(true);
   hgr->GetXaxis()->SetLabelSize(0.05);
   hgr->GetXaxis()->SetTitleSize(0.06);
   hgr->GetXaxis()->SetTitleOffset(0.62);
   hgr->GetYaxis()->SetLabelSize(0.06);
   gr->SetHistogram(hgr);

   gr->Print();
   
   gr->Draw("alp");
  
   fout->Close();
}

//______________________________________________________________________________
Int_t TProofBench::RunDataSet(const char *dset,
                              Int_t start, Int_t stop, Int_t step)
{
   // Perform a test using dataset 'dset'
   // Return 0 on success, -1 on error
   // Open the file for the results

   if (OpenOutFile(kTRUE) != 0) {
      Error("RunDataSet", "problems opening '%s' to save the result", fOutFileName.Data());
      return -1;
   }
   fUnlinkOutfile = kFALSE;

   ReleaseCache(dset);
   SafeDelete(fRunDS);
   TPBReadType *readType = fReadType;
   if (!readType) readType = new TPBReadType(TPBReadType::kReadOpt);
   fRunDS = new TProofBenchRunDataRead(fDS, readType, fOutFile); 
   if (!fDataSel.IsNull()) fRunDS->SetSelName(fDataSel);
   if (!fDataPar.IsNull()) fRunDS->SetParList(fDataPar);
   fRunDS->Run(dset, start, stop, step, fNtries, -1, -1);
   if (!fReadType) SafeDelete(readType);
   
   // Close the file
   if (SetOutFile(0) != 0)
      Warning("RunDataSet", "problems closing '%s'", fOutFileName.Data());
   
   // Done
   return 0;
}

//______________________________________________________________________________
Int_t TProofBench::RunDataSetx(const char *dset, Int_t start, Int_t stop)
{
   // Perform a test using dataset 'dset' scanning over the number of workers
   // per node.
   // Return 0 on success, -1 on error
   // Open the file for the results

   if (OpenOutFile(kTRUE) != 0) {
      Error("RunDataSetx", "problems opening '%s' to save the result", fOutFileName.Data());
      return -1;
   }
   fUnlinkOutfile = kFALSE;

   ReleaseCache(dset);
   SafeDelete(fRunDS);
   TPBReadType *readType = fReadType;
   if (!readType) readType = new TPBReadType(TPBReadType::kReadOpt);
   fRunDS = new TProofBenchRunDataRead(fDS, readType, fOutFile); 
   if (!fDataSel.IsNull()) fRunDS->SetSelName(fDataSel);
   if (!fDataPar.IsNull()) fRunDS->SetParList(fDataPar);
   fRunDS->Run(dset, start, stop, -2, fNtries, -1, -1);
   if (!fReadType) SafeDelete(readType);

   // Close the file
   if (SetOutFile(0) != 0)
      Warning("RunDataSetx", "problems closing '%s'", fOutFileName.Data());
   
   // Done
   return 0;
}

//______________________________________________________________________________
void TProofBench::DrawDataSet(const char *outfile, const char *opt, const char *type)
{
   // Draw the CPU speedup plot.
   //  opt =    'std:'          Standard scaling plot
   //           'norm:'         Normalized scaling plot
   //           'stdx:'         Standard scaling plot, 1 worker per node
   //           'normx:'        Normalized scaling plot, 1 worker per node
   // type =    'mbs'           MB/s scaling plots (default)
   //           'evts'          Event/s scaling plots
   //

   // Get the TProfile an create the graphs
   TFile *fout = TFile::Open(outfile, "READ");
   if (!fout || (fout && fout->IsZombie())) {
      ::Error("DrawDataSet", "could not open file '%s' ...", outfile);
      return;
   }
   TDirectory *d = (TDirectory *) fout->Get("RunDataRead");
   if (!d) {
      ::Error("DrawDataSet", "could not find directory 'RunDataRead' ...");
      fout->Close();
      delete fout;
      return;
   }
   d->cd();

   TString hprofn, typ("QR_IO");
   if (type && !strcmp(type, "evts")) typ = "QR_Evts";

   if (!strcmp(opt, "std:")) {
      hprofn.Form("Prof_DataRead_%s", typ.Data());
   } else if (!strcmp(opt, "stdx:")) {
      hprofn.Form("Prof_x_DataRead_%s", typ.Data());
   } else if (!strcmp(opt, "norm:")) {
      hprofn.Form("Norm_DataRead_%s", typ.Data());
   } else if (!strcmp(opt, "absx:")) {
      hprofn.Form("Norm_x_DataRead_%s", typ.Data());
   } else {
      ::Error("DrawDataSet", "unknown option '%s'", opt);
      fout->Close();
      delete fout;
      return;
   }
   TProfile *pf = 0;
   TList *keylist = d->GetListOfKeys();
   TKey *key = 0;
   TIter nxk(keylist);
   while ((key = (TKey *) nxk())) {
      if (TString(key->GetName()).BeginsWith(hprofn)) {
         pf = (TProfile *) d->Get(key->GetName());
         break;
      }
   }
   if (!pf) {
      ::Error("DrawDataSet", "could not find '%s' ...", hprofn.Data());
      fout->Close();
      delete fout;
      return;
   }
   Int_t nbins = pf->GetNbinsX();
   TGraphErrors *gr = new TGraphErrors(nbins);
   Double_t xx, ex, yy, ey, ymi = pf->GetBinContent(1), ymx = ymi;
   Int_t k =1;
   for (;k <= nbins; k++) {
      xx = pf->GetBinCenter(k);
      ex = pf->GetBinWidth(k) * .001;
      yy = pf->GetBinContent(k);
      ey = pf->GetBinError(k);
      if (k == 1) {
         ymi = yy;
         ymx = yy;
      } else {
         if (yy < ymi) ymi = yy; 
         if (yy > ymx) ymx = yy;
      }
      gr->SetPoint(k-1, xx, yy);
      gr->SetPointError(k-1, ex, ey);
      Printf("%d %f %f", (Int_t)xx, yy, ey);
   }

   // Create the canvas
   TCanvas *cpu = new TCanvas("dataset", "Rate vs wrks",204,69,1050,502);
   cpu->Range(-3.106332,0.7490716,28.1362,1.249867);

   gStyle->SetOptTitle(0);
   gr->SetFillColor(1);
   gr->SetLineColor(13);
   gr->SetMarkerStyle(21);
   gr->SetMarkerSize(1.2);

   TH1F *hgr = new TH1F("Graph-DataSet"," Data Read speed-up", nbins*4,0,nbins+1);
   hgr->SetMaximum(ymx + (ymx-ymi)*0.2);
   hgr->SetMinimum(0);
   hgr->SetDirectory(0);
   hgr->SetStats(0);
//   hgr->CenterTitle(true);
   hgr->GetXaxis()->SetTitle(pf->GetXaxis()->GetTitle());
   hgr->GetXaxis()->CenterTitle(true);
   hgr->GetXaxis()->SetLabelSize(0.05);
   hgr->GetXaxis()->SetTitleSize(0.06);
   hgr->GetXaxis()->SetTitleOffset(0.62);
   hgr->GetYaxis()->SetLabelSize(0.06);
   gr->SetHistogram(hgr);

   gr->Print();
   
   gr->Draw("alp");
  
   fout->Close();
}

//______________________________________________________________________________
Int_t TProofBench::ReleaseCache(const char *dset)
{
   // Release memory cache for dataset 'dset'
   // Return 0 on success, -1 on error

   // Do it via the dataset handler
   if (!fDS) fDS = new TProofBenchDataSet(fProofDS);
   return fDS ? fDS->ReleaseCache(dset) : -1;
}

//______________________________________________________________________________
Int_t TProofBench::RemoveDataSet(const char *dset)
{
   // Physically remove the dataset 'dset', i.e. remove the dataset and the files
   // it describes
   // Return 0 on success, -1 on error

   // Do it via the dataset handler
   if (!fDS) fDS = new TProofBenchDataSet(fProofDS);
   return fDS ? fDS->RemoveFiles(dset) : -1;
}

//______________________________________________________________________________
Int_t TProofBench::MakeDataSet(const char *dset, Long64_t nevt, const char *fnroot)
{
   // Create the largest dataset for the run.
   // Defaults for
   //          dataset name, filename root
   // are
   //          "BenchDataSet", "event"
   // respectively.
   // Default selecor is TSelEventGen. Use SetDataGenSel and SetDataGenOar to change it
   // and to pass the list of PARs defining the alternative selector.
   // These can be changed via dset, sel and fnroot, respectively.
   // The argument 'nevt' controls the number of events per file (-1 for the default,
   // which is 30000).
   // Return 0 on success, -1 on error

   if (dset && strlen(dset) > 0) fDataSet = dset;

   // Load the selector, if needed
   if (!TClass::GetClass(fDataGenSel)) {
      // Is it the default selector?
      if (fDataGenSel == kPROOF_BenchSelDataGenDef) {
         // Load the parfile
         TString par = TString::Format("%s%s.par", kPROOF_BenchSrcDir, kPROOF_BenchDataSelPar);
         Info("MakeDataSet", "uploading '%s' ...", par.Data());
         if (fProof->UploadPackage(par) != 0) {
            Error("MakeDataSet", "problems uploading '%s' - cannot continue", par.Data());
            return -1;
         }
         Info("MakeDataSet", "enabling '%s' ...", kPROOF_BenchDataSelPar);
         if (fProof->EnablePackage(kPROOF_BenchDataSelPar) != 0) {
            Error("MakeDataSet", "problems enabling '%s' - cannot continue", kPROOF_BenchDataSelPar);
            return -1;
         }
      } else {
         if (fDataGenPar.IsNull()) {
            Error("MakeDataSet", "you should load the class '%s' before running the benchmark", fDataGenSel.Data());
            return -1;
         } else {
            TString par;
            Int_t from = 0;
            while (fDataGenPar.Tokenize(par, from, ",")) {
               Info("MakeDataSet", "Uploading '%s' ...", par.Data());
               if (fProof->UploadPackage(par) != 0) {
                  Error("MakeDataSet", "problems uploading '%s' - cannot continue", par.Data());
                  return -1;
               }
               Info("MakeDataSet", "Enabling '%s' ...", par.Data());
               if (fProof->EnablePackage(par) != 0) {
                  Error("MakeDataSet", "problems enabling '%s' - cannot continue", par.Data());
                  return -1;
               }
            }
         }
      }
      // Check
      if (!TClass::GetClass(fDataGenSel)) {
         Error("MakeDataSet", "failed to load '%s'", fDataGenSel.Data());
         return -1;
      }
   }

   // For files, 30000 evst each (about 600 MB total) per worker
   Int_t files_per_node = 4;
   TString fn, fnr = (fnroot && strlen(fnroot) > 0) ? fnroot : "event";
   TProofNodes pn(fProof);
   TMap *filesmap = new TMap;
   TMap *nodesmap = pn.GetMapOfNodes();
   TIter nxnd(nodesmap);
   TList *wli = 0;
   TObject *obj = 0;
   Int_t kf = 1;
   while ((obj = nxnd()) != 0) {
      if ((wli = dynamic_cast<TList *>(nodesmap->GetValue(obj)))) {
         THashList *fli = new THashList;
         Int_t nf = wli->GetSize() * files_per_node;
         TSlaveInfo *wi = (TSlaveInfo *) wli->First();
         while (nf--) {
            fn.Form("%s-%s-%d.root", fnr.Data(), wi->GetName(), kf++);
            // Add to the node list for generation
            fli->Add(new TObjString(fn));
         }
         filesmap->Add(new TObjString(obj->GetName()), fli);
      }
   }
   filesmap->Print();
   // Prepare for file generation ... add map in the input list
   filesmap->SetName("PROOF_FilesToProcess");
   fProof->AddInput(filesmap);

   // Set parameters for processing
   TString oldpack;
   if (TProof::GetParameter(fProof->GetInputList(), "PROOF_Packetizer", oldpack) != 0) oldpack = "";
   fProof->SetParameter("PROOF_Packetizer", "TPacketizerFile");
   Int_t oldnotass = -1;
   if (TProof::GetParameter(fProof->GetInputList(), "PROOF_ProcessNotAssigned", oldnotass) != 0) oldnotass = -1;
   fProof->SetParameter("PROOF_ProcessNotAssigned", (Int_t)0);

   // Process
   Long64_t ne = (nevt > 0) ? nevt : 30000;
   fProof->SetParameter("PROOF_BenchmarkNEvents", ne);
   fProof->Process(fDataGenSel, (Long64_t) 1);
   fProof->DeleteParameters("PROOF_BenchmarkNEvents");

   // Restore parameters
   if (!oldpack.IsNull())
      fProof->SetParameter("PROOF_Packetizer", oldpack);
   else
      fProof->DeleteParameters("PROOF_Packetizer");
   if (oldnotass != -1)
      fProof->SetParameter("PROOF_ProcessNotAssigned", oldnotass);
   else
      fProof->DeleteParameters("PROOF_ProcessNotAssigned");
   
   // Cleanup
   fProof->GetInputList()->Remove(filesmap);
   filesmap->SetOwner(kTRUE);
   delete filesmap;

   // The dataset to be registered in the end with proper port
   TFileCollection *fc = new TFileCollection("dum", "dum");

   fProof->GetOutputList()->Print();
   TIter nxout(fProof->GetOutputList());
   while ((obj = nxout())) {
      TList *fli = dynamic_cast<TList *>(obj);
      if (fli && TString(fli->GetName()).BeginsWith("PROOF_FilesGenerated_")) {
         TIter nxfg(fli);
         TFileInfo *fi = 0;
         while ((fi = (TFileInfo *) nxfg()))
            fc->Add(fi);
         fli->SetOwner(kFALSE);
      }
   }
   fc->Print("F");
   // Register the dataset, overwriting any existing dataset wth the same name and verifying it
   if (!(fProof->RegisterDataSet(fDataSet, fc, "OV")))
      Warning("MakeDataSet", "problems registering and verifying '%s'", dset);
   delete fc;
   
   // Done
   return 0;
}

//______________________________________________________________________________
Int_t TProofBench::CopyDataSet(const char *dset, const char *dsetdst, const char *destdir)
{
   // Copy the files of dataset 'dset' to 'destdir' and create a new dataset named 'dsetdst'
   // decribing them.
   // Return 0 on success, -1 on error

   // Make some checks
   if (!fProof) {
      Error("CopyDataSet", "no PROOF found - cannot continue");
      return -1;
   }
   if (!dset || (dset && !fProof->ExistsDataSet(dset))) {
      Error("CopyDataSet", "dataset '%s' does not exist", dset);
      return -1;
   }
   if (!dsetdst || (dsetdst && fProof->ExistsDataSet(dsetdst))) {
      if (isatty(0) != 0 && isatty(1) != 0) {
         Printf("Target dataset '%s' exists already:"
                                          " do you want to remove it first?", dsetdst);
         char *a = Getline("[Y,n] ");
         Printf("a: %s", a);
         if (a[0] == 'Y' || a[0] == 'y' || a[0] == '\n') {
            Info("CopyDataSet", "removing dataset '%s' ...", dsetdst);
            RemoveDataSet(dsetdst);
         } else {
            return -1;
         }
      } else {
         Error("CopyDataSet", "destination dataset '%s' does already exist: remove it first", dsetdst);
         return -1;
      }
   }

   // The TFileCollection object for the new dataset
   TFileCollection *fc = fProof->GetDataSet(dset);
   if (!fc) {
      Error("CopyDataSet", "problems retrieving TFileCollection for dataset '%s'", dset);
      return -1;
   }
   TFileCollection *fcn = new TFileCollection(dsetdst, "");
   TString fn;
   TFileInfo *fi = 0;
   TIter nxfi(fc->GetList());
   while ((fi = (TFileInfo *) nxfi())) {
      fn.Form("%s/%s", destdir, gSystem->BaseName(fi->GetCurrentUrl()->GetFile()));
      Info("CopyDataSet", "adding info for file '%s'", fn.Data());
      fcn->Add(new TFileInfo(fn));
   }
   delete fc;
      
   // Do it via the dataset handler
   if (!fDS) fDS = new TProofBenchDataSet(fProofDS);
   if (fDS->CopyFiles(dset, destdir) != 0) {
      Error("CopyDataSet", "problems copying files of dataset '%s' to dest dir '%s'", dset, destdir);
      delete fcn;
      return -1;
   }

   // Register the new dataset, overwriting any existing dataset wth the same name and verifying it
   Int_t rc = 0;
   if (!(fProof->RegisterDataSet(dsetdst, fcn, "OV"))) {
      Error("CopyDataSet", "problems registering and verifying '%s'", dsetdst);
      rc = -1;
   }
   delete fcn;

   // Done
   return rc;
}

//______________________________________________________________________________
void TProofBench::SetProofDS(TProof *pds)
{
   // Set the PROOF instance to be used for dataset operations, like releasing
   // cache ...
   // Use SetProofDS(0) to reset and using the default PROOF
   
   if (pds && !pds->IsValid()) {
      Error("SetProofDS", "trying to set an invalid PROOF instance");
      return;
   }
   fProofDS = pds ? pds : fProof;
   if (fProofDS) {
      SafeDelete(fDS);
      fDS = new TProofBenchDataSet(fProofDS);
   }
   // Done
   return;
}

