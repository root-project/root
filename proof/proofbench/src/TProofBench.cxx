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

#include "RConfigure.h"

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
#include "TSortedList.h"
#include "TTimeStamp.h"
#include "TUrl.h"

#include "TCanvas.h"
#include "TF1.h"
#include "TGraphErrors.h"
#include "TH1F.h"
#include "TMath.h"
#include "TProfile.h"
#include "TStyle.h"

ClassImp(TProofBench)

// Functions for fitting

TF1 *TProofBench::fgFp1 = 0;
TF1 *TProofBench::fgFp1n = 0;
TF1 *TProofBench::fgFp2 = 0;
TF1 *TProofBench::fgFp2n = 0;

//_____________________________________________________________________
Double_t funp1(Double_t *xx, Double_t *par)
{
   // Simple polynomial 1st degree
   
   Double_t res = par[0] + par[1] * xx[0];
   return res;
}

//_____________________________________________________________________
Double_t funp2(Double_t *xx, Double_t *par)
{
   // Simple polynomial 2nd degree
   
   Double_t res = par[0] + par[1] * xx[0] + par[2] * xx[0] * xx[0];
   return res;
}

//_____________________________________________________________________
Double_t funp1n(Double_t *xx, Double_t *par)
{
   // Normalized 1st degree
   
   Double_t res = par[0] / xx[0] + par[1];
   return res;
}

//_____________________________________________________________________
Double_t funp2n(Double_t *xx, Double_t *par)
{
   // Normalized 2nd degree
   
   Double_t res = par[0] / xx[0] + par[1] + par[2] * xx[0];
   return res;
}

//______________________________________________________________________________
TProofBench::TProofBench(const char *url, const char *outfile, const char *proofopt)
            : fUnlinkOutfile(kFALSE), fProofDS(0), fOutFile(0),
              fNtries(4), fHistType(0), fNHist(16), fReadType(0),
              fDataSet("BenchDataSet"), fNFilesWrk(4),
              fDataGenSel(kPROOF_BenchSelDataGenDef),
              fRunCPU(0), fRunDS(0), fDS(0), fDebug(kFALSE), fDescription(0)
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
   // Get the size of the cluster
   fNumWrkMax = fProof->GetParallel();
   if (fProof->UseDynamicStartup() && TProof::GetEnvVars()) {
      // It must be passed as PROOF option 'workers=N' and recorded in the envs vars
      TNamed *n = (TNamed *) TProof::GetEnvVars()->FindObject("PROOF_NWORKERS");
      if (!n) {
         Error("TProofBench", "dynamic mode: you must specify the max number of workers");
         fProof->Close();
         SafeDelete(fProof);
         return;
      }
      TString sn(n->GetTitle());
      if (sn.IsDigit()) fNumWrkMax = sn.Atoi();
      if (!sn.IsDigit()) {
         Error("TProofBench", "dynamic mode: wrong specification of the max number of"
                              " workers ('%s')", n->GetTitle());
         fProof->Close();
         SafeDelete(fProof);
         return;
      }
   }
   if (fNumWrkMax <= 0) {
      Error("TProofBench", "wrong max number of workers ('%d')", fNumWrkMax);
      fProof->Close();
      SafeDelete(fProof);
      return;
   }
   // By default we use the same instance for dataset actions
   fProofDS = fProof;
   // The object is now valid
   ResetBit(kInvalidObject);
   // Identifying string
   TUrl u(url);
   TString host(TString::Format("PROOF at %s", u.GetHost()));
   if (!strcmp(u.GetProtocol(), "lite")) host.Form("PROOF-Lite on %s", gSystem->HostName());
   fDescription = new TNamed("PB_description",
                             TString::Format("%s, %d workers", host.Data(), fNumWrkMax).Data());
   Printf(" Run description: %s", fDescription->GetTitle());
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
   SafeDelete(fDescription);
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
      if (fOutFile) {
         gROOT->GetListOfFiles()->Remove(fOutFile);
         if (!strcmp(mode, "RECREATE")) {
            // Save the description string
            fOutFile->cd();
            fDescription->Write();
         }
      }
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
                        fProof->GetMaster(), lite, fNumWrkMax,
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
void TProofBench::CloseOutFile()
{
   // Close output file

   if (SetOutFile(0) != 0)
      Warning("CloseOutFile", "problems closing '%s'", fOutFileName.Data());
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
   fRunCPU->Run(nevents, start, stop, step, fNtries, fDebug, -1);

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
   fRunCPU->Run(nevents, start, stop, -2, fNtries, fDebug, -1);

   // Close the file
   if (SetOutFile(0) != 0)
      Warning("RunCPUx", "problems closing '%s'", fOutFileName.Data());

   // Done
   return 0;
}

//______________________________________________________________________________
void TProofBench::DrawCPU(const char *outfile, const char *opt, Bool_t verbose, Int_t dofit)
{
   // Draw the CPU speedup plot.
   //  opt =    'std:'      draw standard evt/s plot
   //           'stdx:'     draw standard evt/s plot, 1 worker per node
   //           'norm:'     draw normalized plot
   //           'normx:'    draw normalized plot, 1 worker per node
   //  dofit =  0           no fit
   //           1           fit with the relevant '1st degree related' function
   //           2           fit with the relevant '2nd degree related' function
   //

   // Get the TProfile an create the graphs
   TFile *fout = TFile::Open(outfile, "READ");
   if (!fout || (fout && fout->IsZombie())) {
      ::Error("DrawCPU", "could not open file '%s' ...", outfile);
      return;
   }
   
   // Get description
   TString description("<not available>");
   TNamed *nmdesc = (TNamed *) fout->Get("PB_description");
   if (nmdesc) description = nmdesc->GetTitle();

   TString oo(opt);
   const char *dirn = (oo.Contains("x:")) ? "RunCPUx" : "RunCPU";
   TDirectory *d = (TDirectory *) fout->Get(dirn);
   if (!d) {
      ::Error("DrawCPU", "could not find directory 'RunCPU' ...");
      fout->Close();
      delete fout;
      return;
   }
   d->cd();
   TString hprofn;
   Bool_t isnorm = kFALSE;
   if (!strcmp(opt, "std:")) {
      hprofn = "Prof_CPU_QR_Evts";
   } else if (!strcmp(opt, "stdx:")) {
      hprofn = "Prof_x_CPU_QR_Evts";
   } else if (!strcmp(opt, "norm:")) {
      hprofn = "Norm_CPU_QR_Evts";
      isnorm = kTRUE;
    } else if (!strcmp(opt, "normx:")) {
      hprofn = "Norm_x_CPU_QR_Evts";
      isnorm = kTRUE;
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
   Double_t xlow = pf->GetBinCenter(1) - pf->GetBinWidth(1)/2. ;
   Double_t xhigh = pf->GetBinCenter(nbins) + pf->GetBinWidth(nbins)/2. ;
   Int_t k =1, kmx = -1;
   for (;k <= nbins; k++) {
      xx = pf->GetBinCenter(k);
      ex = pf->GetBinWidth(k) * .001;
      yy = pf->GetBinContent(k);
      ey = pf->GetBinError(k);
      if (k == 1) {
         ymi = yy;
         ymx = yy;
         kmx = k;
      } else {
         if (yy < ymi) ymi = yy; 
         if (yy > ymx) { ymx = yy; kmx = k; }
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

   TH1F *hgr = new TH1F("Graph-CPU"," CPU speed-up", nbins*4, xlow, xhigh);
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

   if (verbose) gr->Print();
   
   gr->Draw("alp");

   if (dofit > 0) {
      // Make sure the fitting functions are defined
      Double_t xmi = 0.9;
      if (nbins > 5) xmi = 1.5;
      AssertFittingFun(xmi, nbins + .1);

      // Starting point for the parameters and fit
      Double_t normrate = -1.;
      if (dofit == 1) {
         if (isnorm) {
            fgFp1n->SetParameter(0, pf->GetBinContent(1));
            fgFp1n->SetParameter(1, pf->GetBinContent(nbins-1));
            gr->Fit(fgFp1n);
            if (verbose) fgFp1n->Print();
            normrate = fgFp1n->GetParameter(1);
         } else {
            fgFp1->SetParameter(0, 0.);
            fgFp1->SetParameter(1, pf->GetBinContent(1));
            gr->Fit(fgFp1);
            if (verbose) fgFp1->Print();
            normrate = fgFp1->GetParameter(1);
         }
      } else {
         if (isnorm) {
            fgFp2n->SetParameter(0, pf->GetBinContent(1));
            fgFp2n->SetParameter(1, pf->GetBinContent(nbins-1));
            fgFp2n->SetParameter(2, 0.);
            gr->Fit(fgFp2n);
            if (verbose) fgFp2n->Print();
            normrate = fgFp2n->GetParameter(1);
         } else {
            fgFp2->SetParameter(0, 0.);
            fgFp2->SetParameter(1, pf->GetBinContent(1));
            fgFp2->SetParameter(2, 0.);
            gr->Fit(fgFp2);
            if (verbose) fgFp2->Print();
            normrate = fgFp2->GetParameter(1);
         }
      }

      // Notify the cluster performance parameters
      if (!isnorm) {
         printf("* ************************************************************ *\n");
         printf("*                                                              *\r");
         printf("* Cluster: %s\n", description.Data());
         printf("* Performance measurement from scalability plot:               *\n");
         printf("*                                                              *\r");
         printf("*    rate max:         %.3f\tmegaRNGPS (@ %d workers)\n", ymx/1000000, kmx);
         printf("*                                                              *\r");
         printf("*    per-worker rate:  %.3f\tmegaRNGPS \n", normrate/1000000);
         printf("* ************************************************************ *\n");
      } else {
         printf("* ************************************************************ *\n");
         printf("*                                                              *\r");
         printf("* Cluster: %s\n", description.Data());
         printf("*                                                              *\r");
         printf("* Per-worker rate from normalized plot:  %.3f\tmegaRNGPS\n", normrate/1000000);
         printf("* ************************************************************ *\n");
      }
   }
   // Close the file
   fout->Close();
}

//______________________________________________________________________________
void TProofBench::AssertFittingFun(Double_t mi, Double_t mx)
{
   // Make sure that the fitting functions are defined

   if (!fgFp1) {
      fgFp1 = new TF1("funp1", funp1, mi, mx, 2);
      fgFp1->SetParNames("offset", "slope");
   }

   if (!fgFp1n) {
      fgFp1n = new TF1("funp1n", funp1n, mi, mx, 2);
      fgFp1n->SetParNames("decay", "norm rate");
   }

   if (!fgFp2) {
      fgFp2 = new TF1("funp2", funp2, mi, mx, 3);
      fgFp2->SetParNames("offset", "slope", "deviation");
   }

   if (!fgFp2n) {
      fgFp2n = new TF1("funp2n", funp2n, mi, mx, 3);
      fgFp2n->SetParNames("decay", "norm rate", "deviation");
   }

}

//______________________________________________________________________________
class fileDesc : public TNamed {
public:
   Long_t  fMtime; // Modification time
   TString fDesc; // Test description string, if any
   fileDesc(const char *n, const char *o,
            Long_t t, const char *d) : TNamed(n, o), fMtime(t), fDesc(d) { }
   Int_t   Compare(const TObject *o) const {
      const fileDesc *fd = static_cast<const fileDesc *>(o);
      if (!fd || (fd && fd->fMtime == fMtime)) return 0;
      if (fMtime < fd->fMtime) return -1;
      return 1;
   }
};

//______________________________________________________________________________
void TProofBench::GetPerfSpecs(const char *path, Int_t degfit)
{
   // Get performance specs. Check file 'path', or files in directory 'path'
   // (default current directory).
   // The degree of the polynomial used for the fit is 'degfit' (default 1). 
   
   // Locate the file (ask if many)
   TString pp(path), fn, oo;
   if (pp.IsNull()) pp = gSystem->WorkingDirectory();
   FileStat_t st;
   if (gSystem->GetPathInfo(pp.Data(), st) != 0) {
      ::Error("TProofBench::GetPerfSpecs", "path '%s' could not be stat'ed - abort", pp.Data());
      return;
   }
   TSortedList filels;
   if (R_ISDIR(st.fMode)) {
      // Scan the directory
      void *dirp = gSystem->OpenDirectory(pp.Data());
      if (!dirp) {
         ::Error("TProofBench::GetPerfSpecs", "directory path '%s' could nto be open - abort", pp.Data());
         return;
      }
      const char *ent = 0;
      while ((ent = gSystem->GetDirEntry(dirp))) {
         if (!strcmp(ent, ".") || !strcmp(ent, "..")) continue;
         fn.Form("%s/%s", pp.Data(), ent);
         if (gSystem->GetPathInfo(fn.Data(), st) != 0) continue;
         if (!R_ISREG(st.fMode)) continue;
         fn += "?filetype=raw";
         TFile *f = TFile::Open(fn);
         if (!f) continue;
         char rr[5] = {0};
         if (!f->ReadBuffer(rr, 4)) {
            if (!strncmp(rr, "root", 4)) {
               SafeDelete(f);
               fn.ReplaceAll("?filetype=raw", "");
               if ((f = TFile::Open(fn))) {
                  TString desc("<no decription>");
                  TNamed *nmdesc = (TNamed *) f->Get("PB_description");
                  if (nmdesc) desc = nmdesc->GetTitle();
                  if (f->GetListOfKeys()->FindObject("RunCPU"))
                     filels.Add(new fileDesc(fn, "std:", st.fMtime, desc.Data()));
                  if (f->GetListOfKeys()->FindObject("RunCPUx"))
                     filels.Add(new fileDesc(fn, "stdx:", st.fMtime, desc.Data()));
               } else {
                  ::Warning("TProofBench::GetPerfSpecs", "problems opening '%s'", fn.Data());
               }
            }
         }
         SafeDelete(f);
      }
   } else if (!R_ISREG(st.fMode)) {
      ::Error("TProofBench::GetPerfSpecs",
              "path '%s' not a regular file nor a directory - abort", pp.Data());
      return;
   } else {
      // This is the file
      fn = pp;
      // Check it
      TString emsg;
      Bool_t isOk = kFALSE;
      if (gSystem->GetPathInfo(fn.Data(), st) == 0) {
         fn += "?filetype=raw";
         TFile *f = TFile::Open(fn);
         if (f) {
            char rr[5] = {0};
            if (!(f->ReadBuffer(rr, 4))) {
               if (!strncmp(rr, "root", 4)) {
                  fn.ReplaceAll("?filetype=raw", "");
                  if ((f = TFile::Open(fn))) {
                     if (f->GetListOfKeys()->FindObject("RunCPU")) oo = "std:";
                     if (f->GetListOfKeys()->FindObject("RunCPUx")) oo = "stdx:";
                     SafeDelete(f);
                     if (!oo.IsNull()) {
                        isOk = kTRUE;
                     } else {
                        emsg.Form("path '%s' does not contain the relevant dirs - abort", fn.Data());
                     }
                  } else {
                     emsg.Form("path '%s' cannot be open - abort", fn.Data());
                  }
               } else {
                  emsg.Form("'%s' is not a ROOT file - abort", fn.Data());
               }
            } else { 
               emsg.Form("could not read first 4 bytes from '%s' - abort", fn.Data());
            }
            SafeDelete(f);
         } else {
            emsg.Form("path '%s' cannot be open in raw mode - abort", fn.Data());
         }
      } else {
         emsg.Form("path '%s' cannot be stated - abort", fn.Data());
      }
      if (!isOk) {
         ::Error("TProofBench::GetPerfSpecs", "%s", emsg.Data());
         return;
      }
   }

   fileDesc *nm = 0;
   // Ask the user, if more then 1
   if (filels.GetSize() == 1) {
      nm = (fileDesc *) filels.First();
      fn = nm->GetName();
      oo = nm->GetTitle();
   } else if (filels.GetSize() > 1) {
      TIter nxf(&filels);
      Int_t idx = 0;
      Printf("Several possible files found:"); 
      while ((nm = (fileDesc *) nxf())) {
         Printf("  %d\t%s\t%s\t%s (file: %s)", idx++, nm->GetTitle(),
                TTimeStamp(nm->fMtime).AsString("s"), nm->fDesc.Data(), nm->GetName());
      }
      TString a(Getline(TString::Format("Make your choice [%d] ", idx-1)));
      if (a.IsNull() || a[0] == '\n') a.Form("%d", idx-1);
      idx = a.Atoi();
      if ((nm = (fileDesc *) filels.At(idx))) {
         fn = nm->GetName();
         oo = nm->GetTitle();
      } else {
         ::Error("TProofBench::GetPerfSpecs", "chosen index '%d' does not exist - abort", idx);
         return;
      }
   } else {
      if (fn.IsNull()) {
         ::Error("TProofBench::GetPerfSpecs",
                 "path '%s' is a directory but no ROOT file found in it - abort", pp.Data());
         return;
      }
   }

   // Now get the specs
   TProofBench::DrawCPU(fn.Data(), oo.Data(), kFALSE, degfit);
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
   fRunDS->Run(dset, start, stop, step, fNtries, fDebug, -1);
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
   fRunDS->Run(dset, start, stop, -2, fNtries, fDebug, -1);
   if (!fReadType) SafeDelete(readType);

   // Close the file
   if (SetOutFile(0) != 0)
      Warning("RunDataSetx", "problems closing '%s'", fOutFileName.Data());
   
   // Done
   return 0;
}

//______________________________________________________________________________
void TProofBench::DrawDataSet(const char *outfile,
                              const char *opt, const char *type, Bool_t verbose)
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
   TString oo(opt);
   const char *dirn = (oo.Contains("x:")) ? "RunDataReadx" : "RunDataRead";
   TDirectory *d = (TDirectory *) fout->Get(dirn);
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
   } else if (!strcmp(opt, "normx:")) {
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
   Double_t xlow = pf->GetBinCenter(1) - pf->GetBinWidth(1)/2. ;
   Double_t xhigh = pf->GetBinCenter(nbins) + pf->GetBinWidth(nbins)/2. ;
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

   TH1F *hgr = new TH1F("Graph-DataSet"," Data Read speed-up", nbins*4, xlow, xhigh);
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

   if (verbose) gr->Print();
   
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
Int_t TProofBench::MakeDataSet(const char *dset, Long64_t nevt, const char *fnroot,
                               Bool_t regenerate)
{
   // Create the largest dataset for the run.
   // Defaults for
   //          dataset name, filename root
   // are
   //          "BenchDataSet", "event"
   // respectively.
   // These can be changed via dset and fnroot, respectively.
   // The string 'fnroot' defines the location of the files, interpreted as an URL.
   // Examples:
   //          fnroot             files
   //          'event'            <datadir>/event_<ord>_<#>.root
   //          '/mss/event'       /mss/event_<ord>_<#>.root
   //          'root://srv//mss/event?remote=1'
   //                             root://srv//mss/event_<ord>_<#>?remote=1.root          
   // Default selector is TSelEventGen. Use SetDataGenSel and SetDataGenPar to change it
   // and to pass the list of PARs defining the alternative selector.
   // The argument 'nevt' controls the number of events per file (-1 for the default,
   // which is 30000).
   // Return 0 on success, -1 on error

   if (dset && strlen(dset) > 0) fDataSet = dset;

   // Load the selector, if needed
   if (!TClass::GetClass(fDataGenSel)) {
      // Is it the default selector?
      if (fDataGenSel == kPROOF_BenchSelDataGenDef) {
         // Load the parfile
#ifdef R__HAVE_CONFIG
         TString par = TString::Format("%s/%s%s.par", ROOTETCDIR, kPROOF_BenchParDir, kPROOF_BenchDataSelPar);
#else
         TString par = TString::Format("$ROOTSYS/etc/%s%s.par", kPROOF_BenchParDir, kPROOF_BenchDataSelPar);
#endif
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
         }
      }
      // Load additional PAR files, if any or required by the alternative selector
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
      // Check
      if (!TClass::GetClass(fDataGenSel)) {
         Error("MakeDataSet", "failed to load '%s'", fDataGenSel.Data());
         return -1;
      }
   }

   // For files, 30000 evst each (about 600 MB total) per worker
   TString fn, fnr("event");
   Bool_t remote = kFALSE;
   if (fnroot && strlen(fnroot) > 0) {
      TUrl ur(fnroot, kTRUE);
      if (!strcmp(ur.GetProtocol(), "file") &&
         !gSystem->IsAbsoluteFileName(ur.GetFile())) {
         fnr = fnroot;
      } else {
         fnr = gSystem->BaseName(ur.GetFile());
         // We need to set the basedir
         TString bdir(gSystem->DirName(fnroot));
         bdir += "/<fn>";
         fProof->SetParameter("PROOF_BenchmarkBaseDir", bdir.Data());
         // Flag as remote, if so
         if (strcmp(ur.GetProtocol(), "file")) remote = kTRUE;
      }
   }
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
         Int_t nf = wli->GetSize() * fNFilesWrk;
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
   fProof->SetParameter("PROOF_BenchmarkRegenerate", Int_t(regenerate));
   fProof->Process(fDataGenSel, (Long64_t) 1);
   fProof->DeleteParameters("PROOF_BenchmarkNEvents");
   fProof->DeleteParameters("PROOF_BenchmarkRegenerate");
   fProof->DeleteParameters("PROOF_BenchmarkBaseDir");

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
   if (fProof->GetInputList()) fProof->GetInputList()->Remove(filesmap);
   filesmap->SetOwner(kTRUE);
   delete filesmap;

   // The dataset to be registered in the end with proper port
   TFileCollection *fc = new TFileCollection("dum", "dum");

   if (fProof->GetOutputList()) {
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
      // Register the new dataset, overwriting any existing dataset wth the same name
      // trusting the existing information
      fc->Update();
      if (fc->GetNFiles() > 0) {
         if (remote) fc->SetBit(TFileCollection::kRemoteCollection);
         if (!(fProof->RegisterDataSet(fDataSet, fc, "OT")))
            Warning("MakeDataSet", "problems registering '%s'", dset);
      } else {
         Warning("MakeDataSet", "dataset '%s' is empty!", dset);
      }
   } else {
      Warning("MakeDataSet", "PROOF output list is empty!");
   }

   SafeDelete(fc);

   // Get updated information
   fc = fProof->GetDataSet(fDataSet);
   if (fc) {
      fc->Print("F");
   } else {
      Warning("MakeDataSet", "dataset '%s' was not generated!", fDataSet.Data());
   }
   
   SafeDelete(fc);

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
         const char *a = Getline("[Y,n] ");
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

   // Register the new dataset, overwriting any existing dataset wth the same name
   // trusting the existing information
   Int_t rc = 0;
   if (!(fProof->RegisterDataSet(dsetdst, fcn, "OT"))) {
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

