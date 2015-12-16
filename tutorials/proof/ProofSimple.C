/// \file
/// \ingroup tutorial_ProofSimple
///
/// Selector to fill a set of histograms
///
/// \macro_code
///
/// \author Gerardo Ganis (gerardo.ganis@cern.ch)

#define ProofSimple_cxx

#include "ProofSimple.h"
#include <TCanvas.h>
#include <TFrame.h>
#include <TPaveText.h>
#include <TFormula.h>
#include <TF1.h>
#include <TH1F.h>
#include <TH3F.h>
#include <TMath.h>
#include <TRandom3.h>
#include <TString.h>
#include <TStyle.h>
#include <TSystem.h>
#include <TParameter.h>
#include <TSortedList.h>
#include "TProof.h"
#include <TFile.h>
#include <TProofOutputFile.h>
#include <TNtuple.h>
#include <TFileCollection.h>
#include <TFileInfo.h>
#include <THashList.h>

//_____________________________________________________________________________
ProofSimple::ProofSimple()
{
   // Constructor

   fNhist = -1;
   fHist = 0;
   fNhist3 = -1;
   fHist3 = 0;
   fRandom = 0;
   fHLab = 0;
   fFile = 0;
   fProofFile = 0;
   fNtp = 0;
   fHasNtuple = 0;
   fPlotNtuple = kFALSE;
}

//_____________________________________________________________________________
ProofSimple::~ProofSimple()
{
   // Destructor

   if (fFile) {
      SafeDelete(fNtp);
      SafeDelete(fFile);
   }
   SafeDelete(fRandom);
}

//_____________________________________________________________________________
void ProofSimple::Begin(TTree * /*tree*/)
{
   // The Begin() function is called at the start of the query.
   // When running with PROOF Begin() is only called on the client.
   // The tree argument is deprecated (on PROOF 0 is passed).

   TString option = GetOption();
   Ssiz_t iopt = kNPOS;

   // Histos array
   if (fInput->FindObject("ProofSimple_NHist")) {
      TParameter<Long_t> *p =
         dynamic_cast<TParameter<Long_t>*>(fInput->FindObject("ProofSimple_NHist"));
      fNhist = (p) ? (Int_t) p->GetVal() : fNhist;
   } else if ((iopt = option.Index("nhist=")) != kNPOS) {
      TString s;
      Ssiz_t from = iopt + strlen("nhist=");
      if (option.Tokenize(s, from, ";") && s.IsDigit()) fNhist = s.Atoi();
   }
   if (fNhist < 1) {
      Abort("fNhist must be > 0! Hint: proof->SetParameter(\"ProofSimple_NHist\","
            " (Long_t) <nhist>)", kAbortProcess);
      return;
   }

   if (fInput->FindObject("ProofSimple_NHist3")) {
      TParameter<Long_t> *p =
         dynamic_cast<TParameter<Long_t>*>(fInput->FindObject("ProofSimple_NHist3"));
      fNhist3 = (p) ? (Int_t) p->GetVal() : fNhist3;
   } else if ((iopt = option.Index("nhist3=")) != kNPOS) {
      TString s;
      Ssiz_t from = iopt + strlen("nhist3=");
      if (option.Tokenize(s, from, ";") && s.IsDigit()) fNhist3 = s.Atoi();
   }

   // Ntuple
   TNamed *nm = dynamic_cast<TNamed *>(fInput->FindObject("ProofSimple_Ntuple"));
   if (nm) {

      // Title is in the form
      //         merge                  merge via file
      //           |<fout>                      location of the output file if merge
      //           |retrieve                    retrieve to client machine
      //         dataset                create a dataset
      //           |<dsname>                    dataset name (default: dataset_ntuple)
      //         |plot                  for a final plot
      //         <empty> or other       keep in memory

      fHasNtuple = 1;

      TString ontp(nm->GetTitle());
      if (ontp.Contains("|plot") || ontp == "plot") {
         fPlotNtuple = kTRUE;
         ontp.ReplaceAll("|plot", "");
         if (ontp == "plot") ontp = "";
      }
      if (ontp.BeginsWith("dataset")) fHasNtuple = 2;
   }
}

//_____________________________________________________________________________
void ProofSimple::SlaveBegin(TTree * /*tree*/)
{
   // The SlaveBegin() function is called after the Begin() function.
   // When running with PROOF SlaveBegin() is called on each slave server.
   // The tree argument is deprecated (on PROOF 0 is passed).

   TString option = GetOption();
   Ssiz_t iopt = kNPOS;

   // Histos array
   if (fInput->FindObject("ProofSimple_NHist")) {
      TParameter<Long_t> *p =
         dynamic_cast<TParameter<Long_t>*>(fInput->FindObject("ProofSimple_NHist"));
      fNhist = (p) ? (Int_t) p->GetVal() : fNhist;
   } else if ((iopt = option.Index("nhist=")) != kNPOS) {
      TString s;
      Ssiz_t from = iopt + strlen("nhist=");
      if (option.Tokenize(s, from, ";") && s.IsDigit()) fNhist = s.Atoi();
   }
   if (fNhist < 1) {
      Abort("fNhist must be > 0! Hint: proof->SetParameter(\"ProofSimple_NHist\","
            " (Long_t) <nhist>)", kAbortProcess);
      return;
   }
   fHist = new TH1F*[fNhist];

   TString hn;
   // Create the histogram
   for (Int_t i=0; i < fNhist; i++) {
      hn.Form("h%d",i);
      fHist[i] = new TH1F(hn.Data(), hn.Data(), 100, -3., 3.);
      fHist[i]->SetFillColor(kRed);
      fOutput->Add(fHist[i]);
   }

   // 3D Histos array
   if (fInput->FindObject("ProofSimple_NHist3")) {
      TParameter<Long_t> *p =
         dynamic_cast<TParameter<Long_t>*>(fInput->FindObject("ProofSimple_NHist3"));
      fNhist3 = (p) ? (Int_t) p->GetVal() : fNhist3;
   } else if ((iopt = option.Index("nhist3=")) != kNPOS) {
      TString s;
      Ssiz_t from = iopt + strlen("nhist3=");
      if (option.Tokenize(s, from, ";") && s.IsDigit()) fNhist3 = s.Atoi();
   }
   if (fNhist3 > 0) {
      fHist3 = new TH3F*[fNhist3];
      Info("Begin", "%d 3D histograms requested", fNhist3);
      // Create the 3D histogram
      for (Int_t i=0; i < fNhist3; i++) {
         hn.Form("h%d_3d",i);
         fHist3[i] = new TH3F(hn.Data(), hn.Data(),
                              100, -3., 3., 100, -3., 3., 100, -3., 3.);
         fOutput->Add(fHist3[i]);
      }
   }

   // Histo with labels
   if (fInput->FindObject("ProofSimple_TestLabelMerging")) {
      fHLab = new TH1F("hlab", "Test merging of histograms with automatic labels", 10, 0., 10.);
      fOutput->Add(fHLab);
   }

   // Ntuple
   TNamed *nm = dynamic_cast<TNamed *>(fInput->FindObject("ProofSimple_Ntuple"));
   if (nm) {

      // Title is in the form
      //         merge                  merge via file
      //           |<fout>                      location of the output file if merge
      //           |retrieve                    retrieve to client machine
      //         dataset                create a dataset
      //           |<dsname>                    dataset name (default: dataset_ntuple)
      //         |plot                  for a final plot
      //         <empty> or other       keep in memory

      fHasNtuple = 1;

      TString ontp(nm->GetTitle());
      if (ontp.Contains("|plot") || ontp == "plot") {
         fPlotNtuple = kTRUE;
         ontp.ReplaceAll("|plot", "");
         if (ontp == "plot") ontp = "";
      }
      TString locfn("SimpleNtuple.root");
      if (ontp.BeginsWith("merge")) {
         ontp.Replace(0,5,"");
         fProofFile = new TProofOutputFile(locfn, "M");
         TString fn;
         Ssiz_t iret = ontp.Index("|retrieve");
         if (iret != kNPOS) {
            fProofFile->SetRetrieve(kTRUE);
            TString rettag("|retrieve");
            if ((iret = ontp.Index("|retrieve=")) != kNPOS) {
               rettag += "=";
               fn = ontp(iret + rettag.Length(), ontp.Length() - iret - rettag.Length());
               if ((iret = fn.Index('|')) != kNPOS) fn.Remove(iret);
               rettag += fn;
            }
            ontp.ReplaceAll(rettag, "");
         }
         Ssiz_t iof = ontp.Index('|');
         if (iof != kNPOS) ontp.Remove(0, iof + 1);
         if (!ontp.IsNull()) {
            fProofFile->SetOutputFileName(ontp.Data());
            if (fn.IsNull()) fn = gSystem->BaseName(TUrl(ontp.Data(), kTRUE).GetFile());
         }
         if (fn.IsNull()) fn = locfn;
         // This will be the final file on the client, the case there is one
         fProofFile->SetTitle(fn);
      } else if (ontp.BeginsWith("dataset")) {
         ontp.Replace(0,7,"");
         Ssiz_t iof = ontp.Index("|");
         if (iof != kNPOS) ontp.Remove(0, iof + 1);
         TString dsname = (!ontp.IsNull()) ? ontp.Data() : "dataset_ntuple";
         UInt_t opt = TProofOutputFile::kRegister | TProofOutputFile::kOverwrite | TProofOutputFile::kVerify;
         fProofFile = new TProofOutputFile("SimpleNtuple.root",
                                          TProofOutputFile::kDataset, opt, dsname.Data());
         fHasNtuple = 2;
      } else if (!ontp.IsNull()) {
         Warning("SlaveBegin", "ntuple options unknown: ignored (%s)", ontp.Data());
      }

      // Open the file, if required
      if (fProofFile) {
         // Open the file
         fFile = fProofFile->OpenFile("RECREATE");
         if (fFile && fFile->IsZombie()) SafeDelete(fFile);

         // Cannot continue
         if (!fFile) {
            Info("SlaveBegin", "could not create '%s': instance is invalid!", fProofFile->GetName());
            return;
         }
      }

      // Now we create the ntuple
      fNtp = new TNtuple("ntuple","Demo ntuple","px:py:pz:random:i");
      // File resident, if required
      if (fFile) {
         fNtp->SetDirectory(fFile);
         fNtp->AutoSave();
      } else {
         fOutput->Add(fNtp);
      }
   }

   // Set random seed
   fRandom = new TRandom3(0);
}

//_____________________________________________________________________________
Bool_t ProofSimple::Process(Long64_t entry)
{
   // The Process() function is called for each entry in the tree (or possibly
   // keyed object in the case of PROOF) to be processed. The entry argument
   // specifies which entry in the currently loaded tree is to be processed.
   // It can be passed to either ProofSimple::GetEntry() or TBranch::GetEntry()
   // to read either all or the required parts of the data. When processing
   // keyed objects with PROOF, the object is already loaded and is available
   // via the fObject pointer.
   //
   // This function should contain the "body" of the analysis. It can contain
   // simple or elaborate selection criteria, run algorithms on the data
   // of the event and typically fill histograms.
   //
   // The processing can be stopped by calling Abort().
   //
   // Use fStatus to set the return value of TTree::Process().
   //
   // The return value is currently not used.

   for (Int_t i=0; i < fNhist; i++) {
      if (fRandom && fHist[i]) {
         Double_t x = fRandom->Gaus(0.,1.);
         fHist[i]->Fill(x);
      }
   }
   for (Int_t i=0; i < fNhist3; i++) {
      if (fRandom && fHist3[i]) {
         Double_t x = fRandom->Gaus(0.,1.);
         fHist3[i]->Fill(x,x,x);
      }
   }
   if (fHLab && fRandom) {
      TSortedList sortl;
      Float_t rr[10];
      fRandom->RndmArray(10, rr);
      for (Int_t i=0; i < 10; i++) {
         sortl.Add(new TParameter<Int_t>(TString::Format("%f",rr[i]), i));
      }
      TIter nxe(&sortl);
      TParameter<Int_t> *pi = 0;
      while ((pi = (TParameter<Int_t> *) nxe())) {
         fHLab->Fill(TString::Format("hl%d", pi->GetVal()), pi->GetVal());
      }
   }
   if (fNtp) FillNtuple(entry);

   return kTRUE;
}

//_____________________________________________________________________________
void ProofSimple::FillNtuple(Long64_t entry)
{
   // The Process() function is called for each entry in the tree (or possibly
   // keyed object in the case of PROOF) to be processed. The entry argument
   // specifies which entry in the currently loaded tree is to be processed.
   // It can be passed to either ProofNtuple::GetEntry() or TBranch::GetEntry()
   // to read either all or the required parts of the data. When processing
   // keyed objects with PROOF, the object is already loaded and is available
   // via the fObject pointer.
   //
   // This function should contain the "body" of the analysis. It can contain
   // simple or elaborate selection criteria, run algorithms on the data
   // of the event and typically fill histograms.
   //
   // The processing can be stopped by calling Abort().
   //
   // Use fStatus to set the return value of TTree::Process().
   //
   // The return value is currently not used.

   if (!fNtp) return;

   // Fill ntuple
   Float_t px, py, random;
   if (fRandom) {
      fRandom->Rannor(px,py);
      random = fRandom->Rndm();
   } else {
      Abort("no way to get random numbers! Stop processing", kAbortProcess);
      return;
   }
   Float_t pz = px*px + py*py;
   Int_t i = (Int_t) entry;
   fNtp->Fill(px,py,pz,random,i);

   return;
}


//_____________________________________________________________________________
void ProofSimple::SlaveTerminate()
{
   // The SlaveTerminate() function is called after all entries or objects
   // have been processed. When running with PROOF SlaveTerminate() is called
   // on each slave server.

   // Write the ntuple to the file
   if (fFile) {
      if (!fNtp) {
         Error("SlaveTerminate", "'ntuple' is undefined!");
         return;
      }
      Bool_t cleanup = kFALSE;
      TDirectory *savedir = gDirectory;
      if (fNtp->GetEntries() > 0) {
         fFile->cd();
         fNtp->Write();
         fProofFile->Print();
         fOutput->Add(fProofFile);
      } else {
         cleanup = kTRUE;
      }
      fNtp->SetDirectory(0);
      gDirectory = savedir;
      fFile->Close();
      // Cleanup, if needed
      if (cleanup) {
         TUrl uf(*(fFile->GetEndpointUrl()));
         SafeDelete(fFile);
         gSystem->Unlink(uf.GetFile());
         SafeDelete(fProofFile);
      }
   }
}

//_____________________________________________________________________________
void ProofSimple::Terminate()
{
   // The Terminate() function is the last function to be called during
   // a query. It always runs on the client, it can be used to present
   // the results graphically or save the results to file.

   //
   // Create a canvas, with 100 pads
   //
   TCanvas *c1 = (TCanvas *) gDirectory->FindObject("c1");
   if (c1) {
      gDirectory->Remove(c1);
      delete c1;
   }
   c1 = new TCanvas("c1","Proof ProofSimple canvas",200,10,700,700);
   Int_t nside = (Int_t)TMath::Sqrt((Float_t)fNhist);
   nside = (nside*nside < fNhist) ? nside+1 : nside;
   c1->Divide(nside,nside,0,0);

   Bool_t tryfc = kFALSE;
   TH1F *h = 0;
   for (Int_t i=0; i < fNhist; i++) {
      if (!(h = dynamic_cast<TH1F *>(TProof::GetOutput(Form("h%d",i), fOutput)))) {
         // Not found: try TFileCollection
         tryfc = kTRUE;
         break;
      }
      c1->cd(i+1);
      h->DrawCopy();
   }

   // If the histograms are not found they may be in files: is there a file collection?
   if (tryfc && GetHistosFromFC(c1) != 0) {
      Warning("Terminate", "histograms not found");
   } else {
      // Final update
      c1->cd();
      c1->Update();
   }

   // Analyse hlab, if there
   if (fHLab && !gROOT->IsBatch()) {
      // Printout
      Int_t nb = fHLab->GetNbinsX();
      if (nb > 0) {
         Double_t entb = fHLab->GetEntries() / nb;
         if (entb) {
            for (Int_t i = 0; i < nb; i++) {
               TString lab = TString::Format("hl%d", i);
               Int_t ib = fHLab->GetXaxis()->FindBin(lab);
               Info("Terminate","  %s [%d]:\t%f", lab.Data(), ib, fHLab->GetBinContent(ib)/entb);
            }
         } else
            Warning("Terminate", "no entries in the hlab histogram!");
      }
   }

   // Process the ntuple, if required
   if (fHasNtuple != 1 || !fPlotNtuple) return;

   if (!(fNtp = dynamic_cast<TNtuple *>(TProof::GetOutput("ntuple", fOutput)))) {
      // Get the ntuple from the file
      if ((fProofFile =
            dynamic_cast<TProofOutputFile*>(fOutput->FindObject("SimpleNtuple.root")))) {

         TString outputFile(fProofFile->GetOutputFileName());
         TString outputName(fProofFile->GetName());
         outputName += ".root";
         Printf("outputFile: %s", outputFile.Data());

         // Read the ntuple from the file
         fFile = TFile::Open(outputFile);
         if (fFile) {
            Printf("Managed to open file: %s", outputFile.Data());
            fNtp = (TNtuple *) fFile->Get("ntuple");
         } else {
            Error("Terminate", "could not open file: %s", outputFile.Data());
         }
         if (!fFile) return;

      } else {
         Error("Terminate", "TProofOutputFile not found");
         return;
      }
   }
   // Plot ntuples
   if (fNtp) PlotNtuple(fNtp, "proof ntuple");
}

//_____________________________________________________________________________
void ProofSimple::PlotNtuple(TNtuple *ntp, const char *ntptitle)
{
   // Make some plots from the ntuple 'ntp'

   //
   // Create a canvas, with 2 pads
   //
   TCanvas *c1 = new TCanvas(Form("cv-%s", ntp->GetName()), ntptitle,800,10,700,780);
   c1->Divide(1,2);
   TPad *pad1 = (TPad *) c1->GetPad(1);
   TPad *pad2 = (TPad *) c1->GetPad(2);
   //
   // Display a function of one ntuple column imposing a condition
   // on another column.
   pad1->cd();
   pad1->SetGrid();
   pad1->SetLogy();
   pad1->GetFrame()->SetFillColor(15);
   ntp->SetLineColor(1);
   ntp->SetFillStyle(1001);
   ntp->SetFillColor(45);
   ntp->Draw("3*px+2","px**2+py**2>1");
   ntp->SetFillColor(38);
   ntp->Draw("2*px+2","pz>2","same");
   ntp->SetFillColor(5);
   ntp->Draw("1.3*px+2","(px^2+py^2>4) && py>0","same");
   pad1->RedrawAxis();

   //
   // Display a 3-D scatter plot of 3 columns. Superimpose a different selection.
   pad2->cd();
   ntp->Draw("pz:py:px","(pz<10 && pz>6)+(pz<4 && pz>3)");
   ntp->SetMarkerColor(4);
   ntp->Draw("pz:py:px","pz<6 && pz>4","same");
   ntp->SetMarkerColor(5);
   ntp->Draw("pz:py:px","pz<4 && pz>3","same");
   TPaveText *l2 = new TPaveText(0.,0.6,0.9,0.95);
   l2->SetFillColor(42);
   l2->SetTextAlign(12);
   l2->AddText("You can interactively rotate this view in 2 ways:");
   l2->AddText("  - With the RotateCube in clicking in this pad");
   l2->AddText("  - Selecting View with x3d in the View menu");
   l2->Draw();

   // Final update
   c1->cd();
   c1->Update();
}

//_____________________________________________________________________________
Int_t ProofSimple::GetHistosFromFC(TCanvas *cv)
{
   // Check for the histograms in the files of a possible TFileCollection

   TIter nxo(fOutput);
   TFileCollection *fc = 0;
   Bool_t fc_found = kFALSE, hs_found = kFALSE;
   while ((fc = (TFileCollection *) nxo())) {
      if (strcmp(fc->ClassName(), "TFileCollection")) continue;
      fc_found = kTRUE;
      if (!fHist) {
         fHist = new TH1F*[fNhist];
         for (Int_t i = 0; i < fNhist; i++) { fHist[i] = 0; }
      } else {
         for (Int_t i = 0; i < fNhist; i++) { SafeDelete(fHist[i]); }
      }
      // Go through the list of files
      TIter nxf(fc->GetList());
      TFileInfo *fi = 0;
      while ((fi = (TFileInfo *) nxf())) {
         TFile *f = TFile::Open(fi->GetCurrentUrl()->GetUrl());
         if (f) {
            for (Int_t i = 0; i < fNhist; i++) {
               TString hn = TString::Format("h%d", i);
               TH1F *h = (TH1F *) f->Get(hn);
               if (h) {
                  hs_found = kTRUE;
                  if (!fHist[i]) {
                     fHist[i] = (TH1F *) h->Clone();
                     fHist[i]->SetDirectory(0);
                  } else {
                     fHist[i]->Add(h);
                  }
               } else {
                  Error("GetHistosFromFC", "histo '%s' not found in file '%s'",
                        hn.Data(), fi->GetCurrentUrl()->GetUrl());
               }
            }
            f->Close();
         } else {
            Error("GetHistosFromFC", "file '%s' could not be open", fi->GetCurrentUrl()->GetUrl());
         }
      }
      if (hs_found) break;
   }
   if (!fc_found) return -1;
   if (!hs_found) return -1;

   for (Int_t i = 0; i < fNhist; i++) {
      cv->cd(i+1);
      if (fHist[i]) {
         fHist[i]->DrawCopy();
      }
   }
   Info("GetHistosFromFC", "histograms read from %d files in TFileCollection '%s'",
                           fc->GetList()->GetSize(), fc->GetName());
   // Done
   return 0;
}
