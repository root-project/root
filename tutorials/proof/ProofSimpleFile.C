/// \file
/// \ingroup tutorial_ProofSimpleFile
///
/// Selector to fill a set of histograms and merging via file
///
/// \macro_code
///
/// \author Gerardo Ganis (gerardo.ganis@cern.ch)

#define ProofSimpleFile_cxx

#include "ProofSimpleFile.h"
#include <TCanvas.h>
#include <TFrame.h>
#include <TPaveText.h>
#include <TFormula.h>
#include <TF1.h>
#include <TH1F.h>
#include <TMath.h>
#include <TRandom3.h>
#include <TString.h>
#include <TStyle.h>
#include <TSystem.h>
#include <TParameter.h>
#include <TFile.h>
#include <TProofOutputFile.h>

//_____________________________________________________________________________
ProofSimpleFile::ProofSimpleFile()
{
   // Constructor

   fNhist = 16;
   fHistTop = 0;
   fHistDir = 0;
   fRandom = 0;
   fFile = 0;
   fProofFile = 0;
   fFileDir = 0;
}

//_____________________________________________________________________________
ProofSimpleFile::~ProofSimpleFile()
{
   // Destructor

   if (fRandom) delete fRandom;
}

//_____________________________________________________________________________
Int_t ProofSimpleFile::CreateHistoArrays()
{
   // Create the histogram arrays

   if (fNhist <= 0) {
      Error("CreateHistoArrays", "fNhist must be positive!");
      return -1;
   }
   // Histos array
   fHistTop = new TH1F*[fNhist];
   fHistDir = new TH1F*[fNhist];
   // Done
   return 0;
}

//_____________________________________________________________________________
void ProofSimpleFile::Begin(TTree * /*tree*/)
{
   // The Begin() function is called at the start of the query.
   // When running with PROOF Begin() is only called on the client.
   // The tree argument is deprecated (on PROOF 0 is passed).

   TString option = GetOption();

   // Number of histograms (needed in terminate)
   Ssiz_t iopt = kNPOS;
   if (fInput->FindObject("ProofSimpleFile_NHist")) {
      TParameter<Long_t> *p =
         dynamic_cast<TParameter<Long_t>*>(fInput->FindObject("ProofSimpleFile_NHist"));
      fNhist = (p) ? (Int_t) p->GetVal() : fNhist;
   } else if ((iopt = option.Index("nhist=")) != kNPOS) {
      TString s;
      Ssiz_t from = iopt + strlen("nhist=");
      if (option.Tokenize(s, from, ";") && s.IsDigit()) fNhist = s.Atoi();
   }
}

//_____________________________________________________________________________
void ProofSimpleFile::SlaveBegin(TTree * /*tree*/)
{
   // The SlaveBegin() function is called after the Begin() function.
   // When running with PROOF SlaveBegin() is called on each slave server.
   // The tree argument is deprecated (on PROOF 0 is passed).

   TString option = GetOption();

   // Number of histograms (needed in terminate)
   Ssiz_t iopt = kNPOS;
   if (fInput->FindObject("ProofSimpleFile_NHist")) {
      TParameter<Long_t> *p =
         dynamic_cast<TParameter<Long_t>*>(fInput->FindObject("ProofSimpleFile_NHist"));
      fNhist = (p) ? (Int_t) p->GetVal() : fNhist;
   } else if ((iopt = option.Index("nhist=")) != kNPOS) {
      TString s;
      Ssiz_t from = iopt + strlen("nhist=");
      if (option.Tokenize(s, from, ";") && s.IsDigit()) fNhist = s.Atoi();
   }

   // The file for merging
   fProofFile = new TProofOutputFile("SimpleFile.root", "M");
   TNamed *out = (TNamed *) fInput->FindObject("PROOF_OUTPUTFILE");
   if (out) fProofFile->SetOutputFileName(out->GetTitle());
   TDirectory *savedir = gDirectory;
   fFile = fProofFile->OpenFile("RECREATE");
   if (fFile && fFile->IsZombie()) SafeDelete(fFile);
   savedir->cd();

   // Cannot continue
   if (!fFile) {
      TString amsg = TString::Format("ProofSimpleFile::SlaveBegin: could not create '%s':"
                                     " instance is invalid!", fProofFile->GetName());
      Abort(amsg, kAbortProcess);
      return;
   }

   // Histos arrays
   if (CreateHistoArrays() != 0) {
      Abort("ProofSimpleFile::SlaveBegin: could not create histograms", kAbortProcess);
      return;
   }

   // Create directory
   if (!(fFileDir = fFile->mkdir("blue"))) {
      Abort("ProofSimpleFile::SlaveBegin: could not create directory 'blue' in file!",
            kAbortProcess);
      return;
   }

   // Create the histograms
   for (Int_t i=0; i < fNhist; i++) {
      fHistTop[i] = new TH1F(Form("ht%d",i), Form("ht%d",i), 100, -3., 3.);
      fHistTop[i]->SetFillColor(kRed);
      fHistTop[i]->SetDirectory(fFile);
      fHistDir[i] = new TH1F(Form("hd%d",i), Form("hd%d",i), 100, -3., 3.);
      fHistDir[i]->SetFillColor(kBlue);
      fHistDir[i]->SetDirectory(fFileDir);
   }

   // Set random seed
   fRandom = new TRandom3(0);
}

//_____________________________________________________________________________
Bool_t ProofSimpleFile::Process(Long64_t)
{
   // The Process() function is called for each entry in the tree (or possibly
   // keyed object in the case of PROOF) to be processed. The entry argument
   // specifies which entry in the currently loaded tree is to be processed.
   // It can be passed to either ProofSimpleFile::GetEntry() or TBranch::GetEntry()
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
      if (fRandom && fHistTop[i] && fHistDir[i]) {
         fHistTop[i]->Fill(fRandom->Gaus(0.,1.));
         fHistDir[i]->Fill(fRandom->Gaus(0.,1.));
      }
   }

   return kTRUE;
}

//_____________________________________________________________________________
void ProofSimpleFile::SlaveTerminate()
{
   // The SlaveTerminate() function is called after all entries or objects
   // have been processed. When running with PROOF SlaveTerminate() is called
   // on each slave server.

   // Write histos to file
   if (fFile) {
      Bool_t cleanup = kTRUE;
      TDirectory *savedir = gDirectory;
      fFile->cd();
      for (Int_t i=0; i < fNhist; i++) {
         if (fHistTop[i] && fHistTop[i]->GetEntries() > 0) {
            fHistTop[i]->Write();
            fHistTop[i]->SetDirectory(0);
            cleanup = kFALSE;
         }
      }
      // Change to subdirectory
      fFileDir->cd();
      for (Int_t i=0; i < fNhist; i++) {
         if (fHistDir[i] && fHistDir[i]->GetEntries() > 0) {
            fHistDir[i]->Write();
            fHistDir[i]->SetDirectory(0);
            cleanup = kFALSE;
         }
      }
      gDirectory = savedir;
      fFile->Close();
      // Cleanup or register
      if (cleanup) {
         Info("SlaveTerminate", "nothing to save: just cleanup everything ...");
         TUrl uf(*(fFile->GetEndpointUrl()));
         SafeDelete(fFile);
         gSystem->Unlink(uf.GetFile());
         SafeDelete(fProofFile);
      } else {
         Info("SlaveTerminate", "objects saved into '%s%s': sending related TProofOutputFile ...",
                                fProofFile->GetFileName(), fProofFile->GetOptionsAnchor());
         fProofFile->Print();
         fOutput->Add(fProofFile);
      }
   }

}

//_____________________________________________________________________________
void ProofSimpleFile::Terminate()
{
   // The Terminate() function is the last function to be called during
   // a query. It always runs on the client, it can be used to present
   // the results graphically or save the results to file.

   // Get the histos from the file
   if ((fProofFile =
           dynamic_cast<TProofOutputFile*>(fOutput->FindObject("SimpleFile.root")))) {

      TString outputFile(fProofFile->GetOutputFileName());
      TString outputName(fProofFile->GetName());
      outputName += ".root";
      Printf("outputFile: %s", outputFile.Data());

      // Read the ntuple from the file
      if (!(fFile = TFile::Open(outputFile))) {
         Error("Terminate", "could not open file: %s", outputFile.Data());
         return;
      }

   } else {
      Error("Terminate", "TProofOutputFile not found");
      return;
   }

   // Histos arrays
   if (CreateHistoArrays() != 0) {
      Error("Terminate", "could not create histograms");
      return;
   }

   // Top histos
   PlotHistos(0);
   // Dir histos
   PlotHistos(1);
}

//_____________________________________________________________________________
void ProofSimpleFile::PlotHistos(Int_t opt)
{
   // Plot the histograms ina dedicated canvas

   // Create a canvas, with fNhist pads
   if (opt == 0) {
      TCanvas *c1 = new TCanvas("c1","ProofSimpleFile top dir canvas",200,10,700,700);
      Int_t nside = (Int_t)TMath::Sqrt((Float_t)fNhist);
      nside = (nside*nside < fNhist) ? nside+1 : nside;
      c1->Divide(nside,nside,0,0);

      for (Int_t i=0; i < fNhist; i++) {
         fHistTop[i] = (TH1F *) fFile->Get(TString::Format("ht%d",i));
         c1->cd(i+1);
         if (fHistTop[i])
            fHistTop[i]->Draw();
      }

      // Final update
      c1->cd();
      c1->Update();
   } else if (opt == 1) {
      TCanvas *c2 = new TCanvas("c2","ProofSimpleFile 'blue' sub-dir canvas",400,60,700,700);
      Int_t nside = (Int_t)TMath::Sqrt((Float_t)fNhist);
      nside = (nside*nside < fNhist) ? nside+1 : nside;
      c2->Divide(nside,nside,0,0);

      if ((fFileDir = (TDirectory *) fFile->Get("blue"))) {
         for (Int_t i=0; i < fNhist; i++) {
            fHistDir[i] = (TH1F *) fFileDir->Get(TString::Format("hd%d",i));
            c2->cd(i+1);
            if (fHistDir[i])
               fHistDir[i]->Draw();
         }
      } else {
         Error("PlotHistos", "directory 'blue' not found in output file");
      }

      // Final update
      c2->cd();
      c2->Update();
   } else {
      Error("PlotHistos", "unknown option: %d", opt);
   }
}
