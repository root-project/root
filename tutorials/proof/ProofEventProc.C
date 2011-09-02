#define ProofEventProc_cxx

//////////////////////////////////////////////////////////
//
// Example of TSelector implementation to process trees
// containing 'Event' structures, e.g. the files under
// http://root.cern.ch/files/data .
// See tutorials/proof/runProof.C, option "eventproc", for
// an example of how to run this selector.
//
//////////////////////////////////////////////////////////

#include "ProofEventProc.h"
#include <TStyle.h>
#include "TCanvas.h"
#include "TPad.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TParameter.h"
#include "TRandom.h"


void ProofEventProc::Begin(TTree *)
{
   // The Begin() function is called at the start of the query.
   // When running with PROOF Begin() is only called on the client.
   // The tree argument is deprecated (on PROOF 0 is passed).

   TString option = GetOption();

}

void ProofEventProc::SlaveBegin(TTree *tree)
{
   // The SlaveBegin() function is called after the Begin() function.
   // When running with PROOF SlaveBegin() is called on each slave server.
   // The tree argument is deprecated (on PROOF 0 is passed).

   Init(tree);

   // How much to read
   fFullRead = kFALSE;
   TNamed *nm = 0;
   if (fInput) 
      nm = dynamic_cast<TNamed *>(fInput->FindObject("ProofEventProc_Read"));
   if (nm && !strcmp(nm->GetTitle(), "readall"))
      fFullRead = kTRUE;
   Info("SlaveBegin", "'%s' reading", (fFullRead ? "full" : "optimized"));

   TString option = GetOption();

   fPtHist = new TH1F("pt_dist","p_{T} Distribution",100,0,5);
   fPtHist->SetDirectory(0);
   fPtHist->GetXaxis()->SetTitle("p_{T}");
   fPtHist->GetYaxis()->SetTitle("dN/p_{T}dp_{T}");

   fOutput->Add(fPtHist);

   fPzHist = new TH1F("pz_dist","p_{Z} Distribution",100,0,5.);
   fPzHist->SetDirectory(0);
   fPzHist->GetXaxis()->SetTitle("p_{Z}");
   fPzHist->GetYaxis()->SetTitle("dN/dp_{Z}");

   fOutput->Add(fPzHist);

   fPxPyHist = new TH2F("px_py","p_{X} vs p_{Y} Distribution",100,-5.,5.,100,-5.,5.);
   fPxPyHist->SetDirectory(0);
   fPxPyHist->GetXaxis()->SetTitle("p_{X}");
   fPxPyHist->GetYaxis()->SetTitle("p_{Y}");

   fOutput->Add(fPxPyHist);
   
   // Abort test, if any
   TParameter<Int_t> *pi = 0;
   if (fInput) 
      pi = dynamic_cast<TParameter<Int_t> *>(fInput->FindObject("ProofEventProc_TestAbort"));
   if (pi) fTestAbort = pi->GetVal();
   if (fTestAbort < -1 || fTestAbort > 1) {
      Info("SlaveBegin", "unsupported value for the abort test: %d not in [-1,1] - ignore", fTestAbort);
      fTestAbort = -1;
   } else if (fTestAbort > -1) {
      Info("SlaveBegin", "running abort test: %d", fTestAbort);
   }

   if (fTestAbort == 0) 
      Abort("Test abortion during init", kAbortProcess);
}

Bool_t ProofEventProc::Process(Long64_t entry)
{
   // The Process() function is called for each entry in the tree (or possibly
   // keyed object in the case of PROOF) to be processed. The entry argument
   // specifies which entry in the currently loaded tree is to be processed.
   // It can be passed to either TTree::GetEntry() or TBranch::GetEntry()
   // to read either all or the required parts of the data. When processing
   // keyed objects with PROOF, the object is already loaded and is available
   // via the fObject pointer.
   //
   // This function should contain the "body" of the analysis. It can contain
   // simple or elaborate selection criteria, run algorithms on the data
   // of the event and typically fill histograms.

   // WARNING when a selector is used with a TChain, you must use
   //  the pointer to the current TTree to call GetEntry(entry).
   //  The entry is always the local entry number in the current tree.
   //  Assuming that fChain is the pointer to the TChain being processed,
   //  use fChain->GetTree()->GetEntry(entry).

   if (fTestAbort == 1) {
      Double_t rr = gRandom->Rndm();
      if (rr > 0.999) {
         Info("Process", "%lld -> %f", entry, rr); 
         Abort("Testing file abortion", kAbortFile);
         return kTRUE;
      }
   }

   if (fFullRead) {
      fChain->GetTree()->GetEntry(entry);
   } else {
      b_event_fNtrack->GetEntry(entry);
   }

   if (fNtrack > 0) {
      if (!fFullRead) b_fTracks->GetEntry(entry);
      if (fTracks) {
         for (Int_t j=0;j<fTracks->GetEntries();j++){
            Track *curtrack = dynamic_cast<Track*>(fTracks->At(j));
            if (curtrack) {
               fPtHist->Fill(curtrack->GetPt(),1./curtrack->GetPt());
               fPxPyHist->Fill(curtrack->GetPx(),curtrack->GetPy());
               if (j == 0) fPzHist->Fill(curtrack->GetPz());
            }
         }
         fTracks->Clear("C");
      }
   }

   return kTRUE;
}

void ProofEventProc::SlaveTerminate()
{
   // The SlaveTerminate() function is called after all entries or objects
   // have been processed. When running with PROOF SlaveTerminate() is called
   // on each slave server.

}

void ProofEventProc::Terminate()
{
   // The Terminate() function is the last function to be called during
   // a query. It always runs on the client, it can be used to present
   // the results graphically or save the results to file.

   TCanvas* canvas = new TCanvas("event","event",800,10,700,780);
   canvas->Divide(2,2);
   TPad *pad1 = (TPad *) canvas->GetPad(1);
   TPad *pad2 = (TPad *) canvas->GetPad(2);
   TPad *pad3 = (TPad *) canvas->GetPad(3);
   TPad *pad4 = (TPad *) canvas->GetPad(4);

   // The number of tracks
   pad1->cd();
   pad1->SetLogy();
   TH1F *hi = dynamic_cast<TH1F*>(fOutput->FindObject("pz_dist"));
   if (hi) {
      hi->SetFillColor(30);
      hi->SetLineColor(9);
      hi->SetLineWidth(2);
      hi->DrawCopy();
   } else { Warning("Terminate", "no pz dist found"); }

   // The Pt distribution
   pad2->cd();
   pad2->SetLogy();
   TH1F *hf = dynamic_cast<TH1F*>(fOutput->FindObject("pt_dist"));
   if (hf) {
      hf->SetFillColor(30);
      hf->SetLineColor(9);
      hf->SetLineWidth(2);
      hf->DrawCopy();
   } else { Warning("Terminate", "no pt dist found"); }

   // The Px,Py distribution, color surface
   TH2F *h2f = dynamic_cast<TH2F*>(fOutput->FindObject("px_py"));
   if (h2f) {
      // Color surface
      pad3->cd();
      h2f->DrawCopy("SURF1 ");
      // Lego
      pad4->cd();
      h2f->DrawCopy("CONT2COL");
   } else {
      Warning("Terminate", "no px py found");
   }

   // Final update
   canvas->cd();
   canvas->Update();
}
