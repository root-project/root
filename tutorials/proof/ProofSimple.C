#define ProofSimple_cxx
// The class definition in ProofSimple.h has been generated automatically
// by the ROOT utility TTree::MakeSelector(). This class is derived
// from the ROOT class TSelector. For more information on the TSelector
// framework see $ROOTSYS/README/README.SELECTOR or the ROOT User Manual.
//
// The following methods are defined in this file:
//    Begin():        called everytime a loop on the tree starts,
//                    a convenient place to create your histograms.
//    SlaveBegin():   called after Begin(), when on PROOF called only on the
//                    slave servers.
//    Process():      called for each event, in this function you decide what
//                    to read and fill your histograms.
//    SlaveTerminate: called at the end of the loop on the tree, when on PROOF
//                    called only on the slave servers.
//    Terminate():    called at the end of the loop on the tree,
//                    a convenient place to draw/fit your histograms.
//
// To use this file, try the following session on your Tree T:
//
// Root > T->Process("ProofSimple.C")
// Root > T->Process("ProofSimple.C","some options")
// Root > T->Process("ProofSimple.C+")
//

#include "ProofSimple.h"
#include <TCanvas.h>
#include <TFrame.h>
#include <TPaveText.h>
#include <TFormula.h>
#include <TF1.h>
#include <TH1D.h>
#include <TMath.h>
#include <TNtuple.h>
#include <TRandom.h>
#include <TString.h>
#include <TStyle.h>
#include <TSystem.h>

// Global functions
static TFormula *gForm1 = 0;
static TF1 *gSQroot = 0;

void ProofSimple::Begin(TTree * /*tree*/)
{
   // The Begin() function is called at the start of the query.
   // When running with PROOF Begin() is only called on the client.
   // The tree argument is deprecated (on PROOF 0 is passed).

   TString option = GetOption();

}

void ProofSimple::SlaveBegin(TTree * /*tree*/)
{
   // The SlaveBegin() function is called after the Begin() function.
   // When running with PROOF SlaveBegin() is called on each slave server.
   // The tree argument is deprecated (on PROOF 0 is passed).

   TString option = GetOption();

   // Create the formulas
   if (!gForm1)
      gForm1 = new TFormula("form1", "abs(sin(x)/x)");
   if (!gSQroot) {
      gSQroot = new TF1("sqroot","x*gaus(0) + [3]*form1",0,10);
      gSQroot->SetParameters(10,4,1,20);
   }

   // Create the histogram
   fHgaus = new TH1D("hgaus","Random gaussian numbers",100, -5., 5.);
   fHsqr = new TH1D("hsqr","x*gaus(0) + [3]*abs(sin(x)/x)",200, 0, 10.);
   fNtp = new TNtuple("ntuple","Demo ntuple","px:py:pz:random:i");

   // Owned by fOutput: must not be deleted
   fOutput->Add(fHgaus);
   fOutput->Add(fHsqr);
   fOutput->Add(fNtp);

   // Set random seed
   ULong_t now = gSystem->Now();
   TString seed = Form("%ld", now);
   Info("SlaveBegin","seed set to: %s", seed.Data());
   gRandom->SetSeed(static_cast<UInt_t>(TMath::Hash(seed)));
}

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

   // Fill the first histo with a Gaussian
   fHgaus->Fill(gRandom->Gaus(0.,1.));

   // Fill the second usi
   fHsqr->Fill(gSQroot->GetRandom());

   // Fill ntuple
   Float_t px, py;
   gRandom->Rannor(px,py);
   Float_t pz = px*px + py*py;
   Float_t random = gRandom->Rndm(1);
   Int_t i = (Int_t) entry;
   fNtp->Fill(px,py,pz,random,i);

   return kTRUE;
}

void ProofSimple::SlaveTerminate()
{
   // The SlaveTerminate() function is called after all entries or objects
   // have been processed. When running with PROOF SlaveTerminate() is called
   // on each slave server.

}

void ProofSimple::Terminate()
{
   // The Terminate() function is the last function to be called during
   // a query. It always runs on the client, it can be used to present
   // the results graphically or save the results to file.

   // Get 'hgaus' histogram
   fHgaus = dynamic_cast<TH1D *>(fOutput->FindObject("hgaus"));
   if (!fHgaus) {
      Error("Terminate", "Could not find histogram 'hgaus'");
      return;
   }

   // Get 'hsqr' histogram
   fHsqr = dynamic_cast<TH1D *>(fOutput->FindObject("hsqr"));
   if (!fHsqr) {
      Error("Terminate", "Could not find histogram 'hsqr'");
      return;
   }

   // Get 'ntuple'
   fNtp = dynamic_cast<TNtuple *>(fOutput->FindObject("ntuple"));
   if (!fNtp) {
      Error("Terminate", "Could not find 'ntuple'");
      return;
   }
   //
   // Create a canvas, with 4 pads
   //
   TCanvas *c1 = new TCanvas("c1","Proof ProofSimple canvas",200,10,700,780);
   TPad *pad1 = new TPad("pad1","Hgaus",0.02,0.52,0.48,0.98,21);
   TPad *pad2 = new TPad("pad2","Hsqr",0.52,0.52,0.98,0.98,21);
   TPad *pad3 = new TPad("pad3","Ntuple 1",0.02,0.02,0.48,0.48,21);
   TPad *pad4 = new TPad("pad4","Ntuple 2",0.52,0.02,0.98,0.48,1);
   pad1->Draw();
   pad2->Draw();
   pad3->Draw();
   pad4->Draw();

   // Draw 'hgaus'
   pad1->cd();
   pad1->SetFillColor(42);
/*   pad1->GetFrame()->SetFillColor(21);
   pad1->GetFrame()->SetBorderSize(6);
   pad1->GetFrame()->SetBorderMode(-1);*/
   fHgaus->SetFillColor(48);
   fHgaus->Draw();

   // Draw + Fit 'hsqr'
   pad2->cd();
   pad2->SetFillColor(0);
   pad2->SetGridx();
   pad2->SetGridy();
   pad2->GetFrame()->SetFillColor(21);
   pad2->GetFrame()->SetBorderMode(-1);
   pad2->GetFrame()->SetBorderSize(5);
   fHsqr->SetFillColor(45);
   // Create the formulas
   if (!gForm1)
      gForm1 = new TFormula("form1", "abs(sin(x)/x)");
   if (!gSQroot) {
      gSQroot = new TF1("sqroot","x*gaus(0) + [3]*form1",0,10);
      gSQroot->SetParameters(100,4,1,200);
   }
   gSQroot->Print();
   fHsqr->SetFillColor(45);
   fHsqr->Fit("sqroot");
   gStyle->SetOptFit(1);
   fHsqr->Draw();

   //
   // Display a function of one ntuple column imposing a condition
   // on another column.
   pad3->cd();
   pad3->SetGrid();
   pad3->SetLogy();
   pad3->GetFrame()->SetFillColor(15);
   fNtp->SetLineColor(1);
   fNtp->SetFillStyle(1001);
   fNtp->SetFillColor(45);
   fNtp->Draw("3*px+2","px**2+py**2>1");
   fNtp->SetFillColor(38);
   fNtp->Draw("2*px+2","pz>2","same");
   fNtp->SetFillColor(5);
   fNtp->Draw("1.3*px+2","(px^2+py^2>4) && py>0","same");
   pad3->RedrawAxis();

   //
   // Display a 3-D scatter plot of 3 columns. Superimpose a different selection.
   pad4->cd();
   fNtp->Draw("pz:py:px","(pz<10 && pz>6)+(pz<4 && pz>3)");
   fNtp->SetMarkerColor(4);
   fNtp->Draw("pz:py:px","pz<6 && pz>4","same");
   fNtp->SetMarkerColor(5);
   fNtp->Draw("pz:py:px","pz<4 && pz>3","same");
   TPaveText *l4 = new TPaveText(-0.9,0.5,0.9,0.95);
   l4->SetFillColor(42);
   l4->SetTextAlign(12);
   l4->AddText("You can interactively rotate this view in 2 ways:");
   l4->AddText("  - With the RotateCube in clicking in this pad");
   l4->AddText("  - Selecting View with x3d in the View menu");
   l4->Draw();

   // Final update
   c1->cd();
   c1->Update();
}
