#define ProofFriends_cxx
// The class definition in ProofFriends.h has been generated automatically
// by the ROOT utility TTree::MakeSelector(). This class is derived
// from the ROOT class TSelector. For more information on the TSelector
// framework see $ROOTSYS/README/README.SELECTOR or the ROOT User Manual.

// The following methods are defined in this file:
//    Begin():        called every time a loop on the tree starts,
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
// Root > T->Process("ProofFriends.C")
// Root > T->Process("ProofFriends.C","some options")
// Root > T->Process("ProofFriends.C+")
//

#include "ProofFriends.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TList.h"
#include "TMath.h"
#include "TString.h"
#include "TStyle.h"

//_____________________________________________________________________________
ProofFriends::ProofFriends()
{
   // Constructor

   fXY = 0;
   fZ = 0;
   fR = 0;
   fRZ = 0;
}

void ProofFriends::Begin(TTree * /*tree*/)
{
   // The Begin() function is called at the start of the query.
   // When running with PROOF Begin() is only called on the client.
   // The tree argument is deprecated (on PROOF 0 is passed).

   TString option = GetOption();

}

void ProofFriends::SlaveBegin(TTree * /*tree*/)
{
   // The SlaveBegin() function is called after the Begin() function.
   // When running with PROOF SlaveBegin() is called on each slave server.
   // The tree argument is deprecated (on PROOF 0 is passed).

   TString option = GetOption();

   // Histograms
   fXY = new TH2F("histo1", "y:x", 50, 5., 15., 50, 10., 30.);
   fZ = new TH1F("histo2", "z , sqrt(dx*dx+dy*dy) < 1", 50, 0., 5.);
   fR = new TH1F("histo3", "Tfrnd.r , sqrt(dx*dx+dy*dy) < 1, z < 1", 50, 5., 15.);
   fRZ = new TH2F("histo4", "Tfrnd.r:z , sqrt(dx*dx+dy*dy) < 1, z < 1", 50, 0., 1., 50, 5., 15.);
   fZ->SetFillColor(kBlue);
   fR->SetFillColor(kRed);
   fOutput->Add(fXY);
   fOutput->Add(fZ);
   fOutput->Add(fR);
   fOutput->Add(fRZ);

}

Bool_t ProofFriends::Process(Long64_t entry)
{
   // The Process() function is called for each entry in the tree (or possibly
   // keyed object in the case of PROOF) to be processed. The entry argument
   // specifies which entry in the currently loaded tree is to be processed.
   // It can be passed to either ProofFriends::GetEntry() or TBranch::GetEntry()
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

   // Read x and y, fill and apply cut
   b_x->GetEntry(entry);
   b_y->GetEntry(entry);
   fXY->Fill(x, y, 1.);
   Double_t dx = x-10.;
   Double_t dy = y-20.;
   Double_t xpy = TMath::Sqrt(dx*dx + dy*dy);
   if (xpy > 1.) return kFALSE;

   // Read z, fill and apply cut
   b_z->GetEntry(entry);
   fZ->Fill(z, 1.);
   if (z > 1.) return kFALSE;

   // Read r and fill
   b_r->GetEntry(entry);
   fR->Fill(r, 1.);
   fRZ->Fill(z, r, 1.);

   return kTRUE;
}

void ProofFriends::SlaveTerminate()
{
   // The SlaveTerminate() function is called after all entries or objects
   // have been processed. When running with PROOF SlaveTerminate() is called
   // on each slave server.

}

void ProofFriends::Terminate()
{
   // The Terminate() function is the last function to be called during
   // a query. It always runs on the client, it can be used to present
   // the results graphically or save the results to file.

   gStyle->SetOptStat(1110);
   // Create canvas
   TCanvas *c1 = new TCanvas("c1","Proof ProofFriends canvas",200,10,700,700);
   // Overall background
   Int_t cb = TColor::GetColor("#ccffff");
   c1->SetFillColor(cb);
   c1->SetBorderMode(0);
   // 4 pads
   c1->Divide(2, 2);

   Int_t cf = TColor::GetColor("#99cccc");
   TPad *p1 = 0;
   if ((fXY = dynamic_cast<TH2F *>(fOutput->FindObject("histo1")))) {
      p1 = (TPad *) c1->cd(1);
      p1->SetBorderMode(0);
      p1->SetFrameFillColor(cf);
      fXY->GetXaxis()->SetTitle("x");
      fXY->GetYaxis()->SetTitle("y");
      fXY->Draw("");
   }

   if ((fZ = dynamic_cast<TH1F *>(fOutput->FindObject("histo2")))) {
      p1 = (TPad *) c1->cd(2);
      p1->SetBorderMode(0);
      p1->SetFrameFillColor(cf);
      fZ->GetXaxis()->SetTitle("z");
      fZ->GetYaxis()->SetTitle("N / 0.1");
      fZ->Draw("");
   }

   if ((fR = dynamic_cast<TH1F *>(fOutput->FindObject("histo3")))) {
      p1 = (TPad *) c1->cd(3);
      p1->SetBorderMode(0);
      p1->SetFrameFillColor(cf);
      fR->GetXaxis()->SetTitle("Tfrnd.r");
      fR->GetYaxis()->SetTitle("N / 0.2");
      fR->Draw();
   }

   if ((fRZ = dynamic_cast<TH2F *>(fOutput->FindObject("histo4")))) {
      p1 = (TPad *) c1->cd(4);
      p1->SetBorderMode(0);
      p1->SetFrameFillColor(cf);
      fRZ->GetXaxis()->SetTitle("z");
      fRZ->GetXaxis()->CenterTitle(1);
      fRZ->GetXaxis()->SetTitleOffset(1.5);
      fRZ->GetYaxis()->SetTitle("Tfrnd.r");
      fRZ->GetYaxis()->CenterTitle(1);
      fRZ->GetYaxis()->SetTitleOffset(1.75);
      fRZ->GetZaxis()->SetTitle("N / 0.1 / 0.2");
      fRZ->GetZaxis()->SetTitleOffset(1.25);
      fRZ->Draw("lego");
   }

   // Final update
   c1->cd();
   c1->Update();

}
