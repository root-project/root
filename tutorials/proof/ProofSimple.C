#define ProofSimple_cxx
// The class definition in ProofSimple.h has been generated automatically
// by the ROOT utility TTree::MakeSelector(). This class is derived
// from the ROOT class TSelector. For more information on the TSelector
// framework see $ROOTSYS/README/README.SELECTOR or the ROOT User Manual.

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
#include <TH1F.h>
#include <TMath.h>
#include <TRandom3.h>
#include <TString.h>
#include <TStyle.h>
#include <TSystem.h>
#include <TParameter.h>

//_____________________________________________________________________________
ProofSimple::ProofSimple()
{
   // Constructor

   fNhist = -1;
   fHist = 0;
   fRandom = 0;
}

//_____________________________________________________________________________
ProofSimple::~ProofSimple()
{
   // Destructor

   if (fRandom) delete fRandom;
}

//_____________________________________________________________________________
void ProofSimple::Begin(TTree * /*tree*/)
{
   // The Begin() function is called at the start of the query.
   // When running with PROOF Begin() is only called on the client.
   // The tree argument is deprecated (on PROOF 0 is passed).

   TString option = GetOption();

   // Histos array
   fNhist = 100;
   if (fInput->FindObject("ProofSimple_NHist")) {
      TParameter<Long_t> *p =
         dynamic_cast<TParameter<Long_t>*>(fInput->FindObject("ProofSimple_NHist"));
      fNhist = (p) ? (Int_t) p->GetVal() : fNhist;
   }
   fHist = new TH1F*[fNhist];
}

//_____________________________________________________________________________
void ProofSimple::SlaveBegin(TTree * /*tree*/)
{
   // The SlaveBegin() function is called after the Begin() function.
   // When running with PROOF SlaveBegin() is called on each slave server.
   // The tree argument is deprecated (on PROOF 0 is passed).

   TString option = GetOption();

   // Histos array
   fNhist = 100;
   if (fInput->FindObject("ProofSimple_NHist")) {
      TParameter<Long_t> *p =
         dynamic_cast<TParameter<Long_t>*>(fInput->FindObject("ProofSimple_NHist"));
      fNhist = (p) ? (Int_t) p->GetVal() : fNhist;
   }
   fHist = new TH1F*[fNhist];

   // Create the histogram
   for (Int_t i=0; i < fNhist; i++) {
      fHist[i] = new TH1F(Form("h%d",i), Form("h%d",i), 100, -3., 3.);
      fHist[i]->SetFillColor(kRed);
      fOutput->Add(fHist[i]);
   }

   // Set random seed
   fRandom = new TRandom3(0);
}

//_____________________________________________________________________________
Bool_t ProofSimple::Process(Long64_t)
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

   return kTRUE;
}

//_____________________________________________________________________________
void ProofSimple::SlaveTerminate()
{
   // The SlaveTerminate() function is called after all entries or objects
   // have been processed. When running with PROOF SlaveTerminate() is called
   // on each slave server.

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
   TCanvas *c1 = new TCanvas("c1","Proof ProofSimple canvas",200,10,700,700);
   Int_t nside = (Int_t)TMath::Sqrt((Float_t)fNhist);
   nside = (nside*nside < fNhist) ? nside+1 : nside;
   c1->Divide(nside,nside,0,0);

   for (Int_t i=0; i < fNhist; i++) {
      fHist[i] = dynamic_cast<TH1F *>(fOutput->FindObject(Form("h%d",i)));
      c1->cd(i+1);
      if (fHist[i])
         fHist[i]->Draw();
   }

   // Final update
   c1->cd();
   c1->Update();
}
