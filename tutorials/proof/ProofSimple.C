#define ProofSimple_cxx
//////////////////////////////////////////////////////////
//
// Example of TSelector implementation to do generic
// processing (filling a set of histograms in this case).
// See tutorials/proof/runProof.C, option "simple", for an
// example of how to run this selector.
//
//////////////////////////////////////////////////////////

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

//_____________________________________________________________________________
ProofSimple::ProofSimple()
{
   // Constructor

   fNhist = -1;
   fHist = 0;
   fNhist3 = -1;
   fHist3 = 0;
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
   
   if (fInput->FindObject("ProofSimple_NHist3")) {
      TParameter<Long_t> *p =
         dynamic_cast<TParameter<Long_t>*>(fInput->FindObject("ProofSimple_NHist3"));
      fNhist3 = (p) ? (Int_t) p->GetVal() : fNhist3;
   }
   if (fNhist3 > 0) fHist3 = new TH3F*[fNhist3];
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
   
   // 3D Histos array
   if (fInput->FindObject("ProofSimple_NHist3")) {
      TParameter<Long_t> *p =
         dynamic_cast<TParameter<Long_t>*>(fInput->FindObject("ProofSimple_NHist3"));
      fNhist3 = (p) ? (Int_t) p->GetVal() : fNhist3;
   }
   if (fNhist3 > 0) {
      fHist3 = new TH3F*[fNhist3];
      // Create the 3D histogram
      for (Int_t i=0; i < fNhist3; i++) {
         fHist3[i] = new TH3F(Form("h%d_3d",i), Form("h3%d_3d",i),
                              100, -3., 3., 100, -3., 3., 100, -3., 3.);
         fOutput->Add(fHist3[i]);
      }
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
   for (Int_t i=0; i < fNhist3; i++) {
      if (fRandom && fHist3[i]) {
         Double_t x = fRandom->Gaus(0.,1.);
         fHist3[i]->Fill(x,x,x);
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
