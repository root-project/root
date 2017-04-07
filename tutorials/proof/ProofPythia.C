/// \file
/// \ingroup tutorial_proofpythia
///
/// Selector to generate Monte Carlo events with Pythia8
///
/// \macro_code
///
/// \author Gerardo Ganis (gerardo.ganis@cern.ch)

#define ProofPythia_cxx

#include <TCanvas.h>
#include <TFrame.h>
#include <TPaveText.h>
#include <TFormula.h>
#include <TF1.h>
#include <TH1F.h>
#include <TMath.h>
#include <TString.h>
#include <TStyle.h>
#include <TSystem.h>
#include <TParameter.h>
#include "TClonesArray.h"
#include "TParticle.h"
#include "TDatabasePDG.h"

#include "ProofPythia.h"
#include "TPythia8.h"

//_____________________________________________________________________________
ProofPythia::ProofPythia()
{
   // Constructor

   fHist = 0;
   fPt = 0;
   fEta = 0;
   fPythia = 0;
   fP = 0;
}

//_____________________________________________________________________________
ProofPythia::~ProofPythia()
{
   // Destructor

   SafeDelete(fPythia);
   SafeDelete(fP);
}

//_____________________________________________________________________________
void ProofPythia::Begin(TTree * /*tree*/)
{
   // The Begin() function is called at the start of the query.
   // When running with PROOF Begin() is only called on the client.
   // The tree argument is deprecated (on PROOF 0 is passed).

   TString option = GetOption();
   Info("Begin", "starting a simple exercise with process option: %s", option.Data());
}

//_____________________________________________________________________________
void ProofPythia::SlaveBegin(TTree * /*tree*/)
{
   // The SlaveBegin() function is called after the Begin() function.
   // When running with PROOF SlaveBegin() is called on each slave server.
   // The tree argument is deprecated (on PROOF 0 is passed).

   TString option = GetOption();

   // Histograms
   fTot = new TH1F("histo1", "total multiplicity", 25, 0.5, 2500.5);
   fHist = new TH1F("histo2", "charged multiplicity", 20, 0.5, 500.5);
   fPt = new TH1F("histo3", "particles pT", 100, 0., 10);
   fEta = new TH1F("histo4", "particles Eta", 100, -10., 10);
   fTot->SetFillColor(kBlue);
   fHist->SetFillColor(kRed);
   fOutput->Add(fTot);
   fOutput->Add(fHist);
   fOutput->Add(fPt);
   fOutput->Add(fEta);

   fPythia = new TPythia8();
   // Configure
   fPythia->SetName("pythia8");
   fPythia->ReadConfigFile("pythia8/main03.cmnd");

   // Initialize
   fPythia->Initialize( 2212, 2212, 14000.);
   fP = new TClonesArray("TParticle", 1000);

}

//_____________________________________________________________________________
Bool_t ProofPythia::Process(Long64_t entry)
{
   // Main event loop

   fPythia->GenerateEvent();
   if (entry < 2)
      fPythia->EventListing();
   fPythia->ImportParticles(fP, "All");
   Int_t nTot = fPythia->GetN();
   fPythia->ImportParticles(fP, "All");
   Int_t np = fP->GetEntriesFast();
   // Particle loop
   Int_t nCharged = 0;
   for (Int_t ip = 0; ip < np; ip++) {
      TParticle* part = (TParticle*) fP->At(ip);
      Int_t ist = part->GetStatusCode();
      Int_t pdg = part->GetPdgCode();
      if (ist != 1) continue;
      Float_t charge = TDatabasePDG::Instance()->GetParticle(pdg)->Charge();
      if (charge == 0.) continue;
      nCharged++;
      Float_t eta = part->Eta();
      Float_t pt  = part->Pt();
      if (pt > 0.) fPt->Fill(pt);
      if ((eta > -10) && (eta < 10)) fEta->Fill(eta);
   }
   fHist->Fill(nCharged);
   fTot->Fill(nTot);

   return kTRUE;
}

//_____________________________________________________________________________
void ProofPythia::SlaveTerminate()
{
   // The SlaveTerminate() function is called after all entries or objects
   // have been processed. When running with PROOF SlaveTerminate() is called
   // on each slave server.
}

//_____________________________________________________________________________
void ProofPythia::Terminate()
{
   // The Terminate() function is the last function to be called during
   // a query. It always runs on the client, it can be used to present
   // the results graphically or save the results to file.

   //
   // Create canvas
   //
   TCanvas *c1 = new TCanvas("c1","Proof ProofPythia canvas",200,10,700,700);
   c1->Divide(2, 2);

   if ((fTot = dynamic_cast<TH1F *>(fOutput->FindObject("histo1")))) {
      c1->cd(1);
      fTot->Draw("h");
   }

   if ((fHist = dynamic_cast<TH1F *>(fOutput->FindObject("histo2")))) {
      c1->cd(2);
      fHist->Draw("h");
   }

   if ((fPt = dynamic_cast<TH1F *>(fOutput->FindObject("histo3")))) {
      c1->cd(3);
      fPt->Draw("h");
   }

   if ((fEta = dynamic_cast<TH1F *>(fOutput->FindObject("histo4")))) {
      c1->cd(4);
      fEta->Draw("h");
   }

   // Final update
   c1->cd();
   c1->Update();
}
