#define EventTree_Proc_cxx
// The class definition in EventTree_Proc.h has been generated automatically
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
// Root > T->Process("EventTree_Proc.C")
// Root > T->Process("EventTree_Proc.C","some options")
// Root > T->Process("EventTree_Proc.C+")
//

#include "EventTree_Proc.h"
#include <TH2.h>
#include <TStyle.h>
#include "TCanvas.h"


void EventTree_Proc::Begin(TTree *)
{
   // The Begin() function is called at the start of the query.
   // When running with PROOF Begin() is only called on the client.
   // The tree argument is deprecated (on PROOF 0 is passed).

   TString option = GetOption();

}

void EventTree_Proc::SlaveBegin(TTree *tree)
{
   // The SlaveBegin() function is called after the Begin() function.
   // When running with PROOF SlaveBegin() is called on each slave server.
   // The tree argument is deprecated (on PROOF 0 is passed).

   Init(tree);

   TString option = GetOption();

   fPtHist = new TH1F("pt_dist","p_{T} Distribution",100,0,5);
   fPtHist->SetDirectory(0);
   fPtHist->GetXaxis()->SetTitle("p_{T}");
   fPtHist->GetYaxis()->SetTitle("dN/p_{T}dp_{T}");

   fOutput->Add(fPtHist);

   fNTracksHist = new TH1I("ntracks_dist","N_{Tracks} per Event Distribution",5,0,5);
   fNTracksHist->SetDirectory(0);
   fNTracksHist->GetXaxis()->SetTitle("N_{Tracks}");
   fNTracksHist->GetYaxis()->SetTitle("N_{Events}");

   fOutput->Add(fNTracksHist);

}

Bool_t EventTree_Proc::Process(Long64_t entry)
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

   fChain->GetTree()->GetEntry(entry);

   fNTracksHist->Fill(fNtrack);

   for(Int_t j=0;j<fTracks->GetEntries();j++){
     Track* curtrack = dynamic_cast<Track*>(fTracks->At(j));
     fPtHist->Fill(curtrack->GetPt(),1./curtrack->GetPt());
   }
   fTracks->Clear("C");

   return kTRUE;
}

void EventTree_Proc::SlaveTerminate()
{
   // The SlaveTerminate() function is called after all entries or objects
   // have been processed. When running with PROOF SlaveTerminate() is called
   // on each slave server.

}

void EventTree_Proc::Terminate()
{
   // The Terminate() function is the last function to be called during
   // a query. It always runs on the client, it can be used to present
   // the results graphically or save the results to file.

   TCanvas* canvas = new TCanvas("can","can",800,600);
   canvas->SetBorderMode(0);
   canvas->SetLogy();
   TH1F* h = dynamic_cast<TH1F*>(fOutput->FindObject("pt_dist"));
   if (h) h->DrawCopy();
   else Warning("Terminate", "no pt dist found");

}
