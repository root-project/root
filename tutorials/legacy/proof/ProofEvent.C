/// \file
/// \ingroup tutorial_ProofEvent
///
/// Selector for generic processing with Event
///
/// \macro_code
///
/// \author Gerardo Ganis (gerardo.ganis@cern.ch)

#define ProofEvent_cxx

#include "ProofEvent.h"
#include "Event.h"

#include <TCanvas.h>
#include <TH1F.h>
#include <TRandom3.h>

//_____________________________________________________________________________
ProofEvent::ProofEvent()
{
   // Constructor

   fEvent = 0;
   fNtrack = -1;
   fHisto = 0;
   fRandom = 0;
}

//_____________________________________________________________________________
ProofEvent::~ProofEvent()
{
   // Destructor

   SafeDelete(fRandom);
}

//_____________________________________________________________________________
void ProofEvent::Begin(TTree * /*tree*/)
{
   // The Begin() function is called at the start of the query.
   // When running with PROOF Begin() is only called on the client.
   // The tree argument is deprecated (on PROOF 0 is passed).

   TString option = GetOption();
   Info("Begin", "starting a simple exercise with process option: %s", option.Data());
}

//_____________________________________________________________________________
void ProofEvent::SlaveBegin(TTree * /*tree*/)
{
   // The SlaveBegin() function is called after the Begin() function.
   // When running with PROOF SlaveBegin() is called on each slave server.
   // The tree argument is deprecated (on PROOF 0 is passed).

   TString option = GetOption();
   Info("SalveBegin", "starting on a slave with process option: %s", option.Data());

   // Create event
   fEvent = new Event();

   // Create the histogram
   fHisto = new TH1F("histo", "tracks multiplicity", 20, 0, 100);
   fHisto->GetYaxis()->SetTitle("number of events");
   fHisto->GetXaxis()->SetTitle("number of tracks");

   //adding histo to selector output list
   fOutput->Add(fHisto);

   // Set random seed
   fRandom = new TRandom3(0);
}

//_____________________________________________________________________________
Bool_t ProofEvent::Process(Long64_t )
{

  // Start main loop over all events
  // get a random parameter for connstructing event

   int i= (int)(100 * (fRandom->Rndm()));
   fEvent->Build(i,(1+i), 2);
   fNtrack= (fEvent->GetNtrack());
   if ((fNtrack >= 0 )&& (fNtrack <= 100 ))
      fHisto->Fill(fNtrack, 1);

   return kTRUE;
}

//_____________________________________________________________________________
void ProofEvent::SlaveTerminate()
{
   //nothing to be done

}

//_____________________________________________________________________________
void ProofEvent::Terminate()
{
   // The Terminate() function is the last function to be called during
   // a query. It always runs on the client, it can be used to present
   // the results graphically or save the results to file.

   TCanvas *c1 = new TCanvas("c1","Proof ProofEvent canvas",200,10,700,700);
   fHisto = dynamic_cast<TH1F *>(fOutput->FindObject(Form("histo")));
   if (fHisto) {
      fHisto->Draw("h");

      // Final update
      c1->cd();
      c1->Update();
   } else {
      Warning("Terminate", "histogram not found");
   }
}
