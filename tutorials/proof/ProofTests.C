#define ProofTests_cxx
// The class definition in ProofTests.h has been generated automatically
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
// Root > T->Process("ProofTests.C")
// Root > T->Process("ProofTests.C","some options")
// Root > T->Process("ProofTests.C+")
//

#include "ProofTests.h"
#include <TH1F.h>
#include <TH1I.h>
#include <TMath.h>
#include <TString.h>
#include <TSystem.h>
#include <TParameter.h>

//_____________________________________________________________________________
ProofTests::ProofTests()
{
   // Constructor

   fTestType = 0;
   fStat = 0;
}

//_____________________________________________________________________________
ProofTests::~ProofTests()
{
   // Destructor

}

//_____________________________________________________________________________
void ProofTests::Begin(TTree * /*tree*/)
{
   // The Begin() function is called at the start of the query.
   // When running with PROOF Begin() is only called on the client.
   // The tree argument is deprecated (on PROOF 0 is passed).

   TString option = GetOption();

   // Determine the test type
   TNamed *ntype = dynamic_cast<TNamed*>(fInput->FindObject("ProofTests_Type"));
   if (ntype) {
      if (!strcmp(ntype->GetTitle(), "InputData")) {
         fTestType = 0;
      } else {
         Info("Begin", "unknown type: '%s'", ntype->GetTitle());
      }
   }
}

//_____________________________________________________________________________
void ProofTests::SlaveBegin(TTree * /*tree*/)
{
   // The SlaveBegin() function is called after the Begin() function.
   // When running with PROOF SlaveBegin() is called on each slave server.
   // The tree argument is deprecated (on PROOF 0 is passed).

   TString option = GetOption();

   // Output histo
   fStat = new TH1I("TestStat", "Test results", 20, .5, 20.5);
   fOutput->Add(fStat);

   // We were started
   fStat->Fill(1.);

   // Depends on the test
   if (fTestType == 0) {
      // Retrieve objects from the input list and copy them to the output
      // H1
      TList *h1list = dynamic_cast<TList*>(fInput->FindObject("h1list"));
      if (h1list) {
         // Retrieve objects from the input list and copy them to the output
         TH1F *h1 = dynamic_cast<TH1F*>(h1list->FindObject("h1data"));
         if (h1) {
            TParameter<Double_t> *h1avg = dynamic_cast<TParameter<Double_t>*>(h1list->FindObject("h1avg"));
            TParameter<Double_t> *h1rms = dynamic_cast<TParameter<Double_t>*>(h1list->FindObject("h1rms"));
            if (h1avg && h1rms) {
               if (TMath::Abs(h1avg->GetVal() - h1->GetMean()) < 0.0001) {
                  if (TMath::Abs(h1rms->GetVal() - h1->GetRMS()) < 0.0001) {
                     fStat->Fill(2.);
                  }
               }
            } else {
               Info("BeginSlave", "info 'h1avg' or 'h1rms' not found!");
            }
         } else {
            Info("BeginSlave", "input histo 'h1data' not found!");
         }
      } else {
         Info("BeginSlave", "input list 'h1list' not found!");
      }
      // H2
      TList *h2list = dynamic_cast<TList*>(fInput->FindObject("h2list"));
      if (h2list) {
         // Retrieve objects from the input list and copy them to the output
         TH1F *h2 = dynamic_cast<TH1F*>(h2list->FindObject("h2data"));
         if (h2) {
            TParameter<Double_t> *h2avg = dynamic_cast<TParameter<Double_t>*>(h2list->FindObject("h2avg"));
            TParameter<Double_t> *h2rms = dynamic_cast<TParameter<Double_t>*>(h2list->FindObject("h2rms"));
            if (h2avg && h2rms) {
               if (TMath::Abs(h2avg->GetVal() - h2->GetMean()) < 0.0001) {
                  if (TMath::Abs(h2rms->GetVal() - h2->GetRMS()) < 0.0001) {
                     fStat->Fill(3.);
                  }
               }
            } else {
               Info("BeginSlave", "info 'h2avg' or 'h2rms' not found!");
            }
         } else {
            Info("BeginSlave", "input histo 'h2data' not found!");
         }
      } else {
         Info("BeginSlave", "input list 'h2list' not found!");
      }

      TNamed *iob = dynamic_cast<TNamed*>(fInput->FindObject("InputObject"));
      if (iob) {
         fStat->Fill(4.);
      } else {
         Info("BeginSlave", "input histo 'InputObject' not found!");
      }
   }
}

//_____________________________________________________________________________
Bool_t ProofTests::Process(Long64_t)
{
   // The Process() function is called for each entry in the tree (or possibly
   // keyed object in the case of PROOF) to be processed. The entry argument
   // specifies which entry in the currently loaded tree is to be processed.
   // It can be passed to either ProofTests::GetEntry() or TBranch::GetEntry()
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

   return kTRUE;
}

//_____________________________________________________________________________
void ProofTests::SlaveTerminate()
{
   // The SlaveTerminate() function is called after all entries or objects
   // have been processed. When running with PROOF SlaveTerminate() is called
   // on each slave server.

}

//_____________________________________________________________________________
void ProofTests::Terminate()
{
   // The Terminate() function is the last function to be called during
   // a query. It always runs on the client, it can be used to present
   // the results graphically or save the results to file.

}
