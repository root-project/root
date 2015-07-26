#define ProofStdVect_cxx

//////////////////////////////////////////////////////////
//
// Example of TSelector implementation to do generic
// processing with stdlib collections.
// See tutorials/proof/runProof.C, option "stdlib", for an
// example of how to run this selector.
//
//////////////////////////////////////////////////////////

#include "ProofStdVect.h"
#include <TMath.h>
#include <TTree.h>
#include <TRandom3.h>
#include <TROOT.h>
#include <TString.h>
#include <TSystem.h>
#include <TFile.h>
#include <TProofOutputFile.h>
#include <TCanvas.h>
#include <TH1F.h>

//_____________________________________________________________________________
ProofStdVect::ProofStdVect()
{
   // Constructor

   fCreate = kFALSE;
   fTree = 0;
   fFile = 0;
   fProofFile = 0;
   fRandom = 0;
   fHgood = 0;
   fHbad = 0;
}

//_____________________________________________________________________________
ProofStdVect::~ProofStdVect()
{
   // Destructor

   SafeDelete(fTree);
   SafeDelete(fFile);
   SafeDelete(fRandom);
}

//_____________________________________________________________________________
void ProofStdVect::Begin(TTree * /*tree*/)
{
   // The Begin() function is called at the start of the query.
   // When running with PROOF Begin() is only called on the client.
   // The tree argument is deprecated (on PROOF 0 is passed).

   TString option = GetOption();

   // Dataset creation run?
   if (fInput && fInput->FindObject("ProofStdVect_Create")) {
      fCreate = kTRUE;
   } else if (option.Contains("create")) {
      fCreate = kTRUE;
   }
}

//_____________________________________________________________________________
void ProofStdVect::SlaveBegin(TTree * /*tree*/)
{
   // The SlaveBegin() function is called after the Begin() function.
   // When running with PROOF SlaveBegin() is called on each slave server.
   // The tree argument is deprecated (on PROOF 0 is passed).

   TString option = GetOption();

   // Dataset creation run?
   if (fInput && fInput->FindObject("ProofStdVect_Create")) {
      fCreate = kTRUE;
   } else if (option.Contains("create")) {
      fCreate = kTRUE;
   }

   // If yes, create the output file ...
   if (fCreate) {
      // Just create the object
      UInt_t opt = TProofOutputFile::kRegister | TProofOutputFile::kOverwrite | TProofOutputFile::kVerify;
      fProofFile = new TProofOutputFile("ProofStdVect.root",
                                        TProofOutputFile::kDataset, opt, "TestStdVect");

      // Open the file
      fFile = fProofFile->OpenFile("RECREATE");
      if (fFile && fFile->IsZombie()) SafeDelete(fFile);

      // Cannot continue
      if (!fFile) {
         Info("SlaveBegin", "could not create '%s': instance is invalid!", fProofFile->GetName());
         return;
      }

      // Create a TTree
      fTree = new TTree("stdvec", "Tree with std vector");
      fTree->Branch("Vb",&fVb);
      fTree->Branch("Vfx",&fVfx);
      fTree->Branch("Vfy",&fVfy);
      // File resident
      fTree->SetDirectory(fFile);
      fTree->AutoSave();

      // Init the random generator
      fRandom = new TRandom3(0);

   } else {
      // Create two histograms
      fHgood = new TH1F("Hgood", "Good hits", 100., -2.5, 2.5);
      fHbad = new TH1F("Hbad", "Bad hits", 100., -6., 6.);
      fOutput->Add(fHgood);
      fOutput->Add(fHbad);
   }
}

//_____________________________________________________________________________
Bool_t ProofStdVect::Process(Long64_t entry)
{
   // The Process() function is called for each entry in the tree (or possibly
   // keyed object in the case of PROOF) to be processed. The entry argument
   // specifies which entry in the currently loaded tree is to be processed.
   // It can be passed to either ProofStdVect::GetEntry() or TBranch::GetEntry()
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

   if (fCreate) {
      if (!fTree) return kTRUE;

      // Number of vectors
      Int_t nv =  (Int_t) (entry % 10);
      if (nv < 1) nv = 1;

      // Create vectors
      for (Int_t i = 0; i < nv; i++) {
         std::vector<bool> vb;
         std::vector<float> vfx, vfy;
         Int_t np =  (Int_t) (entry % 100);
         if (np < 1) np = 1;
         for (Int_t j = 0; j < np; j++) {
            float x = (float)j;
            float y = 5.*x;
            Double_t sy = (Double_t) (0.1*y);
            Double_t ym = fRandom->Gaus((Double_t)y, sy);
            Double_t c2 = TMath::Abs((ym - y) / sy);
            bool xb = (1. - TMath::Erfc(c2/TMath::Sqrt(2.)) > .95) ? 0 : 1;
            vb.push_back(xb);
            vfx.push_back(x);
            vfy.push_back(float(ym));
         }
         fVb.push_back(vb);
         fVfx.push_back(vfx);
         fVfy.push_back(vfy);
      }

      // Fill the tree
      fTree->Fill();

      // Clear the vectors
      std::vector<std::vector<bool> >::iterator ivb;
      for (ivb = fVb.begin(); ivb != fVb.end(); ivb++) {
         (*ivb).clear();
      }
      fVb.clear();
      std::vector<std::vector<float> >::iterator ivf;
      for (ivf = fVfx.begin(); ivf != fVfx.end(); ivf++) {
         (*ivf).clear();
      }
      fVfx.clear();
      for (ivf = fVfy.begin(); ivf != fVfy.end(); ivf++) {
         (*ivf).clear();
      }
      fVfy.clear();
   } else {
      // Read the entry
      GetEntry(entry);
      // Plot normalized values for bad and good hits
      for (UInt_t i = 0; i < fVfyr->size(); i++) {
         std::vector<bool> &vb = fVbr->at(i);
         std::vector<float> &vfx = fVfxr->at(i);
         std::vector<float> &vfy = fVfyr->at(i);
         for (UInt_t j = 0; j < vfy.size(); j++) {
            Double_t ny = (vfy.at(j) - 5*vfx.at(j)) / (0.1 * 5 * vfx.at(j));
            if (vb.at(j) < 0.5)
               fHbad->Fill(ny);
            else
               fHgood->Fill(ny);
         }
      }
   }

   return kTRUE;
}

//_____________________________________________________________________________
void ProofStdVect::SlaveTerminate()
{
   // The SlaveTerminate() function is called after all entries or objects
   // have been processed. When running with PROOF SlaveTerminate() is called
   // on each slave server.

   // Nothing to do in read mode
   if (!fCreate) return;

   // Write the ntuple to the file
   if (fFile) {
      if (!fTree) {
         Error("SlaveTerminate", "'tree' is undefined!");
         return;
      }
      Bool_t cleanup = kFALSE;
      TDirectory::TContext ctxt;
      if (fTree->GetEntries() > 0) {
         fFile->cd();
         fTree->Write();
         fProofFile->Print();
         fOutput->Add(fProofFile);
      } else {
         cleanup = kTRUE;
      }
      fTree->SetDirectory(0);
      fFile->Close();
      // Cleanup, if needed
      if (cleanup) {
         TUrl uf(*(fFile->GetEndpointUrl()));
         SafeDelete(fFile);
         gSystem->Unlink(uf.GetFile());
         SafeDelete(fProofFile);
      }
   }
}

//_____________________________________________________________________________
void ProofStdVect::Terminate()
{
   // The Terminate() function is the last function to be called during
   // a query. It always runs on the client, it can be used to present
   // the results graphically or save the results to file.

   // Nothing to do in create mode
   if (fCreate) return;

   // Create a canvas, with 2 pads
   TCanvas *c1 = new TCanvas("cvstdvec", "Test StdVec", 800,10,700,780);
   c1->Divide(1,2);
   TPad *pad1 = (TPad *) c1->GetPad(1);
   TPad *pad2 = (TPad *) c1->GetPad(2);
   pad2->cd();
   if (fHbad) fHbad->Draw();
   pad1->cd();
   if (fHgood) fHgood->Draw();
   c1->cd();
   c1->Update();
}

//_____________________________________________________________________________
void ProofStdVect::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Nothing to do in create mode
   if (fCreate) return;

   // Set object pointer
   fVbr = 0;
   fVfxr = 0;
   fVfyr = 0;
   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("Vb", &fVbr, &b_Vb);
   fChain->SetBranchAddress("Vfx", &fVfxr, &b_Vfx);
   fChain->SetBranchAddress("Vfy", &fVfyr, &b_Vfy);
}


//_____________________________________________________________________________
Bool_t ProofStdVect::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   // Nothing to do in create mode
   if (fCreate) return kTRUE;
   Info("Notify","processing file: %s",fChain->GetCurrentFile()->GetName());

   return kTRUE;
}
