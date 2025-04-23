#include <TLorentzVector.h>
#include <TApplication.h>
#include <TTree.h>
#include <TFile.h>
#include <TClonesArray.h>
#include <iostream>

namespace std {} using namespace std;

class Jet : public TLorentzVector {
public:
   Jet(Double_t x = 0.0, Double_t y = 0.0,
       Double_t z = 0.0, Double_t t = 0.0,
       Double_t val = 0.0) : TLorentzVector(x,y,z,t),jetVal(val)  {};
   ~Jet() override {};
   
   Double_t jetVal;

   ClassDefOverride(Jet,1);
};

void writeJet() {
   TClonesArray* caJet=new TClonesArray("Jet");

   TFile *file = new TFile("jet.root","recreate");
   TTree* tree=new TTree("test","test",99);
   tree->Branch("jet", "TClonesArray",&caJet, 32000);

   Int_t event = 0;
   Int_t njet  = 0;
   
   for(event=0; event<3; event++) {
      
      for(njet=0;njet<5; njet++) {
         new ( (*caJet)[njet] ) Jet(event,33,44,55, njet);
      }
      tree->Fill();
      caJet->Clear();
   }
   file->Write();
   delete file;
}

void readJet() {
   TFile *file = new TFile("jet.root","READ");
   TTree *tree = (TTree*)file->Get("test");
   TClonesArray* caJet=0;
   tree->SetBranchAddress("jet",&caJet);

   for(Int_t event=0; event<3; event++) {
      tree->GetEntry(event);
      
      assert(caJet);

      for(Int_t njet=0;njet<5; njet++) {
         Jet *j = (Jet*)caJet->At(njet);
         if (j==0) {
            cerr << "In event " << event 
                 << " missing jet #" << njet << endl;
            gApplication->Terminate(1);
         }
         bool failed = false;
         if (j->X()!= event || j->Y()!=33 || j->Z()!=44 || j->T()!=55
             || j->jetVal != njet ) {
            failed = true;
         }
         cerr << "In event " << event 
              << " jet #" << njet;
         if (failed) cerr << " values are incorrect: ";
         else cerr << " value are correct ";
         cerr << j->X() << "," 
              << j->Y() << ","
              << j->Z() << ","
              << j->T() << ","
              << j->jetVal
              << endl;
         if (failed) {
            gApplication->Terminate(1);
         }
      }
   }

   delete file;
}
