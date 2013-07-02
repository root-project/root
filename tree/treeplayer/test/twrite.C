#include "MyParticle.h"
#include "TLorentzVector.h"
#include "TRandom.h"
#include "TTree.h"
#include "TFile.h"
#include "TInterpreter.h"
#include "TClonesArray.h"

// FOR NOW:
void twrite() {
   gInterpreter->GenerateDictionary("vector<MyParticle*>", "vector;MyParticle.h");
   gInterpreter->GenerateDictionary("vector<MyParticle>", "vector;MyParticle.h");
   TFile* file = TFile::Open("tr.root", "RECREATE");
   TTree* tree = new TTree("T", "TTreeWriter test tree");
   std::vector<MyParticle*>* vpmuons = new std::vector<MyParticle*>();
   std::vector<MyParticle>* vmuons = new std::vector<MyParticle>();
   TClonesArray* /*MyParticle* */ camuons = new TClonesArray("MyParticle");
   ParticleHolder *phmuons = new ParticleHolder();

   //tree->Branch("vp", &vpmuons, 32000, 250);
   printf("WARNING: TBranchProxy does not support BranchSTL yet!\n");

   tree->Branch("v", &vmuons);
   tree->Branch("ca", &camuons);
   tree->Branch("ph", &phmuons);
   for (int i = 0; i < 10; ++i) {
      if (vpmuons) {
         int nOldMuons = (int)vpmuons->size();
         for (int m = 0; m < nOldMuons; ++m) {
            delete vpmuons->at(m);
         }
         vpmuons->clear();
         vmuons->clear();
         camuons->Clear();
         phmuons->Clear();
      }

      int nMuons = gRandom->Uniform(12);
      phmuons->SetN(nMuons);
      for (int m = 0; m < nMuons; ++m) {
         double x[7];
         gRandom->RndmArray(7, x);
         MyPos pos(x);
         vpmuons->push_back(new MyParticle(x+3, pos));
         MyParticle* p = new ((*camuons)[m]) MyParticle(x+3, pos);
         vmuons->push_back(*p);
         phmuons->Set(m, *p);
      }
      tree->Fill();
   }
   tree->Write();
   delete file;
}

// IN THE FUTURE:
#if 0
void twrite() {
   gInterpreter->GenerateDictionary("vector<TParticle*>", "vector;TParticle.h");
   TFile* file = TFile::Open("tr.root", "RECREATE");
   TTreeWriter t(file, "T");
   TTreeWriterValuePtr< std::vector<TParticle*> > muons(t, "muons");
   for (int i = 0; i < 10; ++i) {
      for (int m = 0; m < nMuons; ++m) {
         delete muons->at(m);
      }
      muons->clear();
      nMuons = gRandom->Uniform(12);
      for (int m = 0; m < nMuons; ++m) {
         TLorentzVector p(gRandom->Gaus(), gRandom->Gaus(), gRandom->Gaus(), gRandom->Gaus());
         TLorentzVector v(gRandom->Gaus(), gRandom->Gaus(), gRandom->Gaus(), gRandom->Gaus());
         muons->push_back(new TParticle(1, 0, 0, 0, 0, 0, p, v));
      }
      t.Fill();
    }
    delete file;
}
#endif
