// a test for ROOT-9757

#include <vector>
#include "TTree.h"
#include "TFile.h"

struct A {
   std::vector<int> A1;
};

struct B: public A {
   std::vector<float> B1;
};

void initOffset() {
   TFile* file = new TFile("T.root", "RECREATE");
   TTree* tree = new TTree("T", "T");
   B b;
   b.A1.push_back(42);
   b.B1.push_back(17.);
   tree->Branch("B", &b);
   tree->Fill();
   tree->Write();
   delete file;
}
