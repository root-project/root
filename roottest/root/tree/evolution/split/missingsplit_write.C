#include "TFile.h"
#include "TTree.h"

class Content {
public:
   Content() : a(0),b(0) {}
   void Set(int i) { a = i; b = 2*i; }
   int a;
   int b;
};

class MyContainer {
public:
   MyContainer() : one(), two() {}
   void Set(int i) { one.Set(i); two.Set(2+i); }
   Content one;
   Content two;
};

void missingsplit_write(const char *filename = "missingsplit.root") {
   TFile *f = new TFile(filename,"RECREATE");
   TTree *tree = new TTree("T","T");
   MyContainer cont;
   cont.Set(3);
   tree->Branch("cont.",&cont);
   tree->Fill();
   f->Write();
   delete f;
}