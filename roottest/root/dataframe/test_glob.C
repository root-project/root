#include "TFile.h"
#include "TROOT.h"
#include "TTree.h"

#include "ROOT/RDataFrame.hxx"

// A simple helper function to fill a test tree and save it to file
// This makes the example stand-alone
void FillTree(const char* filename, const char* treeName) {
   TFile f(filename,"RECREATE");
   TTree t(treeName,treeName);
   double b1;
   t.Branch("b1", &b1);
   for(int i = 0; i < 10; ++i) {
      b1 = i;
      t.Fill();
   }
   t.Write();
   f.Close();
   return;
}

const char* treeName = "myTree";

void run() {
   auto ok = []() { return true; };
   auto ko = []() { return false; };

   // Define data-frame
   ROOT::RDataFrame d(treeName, "test_glob_*root");
   auto c1 = d.Count();

   auto dd = d.Filter(ok);
   auto c2 = dd.Count();

   auto ddd = d.Filter(ko);
   auto c3 = ddd.Count();

   std::cout << "c1 " << *c1 << std::endl;
   std::cout << "c2 " << *c2 << std::endl;
   std::cout << "c3 " << *c3 << std::endl;
}

int test_glob() {
   // Prepare an input tree to run on
   for (int i=0;i<5;++i) FillTree(TString::Format("test_glob_%i.root", i),treeName);

   run();
#ifdef R__USE_IMT
   ROOT::EnableImplicitMT();
#endif
   run();

   return 0;
}
