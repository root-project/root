#include "TFile.h"
#include "TTree.h"
#include "ROOT/TDataFrame.hxx"
#include "TROOT.h"
#include <iomanip>

void FillTree(const char* filename, const char* treeName) {
   TFile f(filename, "RECREATE");
   TTree t(treeName, treeName);
   int i;
   t.Branch("i", &i);
   for(i=0; i<100000000; ++i)
      t.Fill();
   t.Write();
   f.Close();
}


int main() {
   // TODO add a section without EnableImplicitMT enabled

   ROOT::EnableImplicitMT(4);

   // Prepare an input tree to run on
   auto fileName = "test_par.root";
   auto treeName = "myTree";
   FillTree(fileName, treeName);
   TFile f(fileName);

   ROOT::Experimental::TDataFrame d(treeName, &f, {"i"});
   auto max = d.Filter([](int i) { return i % 2 == 1; }).Max();
   auto min = d.Min();
   auto mean = d.Mean();
   auto h = d.Histo1D();
   std::cout << std::setprecision(10) << "max " << *max << std::endl;
   std::cout << std::setprecision(10) << "min " << *min << std::endl;
   std::cout << std::setprecision(10) << "mean " << *mean << std::endl;
   std::cout << std::setprecision(10) << "h entries " << h->GetEntries() << std::endl;
   return 0;
}
