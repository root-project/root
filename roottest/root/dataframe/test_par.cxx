#include "TFile.h"
#include "TTree.h"
#include "ROOT/RDataFrame.hxx"
#include "TROOT.h"
#include <iomanip>

void FillTree(const char* filename, const char* treeName) {
   TFile f(filename, "RECREATE");
   TTree t(treeName, treeName);
   int i;
   t.Branch("i", &i);
   for(i=0; i<1000000; ++i)
      t.Fill();
   t.Write();
   f.Close();
}


int main() {
   // reference output must be the same for parallel and sequential execution
#ifdef R__USE_IMT
   ROOT::EnableImplicitMT();
#endif

   // Prepare an input tree to run on
   auto fileName = "test_par.root";
   auto treeName = "myTree";
   FillTree(fileName, treeName);

   ROOT::RDataFrame d(treeName, fileName, {"i"});
   auto count = d.Count();
   auto max = d.Filter([](int i) { return i % 2 == 1; }).Max<int>();
   auto min = d.Min<int>();
   auto mean = d.Mean<int>();
   auto h = d.Histo1D<int>();
   std::cout << std::setprecision(10)
      << "count " << *count << "\n"
      << "max " << *max << "\n"
      << "min " << *min << "\n"
      << "mean " << *mean << "\n"
      << "h entries " << h->GetEntries() << std::endl;
   return 0;
}
