#include "ROOT/TDataFrame.hxx"
#include "TError.h" // Info
#include "TFile.h"
#include "TTree.h"

void FillTree(const char* filename, const char* treeName) {
   TFile f(filename, "RECREATE");
   TTree t(treeName, treeName);
   int i;
   t.Branch("i", &i);
   for(i = 1; i <= 10; ++i)
      t.Fill();
   t.Write();
   f.Close();
}

void test_reduce() {
   auto fileName = "test_reduce.root";
   auto treeName = "reduceTree";
   FillTree(fileName, treeName);

#ifdef R__USE_IMT
   ROOT::EnableImplicitMT();
#endif
   TFile f(fileName);
   ROOT::Experimental::TDataFrame d("reduceTree", &f, {"i"});
   auto r = d.Reduce([](int a, int b) { return a + b; }, {"i"});
   auto rDefBranch = d.Filter([]() { return true; })
                      .Reduce([](int a, int b) { return a*b; }, {}, 1);

   Info("test_reduce", "%d %d", *r, *rDefBranch);
}

int main() {
   test_reduce();
   return 0;
}
