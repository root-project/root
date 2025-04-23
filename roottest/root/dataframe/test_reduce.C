#include "ROOT/RDataFrame.hxx"
#include "TError.h" // Info
#include "TFile.h"
#include "TTree.h"
#include "TROOT.h"

class NoDefCtor {
   int fInt;
public:
   NoDefCtor() = delete;
   NoDefCtor(TRootIOCtor*) {}
   NoDefCtor(int i) : fInt(i) {}
   void SetInt(int i) { fInt = i; }
   int GetInt() const { return fInt; }
};

void FillTree(const char* filename, const char* treeName) {
   TFile f(filename, "RECREATE");
   TTree t(treeName, treeName);
   int i;
   NoDefCtor o(0);
   t.Branch("i", &i);
   t.Branch("o", &o);
   for(i = 1; i <= 10; ++i) {
      o.SetInt(i);
      t.Fill();
   }
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
   ROOT::RDataFrame d("reduceTree", fileName, {"i"});
   auto r = d.Reduce([](int a, int b) { return a + b; }, {"i"});
   auto rDefBranch = d.Filter([]() { return true; })
                      .Reduce([](int a, int b) { return a*b; }, "", 1);
   auto rNoDefCtor = d.Reduce(
      [](const NoDefCtor& a, const NoDefCtor& b) { return NoDefCtor(a.GetInt() + b.GetInt()); },
      {"o"},
      NoDefCtor(0));

   Info("test_reduce", "%d %d %d", *r, *rDefBranch, rNoDefCtor->GetInt());
}
