#include "TFile.h"
#include "TTree.h"
#include "TROOT.h"

#include "ROOT/RDataFrame.hxx"

#include <iostream>

auto filename("test_branchoverwrite.root");

int main()
{
   {
      TFile wf(filename, "RECREATE");
      TTree t("emptyTree", "emptyTree");
      int a;
      t.Branch("a", &a);
      t.Write();
   }

   ROOT::RDataFrame d("emptyTree", filename, {"a"});
   d.Define("b", []() { return 8; });
   try {
      auto c = d.Define("a", []() { return 42; });
   } catch (const std::runtime_error &e) {
      std::cout << "Exception caught: " << e.what() << std::endl;
   }

   return 0;
}
