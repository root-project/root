#include "ROOT/TDataFrame.hxx"
#include "TFile.h"
#include "TString.h"
#include "TTree.h"

#include <stdexcept>

int main() {
   {
      TFile f("testtypeguessing.root", "RECREATE");
      TTree t("myTree", "myTree");
      int b = 42;
      t.Branch("b", &b);
      TString s = "fortytwo";
      t.Branch("s", &s);
      t.Fill();
      t.Write();
   }

   TFile f("testtypeguessing.root");
   ROOT::Experimental::TDataFrame d("myTree", &f);
   // TTreeReader should cause a runtime error (type mismatch) when the event-loop is run
   auto hb = d.Histo1D<double>("b");

   // this should throw an exception because TString is not a guessable type
   try {
      auto hs = d.Histo1D("s");
   } catch (const std::runtime_error& error) {
      std::cout << error.what() << std::endl;
   }
   *hb;

   return 0;
}
