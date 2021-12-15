#include "ROOT/RDataFrame.hxx"
#include "TFile.h"
#include "TTree.h"

#include <string>
#include <stdexcept>
#include <cassert>

auto fileName("testtypeguessing.root");
auto treeName("myTree");

int main() {
   {
      TFile f(fileName, "RECREATE");
      TTree t(treeName, treeName);
      int b = 42;
      t.Branch("b", &b);
      std::string s = "fortytwo";
      t.Branch("s", &s);
      t.Fill();
      t.Write();
   }

   TFile f(fileName);
   ROOT::RDataFrame d(treeName, fileName);

   // TTreeReader should cause a runtime error (type mismatch) when the event-loop is run
   auto hb = d.Histo1D<double>("b");

   bool exception_caught = false;
   try {
      *hb;
   } catch (const std::runtime_error &) {
      exception_caught = true;
   }
   R__ASSERT(exception_caught);

   return 0;
}
