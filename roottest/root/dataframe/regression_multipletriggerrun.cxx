#include "ROOT/RDataFrame.hxx"
#include "TFile.h"
#include "TTree.h"

void fill_tree(const char* filename, const char* treeName) {
   TFile f(filename,"RECREATE");
   TTree t(treeName,treeName);
   double b1;
   t.Branch("b1", &b1);
   b1 = 1;
   t.Fill();
   t.Write();
   f.Close();
   return;
}

int main() {
   // We prepare an input tree to run on
   auto fileName = "regression_multipletriggerrun.root";
   auto treeName = "myTree";
   fill_tree(fileName,treeName);

   ROOT::RDataFrame d(treeName, fileName, {"b1"});
   auto sentinel = []() { std::cout << "filter called" << std::endl; return true; };
   auto f1 = d.Filter(sentinel);
   auto m1 = f1.Min();
   *m1;
   std::cout << "end first run" << std::endl;
   auto f2 = d.Filter(sentinel);
   auto dummy = f2.Max();
   *m1; // this should NOT cause a second printing of "filter called"

   return 0;
}
