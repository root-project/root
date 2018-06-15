#include "ROOT/RDataFrame.hxx"

#include "TFile.h"
#include "TSystem.h"

#include <exception>

const char* fileName("regression_invalidref.root");
const char* treeName("mytree");

void fill_tree() {
   TFile f(fileName,"RECREATE");
   TTree t(treeName,treeName);
   t.Write();
   f.Close();
   return;
}

auto FilteredDFFactory = []() {
   ROOT::RDataFrame d(treeName, fileName);
   auto f = d.Filter([]() { return true; });
   return f;
};

int main() {

   fill_tree();
   auto f = FilteredDFFactory();
   try {
      f.Filter([]() { return true; });
   } catch (const std::runtime_error& e) {
      gSystem->Unlink(fileName);
      std::cout << "Exception caught: the dataframe went out of scope when booking a filter" << std::endl;
   }

   return 0;
}
