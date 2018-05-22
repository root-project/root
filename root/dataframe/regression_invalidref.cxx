#include "TFile.h"
#include "ROOT/RDataFrame.hxx"
#include <exception>

const char* fileName("regression_invalidref.root");
const char* treeName("mytree");

void fill_tree() {
   TFile f(fileName,"RECREATE");
   TTree t(treeName,treeName);
   for(int i = 0; i < 10; ++i) {
      t.Fill();
   }
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

   auto f = FilteredDFFactory();
   try {
      f.Filter([]() { return true; });
   } catch (const std::runtime_error& e) {
      std::cout << "Exception caught: the dataframe went out of scope when booking a filter" << std::endl;
   }

   return 0;
}
