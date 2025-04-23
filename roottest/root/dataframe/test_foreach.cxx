#include "TFile.h"
#include "ROOT/RDataFrame.hxx"
#include "TROOT.h"
#include "TSystem.h"

#include <cmath> // sqrt
#include <iostream>
#include <numeric> // accumulate
#include <vector>

void FillTree(const char* filename, const char* treeName) {
   TFile f(filename, "RECREATE");
   TTree t(treeName, treeName);
   double b1;
   int b2;
   t.Branch("b1", &b1);
   t.Branch("b2", &b2);
   for(int i = 0; i < 100000; ++i) {
      b1 = i / 100000.;
      b2 = i*i;
      t.Fill();
   }
   t.Write();
   f.Close();
}

int main() {
   auto fileName = "test_foreach.root";
   auto treeName = "foreachTree";
   FillTree(fileName, treeName);

   // evaluation of RMS of branch "b" using ForeachSlot
#ifdef R__USE_IMT
   ROOT::EnableImplicitMT(2);
   unsigned int nSlots = ROOT::GetThreadPoolSize();
#else
   unsigned int nSlots = 1;
#endif

   std::vector<double> rmss(nSlots, 0.);
   std::vector<unsigned int> counts(nSlots, 0);
   ROOT::RDataFrame d("foreachTree", fileName, {"b1"});
   auto rmsLambda = [&rmss, &counts](unsigned int slot, double b) {
      rmss[slot] += b*b;
      counts[slot] += 1;
   };
   d.ForeachSlot(rmsLambda);
   double rms = std::accumulate(rmss.begin(), rmss.end(), 0.); // sum all squares
   unsigned int count = std::accumulate(counts.begin(), counts.end(), 0); // sum all counts
   std::cout << "rms: " << std::sqrt(rms / count) << std::endl;

   return 0;
}
