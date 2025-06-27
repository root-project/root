#include "TFile.h"
#include "TTree.h"
#include "ROOT/RDataFrame.hxx"
#include "TROOT.h"
#include "TRandom.h"

#include <iostream>
#include <atomic>

int main() {
   auto fileName = "test_emptysource.root";
   auto treeName = "test_emptysource";
   constexpr Long64_t numEntries = 1002;

   // Build the empty-source dataframe
   ROOT::RDataFrame d(numEntries);

   // Define some temporary branches
   TRandom tr;
   auto dWithB = d.Define("bRndm", [&tr]() { return tr.Rndm(); })
                  .Define("bGaus", [&tr]() { return tr.Gaus(); })
                  .Define("bOnes", []()    { return 1; })
                  .Define("bZeroes", []()  { return 0; });

   // Generate on the fly the temporary branches and save them on disk
   dWithB.Snapshot(treeName, fileName, {"bRndm", "bGaus", "bOnes", "bZeroes"});

   TFile f(fileName);
   TTree *t = (TTree*)f.Get(treeName);
   std::cout << "Number of snapshot entries: "  << t->GetEntries()   << std::endl;
   std::cout << "Number of snapshot branches: " << t->GetNbranches() << std::endl;

#ifdef R__USE_IMT
   constexpr unsigned int numThreads = 4;
   ROOT::EnableImplicitMT(numThreads);
#endif
   // Build the empty-source dataframe
   ROOT::RDataFrame d2(numEntries);
   auto dd2 = d2.Define("bOnes", []() { return 1; }).Define("bZeroes", []() { return 0; });

   // Increment a counter for each entry
   std::atomic_int counter(0);
   dd2.Foreach([&counter](int val1, int val2) { counter += val1 + val2; }, {"bOnes", "bZeroes"});
   std::cout << "Counter value: "  << counter << std::endl;
  
   // Sum up all the ones in bOnes
   auto sum = dd2.Reduce([](int a, int b) { return a + b; }, {"bOnes"});
   std::cout << "Sum value: "  << *sum << std::endl;
    
   return 0;
}
