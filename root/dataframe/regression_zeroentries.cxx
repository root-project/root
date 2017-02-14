#include "TFile.h"
#include "TTree.h"
#include "TROOT.h"

#include "ROOT/TDataFrame.hxx"

#include <atomic>
#include <cassert>
#include <iostream>
#include <limits>

int main() {
   {
      TFile wf("regression_zeroentries.root", "RECREATE");
      TTree t("emptyTree", "emptyTree");
      int a;
      t.Branch("a", &a);
      t.Write();
   }

#ifdef R__USE_IMT
   ROOT::EnableImplicitMT();
#endif
   TFile f("regression_zeroentries.root");
   ROOT::Experimental::TDataFrame d("emptyTree", &f, {"a"});

   // apply all actions to an empty tree, multi-thread case
   auto min = d.Min<int>();
   auto max = d.Max<int>();
   auto mean = d.Mean<int>();
   auto h = d.Histo1D<int>();
   auto c = d.Count();
   auto g = d.Take<int>();
   std::atomic_int fc(0);
   d.Foreach([&fc]() { ++fc; });

   assert(*min == std::numeric_limits<double>::max());
   assert(*max == std::numeric_limits<double>::min());
   assert(*mean == 0);
   assert(h->GetEntries() == 0);
   assert(*c == 0);
   assert(g->size() == 0);
   assert(fc == 0);

   return 0;
}
