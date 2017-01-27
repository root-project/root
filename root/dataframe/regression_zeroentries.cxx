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

   {
      TFile f("regression_zeroentries.root");
      ROOT::Experimental::TDataFrame d("emptyTree", &f, {"a"});

      // apply all actions to an empty tree, single-threaded case
      auto min = d.Min();
      auto max = d.Max();
      auto mean = d.Mean();
      auto h = d.Histo();
      auto c = d.Count();
      auto g = d.Take<int>();
      int fc = 0;
      d.Foreach([&fc]() { ++fc; });

      assert(*min == std::numeric_limits<double>::max());
      assert(*max == std::numeric_limits<double>::min());
      assert(*mean == 0);
      assert(h->GetEntries() == 0);
      assert(*c == 0);
      assert(g->size() == 0);
      assert(fc == 0);
   }

   {
      ROOT::EnableImplicitMT();
      TFile f("regression_zeroentries.root");
      ROOT::Experimental::TDataFrame d("emptyTree", &f, {"a"});

      // apply all actions to an empty tree, multi-thread case
      auto min = d.Min();
      auto max = d.Max();
      auto mean = d.Mean();
      auto h = d.Histo();
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
   }

   return 0;
}
