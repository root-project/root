#include "TFile.h"
#include "TTreeReader.h"
#include <iterator>

int assertReaderIterator() {
   TFile f("hsimple.root");
   if (f.IsZombie()) {
      Error("assertReaderIterator()", "cannot open hsimple.root\n");
      return 2;
   }
   TTreeReader r("ntuple", &f);
   int num = 0;
   for(Long64_t idx: r) {
      ++num;
   }
   if (num != 25000) {
      Error("assertReaderIterator()", "range based for broken (%d)\n", num);
      return 1;
   }

   int dist = std::distance(r.begin(), r.end());
   if (dist != num) {
      Error("assertReaderIterator()", "std::distance broken (%d)\n", dist);
      return 2;
   }
   return 0;
}
