#include <ROOT/TDataFrame.hxx>
#include <TTree.h>
#include <TFile.h>

#include "test_read_leaves.h"

using namespace ROOT::Experimental;

int main()
{
   TTree t("t", "t");

   V v{1, 2};
   t.Branch("v", &v, "a/I:b/I");

   gROOT->ProcessLine(".L test_read_leaves.h+");
   W w;
   t.Branch("w", &w);

   TDataFrame d(t);
   d.Filter([] { return true; }, {"w.v.a", "w.v.b"}).Report();
   return 0;
}
