#include <ROOT/RDataFrame.hxx>
#include <ROOT/RVec.hxx>
#include <TTree.h>
#include <TFile.h>
#include <TSystem.h>
#include <cassert>

#include "test_read_leaves.h"

using namespace ROOT;

int main()
{
   {
      TFile f("test_read_leaves.root", "RECREATE");
      TTree t("t", "t");

      V v{1, 2};
      t.Branch("v", &v, "a/I:b/I");

      // TODO add checks for reading of "w.v.a" when ROOT-9312 is solved and RDF supports "w.v.a" nested notation
      //gROOT->ProcessLine(".L test_read_leaves.h+");
      //W w;
      //t.Branch("w", &w);

      t.Fill();
      t.Write();
   }

   RDataFrame d("t", "test_read_leaves.root");
   auto check_a_b = []([[maybe_unused]] int a, [[maybe_unused]] int b) {
      assert(a == 1);
      assert(b == 2);
      return true;
   };
   auto c = d.Filter(check_a_b, {"v.a", "v.b"}).Count();
   assert(*c == 1u);
   gSystem->Unlink("test_read_leaves.root");
   return 0;
}

