// Regression test for issue https://github.com/root-project/root/issues/8295
// Make sure that Snapshotting goes well when reading/writing data members of types saved as TBranchElements from
// multiple input files (in particular, due to #8295, from the second file onwards Snapshot used to write wrong data).

#include <ROOT/RDataFrame.hxx>
#include <TError.h>
#include <TFile.h>
#include <TSystem.h>
#include <TTree.h>

#include <cstdlib> // exit

constexpr static const char *fnamePrefix = "test_snapshot_copyaddresses";
constexpr static int nfiles = 3;

struct TwoInts {
   int x;
   int y;
};

void write_inputs()
{
   TwoInts ti{1, 2};
   for (int i = 0; i < nfiles; ++i, ti.x += 2, ti.y += 2) {
      const auto fname = fnamePrefix + std::to_string(i) + ".root";
      TFile f(fname.c_str(), "recreate");
      TTree t("t", "t");
      t.Branch("ti", &ti);
      t.Fill();
      t.Write();
   }
}

void test_snapshot_copyaddresses()
{
   write_inputs();

   const std::string outSuffix = "_out.root";
   // On Windows, one cannot delete files while they are still used (not released)
   // by a process. So enclose ROOT::RDataFrame inside braces to make sure the file
   // handles get out of scope, allowing to properly delete the files
   {
      ROOT::RDataFrame df("t", fnamePrefix + std::string("*.root"));
      auto out_df = df.Snapshot<int, int>("t", fnamePrefix + outSuffix, {"x", "y"});

      int expected = 1;
      out_df->Foreach(
         [&](int x, int y) mutable {
            if (x != expected) {
               std::cerr << "Expected x == " << expected << ", found " << x << '\n';
               std::exit(1);
            }
            ++expected;

            if (y != expected) {
               std::cerr << "Expected y == " << expected << ", found " << y << '\n';
               std::exit(2);
            }
            ++expected;
         },
         {"x", "y"});
   }
   // clean up files
   for (int i = 0; i < nfiles; ++i) {
      gSystem->Unlink((fnamePrefix + std::to_string(i) + ".root").c_str());
   }
   gSystem->Unlink((fnamePrefix + outSuffix).c_str());
}

#ifndef __CLING__
int main()
{
   test_snapshot_copyaddresses();
}
#endif
