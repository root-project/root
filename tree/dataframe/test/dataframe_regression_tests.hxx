#include "ROOT/RDataFrame.hxx"
#include "TROOT.h"
#include "TSystem.h"

#include <algorithm>

#include "gtest/gtest.h"

namespace TEST_CATEGORY {

void FillTree(const char *filename, const char *treeName, int nevents = 0)
{
   TFile f(filename, "RECREATE");
   TTree t(treeName, treeName);
   t.SetAutoFlush(1); // yes, one event per cluster: to make MT more meaningful
   int b;
   t.Branch("b1", &b);
   for (int i = 0; i < nevents; ++i) {
      b = i;
      t.Fill();
   }
   t.Write();
   f.Close();
}
}

TEST(TEST_CATEGORY, MultipleTriggerRun)
{
   auto fileName = "dataframe_regression_0.root";
   auto treeName = "t";
#ifndef dataframe_regression_0_CREATED
#define dataframe_regression_0_CREATED
   {
      ROOT::RDataFrame tdf(1);
      tdf.Define("b1", []() { return 1U; }).Snapshot<unsigned int>(treeName, fileName, {"b1"});
   }
#endif

   ROOT::RDataFrame d(treeName, fileName, {"b1"});
   int i = 0;
   auto sentinel = [&i]() {
      ++i;
      return true;
   };
   auto f1 = d.Filter(sentinel);
   auto m1 = f1.Min();
   *m1; // this should cause i to be incremented
   EXPECT_EQ(1, i) << "The filter was not correctly executed.";

   auto f2 = d.Filter(sentinel);
   auto dummy = f2.Max();
   *m1; // this should NOT cause i to be incremented
   EXPECT_EQ(1, i) << "The filter was executed also when it was not supposed to.";

   *dummy; // this should cause i to be incremented
   EXPECT_EQ(2, i) << "The filter was not correctly executed for the second time.";
}

TEST(TEST_CATEGORY, EmptyTree)
{
   auto fileName = "dataframe_regression_2.root";
   auto treeName = "t";
#ifndef dataframe_regression_2_CREATED
#define dataframe_regression_2_CREATED
   {
      TFile wf(fileName, "RECREATE");
      TTree t(treeName, treeName);
      int a;
      t.Branch("a", &a);
      t.Write();
   }
#endif
   ROOT::RDataFrame d(treeName, fileName, {"a"});
   auto min = d.Min<int>();
   auto max = d.Max<int>();
   auto mean = d.Mean<int>();
   auto h = d.Histo1D<int>();
   auto c = d.Count();
   auto g = d.Take<int>();
   std::atomic_int fc(0);
   d.Foreach([&fc]() { ++fc; });

   EXPECT_DOUBLE_EQ(*min, std::numeric_limits<int>::max());
   EXPECT_DOUBLE_EQ(*max, std::numeric_limits<int>::lowest());
   EXPECT_DOUBLE_EQ(*mean, 0);
   EXPECT_EQ(h->GetEntries(), 0);
   EXPECT_EQ(*c, 0U);
   EXPECT_EQ(g->size(), 0U);
   EXPECT_EQ(fc.load(), 0);
}

// check that rdfentry_ contains all expected values,
// also in multi-thread runs over multiple ROOT files
TEST(TEST_CATEGORY, UniqueEntryNumbers)
{
   const auto treename = "t";
   const auto fname = "df_uniqueentrynumbers.root";
   ROOT::RDataFrame(10).Snapshot<unsigned int>(treename, fname, {"rdfslot_"}); // does not matter what column we write

   ROOT::RDataFrame df(treename, {fname, fname});
   auto entries = *df.Take<ULong64_t>("rdfentry_");
   std::sort(entries.begin(), entries.end());
   const auto nEntries = entries.size();
   for (auto i = 0u; i < nEntries; ++i)
      EXPECT_EQ(i, entries[i]);

   gSystem->Unlink(fname);
}
