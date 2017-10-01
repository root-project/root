#include "ROOT/TDataFrame.hxx"
#include "TROOT.h"
#include "TSystem.h"
#include "gtest/gtest.h"
#include <limits>
#include <memory>
using namespace ROOT::Experimental; // TDataFrame
using namespace ROOT::Experimental::TDF; // TInterface
using namespace ROOT::Detail::TDF; // TLoopManager

/********* FIXTURES *********/
// fixture that provides a TDF with no data-source and a single integer column "ans" with value 42
class TDFSnapshot : public ::testing::Test {
protected:
   const ULong64_t nEvents = 100ull; // must be initialized before fLoopManager

private:
   TDataFrame fTdf;
   TInterface<TLoopManager> DefineAns()
   {
      return fTdf.Define("ans", []() { return 42; });
   }

protected:
   TDFSnapshot() : fTdf(nEvents), tdf(DefineAns()) {}
   TInterface<TLoopManager> tdf;
};

// fixture that enables implicit MT and provides a TDF with no data-source and a single column "x" containing
// normal-distributed doubles
class TDFSnapshotMT : public ::testing::Test {
   class TIMTEnabler {
   public:
      TIMTEnabler(unsigned int nSlots) { ROOT::EnableImplicitMT(nSlots); }
      ~TIMTEnabler() { ROOT::DisableImplicitMT(); }
   };

protected:
   const ULong64_t nEvents = 100ull; // must be initialized before fLoopManager
   const unsigned int nSlots = 4u;

private:
   TIMTEnabler fIMTEnabler;
   TDataFrame fTdf;
   TInterface<TLoopManager> DefineAns()
   {
      return fTdf.Define("ans", []() { return 42; });
   }

protected:
   TDFSnapshotMT() : fIMTEnabler(nSlots), fTdf(nEvents), tdf(DefineAns()) {}
   TInterface<TLoopManager> tdf;
};


/********* TESTS ***********/
void test_snapshot_update(TInterface<TLoopManager> &tdf)
{
   // test snapshotting two trees to the same file with two snapshots and the "UPDATE" option
   const auto outfile = "snapshot_test_update.root";
   auto s1 = tdf.Snapshot<int>("t", outfile, {"ans"});

   auto c1 = s1.Count();
   auto min1 = s1.Min<int>("ans");
   auto max1 = s1.Max<int>("ans");
   auto mean1 = s1.Mean<int>("ans");
   EXPECT_EQ(100ull, *c1);
   EXPECT_EQ(42, *min1);
   EXPECT_EQ(42, *max1);
   EXPECT_EQ(42, *mean1);

   TSnapshotOptions opts;
   opts.fMode = "UPDATE";
   auto s2 = tdf.Define("two", []() { return 2.; }).Snapshot<double>("t2", outfile, {"two"}, opts);

   auto c2 = s2.Count();
   auto min2 = s2.Min<double>("two");
   auto max2 = s2.Max<double>("two");
   auto mean2 = s2.Mean<double>("two");
   EXPECT_EQ(100ull, *c2);
   EXPECT_DOUBLE_EQ(2., *min2);
   EXPECT_DOUBLE_EQ(2., *min2);
   EXPECT_DOUBLE_EQ(2., *mean2);

   // check that the output file contains both trees
   std::unique_ptr<TFile> f(TFile::Open(outfile));
   EXPECT_NE(nullptr, f->Get("t"));
   EXPECT_NE(nullptr, f->Get("t2"));

   // clean-up
   gSystem->Unlink(outfile);
}

TEST_F(TDFSnapshot, Snapshot_update)
{
   test_snapshot_update(tdf);
}

TEST_F(TDFSnapshotMT, Snapshot_update)
{
   test_snapshot_update(tdf);
}

void test_snapshot_options(TInterface<TLoopManager> &tdf)
{
   TSnapshotOptions opts;
   opts.fAutoFlush = 10;
   opts.fMode = "RECREATE";
   opts.fCompressionLevel = 6;

   const auto outfile = "snapshot_test_opts.root";
   for (auto algorithm : {ROOT::kZLIB, ROOT::kLZMA, ROOT::kLZ4}) {
      opts.fCompressionAlgorithm = algorithm;

      auto s = tdf.Snapshot<int>("t", outfile, {"ans"}, opts);

      auto c = s.Count();
      auto min = s.Min<int>("ans");
      auto max = s.Max<int>("ans");
      auto mean = s.Mean<int>("ans");
      EXPECT_EQ(100ull, *c);
      EXPECT_EQ(42, *min);
      EXPECT_EQ(42, *max);
      EXPECT_EQ(42, *mean);

      std::unique_ptr<TFile> f(TFile::Open("snapshot_test_opts.root"));

      EXPECT_EQ(algorithm, f->GetCompressionAlgorithm());
      EXPECT_EQ(6, f->GetCompressionLevel());
   }

   // clean-up
   gSystem->Unlink(outfile);
}

TEST_F(TDFSnapshot, Snapshot_action_with_options)
{
   test_snapshot_options(tdf);
}

TEST_F(TDFSnapshotMT, Snapshot_action_with_options)
{
   test_snapshot_options(tdf);
}
