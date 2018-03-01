#include "ROOT/TDataFrame.hxx"
#include "ROOT/TSeq.hxx"
#include "TFile.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TTree.h"
#include "gtest/gtest.h"
#include <limits>
#include <memory>
using namespace ROOT::Experimental;         // TDataFrame
using namespace ROOT::Experimental::TDF;    // TInterface
using namespace ROOT::Experimental::VecOps; // TVec
using namespace ROOT::Detail::TDF;          // TLoopManager

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

#ifdef R__USE_IMT
// fixture that enables implicit MT and provides a TDF with no data-source and a single column "x" containing
// normal-distributed doubles
class TDFSnapshotMT : public ::testing::Test {
   class TIMTEnabler {
   public:
      TIMTEnabler(unsigned int nSlots) { ROOT::EnableImplicitMT(nSlots); }
      ~TIMTEnabler() { ROOT::DisableImplicitMT(); }
   };

protected:
   const ULong64_t kNEvents = 100ull; // must be initialized before fLoopManager
   const unsigned int kNSlots = 4u;

private:
   TIMTEnabler fIMTEnabler;
   TDataFrame fTdf;
   TInterface<TLoopManager> DefineAns()
   {
      return fTdf.Define("ans", []() { return 42; });
   }

protected:
   TDFSnapshotMT() : fIMTEnabler(kNSlots), fTdf(kNEvents), tdf(DefineAns()) {}
   TInterface<TLoopManager> tdf;
};
#endif

// fixture that provides fixed and variable sized arrays as TDF columns
class TDFSnapshotArrays : public ::testing::Test {
protected:
   const static unsigned int kNEvents = 10u;
   static const std::vector<std::string> kFileNames;

   static void SetUpTestCase()
   {
      // write files containing c-arrays
      const auto eventsPerFile = kNEvents / kFileNames.size();
      auto curEvent = 0u;
      for (const auto &fname : kFileNames) {
         TFile f(fname.c_str(), "RECREATE");
         TTree t("arrayTree", "arrayTree");
         const unsigned int fixedSize = 4u;
         float fixedSizeArr[fixedSize];
         t.Branch("fixedSizeArr", fixedSizeArr, ("fixedSizeArr[" + std::to_string(fixedSize) + "]/F").c_str());
         unsigned int size = 0u;
         t.Branch("size", &size);
         double varSizeArr[kNEvents + 1];
         t.Branch("varSizeArr", varSizeArr, "varSizeArr[size]/D");
         // for each event, fill array elements
         for (auto i : ROOT::TSeqU(eventsPerFile)) {
            for (auto j : ROOT::TSeqU(4))
               fixedSizeArr[j] = curEvent * j;
            size = eventsPerFile - i;
            for (auto j : ROOT::TSeqU(size))
               varSizeArr[j] = curEvent * j;
            t.Fill();
            ++curEvent;
         }
         t.Write();
      }
   }

   static void TearDownTestCase()
   {
      for (const auto &fname : kFileNames)
         gSystem->Unlink(fname.c_str());
   }
};
const std::vector<std::string> TDFSnapshotArrays::kFileNames = {"test_snapshotarray1.root", "test_snapshotarray2.root"};

/********* SINGLE THREAD TESTS ***********/

// Test for ROOT-9210
TEST_F(TDFSnapshot, Snapshot_aliases)
{
   std::string alias0("myalias0");
   auto tdfa = tdf.Alias(alias0, "ans");
   testing::internal::CaptureStderr();
   auto snap = tdfa.Snapshot<int>("mytree", "Snapshot_aliases.root", {alias0});
   std::string err = testing::internal::GetCapturedStderr();
   EXPECT_TRUE(err.empty()) << err;
   EXPECT_STREQ(snap.GetColumnNames()[0].c_str(), alias0.c_str());

   auto takenCol = snap.Alias("a", alias0).Take<int>("a");
   for (auto i : takenCol) {
      EXPECT_EQ(42, i);
   }
}

// Test for ROOT-9122
TEST_F(TDFSnapshot, Snapshot_nocolumnmatch)
{
   const auto fname = "snapshotnocolumnmatch.root";
   TDataFrame d(1);
   int ret(1);
   try {
      testing::internal::CaptureStderr();
      d.Snapshot("t", fname, "x");
   } catch (const std::runtime_error &e) {
      ret = 0;
   }
   EXPECT_EQ(0, ret);
   gSystem->Unlink(fname);
}

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

void checkSnapshotArrayFile(TInterface<TLoopManager> &df, unsigned int kNEvents)
{
   // fixedSizeArr and varSizeArr are TResultProxy<vector<vector<T>>>
   auto fixedSizeArr = df.Take<TVec<float>>("fixedSizeArr");
   auto varSizeArr = df.Take<TVec<double>>("varSizeArr");
   auto size = df.Take<unsigned int>("size");

   // check contents of fixedSizeArr
   const auto nEvents = fixedSizeArr->size();
   const auto fixedSizeSize = fixedSizeArr->front().size();
   EXPECT_EQ(nEvents, kNEvents);
   EXPECT_EQ(fixedSizeSize, 4u);
   for (auto i = 0u; i < nEvents; ++i) {
      for (auto j = 0u; j < fixedSizeSize; ++j)
         EXPECT_DOUBLE_EQ(fixedSizeArr->at(i).at(j), i * j);
   }

   // check contents of varSizeArr
   for (auto i = 0u; i < nEvents; ++i) {
      const auto &v = varSizeArr->at(i);
      const auto thisSize = size->at(i);
      EXPECT_EQ(thisSize, v.size());
      for (auto j = 0u; j < thisSize; ++j)
         EXPECT_DOUBLE_EQ(v[j], i * j);
   }
}

TEST_F(TDFSnapshotArrays, SingleThread)
{
   TDataFrame tdf("arrayTree", kFileNames);
   // template Snapshot
   // "size" _must_ be listed before "varSizeArr"!
   auto dt = tdf.Snapshot<TVec<float>, unsigned int, TVec<double>>(
      "outTree", "test_snapshotTVecout.root", {"fixedSizeArr", "size", "varSizeArr"});

   checkSnapshotArrayFile(dt, kNEvents);
}

TEST_F(TDFSnapshotArrays, SingleThreadJitted)
{
   TDataFrame tdf("arrayTree", kFileNames);
   // jitted Snapshot
   // "size" _must_ be listed before "varSizeArr"!
   auto dj = tdf.Snapshot("outTree", "test_snapshotTVecout.root", {"fixedSizeArr", "size", "varSizeArr"});

   checkSnapshotArrayFile(dj, kNEvents);
}

void WriteColsWithCustomTitles(const std::string &tname, const std::string &fname)
{
   int i;
   float f;
   int a[2];
   TFile file(fname.c_str(), "RECREATE");
   TTree t(tname.c_str(), tname.c_str());
   auto b = t.Branch("float", &f);
   b->SetTitle("custom title");
   b = t.Branch("i", &i);
   b->SetTitle("custom title");
   b = t.Branch("arrint", &a, "arrint[2]/I");
   b->SetTitle("custom title");
   b = t.Branch("vararrint", &a, "vararrint[i]/I");
   b->SetTitle("custom title");

   i = 1;
   a[0] = 42;
   a[1] = 84;
   f = 4.2;
   t.Fill();

   i = 2;
   f = 8.4;
   t.Fill();

   t.Write();
}

void CheckColsWithCustomTitles(unsigned long long int entry, int i, const VecOps::TVec<int> &arrint,
                               const VecOps::TVec<int> &vararrint, float f)
{
   if (entry == 0) {
      EXPECT_EQ(i, 1);
      EXPECT_EQ(arrint.size(), 2u);
      EXPECT_EQ(arrint[0], 42);
      EXPECT_EQ(arrint[1], 84);
      EXPECT_EQ(vararrint.size(), 1u);
      EXPECT_EQ(vararrint[0], 42);
      EXPECT_FLOAT_EQ(f, 4.2f);
   } else if (entry == 1) {
      EXPECT_EQ(i, 2);
      EXPECT_EQ(arrint.size(), 2u);
      EXPECT_EQ(arrint[0], 42);
      EXPECT_EQ(arrint[1], 84);
      EXPECT_EQ(vararrint.size(), 2u);
      EXPECT_EQ(vararrint[0], 42);
      EXPECT_EQ(vararrint[1], 84);
      EXPECT_FLOAT_EQ(f, 8.4f);
   } else
      throw std::runtime_error("tree has more entries than expected");
}

TEST(TDFSnapshotMore, ColsWithCustomTitles)
{
   const auto fname = "colswithcustomtitles.root";
   const auto tname = "t";

   // write test tree
   WriteColsWithCustomTitles(tname, fname);

   // read and write test tree with TDF
   TDataFrame d(tname, fname);
   const std::string prefix = "snapshotted_";
   auto res_tdf = d.Snapshot(tname, prefix + fname);

   // check correct results have been written out
   res_tdf.Foreach(CheckColsWithCustomTitles, {"tdfentry_", "i", "arrint", "vararrint", "float"});

   // clean-up
   gSystem->Unlink(fname);
   gSystem->Unlink((prefix + fname).c_str());
}

/********* MULTI THREAD TESTS ***********/
#ifdef R__USE_IMT
TEST_F(TDFSnapshotMT, Snapshot_update)
{
   test_snapshot_update(tdf);
}

TEST_F(TDFSnapshotMT, Snapshot_action_with_options)
{
   test_snapshot_options(tdf);
}

TEST(TDFSnapshotMore, ManyTasksPerThread)
{
   const auto nSlots = 4u;
   ROOT::EnableImplicitMT(nSlots);

   // easiest way to be sure reading requires spawning of several tasks: create several input files
   const std::string inputFilePrefix = "snapshot_manytasks_";
   const auto tasksPerThread = 8u;
   const auto nInputFiles = nSlots * tasksPerThread;
   ROOT::Experimental::TDataFrame d(1);
   auto dd = d.Define("x", []() { return 42; });
   for (auto i = 0u; i < nInputFiles; ++i)
      dd.Snapshot<int>("t", inputFilePrefix + std::to_string(i) + ".root", {"x"});

   // test multi-thread Snapshotting from many tasks per worker thread
   const auto outputFile = "snapshot_manytasks_out.root";
   ROOT::Experimental::TDataFrame tdf("t", (inputFilePrefix + "*.root").c_str());
   tdf.Snapshot<int>("t", outputFile, {"x"});

   // check output contents
   ROOT::Experimental::TDataFrame checkTdf("t", outputFile);
   auto c = checkTdf.Count();
   auto t = checkTdf.Take<int>("x");
   for (auto v : t)
      EXPECT_EQ(v, 42);
   EXPECT_EQ(*c, nInputFiles);

   // clean-up input files
   for (auto i = 0u; i < nInputFiles; ++i)
      gSystem->Unlink((inputFilePrefix + std::to_string(i) + ".root").c_str());
   gSystem->Unlink(outputFile);

   ROOT::DisableImplicitMT();
}

void checkSnapshotArrayFileMT(TInterface<TLoopManager> &df, unsigned int kNEvents)
{
   // fixedSizeArr and varSizeArr are TResultProxy<vector<vector<T>>>
   auto fixedSizeArr = df.Take<TVec<float>>("fixedSizeArr");
   auto varSizeArr = df.Take<TVec<double>>("varSizeArr");
   auto size = df.Take<unsigned int>("size");

   // multi-thread execution might have scrambled events w.r.t. the original file, so we just check overall properties
   const auto nEvents = fixedSizeArr->size();
   EXPECT_EQ(nEvents, kNEvents);
   // TODO check more!
}

TEST_F(TDFSnapshotArrays, MultiThread)
{
   ROOT::EnableImplicitMT(4);

   TDataFrame tdf("arrayTree", kFileNames);
   auto dt = tdf.Snapshot<TVec<float>, unsigned int, TVec<double>>(
      "outTree", "test_snapshotTVecout.root", {"fixedSizeArr", "size", "varSizeArr"});

   checkSnapshotArrayFileMT(dt, kNEvents);

   ROOT::DisableImplicitMT();
}

TEST_F(TDFSnapshotArrays, MultiThreadJitted)
{
   ROOT::EnableImplicitMT(4);

   TDataFrame tdf("arrayTree", kFileNames);
   auto dj = tdf.Snapshot("outTree", "test_snapshotTVecout.root", {"fixedSizeArr", "size", "varSizeArr"});

   checkSnapshotArrayFileMT(dj, kNEvents);

   ROOT::DisableImplicitMT();
}

TEST(TDFSnapshotMore, ColsWithCustomTitlesMT)
{
   const auto fname = "colswithcustomtitlesmt.root";
   const auto tname = "t";

   // write test tree
   WriteColsWithCustomTitles(tname, fname);

   // read and write test tree with TDF (in parallel)
   ROOT::EnableImplicitMT(4);
   TDataFrame d(tname, fname);
   const std::string prefix = "snapshotted_";
   auto res_tdf = d.Snapshot(tname, prefix + fname);

   // check correct results have been written out
   res_tdf.Foreach(CheckColsWithCustomTitles, {"tdfentry_", "i", "arrint", "vararrint", "float"});

   // clean-up
   gSystem->Unlink(fname);
   gSystem->Unlink((prefix + fname).c_str());
   ROOT::DisableImplicitMT();
}
#endif
