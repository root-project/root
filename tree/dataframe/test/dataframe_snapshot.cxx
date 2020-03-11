#include "ROOT/RDataFrame.hxx"
#include "ROOT/TSeq.hxx"
#include "TFile.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TTree.h"
#include "gtest/gtest.h"
#include <limits>
#include <memory>
using namespace ROOT;         // RDataFrame
using namespace ROOT::RDF;    // RInterface
using namespace ROOT::VecOps; // RVec
using namespace ROOT::Detail::RDF;          // RLoopManager

/********* FIXTURES *********/
// fixture that provides a RDF with no data-source and a single integer column "ans" with value 42
class RDFSnapshot : public ::testing::Test {
protected:
   const ULong64_t nEvents = 100ull; // must be initialized before fLoopManager

private:
   RDataFrame fTdf;
   RInterface<RLoopManager> DefineAns()
   {
      return fTdf.Define("ans", []() { return 42; });
   }

protected:
   RDFSnapshot() : fTdf(nEvents), tdf(DefineAns()) {}
   RInterface<RLoopManager> tdf;
};

#ifdef R__USE_IMT
// fixture that enables implicit MT and provides a RDF with no data-source and a single column "x" containing
// normal-distributed doubles
class RDFSnapshotMT : public ::testing::Test {
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
   RDataFrame fTdf;
   RInterface<RLoopManager> DefineAns()
   {
      return fTdf.Define("ans", []() { return 42; });
   }

protected:
   RDFSnapshotMT() : fIMTEnabler(kNSlots), fTdf(kNEvents), tdf(DefineAns()) {}
   RInterface<RLoopManager> tdf;
};
#endif // R__USE_IMT

// fixture that provides fixed and variable sized arrays as RDF columns
class RDFSnapshotArrays : public ::testing::Test {
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

         // doubles, floats
         const unsigned int fixedSize = 4u;
         float fixedSizeArr[fixedSize];
         t.Branch("fixedSizeArr", fixedSizeArr, ("fixedSizeArr[" + std::to_string(fixedSize) + "]/F").c_str());
         unsigned int size = 0u;
         t.Branch("size", &size);
         double *varSizeArr = new double[eventsPerFile * 100u];
         t.Branch("varSizeArr", varSizeArr, "varSizeArr[size]/D");

         // bools. std::vector<bool> makes bool treatment in RDF special
         bool fixedSizeBoolArr[fixedSize];
         t.Branch("fixedSizeBoolArr", fixedSizeBoolArr,
                  ("fixedSizeBoolArr[" + std::to_string(fixedSize) + "]/O").c_str());
         bool *varSizeBoolArr = new bool[eventsPerFile * 100u];
         t.Branch("varSizeBoolArr", varSizeBoolArr, "varSizeBoolArr[size]/O");

         // for each event, fill array elements
         for (auto i : ROOT::TSeqU(eventsPerFile)) {
            for (auto j : ROOT::TSeqU(4)) {
               fixedSizeArr[j] = curEvent * j;
               fixedSizeBoolArr[j] = j % 2 == 0;
            }
            size = (i + 1) * 100u;
            for (auto j : ROOT::TSeqU(size)) {
               varSizeArr[j] = curEvent * j;
               varSizeBoolArr[j] = j % 2 == 0;
            }
            t.Fill();
            ++curEvent;
         }
         t.Write();

         delete[] varSizeArr;
         delete[] varSizeBoolArr;
      }
   }

   static void TearDownTestCase()
   {
      for (const auto &fname : kFileNames)
         gSystem->Unlink(fname.c_str());
   }
};
const std::vector<std::string> RDFSnapshotArrays::kFileNames = {"test_snapshotarray1.root", "test_snapshotarray2.root"};

/********* SINGLE THREAD TESTS ***********/

TEST_F(RDFSnapshot, SnapshotCallAmbiguities)
{
   auto filename = "Snapshot_interface.root";

   tdf.Snapshot("t", filename, "an.*");
   tdf.Snapshot("t", filename, {"ans"});
   tdf.Snapshot("t", filename, {{"ans"}});

   gSystem->Unlink(filename);
}

// Test for ROOT-9210
TEST_F(RDFSnapshot, Snapshot_aliases)
{
   const auto alias0 = "myalias0";
   const auto alias1 = "myalias1";
   auto tdfa = tdf.Alias(alias0, "ans");
   auto tdfb = tdfa.Define("vec", [] { return RVec<int>{1,2,3}; }).Alias(alias1, "vec");
   testing::internal::CaptureStderr();
   auto snap = tdfb.Snapshot<int, RVec<int>>("mytree", "Snapshot_aliases.root", {alias0, alias1});
   std::string err = testing::internal::GetCapturedStderr();
   EXPECT_TRUE(err.empty()) << err;
   auto names = snap->GetColumnNames();
   EXPECT_EQ(2U, names.size());
   EXPECT_EQ(names, std::vector<std::string>({alias0, alias1}));

   auto takenCol = snap->Alias("a", alias0).Take<int>("a");
   for (auto i : takenCol) {
      EXPECT_EQ(42, i);
   }
}

// Test for ROOT-9122
TEST_F(RDFSnapshot, Snapshot_nocolumnmatch)
{
   const auto fname = "snapshotnocolumnmatch.root";
   RDataFrame d(1);
   auto op = [&](){
      testing::internal::CaptureStderr();
      d.Snapshot("t", fname, "x");
   };
   EXPECT_ANY_THROW(op());
   gSystem->Unlink(fname);
}

void test_snapshot_update(RInterface<RLoopManager> &tdf)
{
   // test snapshotting two trees to the same file with two snapshots and the "UPDATE" option
   const auto outfile = "snapshot_test_update.root";
   auto s1 = tdf.Snapshot<int>("t", outfile, {"ans"});

   auto c1 = s1->Count();
   auto min1 = s1->Min<int>("ans");
   auto max1 = s1->Max<int>("ans");
   auto mean1 = s1->Mean<int>("ans");
   EXPECT_EQ(100ull, *c1);
   EXPECT_EQ(42, *min1);
   EXPECT_EQ(42, *max1);
   EXPECT_EQ(42, *mean1);

   RSnapshotOptions opts;
   opts.fMode = "UPDATE";
   auto s2 = tdf.Define("two", []() { return 2.; }).Snapshot<double>("t2", outfile, {"two"}, opts);

   auto c2 = s2->Count();
   auto min2 = s2->Min<double>("two");
   auto max2 = s2->Max<double>("two");
   auto mean2 = s2->Mean<double>("two");
   EXPECT_EQ(100ull, *c2);
   EXPECT_DOUBLE_EQ(2., *min2);
   EXPECT_DOUBLE_EQ(2., *min2);
   EXPECT_DOUBLE_EQ(2., *mean2);

   // check that the output file contains both trees
   std::unique_ptr<TFile> f(TFile::Open(outfile));
   EXPECT_NE(nullptr, f->Get<TTree>("t"));
   EXPECT_NE(nullptr, f->Get<TTree>("t2"));

   // clean-up
   gSystem->Unlink(outfile);
}

TEST_F(RDFSnapshot, Snapshot_update)
{
   test_snapshot_update(tdf);
}

void test_snapshot_options(RInterface<RLoopManager> &tdf)
{
   RSnapshotOptions opts;
   opts.fAutoFlush = 10;
   opts.fMode = "RECREATE";
   opts.fCompressionLevel = 6;

   const auto outfile = "snapshot_test_opts.root";
   for (auto algorithm : {ROOT::kZLIB, ROOT::kLZMA, ROOT::kLZ4, ROOT::kZSTD}) {
      opts.fCompressionAlgorithm = algorithm;

      auto s = tdf.Snapshot<int>("t", outfile, {"ans"}, opts);

      auto c = s->Count();
      auto min = s->Min<int>("ans");
      auto max = s->Max<int>("ans");
      auto mean = s->Mean<int>("ans");
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

TEST_F(RDFSnapshot, Snapshot_action_with_options)
{
   test_snapshot_options(tdf);
}

void checkSnapshotArrayFile(RResultPtr<RInterface<RLoopManager>> &df, unsigned int kNEvents)
{
   // fixedSizeArr and varSizeArr are RResultPtr<vector<vector<T>>>
   auto fixedSizeArr = df->Take<RVec<float>>("fixedSizeArr");
   auto varSizeArr = df->Take<RVec<double>>("varSizeArr");
   auto fixedSizeBoolArr = df->Take<RVec<bool>>("fixedSizeBoolArr");
   auto varSizeBoolArr = df->Take<RVec<bool>>("varSizeBoolArr");
   auto size = df->Take<unsigned int>("size");

   // check contents of fixed sized arrays
   const auto nEvents = fixedSizeArr->size();
   const auto fixedSizeSize = fixedSizeArr->front().size();
   EXPECT_EQ(nEvents, kNEvents);
   EXPECT_EQ(fixedSizeSize, 4u);
   for (auto i = 0u; i < nEvents; ++i) {
      for (auto j = 0u; j < fixedSizeSize; ++j) {
         EXPECT_DOUBLE_EQ(fixedSizeArr->at(i).at(j), i * j);
         EXPECT_EQ(fixedSizeBoolArr->at(i).at(j), j % 2 == 0);
      }
   }

   // check contents of variable sized arrays
   for (auto i = 0u; i < nEvents; ++i) {
      const auto thisSize = size->at(i);
      const auto &dv = varSizeArr->at(i);
      const auto &bv = varSizeBoolArr->at(i);
      EXPECT_EQ(thisSize, dv.size());
      EXPECT_EQ(thisSize, bv.size());
      for (auto j = 0u; j < thisSize; ++j) {
         EXPECT_DOUBLE_EQ(dv[j], i * j);
         EXPECT_EQ(bv[j], j % 2 == 0);
      }
   }
}

TEST_F(RDFSnapshotArrays, SingleThread)
{
   RDataFrame tdf("arrayTree", kFileNames);
   // template Snapshot
   // "size" _must_ be listed before "varSizeArr"!
   auto dt = tdf.Snapshot<RVec<float>, unsigned int, RVec<double>, RVec<bool>, RVec<bool>>(
      "outTree", "test_snapshotRVecoutST.root",
      {"fixedSizeArr", "size", "varSizeArr", "varSizeBoolArr", "fixedSizeBoolArr"});

   checkSnapshotArrayFile(dt, kNEvents);
}

TEST_F(RDFSnapshotArrays, SingleThreadJitted)
{
   RDataFrame tdf("arrayTree", kFileNames);
   // jitted Snapshot
   // "size" _must_ be listed before "varSizeArr"!
   auto dj = tdf.Snapshot("outTree", "test_snapshotRVecoutSTJitted.root",
                          {"fixedSizeArr", "size", "varSizeArr", "varSizeBoolArr", "fixedSizeBoolArr"});

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

void CheckColsWithCustomTitles(unsigned long long int entry, int i, const RVec<int> &arrint,
                               const RVec<int> &vararrint, float f)
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

TEST(RDFSnapshotMore, ColsWithCustomTitles)
{
   const auto fname = "colswithcustomtitles.root";
   const auto tname = "t";

   // write test tree
   WriteColsWithCustomTitles(tname, fname);

   // read and write test tree with RDF
   RDataFrame d(tname, fname);
   const std::string prefix = "snapshotted_";
   auto res_tdf =
      d.Snapshot<int, float, RVec<int>, RVec<int>>(tname, prefix + fname, {"i", "float", "arrint", "vararrint"});

   // check correct results have been written out
   res_tdf->Foreach(CheckColsWithCustomTitles, {"tdfentry_", "i", "arrint", "vararrint", "float"});

   // clean-up
   gSystem->Unlink(fname);
   gSystem->Unlink((prefix + fname).c_str());
}

TEST(RDFSnapshotMore, ReadWriteStdVec)
{
   // write a TFile containing a std::vector
   const auto fname = "readwritestdvec.root";
   const auto treename = "t";
   TFile f(fname, "RECREATE");
   TTree t(treename, treename);
   std::vector<int> v({42});
   std::vector<bool> vb({true, false, true}); // std::vector<bool> is special, not in a good way
   t.Branch("v", &v);
   t.Branch("vb", &vb);
   t.Fill();
   // as an extra test, make sure that the vector reallocates between first and second entry
   v = std::vector<int>(100000, 84);
   vb = std::vector<bool>(100000, true);
   t.Fill();
   t.Write();
   f.Close();

   auto outputChecker = [&treename](const char* filename){
      // check snapshot output
      TFile f2(filename);
      TTreeReader r(treename, &f2);
      TTreeReaderArray<int> rv(r, "v");
      TTreeReaderArray<bool> rvb(r, "vb");
      r.Next();
      EXPECT_EQ(rv.GetSize(), 1u);
      EXPECT_EQ(rv[0], 42);
      EXPECT_EQ(rvb.GetSize(), 3u);
      EXPECT_TRUE(rvb[0]);
      EXPECT_FALSE(rvb[1]);
      EXPECT_TRUE(rvb[2]);
      r.Next();
      EXPECT_EQ(rv.GetSize(), 100000u);
      EXPECT_EQ(rvb.GetSize(), 100000u);
      for (auto e : rv)
         EXPECT_EQ(e, 84);
      for (auto e : rvb)
         EXPECT_TRUE(e);
   };

   // read and write using RDataFrame

   const auto outfname1 = "out_readwritestdvec1.root";
   RDataFrame(treename, fname).Snapshot<std::vector<int>, std::vector<bool>>(treename, outfname1, {"v", "vb"});
   outputChecker(outfname1);

   const auto outfname2 = "out_readwritestdvec2.root";
   RDataFrame(treename, fname).Snapshot(treename, outfname2);
   outputChecker(outfname2);

   const auto outfname3 = "out_readwritestdvec3.root";
   RDataFrame(treename, fname).Snapshot<RVec<int>, RVec<bool>>(treename, outfname3, {"v", "vb"});
   outputChecker(outfname3);

   gSystem->Unlink(fname);
   gSystem->Unlink(outfname1);
   gSystem->Unlink(outfname2);
   gSystem->Unlink(outfname3);
}

void ReadWriteCarray(const char *outFileNameBase)
{
   // write a TFile containing a arrays
   std::string outFileNameBaseStr = outFileNameBase;
   const auto fname = outFileNameBaseStr + ".root";
   const auto treename = "t";
   TFile f(fname.c_str(), "RECREATE");
   TTree t(treename, treename);
   const auto maxArraySize = 100000U;
   auto size = 0;
   int v[maxArraySize];
   bool vb[maxArraySize];
   t.Branch("size", &size, "size/I");
   t.Branch("v", v, "v[size]/I");
   t.Branch("vb", vb, "vb[size]/O");

   // Size 1
   size = 1;
   v[0] = 12;
   vb[0] = true;
   t.Fill();

   // Size 0 (see ROOT-9860)
   size = 0;
   t.Fill();

   // Size 100k: this reallocates!
   size = maxArraySize;
   for (auto i : ROOT::TSeqU(size)) {
      v[i] = 84;
      vb[i] = true;
   }
   t.Fill();

   // Size 3
   size = 3;
   v[0] = 42;
   v[1] = 43;
   v[2] = 44;
   vb[0] = true;
   vb[1] = false;
   vb[2] = true;
   t.Fill();

   t.Write();
   f.Close();

   auto outputChecker = [&treename](const char *filename) {
      // check snapshot output
      TFile f2(filename);
      TTreeReader r(treename, &f2);
      TTreeReaderArray<int> rv(r, "v");
      TTreeReaderArray<bool> rvb(r, "vb");

      // Size 1
      r.Next();
      EXPECT_EQ(rv.GetSize(), 1u);
      EXPECT_EQ(rv[0], 12);
      EXPECT_EQ(rvb.GetSize(), 1u);
      EXPECT_TRUE(rvb[0]);

      // Size 0
      r.Next();
      EXPECT_EQ(rv.GetSize(), 0u);
      EXPECT_EQ(rvb.GetSize(), 0u);

      // Size 100k
      r.Next();
      EXPECT_EQ(rv.GetSize(), 100000u);
      EXPECT_EQ(rvb.GetSize(), 100000u);
      for (auto e : rv)
         EXPECT_EQ(e, 84);
      for (auto e : rvb)
         EXPECT_TRUE(e);

      // Size 3
      r.Next();
      EXPECT_EQ(rv.GetSize(), 3u);
      EXPECT_EQ(rv[0], 42);
      EXPECT_EQ(rv[1], 43);
      EXPECT_EQ(rv[2], 44);
      EXPECT_EQ(rvb.GetSize(), 3u);
      EXPECT_TRUE(rvb[0]);
      EXPECT_FALSE(rvb[1]);
      EXPECT_TRUE(rvb[2]);
   };

   // read and write using RDataFrame
   const auto outfname1 = outFileNameBaseStr + "_out1.root";
   RDataFrame(treename, fname).Snapshot(treename, outfname1);
   outputChecker(outfname1.c_str());

   const auto outfname2 = outFileNameBaseStr + "_out2.root";
   RDataFrame(treename, fname).Snapshot<int, RVec<int>, RVec<bool>>(treename, outfname2, {"size", "v", "vb"});
   outputChecker(outfname2.c_str());

   gSystem->Unlink(fname.c_str());
   gSystem->Unlink(outfname1.c_str());
   gSystem->Unlink(outfname2.c_str());
}

TEST(RDFSnapshotMore, ReadWriteCarray)
{
   ReadWriteCarray("ReadWriteCarray");
}

struct TwoInts {
   int a, b;
};

void WriteTreeWithLeaves(const std::string &treename, const std::string &fname)
{
   TFile f(fname.c_str(), "RECREATE");
   TTree t(treename.c_str(), treename.c_str());

   TwoInts ti{1, 2};
   t.Branch("v", &ti, "a/I:b/I");

   // TODO add checks for reading of multiple nested levels ("w.v.a")
   // when ROOT-9312 is solved and RDF supports "w.v.a" nested notation

   t.Fill();
   t.Write();
}

TEST(RDFSnapshotMore, ReadWriteNestedLeaves)
{
   const auto treename = "t";
   const auto fname = "readwritenestedleaves.root";
   WriteTreeWithLeaves(treename, fname);
   RDataFrame d(treename, fname);
   const auto outfname = "out_readwritenestedleaves.root";
   auto d2 = d.Snapshot<int, int>(treename, outfname, {"v.a", "v.b"});
   EXPECT_EQ(d2->GetColumnNames(), std::vector<std::string>({"v_a", "v_b"}));
   auto check_a_b = [](int a, int b) {
      EXPECT_EQ(a, 1);
      EXPECT_EQ(b, 2);
   };
   d2->Foreach(check_a_b, {"v_a", "v_b"});
   gSystem->Unlink(fname);
   gSystem->Unlink(outfname);

   try {
      d.Define("v_a", [] { return 0; }).Snapshot<int, int>(treename, outfname, {"v.a", "v_a"});
   } catch (std::runtime_error &e) {
      const auto error_msg = "Column v.a would be written as v_a but this column already exists. Please use Alias to "
                             "select a new name for v.a";
      EXPECT_STREQ(e.what(), error_msg);
   }
}

TEST(RDFSnapshotMore, Lazy)
{
   const auto treename = "t";
   const auto fname0 = "lazy0.root";
   const auto fname1 = "lazy1.root";
   // make sure the file is not here beforehand
   gSystem->Unlink(fname0);
   RDataFrame d(1);
   auto v = 0U;
   auto genf = [&v](){++v;return 42;};
   RSnapshotOptions opts = {"RECREATE", ROOT::kZLIB, 0, 0, 99, true};
   auto ds = d.Define("c0", genf).Snapshot<int>(treename, fname0, {"c0"}, opts);
   EXPECT_EQ(v, 0U);
   EXPECT_TRUE(gSystem->AccessPathName(fname0)); // This returns FALSE if the file IS there
   auto ds2 = ds->Define("c1", genf).Snapshot<int>(treename, fname1, {"c1"}, opts);
   EXPECT_EQ(v, 1U);
   EXPECT_FALSE(gSystem->AccessPathName(fname0));
   EXPECT_TRUE(gSystem->AccessPathName(fname1));
   *ds2;
   EXPECT_EQ(v, 2U);
   EXPECT_FALSE(gSystem->AccessPathName(fname1));
   gSystem->Unlink(fname0);
   gSystem->Unlink(fname1);
}

TEST(RDFSnapshotMore, LazyNotTriggered)
{
   {
      auto d = ROOT::RDataFrame(1);
      ROOT::RDF::RSnapshotOptions opts;
      opts.fLazy = true;
      d.Snapshot<ULong64_t>("t", "foo.root", {"tdfentry_"}, opts);
   }
}

/********* MULTI THREAD TESTS ***********/
#ifdef R__USE_IMT
TEST_F(RDFSnapshotMT, Snapshot_update)
{
   test_snapshot_update(tdf);
}

TEST_F(RDFSnapshotMT, Snapshot_action_with_options)
{
   test_snapshot_options(tdf);
}

TEST(RDFSnapshotMore, ManyTasksPerThread)
{
   const auto nSlots = 4u;
   ROOT::EnableImplicitMT(nSlots);
   // make sure the file is not here beforehand
   gSystem->Unlink("snapshot_manytasks_out.root");

   // easiest way to be sure reading requires spawning of several tasks: create several input files
   const std::string inputFilePrefix = "snapshot_manytasks_";
   const auto tasksPerThread = 8u;
   const auto nInputFiles = nSlots * tasksPerThread;
   ROOT::RDataFrame d(1);
   auto dd = d.Define("x", []() { return 42; });
   for (auto i = 0u; i < nInputFiles; ++i)
      dd.Snapshot<int>("t", inputFilePrefix + std::to_string(i) + ".root", {"x"});

   // test multi-thread Snapshotting from many tasks per worker thread
   const auto outputFile = "snapshot_manytasks_out.root";
   ROOT::RDataFrame tdf("t", (inputFilePrefix + "*.root").c_str());
   tdf.Snapshot<int>("t", outputFile, {"x"});

   // check output contents
   ROOT::RDataFrame checkTdf("t", outputFile);
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

void checkSnapshotArrayFileMT(RResultPtr<RInterface<RLoopManager>> &df, unsigned int kNEvents)
{
   // fixedSizeArr and varSizeArr are RResultPtr<vector<vector<T>>>
   auto fixedSizeArr = df->Take<RVec<float>>("fixedSizeArr");
   auto varSizeArr = df->Take<RVec<double>>("varSizeArr");
   auto size = df->Take<unsigned int>("size");

   // multi-thread execution might have scrambled events w.r.t. the original file, so we just check overall properties
   const auto nEvents = fixedSizeArr->size();
   EXPECT_EQ(nEvents, kNEvents);
   // TODO check more!
}

TEST_F(RDFSnapshotArrays, MultiThread)
{
   ROOT::EnableImplicitMT(4);

   RDataFrame tdf("arrayTree", kFileNames);
   auto dt = tdf.Snapshot<RVec<float>, unsigned int, RVec<double>, RVec<bool>, RVec<bool>>(
      "outTree", "test_snapshotRVecoutMT.root",
      {"fixedSizeArr", "size", "varSizeArr", "varSizeBoolArr", "fixedSizeBoolArr"});

   checkSnapshotArrayFileMT(dt, kNEvents);

   ROOT::DisableImplicitMT();
}

TEST_F(RDFSnapshotArrays, MultiThreadJitted)
{
   ROOT::EnableImplicitMT(4);

   RDataFrame tdf("arrayTree", kFileNames);
   auto dj = tdf.Snapshot("outTree", "test_snapshotRVecoutMTJitted.root",
                          {"fixedSizeArr", "size", "varSizeArr", "varSizeBoolArr", "fixedSizeBoolArr"});

   checkSnapshotArrayFileMT(dj, kNEvents);

   ROOT::DisableImplicitMT();
}

TEST(RDFSnapshotMore, ColsWithCustomTitlesMT)
{
   const auto fname = "colswithcustomtitlesmt.root";
   const auto tname = "t";

   // write test tree
   WriteColsWithCustomTitles(tname, fname);

   // read and write test tree with RDF (in parallel)
   ROOT::EnableImplicitMT(4);
   RDataFrame d(tname, fname);
   const std::string prefix = "snapshotted_";
   auto res_tdf =
      d.Snapshot<int, float, RVec<int>, RVec<int>>(tname, prefix + fname, {"i", "float", "arrint", "vararrint"});

   // check correct results have been written out
   res_tdf->Foreach(CheckColsWithCustomTitles, {"tdfentry_", "i", "arrint", "vararrint", "float"});
   res_tdf->Foreach(CheckColsWithCustomTitles, {"rdfentry_", "i", "arrint", "vararrint", "float"});

   // clean-up
   gSystem->Unlink(fname);
   gSystem->Unlink((prefix + fname).c_str());
   ROOT::DisableImplicitMT();
}

TEST(RDFSnapshotMore, TreeWithFriendsMT)
{
   const auto fname = "treewithfriendsmt.root";
   ROOT::EnableImplicitMT();
   RDataFrame(10).Define("x", []() { return 0; }).Snapshot<int>("t", fname, {"x"});

   TFile file(fname);
   auto tree = file.Get<TTree>("t");
   TFile file2(fname);
   auto tree2 = file2.Get<TTree>("t");
   tree->AddFriend(tree2);

   const auto outfname = "out_treewithfriendsmt.root";
   RDataFrame df(*tree);
   df.Snapshot<int>("t", outfname, {"x"});
   ROOT::DisableImplicitMT();

   gSystem->Unlink(fname);
   gSystem->Unlink(outfname);
}

TEST(RDFSnapshotMore, JittedSnapshotAndAliasedColumns)
{
   ROOT::RDataFrame df(1);
   const auto fname = "out_aliasedcustomcolumn.root";
   // aliasing a custom column
   auto df2 = df.Define("x", [] { return 42; }).Alias("y", "x").Snapshot("t", fname, "y"); // must be jitted!
   EXPECT_EQ(df2->GetColumnNames(), std::vector<std::string>({"y"}));
   EXPECT_EQ(df2->Take<int>("y")->at(0), 42);

   // aliasing a column from a file
   const auto fname2 = "out_aliasedcustomcolumn2.root";
   df2->Alias("z", "y").Snapshot("t", fname2, "z");

   gSystem->Unlink(fname);
   gSystem->Unlink(fname2);
}


TEST(RDFSnapshotMore, LazyNotTriggeredMT)
{
   ROOT::EnableImplicitMT(4);
   const auto fname = "lazynottriggeredmt.root";
   {
      auto d = ROOT::RDataFrame(8);
      ROOT::RDF::RSnapshotOptions opts;
      opts.fLazy = true;
      d.Snapshot<ULong64_t, ULong64_t>("t", fname, {"tdfentry_", "rdfentry_"}, opts);
   }

   gSystem->Unlink(fname);
   ROOT::DisableImplicitMT();
}

TEST(RDFSnapshotMore, EmptyBuffersMT)
{
   const auto fname = "emptybuffersmt.root";
   const auto treename = "t";
   ROOT::EnableImplicitMT(4);
   ROOT::RDataFrame d(10);
   auto dd = d.DefineSlot("x", [](unsigned int s) { return s == 3 ? 0 : 1; })
               .Filter([](int x) { return x == 0; }, {"x"}, "f");
   auto r = dd.Report();
   dd.Snapshot<int>(treename, fname, {"x"});

   // check test sanity
   const auto passed = r->At("f").GetPass();
   EXPECT_GT(passed, 0u);

   // check result
   TFile f(fname);
   auto t = f.Get<TTree>(treename);
   EXPECT_EQ(t->GetListOfBranches()->GetEntries(), 1);
   EXPECT_EQ(t->GetEntries(), Long64_t(passed));

   ROOT::DisableImplicitMT();
   gSystem->Unlink(fname);
}

TEST(RDFSnapshotMore, ReadWriteCarrayMT)
{
   ROOT::EnableImplicitMT(4);
   ReadWriteCarray("ReadWriteCarrayMT");
   ROOT::DisableImplicitMT();
}

#endif // R__USE_IMT

