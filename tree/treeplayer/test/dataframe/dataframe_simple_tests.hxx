#include "ROOT/TDataFrame.hxx"
#include "Compression.h"
#include "TFile.h"
#include "TInterpreter.h"
#include "TRandom.h"

#include "gtest/gtest.h"

#include <chrono>
#include <thread>
#include <set>

using namespace ROOT::Experimental;

namespace TEST_CATEGORY {

int DefineFunction()
{
   return 1;
}

struct DefineStruct {
   int operator()() { return 1; }
};

void FillTree(const char *filename, const char *treeName, int nevents = 0)
{
   TFile f(filename, "RECREATE");
   TTree t(treeName, treeName);
   t.SetAutoFlush(1); // yes, one event per cluster: to make MT more meaningful
   double b1;
   int b2;
   double b3[2];
   unsigned int n;
   int b4[2] = {21, 42};
   t.Branch("b1", &b1);
   t.Branch("b2", &b2);
   t.Branch("b3", b3, "b3[2]/D");
   t.Branch("n", &n);
   t.Branch("b4", b4, "b4[n]/I");
   for (int i = 0; i < nevents; ++i) {
      b1 = i;
      b2 = i * i;
      b3[0] = b1;
      b3[1] = -b1;
      n = i % 2 + 1;
      t.Fill();
   }
   t.Write();
   f.Close();
}
}

TEST(TEST_CATEGORY, CreateEmpty)
{
   TDataFrame tdf(10);
   auto c = tdf.Count();
   EXPECT_EQ(10U, *c);
}

TEST(TEST_CATEGORY, CreateZeroEntries)
{
   TDataFrame tdf(0);
   auto c = tdf.Count();
   EXPECT_EQ(0U, *c);
}

TEST(TEST_CATEGORY, CreateZeroEntriesWithBranches)
{
   auto filename = "dataframe_simple_0.root";
   auto treename = "t";
#ifndef testTDF_simple_0_CREATED
#define testTDF_simple_0_CREATED
   TEST_CATEGORY::FillTree(filename, treename);
#endif
   TDataFrame tdf(treename, filename);
   auto c = tdf.Count();
   auto m = tdf.Mean("b1");
   EXPECT_EQ(0U, *c);
   EXPECT_EQ(0., *m);
}

TEST(TEST_CATEGORY, BuildWithTDirectory)
{
   auto filename = "dataframe_simple_1.root";
   auto treename = "t";
#ifndef testTDF_simple_1_CREATED
#define testTDF_simple_1_CREATED
   TEST_CATEGORY::FillTree(filename, treename, 50);
#endif
   TFile f(filename);
   TDataFrame tdf(treename, &f);
   auto c = tdf.Count();
   EXPECT_EQ(50U, *c);
}

// Jitting of column types
TEST(TEST_CATEGORY, TypeGuessing)
{
   auto filename = "dataframe_simple_2.root";
   auto treename = "t";
#ifndef testTDF_simple_2_CREATED
#define testTDF_simple_2_CREATED
   TEST_CATEGORY::FillTree(filename, treename, 50);
#endif
   TDataFrame tdf(treename, filename, {"b1"});
   auto hcompiled = tdf.Histo1D<double>();
   auto hjitted = tdf.Histo1D();
   EXPECT_EQ(50, hcompiled->GetEntries());
   EXPECT_EQ(50, hjitted->GetEntries());
   EXPECT_DOUBLE_EQ(hcompiled->GetMean(), hjitted->GetMean());
}

// Define

TEST(TEST_CATEGORY, Define_lambda)
{
   TDataFrame tdf(10);
   auto d = tdf.Define("i", []() { return 1; });
   auto m = d.Mean("i");
   EXPECT_DOUBLE_EQ(1., *m);
}

TEST(TEST_CATEGORY, Define_function)
{
   TDataFrame tdf(10);
   auto d = tdf.Define("i", TEST_CATEGORY::DefineFunction);
   auto m = d.Mean("i");
   EXPECT_DOUBLE_EQ(1., *m);
}

TEST(TEST_CATEGORY, Define_functor)
{
   TDataFrame tdf(10);
   TEST_CATEGORY::DefineStruct def;
   auto d = tdf.Define("i", def);
   auto m = d.Mean("i");
   EXPECT_DOUBLE_EQ(1., *m);
}

TEST(TEST_CATEGORY, Define_jitted)
{
   TDataFrame tdf(10);
   auto d = tdf.Define("i", "1");
   auto m = d.Mean("i");
   EXPECT_DOUBLE_EQ(1., *m);
}

TEST(TEST_CATEGORY, Define_jitted_complex)
{
// The test can be run in sequential and MT mode.
#ifndef RNDM_GEN_CREATED
#define RNDM_GEN_CREATED
   gInterpreter->ProcessLine("TRandom r;");
#endif
   gInterpreter->ProcessLine("r.SetSeed(1);");
   TDataFrame tdf(50);
   auto d = tdf.Define("i", "r.Uniform(0.,8.)");
   auto m = d.Max("i");
   EXPECT_EQ(7.867497533559811628, *m);
}

// Define + Filters
TEST(TEST_CATEGORY, Define_Filter)
{
   TRandom r(1);
   TDataFrame tdf(50);
   auto d = tdf.Define("r", [&r]() { return r.Uniform(0., 8.); });
   auto df = d.Filter([](double x) { return x > 5; }, {"r"});
   auto m = df.Max("r");
   EXPECT_EQ(7.867497533559811628, *m);
}

TEST(TEST_CATEGORY, Define_Filter_jitted)
{
   TRandom r(1);
   TDataFrame tdf(50);
   auto d = tdf.Define("r", [&r]() { return r.Uniform(0., 8.); });
   auto df = d.Filter("r>5");
   auto m = df.Max("r");
   EXPECT_EQ(7.867497533559811628, *m);
}

TEST(TEST_CATEGORY, Define_Filter_named)
{
   TRandom r(1);
   TDataFrame tdf(50);
   auto d = tdf.Define("r", [&r]() { return r.Uniform(0., 8.); });
   auto df = d.Filter([](double x) { return x > 5; }, {"r"}, "myFilter");
   auto m = df.Max("r");
   EXPECT_EQ(7.867497533559811628, *m);
}

TEST(TEST_CATEGORY, Define_Filter_named_jitted)
{
   TRandom r(1);
   TDataFrame tdf(50);
   auto d = tdf.Define("r", [&r]() { return r.Uniform(0., 8.); });
   auto df = d.Filter("r>5", "myFilter");
   auto m = df.Max("r");
   EXPECT_EQ(7.867497533559811628, *m);
}

// jitted Define + Filters
TEST(TEST_CATEGORY, Define_jitted_Filter)
{
   gInterpreter->ProcessLine("r.SetSeed(1);");
   TDataFrame tdf(50);
   auto d = tdf.Define("r", "r.Uniform(0.,8.)");
   auto df = d.Filter([](double x) { return x > 5; }, {"r"});
   auto m = df.Max("r");
   EXPECT_EQ(7.867497533559811628, *m);
}

TEST(TEST_CATEGORY, Define_jitted_Filter_jitted)
{
   gInterpreter->ProcessLine("r.SetSeed(1);");
   TDataFrame tdf(50);
   auto d = tdf.Define("r", "r.Uniform(0.,8.)");
   auto df = d.Filter("r>5");
   auto m = df.Max("r");
   EXPECT_EQ(7.867497533559811628, *m);
}

TEST(TEST_CATEGORY, Define_jitted_Filter_named)
{
   gInterpreter->ProcessLine("r.SetSeed(1);");
   TDataFrame tdf(50);
   auto d = tdf.Define("r", "r.Uniform(0.,8.)");
   auto df = d.Filter([](double x) { return x > 5; }, {"r"}, "myFilter");
   auto m = df.Max("r");
   EXPECT_EQ(7.867497533559811628, *m);
}

TEST(TEST_CATEGORY, Define_jitted_Filter_named_jitted)
{
   gInterpreter->ProcessLine("r.SetSeed(1);");
   TDataFrame tdf(50);
   auto d = tdf.Define("r", "r.Uniform(0.,8.)");
   auto df = d.Filter("r>5", "myFilter");
   auto m = df.Max("r");
   EXPECT_EQ(7.867497533559811628, *m);
}

TEST(TEST_CATEGORY, DefineSlotConsistency)
{
   TDataFrame df(8);
   auto m = df.DefineSlot("x", [](unsigned int) { return 1.; }).Max("x");
   EXPECT_EQ(1., *m);
}

TEST(TEST_CATEGORY, DefineSlot)
{
   std::array<int, NSLOTS> values;
   for (auto i = 0u; i < NSLOTS; ++i)
      values[i] = i;
   TDataFrame df(NSLOTS);
   auto ddf = df.DefineSlot("s", [values](unsigned int slot) { return values[slot]; });
   auto m = ddf.Max("s");
   EXPECT_EQ(*m, NSLOTS - 1); // no matter the order of processing, the higher slot number is always taken at least once
}

TEST(TEST_CATEGORY, DefineSlotCheckMT)
{
   auto nSlots = NSLOTS;

   std::hash<std::thread::id> hasher;
   using H_t = decltype(hasher(std::this_thread::get_id()));

   std::vector<H_t> ids(nSlots, 0);
   TDataFrame d(nSlots);
   auto m = d.DefineSlot("x", [&](unsigned int slot) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                ids[slot] = hasher(std::this_thread::get_id());
                return 1.;
             }).Max("x");

   EXPECT_EQ(1, *m); // just in case

   std::set<H_t> s(ids.begin(), ids.end());
   EXPECT_EQ(nSlots, s.size());
   EXPECT_TRUE(s.end() == s.find(0));
}

TEST(TEST_CATEGORY, Snapshot_update)
{
   using TSnapshotOptions = ROOT::Experimental::TDF::TSnapshotOptions;

   TSnapshotOptions opts;

   opts.fMode = "UPDATE";

   TDataFrame tdf1(1000);
   auto s1 = tdf1.Define("one", []() { return 1.0; }).Snapshot<double>("mytree1", "snapshot_test_update.root", {"one"});

   EXPECT_EQ(1000U, *s1.Count());

   EXPECT_EQ(1.0, *s1.Min("one"));
   EXPECT_EQ(1.0, *s1.Max("one"));
   EXPECT_EQ(1.0, *s1.Mean("one"));

   TDataFrame tdf2(1000);
   auto s2 =
      tdf2.Define("two", []() { return 2.0; }).Snapshot<double>("mytree2", "snapshot_test_update.root", {"two"}, opts);

   EXPECT_EQ(1000U, *s2.Count());

   EXPECT_EQ(2.0, *s2.Min("two"));
   EXPECT_EQ(2.0, *s2.Max("two"));
   EXPECT_EQ(2.0, *s2.Mean("two"));

   TFile *f = TFile::Open("snapshot_test_update.root", "READ");
   auto mytree1 = (TTree *)f->Get("mytree1");
   auto mytree2 = (TTree *)f->Get("mytree2");

   EXPECT_NE(nullptr, mytree1);
   EXPECT_NE(nullptr, mytree2);

   f->Close();
   delete f;
}

TEST(TEST_CATEGORY, Snapshot_action_with_options)
{
   using TSnapshotOptions = ROOT::Experimental::TDF::TSnapshotOptions;

   TSnapshotOptions opts;
   opts.fAutoFlush = 10;
   opts.fMode = "RECREATE";

   for (auto algorithm : {ROOT::kZLIB, ROOT::kLZMA, ROOT::kLZ4}) {
      TDataFrame tdf(1000);

      opts.fCompressionLevel = 6;
      opts.fCompressionAlgorithm = algorithm;

      auto s =
         tdf.Define("one", []() { return 1.0; }).Snapshot<double>("mytree", "snapshot_test_opts.root", {"one"}, opts);

      EXPECT_EQ(1000U, *s.Count());
      EXPECT_EQ(1.0, *s.Min("one"));
      EXPECT_EQ(1.0, *s.Max("one"));
      EXPECT_EQ(1.0, *s.Mean("one"));

      TFile *f = TFile::Open("snapshot_test_opts.root", "READ");

      EXPECT_EQ(algorithm, f->GetCompressionAlgorithm());
      EXPECT_EQ(6, f->GetCompressionLevel());

      f->Close();
      delete f;
   }
}

// This tests the interface but we need to run it both w/ and w/o implicit mt
#ifdef R__USE_IMT
TEST(TEST_CATEGORY, GetNSlots)
{
   EXPECT_EQ(NSLOTS, ROOT::Internal::TDF::GetNSlots());
}
#endif

TEST(TEST_CATEGORY, CArraysFromTree)
{
   auto filename = "dataframe_simple_3.root";
   auto treename = "t";
#ifndef testTDF_simple_3_CREATED
#define testTDF_simple_3_CREATED
   TEST_CATEGORY::FillTree(filename, treename, 10);
#endif
   TDataFrame df(treename, filename);

   // no jitting
   auto h = df.Filter([](double b1, unsigned int n, std::array_view<double> b3,
                         std::array_view<int> b4) { return b3[0] == b1 && b4[0] == 21 && b4.size() == n; },
                      {"b1", "n", "b3", "b4"})
               .Histo1D<std::array_view<double>>("b3");
   EXPECT_EQ(20, h->GetEntries());

   // jitting
   auto h_jit = df.Filter(/*"b3[0] == b1"*/"b4[0] == 21"/*"b4.size() == n"*/).Histo1D("b3");
   EXPECT_EQ(20, h_jit->GetEntries());
}
