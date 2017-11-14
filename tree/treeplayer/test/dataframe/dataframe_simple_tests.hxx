#include "Compression.h"
#include "ROOT/TDataFrame.hxx"
#include "ROOT/TSeq.hxx"
#include "TInterpreter.h"
#include "TFile.h"
#include "TRandom.h"
#include "TSystem.h"

#include "gtest/gtest.h"

#include <algorithm> // std::sort
#include <chrono>
#include <thread>
#include <set>

using namespace ROOT::Experimental;
using namespace ROOT::Experimental::TDF;

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

TEST(TEST_CATEGORY, Define_jitted_complex_array_sum)
{
   TDataFrame tdf(10);
   auto d = tdf.Define("x", "3.0")
               .Define("y", "4.0")
               .Define("z", "12.0")
               .Define("v", "std::array<double, 3> v{x, y, z}; return v;")
               .Define("r", "double r2 = 0.0; for (auto&& w : v) r2 += w*w; return sqrt(r2);");
   auto m = d.Max("r");
   EXPECT_DOUBLE_EQ(13.0, *m);
}

TEST(TEST_CATEGORY, Define_jitted_defines_with_return)
{
   TDataFrame tdf(10);
   auto d = tdf.Define("my_return_x", "3.0")
               .Define("return_y", "4.0 // with a comment")
               .Define("v", "std::array<double, 2> v{my_return_x, return_y}; return v; // also with comment")
               .Define("r", "double r2 = 0.0; for (auto&& w : v) r2 += w*w; return sqrt(r2);");
   auto m = d.Max("r");
   EXPECT_DOUBLE_EQ(5.0, *m);
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

TEST(TEST_CATEGORY, Define_jitted_Filter_complex_array)
{
   gInterpreter->ProcessLine("r.SetSeed(1);");
   TDataFrame tdf(50);
   auto d = tdf.Define("x", "r.Uniform(0.0, 1.0)")
               .Define("y", "r.Uniform(0.0, 1.0)")
               .Define("z", "r.Uniform(0.0, 1.0)")
               .Define("v", "std::array<double, 3> v{x, y, z}; return v;")
               .Define("r", "double r2 = 0.0; for (auto&& w : v) r2 += w*w; return sqrt(r2);");
   auto dfin = d.Filter("r <= 1.0", "inside");
   auto dfout = d.Filter("bool out = r > 1.0; return out;", "outside");
   auto in  = dfin.Count();
   auto out = dfout.Count();

   EXPECT_TRUE(*in  < 50U);
   EXPECT_TRUE(*out < 50U);
   EXPECT_EQ(50U, *in + *out);
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
   const auto nSlots = NSLOTS;

   std::vector<unsigned int> ids(nSlots, 0u);
   TDataFrame d(nSlots);
   auto m = d.DefineSlot("x", [&](unsigned int slot) {
                ids[slot] = 1u;
                return 1;
             }).Max("x");
   EXPECT_EQ(1, *m); // just in case

   const auto nUsedSlots = std::accumulate(ids.begin(), ids.end(), 0u);
   EXPECT_GT(nUsedSlots, 0u);
   EXPECT_LE(nUsedSlots, nSlots);
}

TEST(TEST_CATEGORY, DefineSlotEntry)
{
   const auto nEntries = 8u;
   TDataFrame df(nEntries);
   auto es = df.DefineSlotEntry("e", [](unsigned int, ULong64_t e) { return e; }).Take<ULong64_t>("e");
   auto entries = *es;
   std::sort(entries.begin(), entries.end());
   for (auto i = 0u; i < nEntries; ++i) {
      EXPECT_EQ(i, entries[i]);
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
   auto h = df.Filter([](double b1, unsigned int n, TArrayBranch<double> b3,
                         TArrayBranch<int> b4) { return b3[0] == b1 && b4[0] == 21 && b4.size() == n; },
                      {"b1", "n", "b3", "b4"})
               .Histo1D<TArrayBranch<double>>("b3");
   EXPECT_EQ(20, h->GetEntries());

   // jitting
   auto h_jit = df.Filter(/*"b3[0] == b1"*/"b4[0] == 21"/*"b4.size() == n"*/).Histo1D("b3");
   EXPECT_EQ(20, h_jit->GetEntries());
}


TEST(TEST_CATEGORY, TakeCarrays)
{
   auto treeName = "t";
   auto fileName = "CacheCarrays.root";

   {
      TFile f(fileName, "RECREATE");
      TTree t(treeName, treeName);
      float arr[4];
      t.Branch("arr", arr, "arr[4]/F");
      for (auto i : ROOT::TSeqU(4)) {
         for (auto j : ROOT::TSeqU(4)) {
            arr[j] = i + j;
         }
         t.Fill();
      }
      t.Write();
   }

   TDataFrame tdf(treeName, fileName);
   // no auto here: we check that the type is a COLL<vector<float>>!
   using ColType_t = TDF::TArrayBranch<float>;
   std::vector<std::vector<float>> v = *tdf.Take<ColType_t>("arr");
   std::deque<std::vector<float>> d = *tdf.Take<ColType_t, std::deque<ColType_t>>("arr");
   std::list<std::vector<float>> l = *tdf.Take<ColType_t, std::list<ColType_t>>("arr");

   auto lit = l.begin();
   auto ifloat = 0.f;
   for (auto i : ROOT::TSeqU(4)) {
      const auto &vv = v[i];
      const auto &dv = d[i];
      const auto &lv = *lit;
      for (auto j : ROOT::TSeqU(4)) {
         const auto ref = ifloat + j;
         EXPECT_EQ(ref, vv[j]);
         EXPECT_EQ(ref, dv[j]);
         EXPECT_EQ(ref, lv[j]);
      }
      ifloat++;
      lit++;
   }

   gSystem->Unlink(fileName);
}
