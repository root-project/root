/****** Run TDataFrame tests both with and without IMT enabled *******/
#include <gtest/gtest.h>
#include <ROOT/TDataFrame.hxx>
#include <ROOT/TSeq.hxx>
#include <TFile.h>
#include <TGraph.h>
#include <TInterpreter.h>
#include <TRandom.h>
#include <TROOT.h>
#include <TSystem.h>
#include <TTree.h>

#include <algorithm> // std::sort
#include <chrono>
#include <thread>
#include <set>

using namespace ROOT::Experimental;
using namespace ROOT::Experimental::TDF;
using namespace ROOT::Experimental::VecOps;

// Fixture for all tests in this file. If parameter is true, run with implicit MT, else run sequentially
class TDFSimpleTests : public ::testing::TestWithParam<bool> {
protected:
   TDFSimpleTests() : NSLOTS(GetParam() ? 4u : 1u)
   {
      if (GetParam())
         ROOT::EnableImplicitMT(NSLOTS);
   }
   ~TDFSimpleTests()
   {
      if (GetParam())
         ROOT::DisableImplicitMT();
   }
   const unsigned int NSLOTS;
};

// Create file `filename` containing a test tree `treeName` with `nevents` events
// TODO: create just one file at the beginning of the test execution, delete the file at test exit
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

TEST_P(TDFSimpleTests, CreateEmpty)
{
   TDataFrame tdf(10);
   auto c = tdf.Count();
   EXPECT_EQ(10U, *c);
}

TEST_P(TDFSimpleTests, CreateZeroEntries)
{
   TDataFrame tdf(0);
   auto c = tdf.Count();
   EXPECT_EQ(0U, *c);
}

TEST_P(TDFSimpleTests, CreateZeroEntriesWithBranches)
{
   auto filename = "dataframe_simple_0.root";
   auto treename = "t";
   // create input file (at most once per execution of the parametrized gtest)
   static bool hasFile = false;
   if (!hasFile) {
      FillTree(filename, treename);
      hasFile = true;
   }
   TDataFrame tdf(treename, filename);
   auto c = tdf.Count();
   auto m = tdf.Mean("b1");
   EXPECT_EQ(0U, *c);
   EXPECT_EQ(0., *m);
}

TEST_P(TDFSimpleTests, BuildWithTDirectory)
{
   auto filename = "dataframe_simple_1.root";
   auto treename = "t";
   // create input file (at most once per execution of the parametrized gtest)
   static bool hasFile = false;
   if (!hasFile) {
      FillTree(filename, treename, 50);
      hasFile = true;
   }
   TFile f(filename);
   TDataFrame tdf(treename, &f);
   auto c = tdf.Count();
   EXPECT_EQ(50U, *c);
}

// Jitting of column types
TEST_P(TDFSimpleTests, TypeGuessing)
{
   auto filename = "dataframe_simple_2.root";
   auto treename = "t";
   // create input file (at most once per execution of the parametrized gtest)
   static bool hasFile = false;
   if (!hasFile) {
      FillTree(filename, treename, 50);
      hasFile = true;
   }
   TDataFrame tdf(treename, filename, {"b1"});
   auto hcompiled = tdf.Histo1D<double>();
   auto hjitted = tdf.Histo1D();
   EXPECT_EQ(50, hcompiled->GetEntries());
   EXPECT_EQ(50, hjitted->GetEntries());
   EXPECT_DOUBLE_EQ(hcompiled->GetMean(), hjitted->GetMean());
}

// Define

TEST_P(TDFSimpleTests, Define_lambda)
{
   TDataFrame tdf(10);
   auto d = tdf.Define("i", []() { return 1; });
   auto m = d.Mean("i");
   EXPECT_DOUBLE_EQ(1., *m);
}

int DefineFunction()
{
   return 1;
}

TEST_P(TDFSimpleTests, Define_function)
{
   TDataFrame tdf(10);
   auto d = tdf.Define("i", DefineFunction);
   auto m = d.Mean("i");
   EXPECT_DOUBLE_EQ(1., *m);
}

struct DefineStruct {
   int operator()() { return 1; }
};

TEST_P(TDFSimpleTests, Define_functor)
{
   TDataFrame tdf(10);
   DefineStruct def;
   auto d = tdf.Define("i", def);
   auto m = d.Mean("i");
   EXPECT_DOUBLE_EQ(1., *m);
}

TEST_P(TDFSimpleTests, Define_jitted)
{
   TDataFrame tdf(10);
   auto d = tdf.Define("i", "1");
   auto m = d.Mean("i");
   EXPECT_DOUBLE_EQ(1., *m);
}

TEST_P(TDFSimpleTests, Define_jitted_complex)
{
   // this test case (as all others) is usually run twice, in IMT and non-IMT mode,
   // but we only want to create the TRandom object once.
   static bool hasJittedTRandom = false;
   if (!hasJittedTRandom) {
      gInterpreter->ProcessLine("TRandom r;");
      hasJittedTRandom = true;
   }
   gInterpreter->ProcessLine("r.SetSeed(1);");
   TDataFrame tdf(50);
   auto d = tdf.Define("i", "r.Uniform(0.,8.)");
   auto m = d.Max("i");
   EXPECT_EQ(7.867497533559811628, *m);
}

TEST_P(TDFSimpleTests, Define_jitted_complex_array_sum)
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

TEST_P(TDFSimpleTests, Define_jitted_defines_with_return)
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
TEST_P(TDFSimpleTests, Define_Filter)
{
   TRandom r(1);
   TDataFrame tdf(50);
   auto d = tdf.Define("r", [&r]() { return r.Uniform(0., 8.); });
   auto df = d.Filter([](double x) { return x > 5; }, {"r"});
   auto m = df.Max("r");
   EXPECT_EQ(7.867497533559811628, *m);
}

TEST_P(TDFSimpleTests, Define_Filter_jitted)
{
   TRandom r(1);
   TDataFrame tdf(50);
   auto d = tdf.Define("r", [&r]() { return r.Uniform(0., 8.); });
   auto df = d.Filter("r>5");
   auto m = df.Max("r");
   EXPECT_EQ(7.867497533559811628, *m);
}

TEST_P(TDFSimpleTests, Define_Filter_named)
{
   TRandom r(1);
   TDataFrame tdf(50);
   auto d = tdf.Define("r", [&r]() { return r.Uniform(0., 8.); });
   auto df = d.Filter([](double x) { return x > 5; }, {"r"}, "myFilter");
   auto m = df.Max("r");
   EXPECT_EQ(7.867497533559811628, *m);
}

TEST_P(TDFSimpleTests, Define_Filter_named_jitted)
{
   TRandom r(1);
   TDataFrame tdf(50);
   auto d = tdf.Define("r", [&r]() { return r.Uniform(0., 8.); });
   auto df = d.Filter("r>5", "myFilter");
   auto m = df.Max("r");
   EXPECT_EQ(7.867497533559811628, *m);
}

// jitted Define + Filters
TEST_P(TDFSimpleTests, Define_jitted_Filter)
{
   gInterpreter->ProcessLine("r.SetSeed(1);");
   TDataFrame tdf(50);
   auto d = tdf.Define("r", "r.Uniform(0.,8.)");
   auto df = d.Filter([](double x) { return x > 5; }, {"r"});
   auto m = df.Max("r");
   EXPECT_EQ(7.867497533559811628, *m);
}

TEST_P(TDFSimpleTests, Define_jitted_Filter_jitted)
{
   gInterpreter->ProcessLine("r.SetSeed(1);");
   TDataFrame tdf(50);
   auto d = tdf.Define("r", "r.Uniform(0.,8.)");
   auto df = d.Filter("r>5");
   auto m = df.Max("r");
   EXPECT_EQ(7.867497533559811628, *m);
}

TEST_P(TDFSimpleTests, Define_jitted_Filter_named)
{
   gInterpreter->ProcessLine("r.SetSeed(1);");
   TDataFrame tdf(50);
   auto d = tdf.Define("r", "r.Uniform(0.,8.)");
   auto df = d.Filter([](double x) { return x > 5; }, {"r"}, "myFilter");
   auto m = df.Max("r");
   EXPECT_EQ(7.867497533559811628, *m);
}

TEST_P(TDFSimpleTests, Define_jitted_Filter_named_jitted)
{
   gInterpreter->ProcessLine("r.SetSeed(1);");
   TDataFrame tdf(50);
   auto d = tdf.Define("r", "r.Uniform(0.,8.)");
   auto df = d.Filter("r>5", "myFilter");
   auto m = df.Max("r");
   EXPECT_EQ(7.867497533559811628, *m);
}

TEST_P(TDFSimpleTests, Define_jitted_Filter_complex_array)
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
   auto in = dfin.Count();
   auto out = dfout.Count();

   EXPECT_TRUE(*in < 50U);
   EXPECT_TRUE(*out < 50U);
   EXPECT_EQ(50U, *in + *out);
}

TEST_P(TDFSimpleTests, DefineSlotConsistency)
{
   TDataFrame df(8);
   auto m = df.DefineSlot("x", [](unsigned int) { return 1.; }).Max("x");
   EXPECT_EQ(1., *m);
}

TEST_P(TDFSimpleTests, DefineSlot)
{
   std::vector<int> values(NSLOTS);
   for (auto i = 0u; i < NSLOTS; ++i)
      values[i] = i;
   TDataFrame df(NSLOTS);
   auto ddf = df.DefineSlot("s", [values](unsigned int slot) { return values[slot]; });
   auto m = ddf.Max("s");
   EXPECT_EQ(*m, NSLOTS - 1); // no matter the order of processing, the higher slot number is always taken at least once
}

TEST_P(TDFSimpleTests, DefineSlotCheckMT)
{
   std::vector<unsigned int> ids(NSLOTS, 0u);
   TDataFrame d(NSLOTS);
   auto m = d.DefineSlot("x", [&](unsigned int slot) {
                ids[slot] = 1u;
                return 1;
             }).Max("x");
   EXPECT_EQ(1, *m); // just in case

   const auto nUsedSlots = std::accumulate(ids.begin(), ids.end(), 0u);
   EXPECT_GT(nUsedSlots, 0u);
   EXPECT_LE(nUsedSlots, NSLOTS);
}

TEST_P(TDFSimpleTests, DefineSlotEntry)
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

TEST_P(TDFSimpleTests, GetNSlots)
{
   EXPECT_EQ(NSLOTS, ROOT::Internal::TDF::GetNSlots());
}

TEST_P(TDFSimpleTests, CArraysFromTree)
{
   auto filename = "dataframe_simple_3.root";
   auto treename = "t";
   // create input file (at most once per execution of the parametrized gtest)
   static bool hasFile = false;
   if (!hasFile) {
      FillTree(filename, treename, 10);
      hasFile = true;
   }
   TDataFrame df(treename, filename);

   // no jitting
   auto h = df.Filter([](double b1, unsigned int n, TVec<double> &b3,
                         TVec<int> &b4) { return b3[0] == b1 && b4[0] == 21 && b4.size() == n; },
                      {"b1", "n", "b3", "b4"})
               .Histo1D<TVec<double>>("b3");
   EXPECT_EQ(20, h->GetEntries());

   // jitting
   auto h_jit = df.Filter(/*"b3[0] == b1"*/ "b4[0] == 21" /*"b4.size() == n"*/).Histo1D("b3");
   EXPECT_EQ(20, h_jit->GetEntries());
}

TEST_P(TDFSimpleTests, TakeCarrays)
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
   using ColType_t = VecOps::TVec<float>;
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

TEST_P(TDFSimpleTests, Reduce)
{
   auto d = TDataFrame(5).DefineSlotEntry("x", [](unsigned int, ULong64_t e) { return static_cast<int>(e) + 1; });
   auto r1 = d.Reduce([](int x, int y) { return x + y; }, "x");
   auto r2 = d.Reduce([](int x, int y) { return x * y; }, "x", 1);
   EXPECT_EQ(*r1, 15);
   EXPECT_EQ(*r2, 120);
}

TEST_P(TDFSimpleTests, Aggregate)
{
   auto d = TDataFrame(5).DefineSlotEntry("x", [](unsigned int, ULong64_t e) { return static_cast<int>(e) + 1; });
   // acc U(U,T), merge U(U,U), default initValue
   auto r1 = d.Aggregate([](int x, int y) { return x + y; }, [](int x, int y) { return x + y; }, "x");
   // acc U(U,T), merge U(U,U), initValue
   auto r2 = d.Aggregate([](int x, int y) { return x * y; }, [](int x, int y) { return x * y; }, "x", 1);
   EXPECT_EQ(*r1, 15);
   EXPECT_EQ(*r2, 120);
}

TEST_P(TDFSimpleTests, AggregateGraph)
{
   auto d = TDataFrame(20).DefineSlotEntry("x", [](unsigned int, ULong64_t e) { return static_cast<double>(e); });
   auto graph = d.Aggregate([](TGraph &g, double x) { g.SetPoint(g.GetN(), x, x * x); },
                            [](std::vector<TGraph> &graphs) {
                               TList l;
                               for (auto g = graphs.begin() + 1; g != graphs.end(); ++g)
                                  l.Add(&(*g));
                               graphs[0].Merge(&l);
                            },
                            "x");
   EXPECT_EQ(graph->GetN(), 20);
   // collect data-points, sorted by x values (they can be scrambled in MT executions)
   using Point_t = std::pair<double, double>;
   std::vector<Point_t> points;
   Point_t p;
   for (int i = 0; i < 20; ++i) {
      graph->GetPoint(i, p.first, p.second);
      points.emplace_back(p);
   }
   // check that all data-points are retrieved correctly
   std::sort(points.begin(), points.end(), [](const Point_t &p1, const Point_t &p2) { return p1.first < p2.first; });
   for (int i = 0; i < 20; ++i) {
      EXPECT_DOUBLE_EQ(points[i].first, i);
      EXPECT_DOUBLE_EQ(points[i].second, i * i);
   }
}

// run single-thread tests
INSTANTIATE_TEST_CASE_P(Seq, TDFSimpleTests, ::testing::Values(false));

// run multi-thread tests
#ifdef R__USE_IMT
   INSTANTIATE_TEST_CASE_P(MT, TDFSimpleTests, ::testing::Values(true));
#endif
