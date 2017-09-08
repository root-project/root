#include "ROOT/TDataFrame.hxx"
#include "TRandom.h"
#include "TInterpreter.h"

#include "gtest/gtest.h"

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
   t.Branch("b1", &b1);
   t.Branch("b2", &b2);
   for (int i = 0; i < nevents; ++i) {
      b1 = i;
      b2 = i * i;
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
