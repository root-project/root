#include "ROOT/TDataFrame.hxx"
#include "TRandom.h"
#include "TInterpreter.h"

#include "gtest/gtest.h"

using namespace ROOT::Experimental;

namespace TEST_CATEGORY {
   int DefineFunction() {return 1;}

   struct DefineStruct {
      int operator()() {return 1;}
   };
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

// Define

TEST(TEST_CATEGORY, Define_lambda)
{
   TDataFrame tdf(10);
   auto d = tdf.Define("i", [](){return 1;});
   auto m = d.Mean("i");
}

TEST(TEST_CATEGORY, Define_function)
{
   TDataFrame tdf(10);
   auto d = tdf.Define("i", TEST_CATEGORY::DefineFunction);
   auto m = d.Mean("i");
}

TEST(TEST_CATEGORY, Define_functor)
{
   TDataFrame tdf(10);
   TEST_CATEGORY::DefineStruct def;
   auto d = tdf.Define("i", def);
   auto m = d.Mean("i");
}

TEST(TEST_CATEGORY, Define_jitted)
{
   TDataFrame tdf(10);
   auto d = tdf.Define("i", "1");
   auto m = d.Mean("i");
}

TEST(TEST_CATEGORY, Define_jitted_complex)
{
   // The test can be run in sequential and MT mode.
#ifndef RNDM_GEN_CREATED
#define RNDM_GEN_CREATED
   gInterpreter->ProcessLine("TRandom r(1);");
#endif
   TDataFrame tdf(50);
   auto d = tdf.Define("i", "r.Uniform(0.,8.)");
   auto m = d.Mean("i");
}


// Define + Filters
TEST(TEST_CATEGORY, Define_Filter)
{
   TRandom r(1);
   TDataFrame tdf(50);
   auto d = tdf.Define("r", [&r](){return r.Uniform(0.,8.);});
   auto df = d.Filter([](double x){ return x > 5;},{"r"});
   auto m = df.Mean("r");
}

TEST(TEST_CATEGORY, Define_Filter_jitted)
{
   TRandom r(1);
   TDataFrame tdf(50);
   auto d = tdf.Define("r", [&r](){return r.Uniform(0.,8.);});
   auto df = d.Filter("r>5");
   auto m = df.Mean("r");
}

TEST(TEST_CATEGORY, Define_Filter_named)
{
   TRandom r(1);
   TDataFrame tdf(50);
   auto d = tdf.Define("r", [&r](){return r.Uniform(0.,8.);});
   auto df = d.Filter([](double x){ return x > 5;},{"r"},"myFilter");
   auto m = df.Mean("r");
}

TEST(TEST_CATEGORY, Define_Filter_named_jitted)
{
   TRandom r(1);
   TDataFrame tdf(50);
   auto d = tdf.Define("r", [&r](){return r.Uniform(0.,8.);});
   auto df = d.Filter("r>5","myFilter");
   auto m = df.Mean("r");
}

// jitted Define + Filters
TEST(TEST_CATEGORY, Define_jitted_Filter)
{
   gInterpreter->ProcessLine("r.SetSeed(1);");
   TDataFrame tdf(50);
   auto d = tdf.Define("r", "r.Uniform(0.,8.)");
   auto df = d.Filter([](double x){ return x > 5;},{"r"});
   auto m = df.Mean("r");
}

TEST(TEST_CATEGORY, Define_jitted_Filter_jitted)
{
   gInterpreter->ProcessLine("r.SetSeed(1);");
   TDataFrame tdf(50);
   auto d = tdf.Define("r", "r.Uniform(0.,8.)");
   auto df = d.Filter("r>5");
   auto m = df.Mean("r");
}

TEST(TEST_CATEGORY, Define_jitted_Filter_named)
{
   gInterpreter->ProcessLine("r.SetSeed(1);");
   TDataFrame tdf(50);
   auto d = tdf.Define("r", "r.Uniform(0.,8.)");
   auto df = d.Filter([](double x){ return x > 5;},{"r"},"myFilter");
   auto m = df.Mean("r");
}

TEST(TEST_CATEGORY, Define_jitted_Filter_named_jitted)
{
   gInterpreter->ProcessLine("r.SetSeed(1);");
   TDataFrame tdf(50);
   auto d = tdf.Define("r", "r.Uniform(0.,8.)");
   auto df = d.Filter("r>5","myFilter");
   auto m = df.Mean("r");
}

