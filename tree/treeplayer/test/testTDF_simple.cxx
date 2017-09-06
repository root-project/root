#include "ROOT/TDataFrame.hxx"
#include "TRandom.h"
#include "TInterpreter.h"

#include "gtest/gtest.h"

using namespace ROOT::Experimental;

/*
Tests we need to cover:
o define
o jitted define
o filter
o jitted filter
o named filter
o named jitted filter
o (defines) X filters
o list of branches present but empty
 */

TEST(TDataFrame, CreateEmpty)
{
   TDataFrame tdf(10);
}

// Define

TEST(TDataFrame, Define_lambda)
{
   TDataFrame tdf(10);
   auto d = tdf.Define("i", [](){return 1;});
   auto m = d.Mean("i");
}

int DefineFunction() {return 1;}

TEST(TDataFrame, Define_function)
{
   TDataFrame tdf(10);
   auto d = tdf.Define("i", DefineFunction);
   auto m = d.Mean("i");
}

struct DefineStruct {
   int operator()() {return 1;}
};

TEST(TDataFrame, Define_functor)
{
   TDataFrame tdf(10);
   DefineStruct def;
   auto d = tdf.Define("i", def);
   auto m = d.Mean("i");
}

TEST(TDataFrame, Define_jitted)
{
   TDataFrame tdf(10);
   auto d = tdf.Define("i", "1");
   auto m = d.Mean("i");
}

TEST(TDataFrame, Define_jitted_complex)
{
   gInterpreter->ProcessLine("TRandom r(1);");
   TDataFrame tdf(100);
   auto d = tdf.Define("i", "r.Uniform(0.,8.)");
   auto m = d.Mean("i");
}


// Filter

TEST(TDataFrame, Define_Filter)
{
   TRandom r(1);
   TDataFrame tdf(100);
   auto d = tdf.Define("r", [&r](){return r.Uniform(0.,8.);});
   auto df = d.Filter([](double x){ return x > 5;},{"r"});
   auto m = df.Mean("r");
}
