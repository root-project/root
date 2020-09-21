#include <ROOT/RDataFrame.hxx>
#include "gtest/gtest.h"
using namespace ROOT::RDF;

class Dummy {
};

TEST(RResultPtr, DefCtor)
{
   RResultPtr<Dummy> p1, p2;
   EXPECT_TRUE(p1 == nullptr);
   EXPECT_TRUE(nullptr == p1);
   EXPECT_TRUE(p1 == p2);
   EXPECT_TRUE(p2 == p1);
}

TEST(RResultPtr, CopyCtor)
{
   ROOT::RDataFrame d(1);
   auto hasRun = false;
   auto m = d.Define("i", [&hasRun]() {
                hasRun = true;
                return (int)1;
             }).Mean<int>("i");
   auto mc = m;
   EXPECT_TRUE(mc == m);
   EXPECT_TRUE(m == mc);

   EXPECT_FALSE(hasRun);

   EXPECT_EQ(*mc, *m);

   EXPECT_TRUE(hasRun);
}

TEST(RResultPtr, MoveCtor)
{
   ROOT::RDataFrame df(1);
   ROOT::RDF::RResultPtr<ULong64_t> res(df.Count());

   // also test move-assignment
   res = df.Count();

   EXPECT_EQ(*res, 1u);
}

TEST(RResultPtr, ImplConv)
{
   RResultPtr<Dummy> p1;
   EXPECT_FALSE(p1);

   ROOT::RDataFrame d(1);
   auto hasRun = false;
   auto m = d.Define("i", [&hasRun]() {
                hasRun = true;
                return (int)1;
             }).Histo1D<int>("i");

   EXPECT_TRUE(m != nullptr);
   EXPECT_FALSE(hasRun);

   *m;

   EXPECT_TRUE(m != nullptr);
   EXPECT_TRUE(hasRun);
}

TEST(RResultPtr, IsReady)
{
   ROOT::RDataFrame d(1);
   auto p = d.Define("x", "rdfentry_").Sum("x");
   EXPECT_FALSE(p.IsReady());

   p.GetValue();
   EXPECT_TRUE(p.IsReady());
}
