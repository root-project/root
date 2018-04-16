#include <ROOT/TDataFrame.hxx>
#include "gtest/gtest.h"
using namespace ROOT::Experimental::TDF;
using namespace ROOT::Experimental;

class Dummy {
};

TEST(TResultPtr, DefCtor)
{
   TResultPtr<Dummy> p1, p2;
   EXPECT_TRUE(p1 == nullptr);
   EXPECT_TRUE(nullptr == p1);
   EXPECT_TRUE(p1 == p2);
   EXPECT_TRUE(p2 == p1);
}

TEST(TResultPtr, CopyCtor)
{
   TDataFrame d(1);
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

TEST(TResultPtr, ImplConv)
{
   TResultPtr<Dummy> p1;
   EXPECT_FALSE(p1);

   TDataFrame d(1);
   auto hasRun = false;
   auto m = d.Define("i", [&hasRun]() {
                hasRun = true;
                return (int)1;
             }).Histo1D<int>("i");

   EXPECT_TRUE(m);
   EXPECT_FALSE(hasRun);

   *m;

   EXPECT_TRUE(m);
   EXPECT_TRUE(hasRun);
}
