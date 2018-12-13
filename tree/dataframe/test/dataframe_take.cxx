#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"
#include "gtest/gtest.h"

TEST(RDataFrameTake, Bool)
{
   ROOT::RDataFrame df(1);
   auto df2 = df.Define("x", "(bool)true");
   auto result_ptr = df2.Take<bool>("x");
   auto vec = result_ptr.GetValue();
   EXPECT_EQ(vec.size(), 1u);
   EXPECT_EQ(vec[0], true);

   auto result_ptr2 = df2.Take<bool, ROOT::VecOps::RVec<bool>>("x");
   auto vec2 = result_ptr2.GetValue();
   EXPECT_EQ(vec2.size(), 1u);
   EXPECT_EQ(vec2[0], true);
}

TEST(RDataFrameTake, VectorBool)
{
   ROOT::RDataFrame df(1);
   auto df2 = df.Define("x", "std::vector<bool>({true, false})");
   auto result_ptr = df2.Take<std::vector<bool>>("x");
   auto vec = result_ptr.GetValue();
   EXPECT_EQ(vec.size(), 1u);
   EXPECT_EQ(vec[0].size(), 2u);
   EXPECT_EQ(vec[0][0], true);
   EXPECT_EQ(vec[0][1], false);
}

TEST(RDataFrameTake, Float)
{
   ROOT::RDataFrame df(1);
   auto df2 = df.Define("x", "(float)42.0");
   auto result_ptr = df2.Take<float>("x");
   auto vec = result_ptr.GetValue();
   EXPECT_EQ(vec.size(), 1u);
   EXPECT_EQ(vec[0], 42.0);

   auto result_ptr2 = df2.Take<float, ROOT::VecOps::RVec<float>>("x");
   auto vec2 = result_ptr2.GetValue();
   EXPECT_EQ(vec2.size(), 1u);
   EXPECT_EQ(vec2[0], 42.0);
}
