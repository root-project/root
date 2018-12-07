#include "ROOT/RDataFrame.hxx"
#include "gtest/gtest.h"

TEST(RDataFrameTake, Bool)
{
   ROOT::RDataFrame df(1);
   auto df2 = df.Define("x", "(bool)true");
   auto result_ptr = df2.Take<bool>("x");
   auto vec = result_ptr.GetValue();
   EXPECT_EQ(vec.size(), 1u);
   EXPECT_EQ(vec[0], true);
}
