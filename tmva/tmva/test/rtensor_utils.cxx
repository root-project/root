#include <gtest/gtest.h>
#include "TMVA/RTensor.hxx"
#include "TMVA/RTensorUtils.hxx"
#include "ROOT/RDataFrame.hxx"

using namespace ROOT;
using namespace TMVA::Experimental;

TEST(RTensor, AsTensor)
{
   RDataFrame df(10);
   auto df2 = df.Define("a", "1.f * rdfentry_").Define("b", "-1.f * rdfentry_");
   auto x = AsTensor<float>(df2, {"a", "b"});
   EXPECT_EQ(x.GetShape().size(), 2u);
   EXPECT_EQ(x.GetShape()[0], 10u);
   EXPECT_EQ(x.GetShape()[1], 2u);
   for (size_t i = 0; i < 10; i++) {
      EXPECT_EQ(x(i, 0), 1.f * i);
      EXPECT_EQ(x(i, 1), -1.f * i);
   }
}

TEST(RTensor, AsTensorAllColumns)
{
   RDataFrame df(10);
   auto df2 = df.Define("a", "1.f * rdfentry_").Define("b", "-1.f * rdfentry_");
   auto x = AsTensor<float>(df2);
   EXPECT_EQ(x.GetShape().size(), 2u);
   EXPECT_EQ(x.GetShape()[0], 10u);
   EXPECT_EQ(x.GetShape()[1], 2u);
   for (size_t i = 0; i < 10; i++) {
      EXPECT_EQ(x(i, 0), 1.f * i);
      EXPECT_EQ(x(i, 1), -1.f * i);
   }
}

TEST(RTensor, AsTensorColumnMajor)
{
   RDataFrame df(10);
   auto df2 = df.Define("a", "1.f * rdfentry_").Define("b", "-1.f * rdfentry_");
   auto x = AsTensor<float>(df2, {"a", "b"}, MemoryLayout::ColumnMajor);
   EXPECT_EQ(x.GetShape().size(), 2u);
   EXPECT_EQ(x.GetShape()[0], 10u);
   EXPECT_EQ(x.GetShape()[1], 2u);
   for (size_t i = 0; i < 10; i++) {
      EXPECT_EQ(x(i, 0), 1.f * i);
      EXPECT_EQ(x(i, 1), -1.f * i);
   }
}
