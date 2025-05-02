#include "gtest/gtest.h"

#include "TAxis.h"


TEST(TAxis, FindBinExact)
{
   //Test the case where bin edges are exactly represented as floating points
   TAxis ax(88, 1010, 1098);
   for (int i = 1; i <= ax.GetNbins(); i++) {
      double x = ax.GetBinLowEdge(i);
      EXPECT_EQ(i, ax.FindBin(x));
      EXPECT_EQ(i, ax.FindFixBin(x));
      x = ax.GetBinUpEdge(i);
      EXPECT_EQ(i+1, ax.FindBin(x));
      EXPECT_EQ(i+1, ax.FindFixBin(x));
      x -= x * std::numeric_limits<double>::epsilon();
      EXPECT_EQ(i, ax.FindBin(x));
   }
}
TEST(TAxis, FindBinApprox)
{
   TAxis ax(90, 0. , 10.);
   for (int i = 1; i <= ax.GetNbins(); i++) {
      double x = ax.GetBinLowEdge(i);
      EXPECT_EQ(i, ax.FindBin(x));
      EXPECT_EQ(i, ax.FindFixBin(x));
      x = ax.GetBinUpEdge(i);
      EXPECT_EQ(i+1, ax.FindBin(x));
      EXPECT_EQ(i+1, ax.FindFixBin(x));
      x -= x * std::numeric_limits<double>::epsilon();
      EXPECT_EQ(i, ax.FindBin(x));
   }
}