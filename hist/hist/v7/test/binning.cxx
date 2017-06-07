#include "gtest/gtest.h"
#include "ROOT/THist.hxx"
#include "ROOT/THistBinIter.hxx"
#include <cmath>
#include <limits>

using namespace ROOT::Experimental;

// Test FindBin() in all its glory.

// Basic binning on a Equidistant axis.
TEST(AxisBinning, EquiDistBasic) {
   TAxisEquidistant ax("TITLE", 10, -1., 1.);
   EXPECT_EQ(1, ax.FindBin(-.999));
   EXPECT_EQ(5, ax.FindBin(-.001));
   EXPECT_EQ(10, ax.FindBin(0.999));
   EXPECT_EQ(0, ax.FindBin(-2.));
   EXPECT_EQ(11, ax.FindBin(2000.));

   EXPECT_GE(6, ax.FindBin(std::numeric_limits<double>::min()));
   EXPECT_LE(5, ax.FindBin(std::numeric_limits<double>::min()));
   EXPECT_GE(6, ax.FindBin(-std::numeric_limits<double>::min()));
   EXPECT_LE(5, ax.FindBin(-std::numeric_limits<double>::min()));

   EXPECT_EQ(11, ax.FindBin(std::numeric_limits<double>::max()));
   EXPECT_EQ(0, ax.FindBin(-std::numeric_limits<double>::max()));
}

// Epsilon bin widths.
TEST(AxisBinning, EquiDistEps) {
   static constexpr auto eps = std::numeric_limits<double>::epsilon();
   TAxisEquidistant ax("TITLE", 10, 0., eps * 10.);
   EXPECT_LE(0, ax.FindBin(0.5*eps));
   EXPECT_GE(1, ax.FindBin(0.5*eps));

   EXPECT_LE(5, ax.FindBin(5.*eps));
   EXPECT_GE(6, ax.FindBin(5.*eps));

   EXPECT_LE(10, ax.FindBin(10.*eps));
   EXPECT_GE(11, ax.FindBin(10.*eps));

   EXPECT_EQ(0, ax.FindBin(-2000.*eps));
   EXPECT_EQ(11, ax.FindBin(2000.*eps));
   EXPECT_EQ(1, ax.FindBin(std::numeric_limits<double>::min()));
   EXPECT_EQ(0, ax.FindBin(-std::numeric_limits<double>::min()));
   EXPECT_EQ(11, ax.FindBin(std::numeric_limits<double>::max()));
   EXPECT_EQ(0, ax.FindBin(-std::numeric_limits<double>::max()));
}
