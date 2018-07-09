#include "gtest/gtest.h"
#include "ROOT/RHist.hxx"
#include "ROOT/RHistBinIter.hxx"
#include <cmath>
#include <limits>

using namespace ROOT::Experimental;

// Test FindBin() in all its glory.

// Basic binning on a Equidistant axis.
TEST(AxisBinning, EquidistBasic) {
   RAxisEquidistant ax("RITLE", 10, -1., 1.);
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
TEST(AxisBinning, EquidistEpsBins) {
   static constexpr auto eps = std::numeric_limits<double>::min();
   RAxisEquidistant ax("RITLE", 10, 0., eps * 10.);
   EXPECT_LE(0, ax.FindBin(0.5*eps));
   EXPECT_GE(1, ax.FindBin(0.5*eps));

   EXPECT_LE(5, ax.FindBin(5.*eps));
   EXPECT_GE(6, ax.FindBin(5.*eps));

   EXPECT_LE(10, ax.FindBin(10.*eps));
   EXPECT_GE(11, ax.FindBin(10.*eps));

   EXPECT_EQ(0, ax.FindBin(-2000.*eps));
   EXPECT_EQ(11, ax.FindBin(2000.*eps));
   EXPECT_LE(1, ax.FindBin(std::numeric_limits<double>::min()));
   EXPECT_GE(2, ax.FindBin(std::numeric_limits<double>::min()));
   EXPECT_LE(0, ax.FindBin(-std::numeric_limits<double>::min()));
   EXPECT_GE(1, ax.FindBin(-std::numeric_limits<double>::min()));
   EXPECT_EQ(11, ax.FindBin(std::numeric_limits<double>::max()));
   EXPECT_EQ(0, ax.FindBin(-std::numeric_limits<double>::max()));
}


// Basic binning on an Irregular axis.
TEST(AxisBinning, IrregularBasic) {
   RAxisIrregular ax("RITLE", {-5., 0., 0.1, 1., 10., 100.});
   EXPECT_EQ(2, ax.FindBin(.001));
   EXPECT_EQ(1, ax.FindBin(-.001));
   EXPECT_EQ(5, ax.FindBin(99.));
   EXPECT_EQ(0, ax.FindBin(-6.));
   EXPECT_EQ(6, ax.FindBin(2000.));

   EXPECT_GE(2, ax.FindBin(std::numeric_limits<double>::min()));
   EXPECT_LE(1, ax.FindBin(std::numeric_limits<double>::min()));
   EXPECT_GE(2, ax.FindBin(-std::numeric_limits<double>::min()));
   EXPECT_LE(1, ax.FindBin(-std::numeric_limits<double>::min()));

   EXPECT_EQ(6, ax.FindBin(std::numeric_limits<double>::max()));
   EXPECT_EQ(0, ax.FindBin(-std::numeric_limits<double>::max()));
}


// Limit bin widths on an Irregular axis.
TEST(AxisBinning, IrregularEpsBins) {
   static constexpr auto eps = std::numeric_limits<double>::min();
   RAxisIrregular ax("RITLE", {0., eps, 2.*eps, 3.*eps, 4.*eps, 5.*eps});
   EXPECT_LE(0, ax.FindBin(0.5*eps));
   EXPECT_GE(1, ax.FindBin(0.5*eps));

   EXPECT_LE(3, ax.FindBin(3.*eps));
   EXPECT_GE(4, ax.FindBin(3.*eps));

   EXPECT_LE(5, ax.FindBin(5.*eps));
   EXPECT_GE(6, ax.FindBin(5.*eps));

   EXPECT_EQ(0, ax.FindBin(-2000.*eps));
   EXPECT_EQ(6, ax.FindBin(2000.*eps));
   EXPECT_EQ(1, ax.FindBin(std::numeric_limits<double>::min()));
   EXPECT_EQ(0, ax.FindBin(-std::numeric_limits<double>::min()));
   EXPECT_EQ(6, ax.FindBin(std::numeric_limits<double>::max()));
   EXPECT_EQ(0, ax.FindBin(-std::numeric_limits<double>::max()));
}

// Histogram binning on a Equidistant axis.
TEST(HistImplBinning, Equidist1D) {
   Detail::RHistImpl<Detail::RHistData<1, double, std::vector<double>, RHistStatContent>,
                     RAxisEquidistant> hist(RAxisEquidistant(10, 0., 1.));

   EXPECT_EQ(5, hist.GetBinIndex({.45}));
   EXPECT_EQ(10, hist.GetBinIndex({.999}));
   EXPECT_EQ(1, hist.GetBinIndex({.001}));
   EXPECT_EQ(0, hist.GetBinIndex({-.001}));
   EXPECT_EQ(11, hist.GetBinIndex({1.001}));

   EXPECT_GE(1, hist.GetBinIndex({std::numeric_limits<double>::min()}));
   EXPECT_LE(0, hist.GetBinIndex({std::numeric_limits<double>::min()}));
   EXPECT_GE(1, hist.GetBinIndex({-std::numeric_limits<double>::min()}));
   EXPECT_LE(0, hist.GetBinIndex({-std::numeric_limits<double>::min()}));

   EXPECT_EQ(11, hist.GetBinIndex({std::numeric_limits<double>::max()}));
   EXPECT_EQ(0, hist.GetBinIndex({-std::numeric_limits<double>::max()}));
}

TEST(HistImplBinning, EquiDist2D) {
   Detail::RHistImpl<Detail::RHistData<2, double, std::vector<double>, RHistStatContent>,
                     RAxisEquidistant, RAxisEquidistant>
      hist(RAxisEquidistant(2, 0., 2.), RAxisEquidistant(2, -1., 1.));

   EXPECT_EQ( 0, hist.GetBinIndex({-100., -100.}));
   EXPECT_EQ( 1, hist.GetBinIndex({0.5, -100.}));
   EXPECT_EQ( 2, hist.GetBinIndex({1.5, -100.}));
   EXPECT_EQ( 3, hist.GetBinIndex({100., -100.}));
   EXPECT_EQ( 4, hist.GetBinIndex({-100., -0.5}));
   EXPECT_EQ( 5, hist.GetBinIndex({0.5, -0.5}));
   EXPECT_EQ( 6, hist.GetBinIndex({1.5, -0.5}));
   EXPECT_EQ( 7, hist.GetBinIndex({100., -0.5}));
   EXPECT_EQ( 8, hist.GetBinIndex({-100., 0.5}));
   EXPECT_EQ( 9, hist.GetBinIndex({0.5, 0.5}));
   EXPECT_EQ(10, hist.GetBinIndex({1.5, 0.5}));
   EXPECT_EQ(11, hist.GetBinIndex({100., 0.5}));
   EXPECT_EQ(12, hist.GetBinIndex({-100., 100.}));
   EXPECT_EQ(13, hist.GetBinIndex({0.5, 100.}));
   EXPECT_EQ(14, hist.GetBinIndex({1.5, 100.}));
   EXPECT_EQ(15, hist.GetBinIndex({100., 100.}));

   EXPECT_EQ( 0, hist.GetBinIndex({-std::numeric_limits<double>::max(), -std::numeric_limits<double>::max()}));
   EXPECT_EQ( 3, hist.GetBinIndex({ std::numeric_limits<double>::max(), -std::numeric_limits<double>::max()}));
   EXPECT_EQ(12, hist.GetBinIndex({-std::numeric_limits<double>::max(),  std::numeric_limits<double>::max()}));
   EXPECT_EQ(15, hist.GetBinIndex({ std::numeric_limits<double>::max(),  std::numeric_limits<double>::max()}));
}
