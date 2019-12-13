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

   // Here's a visual overview of how binning should work
   //
   //                    Axis 0
   //              UF   0.0  1.0  OF
   //        ------------------------
   //     A   UF  | 0    1    2    3
   //     x  -1.0 | 4    5    6    7
   //     .   0.0 | 8    9    10   11
   //     1   OF  | 12   13   14   15

   // Check that coordinates map into the correct bins

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

   // Check that bins map into the correct coordinates

   const double uf_from = -std::numeric_limits<double>::max();
   const double uf_center_axis0 = (uf_from + 0.0) / 2.0;
   const double uf_center_axis1 = (uf_from - 1.0) / 2.0;
   const double of_to = std::numeric_limits<double>::max();
   const double of_center_axis0 = (2.0 + of_to) / 2.0;
   const double of_center_axis1 = (1.0 + of_to) / 2.0;

   // ... first bin on axis 1 ...

   EXPECT_LE(uf_from,         hist.GetBinFrom(0)[0]);
   EXPECT_LE(uf_center_axis0, hist.GetBinCenter(0)[0]);
   EXPECT_FLOAT_EQ( 0.0,      hist.GetBinTo(0)[0]);
   EXPECT_LE(uf_from,         hist.GetBinFrom(0)[1]);
   EXPECT_LE(uf_center_axis1, hist.GetBinCenter(0)[1]);
   EXPECT_FLOAT_EQ(-1.0,      hist.GetBinTo(0)[1]);

   EXPECT_FLOAT_EQ( 0.0,      hist.GetBinFrom(1)[0]);
   EXPECT_FLOAT_EQ( 0.5,      hist.GetBinCenter(1)[0]);
   EXPECT_FLOAT_EQ( 1.0,      hist.GetBinTo(1)[0]);
   EXPECT_LE(uf_from,         hist.GetBinFrom(1)[1]);
   EXPECT_LE(uf_center_axis1, hist.GetBinCenter(1)[1]);
   EXPECT_FLOAT_EQ(-1.0,      hist.GetBinTo(1)[1]);

   EXPECT_FLOAT_EQ( 1.0,      hist.GetBinFrom(2)[0]);
   EXPECT_FLOAT_EQ( 1.5,      hist.GetBinCenter(2)[0]);
   EXPECT_FLOAT_EQ( 2.0,      hist.GetBinTo(2)[0]);
   EXPECT_LE(uf_from,         hist.GetBinFrom(2)[1]);
   EXPECT_LE(uf_center_axis1, hist.GetBinCenter(2)[1]);
   EXPECT_FLOAT_EQ(-1.0,      hist.GetBinTo(2)[1]);

   EXPECT_FLOAT_EQ( 2.0,      hist.GetBinFrom(3)[0]);
   EXPECT_GE(of_center_axis0, hist.GetBinCenter(3)[0]);
   EXPECT_GE(of_to,           hist.GetBinTo(3)[0]);
   EXPECT_LE(uf_from,         hist.GetBinFrom(3)[1]);
   EXPECT_LE(uf_center_axis1, hist.GetBinCenter(3)[1]);
   EXPECT_FLOAT_EQ(-1.0,      hist.GetBinTo(3)[1]);

   // ... next bin on axis 1 ...

   EXPECT_LE(uf_from,         hist.GetBinFrom(4)[0]);
   EXPECT_LE(uf_center_axis0, hist.GetBinCenter(4)[0]);
   EXPECT_FLOAT_EQ( 0.0,      hist.GetBinTo(4)[0]);
   EXPECT_FLOAT_EQ(-1.0,      hist.GetBinFrom(4)[1]);
   EXPECT_FLOAT_EQ(-0.5,      hist.GetBinCenter(4)[1]);
   EXPECT_FLOAT_EQ( 0.0,      hist.GetBinTo(4)[1]);

   EXPECT_FLOAT_EQ( 0.0,      hist.GetBinFrom(5)[0]);
   EXPECT_FLOAT_EQ( 0.5,      hist.GetBinCenter(5)[0]);
   EXPECT_FLOAT_EQ( 1.0,      hist.GetBinTo(5)[0]);
   EXPECT_FLOAT_EQ(-1.0,      hist.GetBinFrom(5)[1]);
   EXPECT_FLOAT_EQ(-0.5,      hist.GetBinCenter(5)[1]);
   EXPECT_FLOAT_EQ( 0.0,      hist.GetBinTo(5)[1]);

   EXPECT_FLOAT_EQ( 1.0,      hist.GetBinFrom(6)[0]);
   EXPECT_FLOAT_EQ( 1.5,      hist.GetBinCenter(6)[0]);
   EXPECT_FLOAT_EQ( 2.0,      hist.GetBinTo(6)[0]);
   EXPECT_FLOAT_EQ(-1.0,      hist.GetBinFrom(6)[1]);
   EXPECT_FLOAT_EQ(-0.5,      hist.GetBinCenter(6)[1]);
   EXPECT_FLOAT_EQ( 0.0,      hist.GetBinTo(6)[1]);

   EXPECT_FLOAT_EQ( 2.0,      hist.GetBinFrom(7)[0]);
   EXPECT_GE(of_center_axis0, hist.GetBinCenter(7)[0]);
   EXPECT_GE(of_to,           hist.GetBinTo(7)[0]);
   EXPECT_FLOAT_EQ(-1.0,      hist.GetBinFrom(7)[1]);
   EXPECT_FLOAT_EQ(-0.5,      hist.GetBinCenter(7)[1]);
   EXPECT_FLOAT_EQ( 0.0,      hist.GetBinTo(7)[1]);

   // ... next bin on axis 1 ...

   EXPECT_LE(uf_from,         hist.GetBinFrom(8)[0]);
   EXPECT_LE(uf_center_axis0, hist.GetBinCenter(8)[0]);
   EXPECT_FLOAT_EQ( 0.0,      hist.GetBinTo(8)[0]);
   EXPECT_FLOAT_EQ( 0.0,      hist.GetBinFrom(8)[1]);
   EXPECT_FLOAT_EQ( 0.5,      hist.GetBinCenter(8)[1]);
   EXPECT_FLOAT_EQ( 1.0,      hist.GetBinTo(8)[1]);

   EXPECT_FLOAT_EQ( 0.0,      hist.GetBinFrom(9)[0]);
   EXPECT_FLOAT_EQ( 0.5,      hist.GetBinCenter(9)[0]);
   EXPECT_FLOAT_EQ( 1.0,      hist.GetBinTo(9)[0]);
   EXPECT_FLOAT_EQ( 0.0,      hist.GetBinFrom(9)[1]);
   EXPECT_FLOAT_EQ( 0.5,      hist.GetBinCenter(9)[1]);
   EXPECT_FLOAT_EQ( 1.0,      hist.GetBinTo(9)[1]);

   EXPECT_FLOAT_EQ( 1.0,      hist.GetBinFrom(10)[0]);
   EXPECT_FLOAT_EQ( 1.5,      hist.GetBinCenter(10)[0]);
   EXPECT_FLOAT_EQ( 2.0,      hist.GetBinTo(10)[0]);
   EXPECT_FLOAT_EQ( 0.0,      hist.GetBinFrom(10)[1]);
   EXPECT_FLOAT_EQ( 0.5,      hist.GetBinCenter(10)[1]);
   EXPECT_FLOAT_EQ( 1.0,      hist.GetBinTo(10)[1]);

   EXPECT_FLOAT_EQ( 2.0,      hist.GetBinFrom(11)[0]);
   EXPECT_GE(of_center_axis0, hist.GetBinCenter(11)[0]);
   EXPECT_GE(of_to,           hist.GetBinTo(11)[0]);
   EXPECT_FLOAT_EQ( 0.0,      hist.GetBinFrom(11)[1]);
   EXPECT_FLOAT_EQ( 0.5,      hist.GetBinCenter(11)[1]);
   EXPECT_FLOAT_EQ( 1.0,      hist.GetBinTo(11)[1]);

   // ... last bin on axis 1 ...

   EXPECT_LE(uf_from,         hist.GetBinFrom(12)[0]);
   EXPECT_LE(uf_center_axis0, hist.GetBinCenter(12)[0]);
   EXPECT_FLOAT_EQ( 0.0,      hist.GetBinTo(12)[0]);
   EXPECT_FLOAT_EQ( 1.0,      hist.GetBinFrom(12)[1]);
   EXPECT_GE(of_center_axis1, hist.GetBinCenter(12)[1]);
   EXPECT_GE(of_to,           hist.GetBinTo(12)[1]);

   EXPECT_FLOAT_EQ( 0.0,      hist.GetBinFrom(13)[0]);
   EXPECT_FLOAT_EQ( 0.5,      hist.GetBinCenter(13)[0]);
   EXPECT_FLOAT_EQ( 1.0,      hist.GetBinTo(13)[0]);
   EXPECT_FLOAT_EQ( 1.0,      hist.GetBinFrom(13)[1]);
   EXPECT_GE(of_center_axis1, hist.GetBinCenter(13)[1]);
   EXPECT_GE(of_to,           hist.GetBinTo(13)[1]);

   EXPECT_FLOAT_EQ( 1.0,      hist.GetBinFrom(14)[0]);
   EXPECT_FLOAT_EQ( 1.5,      hist.GetBinCenter(14)[0]);
   EXPECT_FLOAT_EQ( 2.0,      hist.GetBinTo(14)[0]);
   EXPECT_FLOAT_EQ( 1.0,      hist.GetBinFrom(14)[1]);
   EXPECT_GE(of_center_axis1, hist.GetBinCenter(14)[1]);
   EXPECT_GE(of_to,           hist.GetBinTo(14)[1]);

   EXPECT_FLOAT_EQ( 2.0,      hist.GetBinFrom(15)[0]);
   EXPECT_GE(of_center_axis0, hist.GetBinCenter(15)[0]);
   EXPECT_GE(of_to,           hist.GetBinTo(15)[0]);
   EXPECT_FLOAT_EQ( 1.0,      hist.GetBinFrom(15)[1]);
   EXPECT_GE(of_center_axis1, hist.GetBinCenter(15)[1]);
   EXPECT_GE(of_to,           hist.GetBinTo(15)[1]);
}
