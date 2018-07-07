#include "gtest/gtest.h"
#include "ROOT/RAxis.hxx"

using namespace ROOT::Experimental;

// Tests the number of bins
TEST(AxisTest, NumBins) {
  constexpr int nOverflow = 2;
  // Through RAxisConfig
  {
    RAxisConfig axis(10, 0., 1.);
    EXPECT_EQ(10 + nOverflow, axis.GetNBins());

    EXPECT_EQ(10, axis.GetNBinsNoOver());
    EXPECT_EQ(0, axis.GetUnderflowBin());
    EXPECT_EQ(11, axis.GetOverflowBin());
    EXPECT_EQ(true, axis.IsUnderflowBin(0));
    EXPECT_EQ(true, axis.IsUnderflowBin(-1));
    EXPECT_EQ(true, axis.IsOverflowBin(11));
    EXPECT_EQ(true, axis.IsOverflowBin(12));

    // RODO: test iterator interface
  }

  // Through RAxisConfig, with title
  {
    RAxisConfig axis("RITLE", 10, 0., 1.);
    EXPECT_EQ(10 + nOverflow, axis.GetNBins());

    EXPECT_EQ(10, axis.GetNBinsNoOver());
    EXPECT_EQ(0, axis.GetUnderflowBin());
    EXPECT_EQ(11, axis.GetOverflowBin());
    EXPECT_EQ(true, axis.IsUnderflowBin(0));
    EXPECT_EQ(true, axis.IsUnderflowBin(-1));
    EXPECT_EQ(true, axis.IsOverflowBin(11));
    EXPECT_EQ(true, axis.IsOverflowBin(12));
  }

  {
    RAxisConfig axis("RITLE", RAxisConfig::Grow, 10, 0., 1.);
    EXPECT_EQ(10 /*NOT + nOverflow*/, axis.GetNBins());
    EXPECT_EQ(10, axis.GetNBinsNoOver());
    EXPECT_EQ(0, axis.GetUnderflowBin());
    EXPECT_EQ(11, axis.GetOverflowBin());
    EXPECT_EQ(true, axis.IsUnderflowBin(0));
    EXPECT_EQ(true, axis.IsUnderflowBin(-1));
    EXPECT_EQ(true, axis.IsOverflowBin(11));
    EXPECT_EQ(true, axis.IsOverflowBin(12));
  }

  {
    RAxisConfig axis({-0.1, 0.2, 0.5, 10.});
    EXPECT_EQ(3 + nOverflow, axis.GetNBins());
    EXPECT_EQ(3, axis.GetNBinsNoOver());
    EXPECT_EQ(0, axis.GetUnderflowBin());
    EXPECT_EQ(4, axis.GetOverflowBin());
    EXPECT_EQ(true, axis.IsUnderflowBin(0));
    EXPECT_EQ(true, axis.IsUnderflowBin(-1));
    EXPECT_EQ(true, axis.IsOverflowBin(4));
    EXPECT_EQ(true, axis.IsOverflowBin(6));
  }

  // Through concrete axis incarnations (and to RAxisConfig)
  {
    RAxisEquidistant ax("RITLE", 10, -1., 1.);

    EXPECT_EQ(10 + nOverflow, ax.GetNBins());
    EXPECT_EQ(10, ax.GetNBinsNoOver());
    EXPECT_EQ(0, ax.GetUnderflowBin());
    EXPECT_EQ(11, ax.GetOverflowBin());
    EXPECT_EQ(true, ax.IsUnderflowBin(0));
    EXPECT_EQ(true, ax.IsUnderflowBin(-10000));
    EXPECT_EQ(true, ax.IsOverflowBin(11));
    EXPECT_EQ(true, ax.IsOverflowBin(16));

    EXPECT_FLOAT_EQ(0.2, ax.GetBinWidth());
    EXPECT_EQ(7, ax.FindBin(0.22));
    EXPECT_EQ(0, ax.FindBin(-2.));
    EXPECT_EQ(10, ax.FindBin(0.99));
    EXPECT_EQ(11, ax.FindBin(1.01));
    EXPECT_EQ(11, ax.FindBin(101.));

    EXPECT_FLOAT_EQ(0.7, ax.GetBinCenter(9));
    EXPECT_FLOAT_EQ(0.6, ax.GetBinFrom(9));
    EXPECT_FLOAT_EQ(0.8, ax.GetBinTo(9));

    EXPECT_LT(ax.GetBinCenter(0), -1.);
    EXPECT_LT(ax.GetBinFrom(0), -1.);
    EXPECT_FLOAT_EQ(-1., ax.GetBinTo(0));

    EXPECT_LT(ax.GetBinCenter(-1), -1.);
    EXPECT_LT(ax.GetBinFrom(-2), -1.);
    EXPECT_LE(ax.GetBinTo(-3), -1.);

    EXPECT_LT(1., ax.GetBinCenter(11));
    EXPECT_FLOAT_EQ(1., ax.GetBinFrom(11));
    EXPECT_LT(1., ax.GetBinTo(11));

    EXPECT_LT(1., ax.GetBinCenter(111));
    EXPECT_LE(1., ax.GetBinFrom(111));
    EXPECT_LT(1., ax.GetBinTo(111));
    

    RAxisConfig axcfg(ax);
    EXPECT_EQ(ax.GetNBins(), axcfg.GetNBins());
    EXPECT_EQ(10, axcfg.GetNBinsNoOver());
    EXPECT_EQ(0, axcfg.GetUnderflowBin());
    EXPECT_EQ(11, axcfg.GetOverflowBin());
    EXPECT_EQ(true, axcfg.IsUnderflowBin(0));
    EXPECT_EQ(true, axcfg.IsUnderflowBin(-10000));
    EXPECT_EQ(true, axcfg.IsOverflowBin(11));
    EXPECT_EQ(true, axcfg.IsOverflowBin(16));
  }

  {
    RAxisGrow ax(10, -1., 1.);
    EXPECT_EQ(10, ax.GetNBins());
    EXPECT_EQ(10, ax.GetNBinsNoOver());
    EXPECT_EQ(0, ax.GetUnderflowBin());
    EXPECT_EQ(11, ax.GetOverflowBin());
    EXPECT_EQ(true, ax.IsUnderflowBin(0));
    EXPECT_EQ(true, ax.IsUnderflowBin(-1));
    EXPECT_EQ(true, ax.IsOverflowBin(11));
    EXPECT_EQ(true, ax.IsOverflowBin(12));

    RAxisConfig axcfg(ax);
    EXPECT_EQ(ax.GetNBins(), axcfg.GetNBins());
    EXPECT_EQ(10, axcfg.GetNBinsNoOver());
    EXPECT_EQ(0, axcfg.GetUnderflowBin());
    EXPECT_EQ(11, axcfg.GetOverflowBin());
    EXPECT_EQ(true, axcfg.IsUnderflowBin(0));
    EXPECT_EQ(true, axcfg.IsUnderflowBin(-1));
    EXPECT_EQ(true, axcfg.IsOverflowBin(11));
    EXPECT_EQ(true, axcfg.IsOverflowBin(12));
  }

  {
    RAxisIrregular ax("RITLE", {-0.1, 0.2, 0.5, 10.});
    EXPECT_EQ(3 + nOverflow, ax.GetNBins());
    EXPECT_EQ(3, ax.GetNBinsNoOver());
    EXPECT_EQ(0, ax.GetUnderflowBin());
    EXPECT_EQ(4, ax.GetOverflowBin());
    EXPECT_EQ(true, ax.IsUnderflowBin(0));
    EXPECT_EQ(true, ax.IsUnderflowBin(-1));
    EXPECT_EQ(true, ax.IsOverflowBin(4));
    EXPECT_EQ(true, ax.IsOverflowBin(5000));

    RAxisConfig axcfg(ax);
    EXPECT_EQ(ax.GetNBins(), axcfg.GetNBins());
    EXPECT_EQ(3, axcfg.GetNBinsNoOver());
    EXPECT_EQ(0, axcfg.GetUnderflowBin());
    EXPECT_EQ(4, axcfg.GetOverflowBin());
    EXPECT_EQ(true, axcfg.IsUnderflowBin(0));
    EXPECT_EQ(true, axcfg.IsUnderflowBin(-1));
    EXPECT_EQ(true, axcfg.IsOverflowBin(5));
    EXPECT_EQ(true, axcfg.IsOverflowBin(4));
  }
}

TEST(AxisTest, ReverseBinLimits) {
  {
    RAxisConfig axiscfg(10, 1., 0.);
    EXPECT_DOUBLE_EQ(0., axiscfg.GetBinBorders()[0]);
    EXPECT_DOUBLE_EQ(1., axiscfg.GetBinBorders()[1]);
    EXPECT_EQ(10, axiscfg.GetNBinsNoOver());
    EXPECT_EQ(0, axiscfg.GetUnderflowBin());
    EXPECT_EQ(11, axiscfg.GetOverflowBin());
    EXPECT_EQ(true, axiscfg.IsUnderflowBin(0));
    EXPECT_EQ(true, axiscfg.IsUnderflowBin(-1));
    EXPECT_EQ(true, axiscfg.IsOverflowBin(12));
    EXPECT_EQ(true, axiscfg.IsOverflowBin(11));


    auto axis = Internal::AxisConfigToType<RAxisConfig::kEquidistant>()(axiscfg);
    EXPECT_DOUBLE_EQ(0., axis.GetMinimum());
    EXPECT_DOUBLE_EQ(1., axis.GetMaximum());
    EXPECT_EQ(10, axis.GetNBinsNoOver());
    EXPECT_EQ(0, axis.GetUnderflowBin());
    EXPECT_EQ(11, axis.GetOverflowBin());
    EXPECT_EQ(true, axis.IsUnderflowBin(0));
    EXPECT_EQ(true, axis.IsUnderflowBin(-1));
    EXPECT_EQ(true, axis.IsOverflowBin(12));
    EXPECT_EQ(true, axis.IsOverflowBin(11));
  }
}
