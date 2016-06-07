#include "gtest/gtest.h"
#include <ROOT/TAxis.h>

using namespace ROOT::Experimental;

// Tests the number of bins
TEST(AxisTest, NumBins) {
  constexpr int nOverflow = 2;
  // Through TAxisConfig
  {
    TAxisConfig axis(10, 0., 1.);
    EXPECT_EQ(10 + nOverflow, axis.GetNBins());

    EXPECT_EQ(10, axis.GetNBinsNoOver());
    EXPECT_EQ(0, axis.GetUnderflowBin());
    EXPECT_EQ(11, axis.GetOverflowBin());
    EXPECT_EQ(true, axis.IsUnderflowBin(0));
    EXPECT_EQ(true, axis.IsUnderflowBin(-1));
    EXPECT_EQ(true, axis.IsOverflowBin(11));
    EXPECT_EQ(true, axis.IsOverflowBin(12));

    // TODO: test iterator interface
  }

  // Through TAxisConfig, with title
  {
    TAxisConfig axis("TITLE", 10, 0., 1.);
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
    TAxisConfig axis("TITLE", TAxisConfig::Grow, 10, 0., 1.);
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
    TAxisConfig axis({-0.1, 0.2, 0.5, 10.});
    EXPECT_EQ(3 + nOverflow, axis.GetNBins());
    EXPECT_EQ(3, axis.GetNBinsNoOver());
    EXPECT_EQ(0, axis.GetUnderflowBin());
    EXPECT_EQ(4, axis.GetOverflowBin());
    EXPECT_EQ(true, axis.IsUnderflowBin(0));
    EXPECT_EQ(true, axis.IsUnderflowBin(-1));
    EXPECT_EQ(true, axis.IsOverflowBin(4));
    EXPECT_EQ(true, axis.IsOverflowBin(6));
  }

  // Through concrete axis incarnations (and to TAxisConfig)
  {
    TAxisEquidistant ax("TITLE", 10, -1., 1.);
    EXPECT_EQ(10 + nOverflow, ax.GetNBins());
    EXPECT_EQ(10, ax.GetNBinsNoOver());
    EXPECT_EQ(0, ax.GetUnderflowBin());
    EXPECT_EQ(11, ax.GetOverflowBin());
    EXPECT_EQ(true, ax.IsUnderflowBin(0));
    EXPECT_EQ(true, ax.IsUnderflowBin(-10000));
    EXPECT_EQ(true, ax.IsOverflowBin(11));
    EXPECT_EQ(true, ax.IsOverflowBin(16));

    TAxisConfig axcfg(ax);
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
    TAxisGrow ax(10, -1., 1.);
    EXPECT_EQ(10, ax.GetNBins());
    EXPECT_EQ(10, ax.GetNBinsNoOver());
    EXPECT_EQ(0, ax.GetUnderflowBin());
    EXPECT_EQ(11, ax.GetOverflowBin());
    EXPECT_EQ(true, ax.IsUnderflowBin(0));
    EXPECT_EQ(true, ax.IsUnderflowBin(-1));
    EXPECT_EQ(true, ax.IsOverflowBin(11));
    EXPECT_EQ(true, ax.IsOverflowBin(12));

    TAxisConfig axcfg(ax);
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
    TAxisIrregular ax("TITLE", {-0.1, 0.2, 0.5, 10.});
    EXPECT_EQ(3 + nOverflow, ax.GetNBins());
    EXPECT_EQ(3, ax.GetNBinsNoOver());
    EXPECT_EQ(0, ax.GetUnderflowBin());
    EXPECT_EQ(4, ax.GetOverflowBin());
    EXPECT_EQ(true, ax.IsUnderflowBin(0));
    EXPECT_EQ(true, ax.IsUnderflowBin(-1));
    EXPECT_EQ(true, ax.IsOverflowBin(4));
    EXPECT_EQ(true, ax.IsOverflowBin(5000));

    TAxisConfig axcfg(ax);
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
    TAxisConfig axiscfg(10, 1., 0.);
    EXPECT_DOUBLE_EQ(0., axiscfg.GetBinBorders()[0]);
    EXPECT_DOUBLE_EQ(1., axiscfg.GetBinBorders()[1]);
    EXPECT_EQ(10, axiscfg.GetNBinsNoOver());
    EXPECT_EQ(0, axiscfg.GetUnderflowBin());
    EXPECT_EQ(11, axiscfg.GetOverflowBin());
    EXPECT_EQ(true, axiscfg.IsUnderflowBin(0));
    EXPECT_EQ(true, axiscfg.IsUnderflowBin(-1));
    EXPECT_EQ(true, axiscfg.IsOverflowBin(12));
    EXPECT_EQ(true, axiscfg.IsOverflowBin(11));


    auto axis = Internal::AxisConfigToType<TAxisConfig::kEquidistant>()(axiscfg);
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
