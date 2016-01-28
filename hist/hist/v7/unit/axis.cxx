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
  }

  {
    TAxisConfig axis("TITLE", TAxisConfig::Grow, 10, 0., 1.);
    EXPECT_EQ(10 + nOverflow, axis.GetNBins());
  }

  {
    TAxisConfig axis({-0.1, 0.2, 0.5, 10.});
    EXPECT_EQ(3 + nOverflow, axis.GetNBins());
  }

  // Through concrete axis incarnations (and to TAxisConfig)
  {
    TAxisEquidistant ax("TITLE", 10, -1., 1.);
    EXPECT_EQ(10 + nOverflow, ax.GetNBins());
    TAxisConfig axcfg(ax);
    EXPECT_EQ(ax.GetNBins(), axcfg.GetNBins());
  }

  {
    TAxisGrow ax(10, -1., 1.);
    EXPECT_EQ(10 + nOverflow, ax.GetNBins());
    TAxisConfig axcfg(ax);
    EXPECT_EQ(ax.GetNBins(), axcfg.GetNBins());
  }

  {
    TAxisIrregular ax("TITLE", {-0.1, 0.2, 0.5, 10.});
    EXPECT_EQ(3 + nOverflow, ax.GetNBins());
    TAxisConfig axcfg(ax);
    EXPECT_EQ(ax.GetNBins(), axcfg.GetNBins());
  }
}

TEST(AxisTest, ReverseBinLimits) {
  {
    TAxisConfig axiscfg(10, 1., 0.);
    auto axisEq = Internal::AxisConfigToType<TAxisConfig::kEquidistant>()(axiscfg);
    EXPECT_DOUBLE_EQ(0., axisEq.GetMinimum());
    EXPECT_DOUBLE_EQ(1., axisEq.GetMaximum());
  }
}
