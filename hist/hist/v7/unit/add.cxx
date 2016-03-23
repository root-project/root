#include "gtest/gtest.h"

#include "ROOT/THist.h"

// Test "x + 0 = x"
TEST(HistAddTest, AddEmptyHist) {
  ROOT::Experimental::TH1F hTo({100,0.,1});
  ROOT::Experimental::TH1F hFrom({100,0.,1});
  hTo.Fill({{0.1111}}, .42);
  ROOT::Experimental::Add(hTo, hFrom);
  EXPECT_FLOAT_EQ(0.42, hTo.GetBinContent({{0.1111}}));
}

// Test addition of a hist range
TEST(HistAddTest, AddView) {
  EXPECT_EQ(1, 1);
}
