#include "gtest/gtest.h"

#include "ROOT/RHist.hxx"

// Test "x + 0 = x"
TEST(HistAddTest, AddEmptyHist) {
  ROOT::Experimental::RH1F hFrom({100,0.,1});
  ROOT::Experimental::RH1F hTo({100,0.,1});
  hFrom.Fill({0.1111}, .42f);
  ROOT::Experimental::Add(hTo, hFrom);
  EXPECT_FLOAT_EQ(0.42f, hTo.GetBinContent({0.1111}));
}

// Test "0 + x = x"
TEST(HistAddTest, AddEmptySelf) {
   ROOT::Experimental::RH1F hFrom({100,0.,1});
   ROOT::Experimental::RH1F hTo({100,0.,1});
   hTo.Fill({0.1111}, .42f);
   ROOT::Experimental::Add(hTo, hFrom);
   EXPECT_FLOAT_EQ(0.42f, hTo.GetBinContent({0.1111}));
}

// Test "x + x = 2*x"
TEST(HistAddTest, AddSelfHist) {
   ROOT::Experimental::RH1F hist({100,0.,1});
   hist.Fill({0.1111}, .42f);
   ROOT::Experimental::Add(hist, hist);
   EXPECT_FLOAT_EQ(0.84f, hist.GetBinContent({0.1111}));
}

// Test "x + y" with less STAT
TEST(HistAddTest, AddHist) {
   ROOT::Experimental::RH1F hFrom({100,0.,1});
   // Implicitly tests ROOT-8485.
   ROOT::Experimental::RHist<1, float> hTo({100,0.,1});
   hFrom.Fill({0.1111}, .42f);
   hTo.Fill({0.1111}, .17f);
   ROOT::Experimental::Add(hTo, hFrom);
   EXPECT_FLOAT_EQ(0.59f, hTo.GetBinContent({0.1111}));
}

// Test addition of a hist range
TEST(HistAddTest, AddView) {
  EXPECT_EQ(1, 1);
}
