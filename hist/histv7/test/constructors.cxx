#include "gtest/gtest.h"

#include "ROOT/RHist.hxx"

// Test RH.. constructors and especially GetNBins() and GetBinContent() from RHistImpl.hxx

// Test RH1F constructor
TEST(HistConstructorTest, ConstructorRH1F) {
  ROOT::Experimental::RH1F hist({100, 0., 1});
  EXPECT_EQ(102, hist.GetImpl()->GetNBins());
  EXPECT_EQ(0, hist.GetImpl()->GetBinContent(42));
  hist.GetImpl()->AddBinContent(42,.42f);
  EXPECT_FLOAT_EQ(.42f, hist.GetImpl()->GetBinContent(42));
}

// Test RH1D constructor
TEST(HistConstructorTest, ConstructorRH1D) {
  ROOT::Experimental::RH1D hist({100, 0, 1});
  EXPECT_EQ(102, hist.GetImpl()->GetNBins());
  EXPECT_EQ(0, hist.GetImpl()->GetBinContent(42));
  hist.GetImpl()->AddBinContent(42,.42f);
  EXPECT_FLOAT_EQ(.42f, hist.GetImpl()->GetBinContent(42));
}

// Test RH1I constructor
TEST(HistConstructorTest, ConstructorRH1I) {
  ROOT::Experimental::RH1I hist({100, 0, 1});
  EXPECT_EQ(102, hist.GetImpl()->GetNBins());
  EXPECT_EQ(0, hist.GetImpl()->GetBinContent(42));
  hist.GetImpl()->AddBinContent(42,3);
  EXPECT_EQ(3, hist.GetImpl()->GetBinContent(42));
}



// Test RH2F constructor
TEST(HistConstructorTest, ConstructorRH2F) {
  ROOT::Experimental::RH2F hist({100, 0., 1},{{0., 1., 2., 3., 10.}});
  EXPECT_EQ(102*6, hist.GetImpl()->GetNBins());
  EXPECT_EQ(0, hist.GetImpl()->GetBinContent(42));
  hist.GetImpl()->AddBinContent(42,.42f);
  EXPECT_FLOAT_EQ(.42f, hist.GetImpl()->GetBinContent(42));
}

// Test RH2D constructor
TEST(HistConstructorTest, ConstructorRH2D) {
  ROOT::Experimental::RH2D hist({100, 0, 1},{{0, 1, 2, 3, 10}});
  EXPECT_EQ(102*6, hist.GetImpl()->GetNBins());
  EXPECT_EQ(0, hist.GetImpl()->GetBinContent(42));
  hist.GetImpl()->AddBinContent(42,.42f);
  EXPECT_FLOAT_EQ(.42f, hist.GetImpl()->GetBinContent(42));
}

// Test RH2I constructor
TEST(HistConstructorTest, ConstructorRH2I) {
  ROOT::Experimental::RH2I hist({100, 0, 1},{{0, 1, 2, 3, 10}});
  EXPECT_EQ(102*6, hist.GetImpl()->GetNBins());
  EXPECT_EQ(0, hist.GetImpl()->GetBinContent(42));
  hist.GetImpl()->AddBinContent(42,3);
  EXPECT_EQ(3, hist.GetImpl()->GetBinContent(42));
}



// Test RH3F constructor
TEST(HistConstructorTest, ConstructorRH3F) {
  ROOT::Experimental::RH3F hist({100, 0., 1},{{0., 1., 2., 3., 10.}},{{0., 1., 2., 3., 4., 10.}});
  EXPECT_EQ(102*6*7, hist.GetImpl()->GetNBins());
  EXPECT_EQ(0, hist.GetImpl()->GetBinContent(42));
  hist.GetImpl()->AddBinContent(42,.42f);
  EXPECT_FLOAT_EQ(.42f, hist.GetImpl()->GetBinContent(42));
}

// Test RH3D constructor
TEST(HistConstructorTest, ConstructorRH3D) {
  ROOT::Experimental::RH3D hist({100, 0, 1},{{0, 1, 2, 3, 10}},{{0, 1, 2, 3, 4, 10}});
  EXPECT_EQ(102*6*7, hist.GetImpl()->GetNBins());
  EXPECT_EQ(0, hist.GetImpl()->GetBinContent(42));
  hist.GetImpl()->AddBinContent(42,.42f);
  EXPECT_FLOAT_EQ(.42f, hist.GetImpl()->GetBinContent(42));
}

// Test RH3I constructor
TEST(HistConstructorTest, ConstructorRH3I) {
  ROOT::Experimental::RH3I hist({100, 0, 1},{{0, 1, 2, 3, 10}},{{0, 1, 2, 3, 4, 10}});
  EXPECT_EQ(102*6*7, hist.GetImpl()->GetNBins());
  EXPECT_EQ(0, hist.GetImpl()->GetBinContent(42));
  hist.GetImpl()->AddBinContent(42,3);
  EXPECT_EQ(3, hist.GetImpl()->GetBinContent(42));
}