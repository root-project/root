#include "gtest/gtest.h"

#include "ROOT/RHist.hxx"

// Test Fill(), FillN(), GetEntries(), GetBinContent(), GetBinUncertainty()


// TODO : add tests to check 0 before hand + test uncertainty after another fill


// Test Fill() with weight and GetEntries()
TEST(HistFillTest, FillCoordWeightEntries) {
  ROOT::Experimental::RH1F hist({100,0.,1});
  EXPECT_EQ(0, hist.GetEntries());
  hist.Fill({0.1111}, .42f);
  EXPECT_EQ(1, hist.GetEntries());
  hist.Fill({0.1111}, .32f);
  EXPECT_EQ(2, hist.GetEntries());
}

// Test Fill() with weight and GetBinContent()
TEST(HistFillTest, FillCoordWeightContent) {
  ROOT::Experimental::RH1F hist({100,0.,1});
  EXPECT_FLOAT_EQ(.0f, hist.GetBinContent({0.1111}));
  hist.Fill({0.1111}, .42f);
  EXPECT_FLOAT_EQ(.42f, hist.GetBinContent({0.1111}));
  hist.Fill({0.1111}, .32f);
  EXPECT_FLOAT_EQ(.42f+.32f, hist.GetBinContent({0.1111}));
}

// Test Fill() with weight and GetBinUncertainty()
TEST(HistFillTest, FillCoordWeightUncertainty) {
  ROOT::Experimental::RH1F hist({100,0.,1});
  float weight1 = .0f;
  EXPECT_FLOAT_EQ(std::sqrt(weight1*weight1), hist.GetBinUncertainty({0.1111}));
  weight1 = .42f;
  hist.Fill({0.1111}, weight1);
  EXPECT_FLOAT_EQ(std::sqrt(weight1*weight1), hist.GetBinUncertainty({0.1111}));
  float weight2 = .32f;
  hist.Fill({0.1111}, weight2);
  EXPECT_FLOAT_EQ(std::sqrt((weight1*weight1)+(weight2*weight2)), hist.GetBinUncertainty({0.1111}));
}

// Test FillN() without weights and GetEntries()
TEST(HistFillTest, FillNCoordsEntries) {
  ROOT::Experimental::RH1F hist({100,0.,1});
  EXPECT_EQ(0, hist.GetEntries());
  hist.FillN({{0.1111},{0.2222},{0.3333}});
  EXPECT_EQ(3, hist.GetEntries());
  hist.FillN({{0.1111},{0.3333}});
  EXPECT_EQ(5, hist.GetEntries());
}

// Test FillN() without weights and GetBinContent()
TEST(HistFillTest, FillNCoordsContent) {
  ROOT::Experimental::RH1F hist({100,0.,1});
  EXPECT_FLOAT_EQ(.0f, hist.GetBinContent({0.1111}));
  EXPECT_FLOAT_EQ(.0f, hist.GetBinContent({0.2222}));
  EXPECT_FLOAT_EQ(.0f, hist.GetBinContent({0.3333}));
  hist.FillN({{0.1111},{0.2222},{0.3333}});
  EXPECT_FLOAT_EQ(1.0f, hist.GetBinContent({0.1111}));
  EXPECT_FLOAT_EQ(1.0f, hist.GetBinContent({0.2222}));
  EXPECT_FLOAT_EQ(1.0f, hist.GetBinContent({0.3333}));
  hist.FillN({{0.1111},{0.3333}});
  EXPECT_FLOAT_EQ(2.0f, hist.GetBinContent({0.1111}));
  EXPECT_FLOAT_EQ(2.0f, hist.GetBinContent({0.3333}));
}

// Test FillN() without weights and GetBinUncertainty()
TEST(HistFillTest, FillNCoordsUncertainty) {
  ROOT::Experimental::RH1F hist({100,0.,1});
  float weight = .0f;
  EXPECT_FLOAT_EQ(std::sqrt(weight*weight), hist.GetBinUncertainty({0.1111}));
  EXPECT_FLOAT_EQ(std::sqrt(weight*weight), hist.GetBinUncertainty({0.2222}));
  EXPECT_FLOAT_EQ(std::sqrt(weight*weight), hist.GetBinUncertainty({0.3333}));
  weight = 1.0f;
  hist.FillN({{0.1111},{0.2222},{0.3333}});
  EXPECT_FLOAT_EQ(std::sqrt(weight*weight), hist.GetBinUncertainty({0.1111}));
  EXPECT_FLOAT_EQ(std::sqrt(weight*weight), hist.GetBinUncertainty({0.2222}));
  EXPECT_FLOAT_EQ(std::sqrt(weight*weight), hist.GetBinUncertainty({0.3333}));
  hist.FillN({{0.1111},{0.3333}});
  EXPECT_FLOAT_EQ(std::sqrt((weight*weight)+(weight*weight)), hist.GetBinUncertainty({0.1111}));
  EXPECT_FLOAT_EQ(std::sqrt((weight*weight)+(weight*weight)), hist.GetBinUncertainty({0.3333}));
}

// Test FillN() with weights and GetEntries()
TEST(HistFillTest, FillNCoordsWeightsEntries) {
  ROOT::Experimental::RH1F hist({100,0.,1});
  EXPECT_EQ(0, hist.GetEntries());
  hist.FillN({{0.1111},{0.2222},{0.3333}},{.42f,.32f,.52f});
  EXPECT_EQ(3, hist.GetEntries());
  hist.FillN({{0.1111},{0.3333}},{.42f,.32f});
  EXPECT_EQ(5, hist.GetEntries());
}

// Test FillN() with weights and GetBinContent()
TEST(HistFillTest, FillNCoordsWeightsContent) {
  ROOT::Experimental::RH1F hist({100,0.,1});
  EXPECT_FLOAT_EQ(.0f, hist.GetBinContent({0.1111}));
  EXPECT_FLOAT_EQ(.0f, hist.GetBinContent({0.2222}));
  EXPECT_FLOAT_EQ(.0f, hist.GetBinContent({0.3333}));
  hist.FillN({{0.1111},{0.2222},{0.3333}},{.42f,.32f,.52f});
  EXPECT_FLOAT_EQ(.42f, hist.GetBinContent({0.1111}));
  EXPECT_FLOAT_EQ(.32f, hist.GetBinContent({0.2222}));
  EXPECT_FLOAT_EQ(.52f, hist.GetBinContent({0.3333}));
  hist.FillN({{0.1111},{0.3333}},{.42f,.32f});
  EXPECT_FLOAT_EQ(.42f+.42f, hist.GetBinContent({0.1111}));
  EXPECT_FLOAT_EQ(.52f+.32f, hist.GetBinContent({0.3333}));
}

// Test FillN() and GetBinUncertainty()
TEST(HistFillTest, FillNCoordsWeightsUncertainty) {
  ROOT::Experimental::RH1F hist({100,0.,1});
  float weight1 = .0f;
  float weight2 = .0f;
  float weight3 = .0f;
  EXPECT_FLOAT_EQ(std::sqrt(weight1*weight1), hist.GetBinUncertainty({0.1111}));
  EXPECT_FLOAT_EQ(std::sqrt(weight2*weight2), hist.GetBinUncertainty({0.2222}));
  EXPECT_FLOAT_EQ(std::sqrt(weight3*weight3), hist.GetBinUncertainty({0.3333}));
  weight1 = .42f;
  weight2 = .32f;
  weight3 = .52f;
  hist.FillN({{0.1111},{0.2222},{0.3333}},{weight1,weight2,weight3});
  EXPECT_FLOAT_EQ(std::sqrt(weight1*weight1), hist.GetBinUncertainty({0.1111}));
  EXPECT_FLOAT_EQ(std::sqrt(weight2*weight2), hist.GetBinUncertainty({0.2222}));
  EXPECT_FLOAT_EQ(std::sqrt(weight3*weight3), hist.GetBinUncertainty({0.3333}));
  hist.FillN({{0.1111},{0.3333}},{weight1,weight2});
  EXPECT_FLOAT_EQ(std::sqrt((weight1*weight1)+(weight1*weight1)), hist.GetBinUncertainty({0.1111}));
  EXPECT_FLOAT_EQ(std::sqrt((weight3*weight3)+(weight2*weight2)), hist.GetBinUncertainty({0.3333}));
}