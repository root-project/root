#include "gtest/gtest.h"
#include "ROOT/THist.h"
#include "ROOT/Canvas.h"

using namespace ROOT::Experimental;

// Test drawing of histograms.
TEST(DrawTest, OneD) {
   TAxisConfig xaxis{10, 0., 1.};
   auto h = std::make_shared<TH1D>(xaxis);
   TCanvas canv;
   canv.Draw(h);
   EXPECT_EQ(canv.GetPrimitives().size(), 1ul);
}

TEST(DrawTest, TwoD) {
   TAxisConfig xaxis{10, 0., 1.};
   TAxisConfig yaxis{{0., 1., 10., 100.}};
   auto h = std::make_shared<TH2I>(xaxis, yaxis);
   TCanvas canv;
   canv.Draw(h);
   EXPECT_EQ(canv.GetPrimitives().size(), 1ul);
}

TEST(DrawTest, ThreeD) {
   TAxisConfig xaxis{{0., 1.}};
   TAxisConfig yaxis{10, 0., 1.};
   TAxisConfig zaxis{{0., 1., 10., 100.}};
   auto h = std::make_shared<TH3F>(xaxis, yaxis, zaxis);
   TCanvas canv;
   canv.Draw(h);
   EXPECT_EQ(canv.GetPrimitives().size(), 1ul);
}
