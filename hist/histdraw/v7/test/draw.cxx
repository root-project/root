#include "gtest/gtest.h"
#include "ROOT/THist.hxx"
#include "ROOT/RCanvas.hxx"
#include "ROOT/RColor.hxx"

using namespace ROOT::Experimental;

// Test drawing of histograms.
TEST(DrawTest, OneD)
{
   TAxisConfig xaxis{10, 0., 1.};
   auto h = std::make_shared<TH1D>(xaxis);
   RCanvas canv;
   canv.Draw(h);
   EXPECT_EQ(canv.GetPrimitives().size(), 1u);
}

TEST(DrawTest, TwoD)
{
   TAxisConfig xaxis{10, 0., 1.};
   TAxisConfig yaxis{{0., 1., 10., 100.}};
   auto h = std::make_shared<TH2I>(xaxis, yaxis);
   RCanvas canv;
   canv.Draw(h);
   // No THist copt c'tor:
   // canv.Draw(TH2F(xaxis, yaxis));
   canv.Draw(std::make_unique<TH2C>(xaxis, yaxis));
   EXPECT_EQ(canv.GetPrimitives().size(), 2u);
}

TEST(DrawTest, ThreeD)
{
   TAxisConfig xaxis{{0., 1.}};
   TAxisConfig yaxis{10, 0., 1.};
   TAxisConfig zaxis{{0., 1., 10., 100.}};
   auto h = std::make_shared<TH3F>(xaxis, yaxis, zaxis);
   RCanvas canv;
   canv.Draw(h);
   EXPECT_EQ(canv.GetPrimitives().size(), 1u);
}

// Drawing options:
TEST(DrawOptTest, OneD)
{
   TAxisConfig xaxis{10, 0., 1.};
   auto h = std::make_shared<TH1D>(xaxis);
   RCanvas canv;
   auto optsPtr = canv.Draw(h);
   optsPtr->SetLineColor(RColor::kRed);
   RColor shouldBeRed = (RColor)optsPtr->GetLineColor();
   EXPECT_EQ(shouldBeRed, RColor::kRed);
}
