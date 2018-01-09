#include "gtest/gtest.h"
#include "ROOT/THist.hxx"
#include "ROOT/TCanvas.hxx"
#include "ROOT/TColor.hxx"

class FixtureBase: public ::testing::Test {
protected:
   virtual void SetUp();
};

struct DrawTest: public FixtureBase {};

// Test drawing of histograms.
TEST_F(DrawTest, OneD)
{
   using namespace ROOT::Experimental;
   TAxisConfig xaxis{10, 0., 1.};
   auto h = std::make_shared<TH1D>(xaxis);
   TCanvas canv;
   canv.Draw(h);
   EXPECT_EQ(canv.GetPrimitives().size(), 1u);
}

TEST_F(DrawTest, TwoD)
{
   using namespace ROOT::Experimental;
   TAxisConfig xaxis{10, 0., 1.};
   TAxisConfig yaxis{{0., 1., 10., 100.}};
   auto h = std::make_shared<TH2I>(xaxis, yaxis);
   TCanvas canv;
   canv.Draw(h);
   // No THist copy c'tor:
   // canv.Draw(TH2F(xaxis, yaxis));
   canv.Draw(std::make_unique<TH2C>(xaxis, yaxis));
   EXPECT_EQ(canv.GetPrimitives().size(), 2u);
}

TEST_F(DrawTest, ThreeD)
{
   using namespace ROOT::Experimental;
   TAxisConfig xaxis{{0., 1.}};
   TAxisConfig yaxis{10, 0., 1.};
   TAxisConfig zaxis{{0., 1., 10., 100.}};
   auto h = std::make_shared<TH3F>(xaxis, yaxis, zaxis);
   TCanvas canv;
   canv.Draw(h);
   EXPECT_EQ(canv.GetPrimitives().size(), 1u);
}


struct DrawOptTest: public FixtureBase {};

// Drawing options:
TEST_F(DrawOptTest, OneD)
{
   using namespace ROOT::Experimental;
   TAxisConfig xaxis{10, 0., 1.};
   auto h = std::make_shared<TH1D>(xaxis);
   TCanvas canv;
   auto &Opts = canv.Draw(h);
   Opts.SetLineColor(TColor::kRed);
   TColor shouldBeRed = Opts.GetLineColor();
   EXPECT_EQ(shouldBeRed, TColor::kRed);
}



#include "TROOT.h" // for gROOT->SetBatch()

void FixtureBase::SetUp() {
   gROOT->SetBatch();
};
