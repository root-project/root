#include "gtest/gtest.h"

#include "TCanvas.h"
#include "TRatioPlot.h"
#include "TH1.h"
#include "TFile.h"


TEST(TRatioPlot, CreateFile)
{
   TCanvas c1("c1", "fit residual simple");
   TH1D h1("h1", "h1", 50, -5, 5);
   h1.FillRandom("gaus", 2000);
   h1.Fit("gaus", "0");
   h1.GetXaxis()->SetTitle("x");
   TRatioPlot rp1(&h1);
   rp1.Draw();
   rp1.GetLowerRefYaxis()->SetTitle("ratio");
   rp1.GetUpperRefYaxis()->SetTitle("entries");

   auto f = TFile::Open("ratioplot.root", "RECREATE");
   f->WriteObject(&c1, "ratioplot");
   delete f;
}

TEST(TRatioPlot, ReadFile)
{
   TCanvas *c1 = nullptr;

   auto f = TFile::Open("ratioplot.root");
   f->GetObject("ratioplot", c1);

   EXPECT_TRUE(c1 != nullptr);

   delete c1;

   delete f;
}

