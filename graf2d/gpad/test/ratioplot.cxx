#include "gtest/gtest.h"

#include "TH1.h"
#include "TF1.h"
#include "TRatioPlot.h"
#include "TCanvas.h"
#include "TFile.h"


TEST(TRatioPlot, CreateFile)
{
   TCanvas c1("c1", "fit residual simple");
   TH1D h1("h1", "TRatioplot example", 50, 0, 10);
   TH1D h2("h2", "TRatioplot example", 50, 0, 10);
   TF1 f1("f1", "exp(-x/[0])");
   f1.SetParameter(0, 3);
   h1.FillRandom("f1", 1900);
   h2.FillRandom("f1", 2000);
   h1.Sumw2();
   h2.Scale(1.9 / 2.);
   h2.SetLineColor(kRed);

   TRatioPlot rp1(&h1, &h2);
   rp1.Draw();

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

