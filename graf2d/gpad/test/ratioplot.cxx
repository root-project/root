#include "gtest/gtest.h"

#include "TH1.h"
#include "TF1.h"
#include "THStack.h"
#include "TRatioPlot.h"
#include "TCanvas.h"
#include "TFile.h"


TEST(TRatioPlot, CreateFile)
{
   TF1 f1("f1", "exp(-x/[0])");
   f1.SetParameter(0, 3);

   TCanvas c1("c1", "fit residual simple");
   TH1D h1("h1", "TRatioplot example", 50, 0, 10);
   TH1D h2("h2", "TRatioplot example", 50, 0, 10);
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

TEST(TRatioPlot, CreateFileStack)
{
   TF1 f1("f1", "exp(-x/[0])");
   f1.SetParameter(0, 3);

   TCanvas c1("c1", "fit residual simple");

   THStack hs("hs","Stacked 1D histograms");
   //create three 1-d histograms
   auto h1st = new TH1D("h1st","test hstack",50, 0, 10);
   h1st->FillRandom("f1", 1000);
   h1st->SetFillColor(kRed);
   h1st->SetMarkerStyle(21);
   h1st->SetMarkerColor(kRed);
   hs.Add(h1st);
   auto h2st = new TH1D("h2st","test hstack",50, 0, 10);
   h2st->FillRandom("f1",1000);
   h2st->SetFillColor(kBlue);
   h2st->SetMarkerStyle(21);
   h2st->SetMarkerColor(kBlue);
   hs.Add(h2st);
   auto h3st = new TH1D("h3st","test hstack",50, 0, 10);
   h3st->FillRandom("f1",1000);
   h3st->SetFillColor(kGreen);
   h3st->SetMarkerStyle(21);
   h3st->SetMarkerColor(kGreen);
   hs.Add(h3st);

   TH1D h2("h2", "TRatioplot example", 50, 0, 10);
   h2.FillRandom("f1", 2000);
   h2.SetLineColor(kRed);

   TRatioPlot rp2(&hs, &h2);
   rp2.Draw();

   auto f = TFile::Open("ratioplotstack.root", "RECREATE");
   f->WriteObject(&c1, "ratioplotstack");
   delete f;
}

TEST(TRatioPlot, ReadFileStack)
{
   TCanvas *c1 = nullptr;

   auto f = TFile::Open("ratioplotstack.root");
   f->GetObject("ratioplotstack", c1);

   EXPECT_TRUE(c1 != nullptr);

   delete c1;

   delete f;
}

