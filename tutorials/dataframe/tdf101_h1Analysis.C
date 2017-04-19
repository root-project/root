/// \file
/// \ingroup tutorial_tdataframe
/// \notebook
/// This tutorial illustrates how to express the H1 analysis with a TDataFrame.
///
/// \macro_code
///
/// \date December 2016
/// \author Axel Naumann

#include "ROOT/TDataFrame.hxx"
#include "TCanvas.h"
#include "TChain.h"
#include "TF1.h"
#include "TH2.h"
#include "TLine.h"
#include "TPaveStats.h"
#include "TStyle.h"

auto Select = [](ROOT::Experimental::TDataFrame &dataFrame) {
   auto ret =
      dataFrame.Filter([](float md0_d) { return TMath::Abs(md0_d - 1.8646) < 0.04; }, {"md0_d"})
         .Filter([](float ptds_d) { return ptds_d > 2.5; }, {"ptds_d"})
         .Filter([](float etads_d) { return TMath::Abs(etads_d) < 1.5; }, {"etads_d"})
         .Filter([](int ik, int ipi, std::array_view<int> nhitrp) { return nhitrp[ik - 1] * nhitrp[ipi - 1] > 1; },
                 {"ik", "ipi", "nhitrp"})
         .Filter([](int ik, std::array_view<float> rstart,
                    std::array_view<float> rend) { return rend[ik - 1] - rstart[ik - 1] > 22; },
                 {"ik", "rstart", "rend"})
         .Filter([](int ipi, std::array_view<float> rstart,
                    std::array_view<float> rend) { return rend[ipi - 1] - rstart[ipi - 1] > 22; },
                 {"ipi", "rstart", "rend"})
         .Filter([](int ik, std::array_view<float> nlhk) { return nlhk[ik - 1] > 0.1; }, {"ik", "nlhk"})
         .Filter([](int ipi, std::array_view<float> nlhpi) { return nlhpi[ipi - 1] > 0.1; }, {"ipi", "nlhpi"})
         .Filter([](int ipis, std::array_view<float> nlhpi) { return nlhpi[ipis - 1] > 0.1; }, {"ipis", "nlhpi"})
         .Filter([](int njets) { return njets >= 1; }, {"njets"});

   return ret;
};

const Double_t dxbin = (0.17 - 0.13) / 40; // Bin-width

Double_t fdm5(Double_t *xx, Double_t *par)
{
   Double_t x = xx[0];
   if (x <= 0.13957) return 0;
   Double_t xp3 = (x - par[3]) * (x - par[3]);
   Double_t res =
      dxbin * (par[0] * pow(x - 0.13957, par[1]) + par[2] / 2.5066 / par[4] * exp(-xp3 / 2 / par[4] / par[4]));
   return res;
}

Double_t fdm2(Double_t *xx, Double_t *par)
{
   static const Double_t sigma = 0.0012;
   Double_t x = xx[0];
   if (x <= 0.13957) return 0;
   Double_t xp3 = (x - 0.1454) * (x - 0.1454);
   Double_t res = dxbin * (par[0] * pow(x - 0.13957, 0.25) + par[1] / 2.5066 / sigma * exp(-xp3 / 2 / sigma / sigma));
   return res;
}

void FitAndPlotHdmd(TH1 &hdmd)
{
   // create the canvas for the h1analysis fit
   gStyle->SetOptFit();
   TCanvas *c1 = new TCanvas("c1", "h1analysis analysis", 10, 10, 800, 600);
   c1->SetBottomMargin(0.15);
   hdmd.GetXaxis()->SetTitle("m_{K#pi#pi} - m_{K#pi}[GeV/c^{2}]");
   hdmd.GetXaxis()->SetTitleOffset(1.4);

   // fit histogram hdmd with function f5 using the loglikelihood option
   if (gROOT->GetListOfFunctions()->FindObject("f5")) delete gROOT->GetFunction("f5");
   TF1 *f5 = new TF1("f5", fdm5, 0.139, 0.17, 5);
   f5->SetParameters(1000000, .25, 2000, .1454, .001);
   hdmd.Fit("f5", "lr");

   // Have the number of entries on the first histogram (to cross check when running
   // with entry lists)
   TPaveStats *psdmd = (TPaveStats *)hdmd.GetListOfFunctions()->FindObject("stats");
   if (psdmd) psdmd->SetOptStat(1110);
   c1->Modified();
}

void FitAndPlotH2(TH2 &h2)
{
   // create the canvas for tau d0
   gStyle->SetOptFit(0);
   gStyle->SetOptStat(1100);
   TCanvas *c2 = new TCanvas("c2", "tauD0", 100, 100, 800, 600);
   c2->SetGrid();
   c2->SetBottomMargin(0.15);

   // Project slices of 2-d histogram h2 along X , then fit each slice
   // with function f2 and make a histogram for each fit parameter
   // Note that the generated histograms are added to the list of objects
   // in the current directory.
   if (gROOT->GetListOfFunctions()->FindObject("f2")) delete gROOT->GetFunction("f2");
   TF1 *f2 = new TF1("f2", fdm2, 0.139, 0.17, 2);
   f2->SetParameters(10000, 10);
   h2.FitSlicesX(f2, 0, -1, 1, "qln");

   TH1D *h2_1 = (TH1D *)gDirectory->Get("h2_1");
   h2_1->GetXaxis()->SetTitle("#tau[ps]");
   h2_1->SetMarkerStyle(21);
   h2_1->Draw();
   c2->Update();
   TLine *line = new TLine(0, 0, 0, c2->GetUymax());
   line->Draw();
}

void tdf101_h1Analysis()
{
   TChain chain("h42");
   chain.Add("http://root.cern.ch/files/h1/dstarmb.root");
   chain.Add("http://root.cern.ch/files/h1/dstarp1a.root");
   chain.Add("http://root.cern.ch/files/h1/dstarp1b.root");
   chain.Add("http://root.cern.ch/files/h1/dstarp2.root");

   ROOT::Experimental::TDataFrame dataFrame(chain);
   auto selected = Select(dataFrame);

   auto hdmdARP = selected.Histo1D(TH1F("hdmd", "Dm_d", 40, 0.13, 0.17), "dm_d");
   auto selectedAddedBranch = selected.Define(
      "h2_y", [](float rpd0_t, float ptd0_d) { return rpd0_t / 0.029979f * 1.8646f / ptd0_d; }, {"rpd0_t", "ptd0_d"});
   auto h2ARP = selectedAddedBranch.Histo2D<float, float>(TH2F("h2", "ptD0 vs Dm_d", 30, 0.135, 0.165, 30, -3, 6),
                                                          "dm_d", "h2_y");

   FitAndPlotHdmd(*hdmdARP);
   FitAndPlotH2(*h2ARP);
}
