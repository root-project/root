/// \file
/// \ingroup tutorial_hist
/// \notebook -js
/// Normalizing a Histogram
///
/// Image produced by `.x NormalizeHistogram.C`
/// Two different methods of normalizing histograms
/// are shown, each with the original histogram.
/// next to the normalized one.
/// \macro_image
/// \macro_code
///
/// \author Advait Dhingra

#include "TH2F.h"
#include "TRandom.h"
#include "TCanvas.h"

void NormalizeHistogram()
{

   // all of the histograms
   std::array<double, 6> binsx{0, 5, 10, 20, 50, 100};
   TH1F *orig = new TH1F("orig", "Original histogram before normalization", binsx.size() - 1, &binsx[0]);

   gStyle->SetTitleFontSize(0.06);

   TRandom2 rand;

   // Filling histograms with random entries
   for (int i = 0; i < 1000; ++i) {
      double r = rand.Rndm() * 100;
      orig->Fill(r);
   }

   TH1F *norm = (TH1F *)orig->Clone("norm");
   norm->SetTitle("Histogram normalized with bin content");
   TH1F *normw = (TH1F *)orig->Clone("normw");
   normw->SetTitle("Histogram normalized with bin content times bin width");

   // method 1: Normalization by division of bin content.
   norm->Scale(1. / norm->Integral());

   // method 2: Normalization by sum of the bins's content times the respective widths.
   normw->Scale(1. / normw->Integral("width"));

   // Drawing everything
   TCanvas *c1 = new TCanvas("c1", "Histogram Normalization", 700, 900);
   c1->Divide(1, 3);

   c1->cd(1);
   orig->Draw();
   c1->cd(2);
   norm->Draw();
   c1->cd(3);
   normw->Draw();
}
