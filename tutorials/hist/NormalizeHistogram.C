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
   TH1F *hist1 = new TH1F("Unnormalized Histogram", "Unnormalized Hist", binsx.size() - 1, &binsx[0]);
   TH1F *hist2 = new TH1F("Method 1 Normalized", "Method 1 Normalized", binsx.size() - 1, &binsx[0]);
   TH1F *hist3 = new TH1F("Method 2 Normalized", "Method 2 Normalized", binsx.size() - 1, &binsx[0]);

   gStyle->SetTitleFontSize(0.1);

   TRandom2 rand;

   // Filling histograms with random entries
   for (int i = 0; i < 1000; ++i) {
      double r = rand.Rndm() * 100;
      hist1->Fill(r);
      hist2->Fill(r);
      hist3->Fill(r);
   }

   // method 1: Normalization by division of bin content.
   hist2->Scale(1. / hist2->Integral());

   // method 2: Normalization by sum of the bins's content times the respective widths.
   hist3->Scale(1. / hist3->Integral("width"));

   // Drawing everything
   TCanvas *c1 = new TCanvas("c1", "Histogram Normalization", 700, 900);
   c1->Divide(1, 3);

   c1->cd(1);
   hist1->Draw();
   c1->cd(2);
   hist2->Draw();
   c1->cd(3);
   hist3->Draw();
}
