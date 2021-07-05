/// \file
/// \ingroup tutorial_hist
/// \notebook -js
/// Normalizing a Histogram
///
/// Image produced by `.x Normalizing_Histogram.C`
/// 2 different methods of normalizing Histograms
/// are shown, each with the original Histogram
/// next to the normalized one.
/// \macro_image
/// \macro_code
///
/// The methods are:
/// 1:
/// ```
/// hist2->Scale(1. / hist2->Integral());
///```
/// 2:
/// ```
///  hist3->Scale(1. / hist3->Integral("width"));
/// ```
///
/// \author Advait Dhingra

#include "TH2F.h"
#include "TRandom.h"
#include "TCanvas.h"

void NormalizeHistogram()
{

   // all of the histograms

   TH1F *hist1 = new TH1F("Unnormalized Histogram", "Unnormalized Hist", 100, 0, 100);
   hist1->SetStats(0);
   TH1F *hist2 = new TH1F("Method 1 Normalized", "Method 1 Normalized", 100, 0, 100);
   hist2->SetStats(0);
   TH1F *hist3 = new TH1F("Method 2 Normalized", "Method 2 Normalized", 100, 0, 100);
   hist3->SetStats(0);

   gStyle->SetTitleFontSize(0.1);

   TRandom2 *rand = new TRandom2();

   // Filling histograms with random entries

   for (int i = 0; i < 1000; i++) {
      double r = rand->Rndm() * 100;
      hist1->Fill(r);
      hist2->Fill(r);
      hist3->Fill(r);
   }

   // method 1
   hist2->Scale(1. / hist2->Integral());

   // method 2
   hist3->Scale(1. / hist3->Integral("width"));

   // Drawing everything

   TCanvas *c1 = new TCanvas("Histogram Normalization Method 1", "Hist Normalization Method 1 ", 1000, 600);
   c1->Divide(1, 3);

   c1->cd(1);
   hist1->Draw();
   c1->cd(2);
   hist2->Draw();
   c1->cd(3);
   hist3->Draw();

   c1->Print("c1.png");
}
