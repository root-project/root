/// \file
/// \ingroup tutorial_hist
/// \notebook -js
/// Normalizing a Histogram
///
/// Image produced by `.x Normalizing_Histogram.C`
/// Two different methods of normalizing histograms
/// are shown, each with the original histogram.
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

   TH1F *orig = new TH1F("orig", "Original histogram before normalization", 100, 0, 100);
   orig->SetStats(0);
   TH1F *norm = new TH1F("norm", "Histogram normalized with bin content", 100, 0, 100);
   norm->SetStats(0);
   TH1F *normw = new TH1F("normw", "Histogram normalized with bin content times bin width", 100, 0, 100);
   hist3->SetStats(0);

   gStyle->SetTitleFontSize(0.1);

   TRandom2 rand;

   // Filling histograms with random entries

   for (int i = 0; i < 1000; ++i) {
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

   TCanvas *c1 = new TCanvas("c1", "Histogram Normalization", 1000, 600);
   c1->Divide(1, 3);

   c1->cd(1);
   hist1->Draw();
   c1->cd(2);
   hist2->Draw();
   c1->cd(3);
   hist3->Draw();

   c1->Print("c1.png");
}
