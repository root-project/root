/// \file
/// \ingroup tutorial_hist
/// \notebook -js
/// \preview Normalizing a Histogram.
///
/// Image produced by `.x NormalizeHistogram.C`
/// Two different methods of normalizing histograms
/// are shown, each with the original histogram.
/// next to the normalized one.
/// \macro_image
/// \macro_code
///
/// \date November 2024
/// \author Advait Dhingra

void hist009_TH1_normalize()
{
   const std::array<double, 6> binsx{0, 5, 10, 20, 50, 100};
   TH1D *orig = new TH1D("orig", "Original histogram before normalization", binsx.size() - 1, binsx.data());

   gStyle->SetTitleFontSize(0.06);

   // Filling histogram with random entries
   TRandom2 rand;
   for (int i = 0; i < 100000; ++i) {
      double r = rand.Rndm() * 100;
      orig->Fill(r);
   }

   TH1D *norm = static_cast<TH1D *>(orig->Clone("norm"));
   norm->SetTitle("Normalized Histogram");

   // Normalizing the Histogram by scaling by 1 / the integral and taking width into account
   norm->Scale(1. / norm->Integral(), "width");

   // Drawing everything
   TCanvas *c1 = new TCanvas("c1", "Histogram Normalization", 700, 900);
   c1->Divide(1, 2);

   c1->cd(1);
   orig->Draw();
   c1->cd(2);
   norm->Draw();
}
