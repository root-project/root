/// \file
/// \ingroup tutorial_hist
/// \notebook -js
/// Changing the Range on the X-Axis of a Histogram
///
/// Image produced by `.x ZoomHistogram.C`
///
/// This demonstrates how to zoom into a histogram by
/// changing the range on one of the axes (or both).
///
/// \macro_image
/// \macro_code
///
/// \date November 2024
/// \author Advait Dhingra

void hist008_TH1_zoom()
{
   // Create and fill a histogram
   TH1F *orig = new TH1F("Normal Histogram", "Normal Histogram", 100, 0, 100);

   TRandom3 rng;
   for (int i = 0; i < 1000; ++i) {
      double x = rng.Gaus(50, 10);
      orig->Fill(x);
   }

   // Clone the histogram into one called "zoom"
   TH1F *zoom = static_cast<TH1F *>(orig->Clone("zoom"));
   zoom->SetTitle("Zoomed-in Histogram");
   // "Zoom" in the histogram by setting a new range to the X axis
   zoom->GetXaxis()->SetRangeUser(50, 100);

   // Draw both histograms to a canvas
   TCanvas *c1 = new TCanvas("c1", "Histogram", 1500, 700);
   // split the canvas horizontally in 2
   int nsubdivX = 2;
   int nsubdivY = 1;
   c1->Divide(nsubdivX, nsubdivY);

   c1->cd(1);
   orig->Draw();
   c1->cd(2);
   zoom->Draw();
}
