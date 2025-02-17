/// \file
/// \ingroup tutorial_hist
/// \notebook
/// Palette coloring for TH1
///
/// Palette coloring for histogram is activated thanks to the options `PFC`
/// (Palette Fill Color), `PLC` (Palette Line Color) and `PMC` (Palette Marker Color).
/// When one of these options is given to `TH1::Draw` the histogram gets its color
/// from the current color palette defined by `gStyle->SetPalette(...)`. The color
/// is determined according to the number of objects having palette coloring in
/// the current pad.
///
/// In this example five histograms are displayed with palette coloring for lines and
/// and marker. The histograms are drawn with markers and error bars and one can see
/// the color of each histogram is picked inside the default palette `kBird`.
///
/// \macro_image
/// \macro_code
///
/// \date November 2024
/// \author Olivier Couet

void hist005_TH1_palettecolor()
{
   auto *canvas = new TCanvas();

   // Disable drawing the title of the canvas
   gStyle->SetOptTitle(kFALSE);
   // Disable drawing the stats box
   gStyle->SetOptStat(0);

   auto *h1 = new TH1D("h1", "Histogram drawn with full circles", 100, -4, 4);
   auto *h2 = new TH1D("h2", "Histogram drawn with full squares", 100, -4, 4);
   auto *h3 = new TH1D("h3", "Histogram drawn with full triangles up", 100, -4, 4);
   auto *h4 = new TH1D("h4", "Histogram drawn with full triangles down", 100, -4, 4);
   auto *h5 = new TH1D("h5", "Histogram drawn with empty circles", 100, -4, 4);

   // Use Mersenne-Twister random number generator
   TRandom3 rng;
   for (int i = 0; i < 25000; i++) {
      // "Rannor" fills the two parameters we pass with RANdom numbers picked from a NORmal distribution.
      // In this case we ignore the second value.
      double val, ignored;
      rng.Rannor(val, ignored);
      // Fill() called with 2 arguments adds the given value (first arg) with the specified weight (second arg)
      h1->Fill(val, 10.);
      h2->Fill(val, 8.);
      h3->Fill(val, 6.);
      h4->Fill(val, 4.);
      h5->Fill(val, 2.);
   }

   // Set different styles for the various histograms
   h1->SetMarkerStyle(kFullCircle);
   h2->SetMarkerStyle(kFullSquare);
   h3->SetMarkerStyle(kFullTriangleUp);
   h4->SetMarkerStyle(kFullTriangleDown);
   h5->SetMarkerStyle(kOpenCircle);

   // Draw all histograms overlapped in the same canvas (thanks to the "SAME" option)
   h1->Draw("PLC PMC");
   h2->Draw("SAME PLC PMC");
   h3->Draw("SAME PLC PMC");
   h4->Draw("SAME PLC PMC");
   h5->Draw("SAME PLC PMC");

   // Build a legend from the objects drawn in the pad, using their description that we specified when constructing them
   gPad->BuildLegend();
}
