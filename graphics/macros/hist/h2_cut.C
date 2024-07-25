/// \file
/// \ingroup tutorial_hist
/// \notebook -js
/// This example demonstrates how to display a 2D histogram and
/// use TCutG object to select bins for drawing.
/// Moving TCutG object one can change displayed region of histogram
///
/// \macro_image
/// \macro_code
///
/// \author Sergey Linev

void h2_cut()
{
   const int n = 6;
   Float_t x[6] = { 1, 2,  1, -1, -2, -1 };
   Float_t y[6] = { 2, 0, -2, -2,  0,  2 };
   TCutG *cut = new TCutG("cut", 6, x, y);
   TH2F *hist = new TH2F("hist", "Histogram with cut", 40, -10., 10., 40, -10., 10.);
   for (int i = 0; i < 100000; i++)
      hist->Fill(gRandom->Gaus(0., 3.), gRandom->Gaus(0., 3.));
   TCanvas *c1 = new TCanvas("c1", "Histogram draw with TCutG", 600, 900);
   hist->Draw("col [cut]");
   cut->Draw("l");
}