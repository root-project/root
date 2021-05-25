/// \file
/// \ingroup tutorial_hist
/// \notebook
/// Example of a fit residual plot.
///
/// Creates a histogram filled with random numbers from a gaussian distribution
/// and fits it with a standard gaussian function. The result is passed to the `TRatioPlot`
/// constructor. Additionally, after calling `TRatioPlot::Draw` the upper and lower y axis
/// titles are modified.
/// Confidence interval bands are automatically drawn on the bottom (but can be disabled by draw option `nobands`.
///
/// \macro_image
/// \macro_code
///
/// \author Paul Gessinger

void ratioplot2() {
   gStyle->SetOptStat(0);
   auto c1 = new TCanvas("c1", "fit residual simple");
   auto h1 = new TH1D("h1", "h1", 50, -5, 5);
   h1->FillRandom("gaus", 2000);
   h1->Fit("gaus", "0");
   h1->GetXaxis()->SetTitle("x");
   auto rp1 = new TRatioPlot(h1);
   rp1->Draw();
   rp1->GetLowerRefYaxis()->SetTitle("ratio");
   rp1->GetUpperRefYaxis()->SetTitle("entries");
}
