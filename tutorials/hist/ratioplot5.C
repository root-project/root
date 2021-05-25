/// \file
/// \ingroup tutorial_hist
/// \notebook
/// Example that shows how you can set the colors of the confidence interval bands by using
/// the method `TRatioPlot::SetConfidenceIntervalColors`.
///
/// \macro_image
/// \macro_code
///
/// \author Paul Gessinger

void ratioplot5()  {
   gStyle->SetOptStat(0);
   auto c1 = new TCanvas("c1", "fit residual simple");
   auto h1 = new TH1D("h1", "h1", 50, -5, 5);
   h1->FillRandom("gaus", 2000);
   h1->Fit("gaus","0");
   h1->GetXaxis()->SetTitle("x");
   h1->GetYaxis()->SetTitle("y");
   auto rp1 = new TRatioPlot(h1);
   rp1->SetConfidenceIntervalColors(kBlue, kRed);
   rp1->Draw();
}
