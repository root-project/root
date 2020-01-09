/// \file
/// \ingroup tutorial_hist
/// \notebook
/// Example showing a fit residual plot, where the separation margin has been set to 0.
/// The last label of the lower plot's y axis is hidden automatically.
///
/// \macro_image
/// \macro_code
///
/// \author Paul Gessinger

void ratioplot6() {
   gStyle->SetOptStat(0);
   auto c1 = new TCanvas("c1", "fit residual simple");
   gPad->SetFrameFillStyle(0);
   auto h1 = new TH1D("h1", "h1", 50, -5, 5);
   h1->FillRandom("gaus", 5000);
   TFitResultPtr fitres = h1->Fit("gaus", "S0");
   h1->Sumw2();
   h1->GetXaxis()->SetTitle("x");
   h1->GetYaxis()->SetTitle("y");
   auto rp1 = new TRatioPlot(h1, "errfunc");
   rp1->SetGraphDrawOpt("L");
   rp1->SetSeparationMargin(0.0);
   rp1->Draw();
   rp1->GetLowerRefGraph()->SetMinimum(-2);
   rp1->GetLowerRefGraph()->SetMaximum(2);
}
