/// \file
/// \ingroup tutorial_hist
///
/// Example showing a fit residual plot, where the separation margin has been set to 0.
/// The last label of the lower plot's y axis is hidden automatically.
///
/// \macro_image
/// \macro_code
///
/// \author Paul Gessinger

{
   gStyle->SetOptStat(0);
   auto c1 = new TCanvas("c1", "fit residual simple");
   gPad->SetFrameFillStyle(0);
   auto h1 = new TH1D("h1", "h1", 50, -5, 5);
   h1->FillRandom("gaus", 5000);
   TFitResultPtr fitres = h1->Fit("gaus", "S");
   h1->Sumw2();

   c1->Clear();
   
   auto rp1 = new TRatioPlot(h1, "errfunc", "", "L");
   rp1->SetSeparationMargin(0.0);
   rp1->Draw();
   rp1->GetLowerRefGraph()->SetMinimum(-2);
   rp1->GetLowerRefGraph()->SetMaximum(2);
   c1->Update();
   return c1;
}
