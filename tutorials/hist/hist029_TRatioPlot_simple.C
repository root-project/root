/// \file
/// \ingroup tutorial_hist
/// \notebook
/// \preview Example creating a simple ratio plot of two histograms using the "pois" division option.
/// Two histograms are set up and filled with random numbers. The constructor of `TRatioPlot`
/// takes the two histograms, name and title for the object, drawing options for the histograms
/// (`hist` and `E` in this case) and a drawing option for the output graph.
/// The histograms drawing options can be changed with `SetH1DrawOpt` and `SetH2DrawOpt`.
///
/// \macro_image
/// \macro_code
///
/// \date February 2023
/// \author Paul Gessinger

void hist029_TRatioPlot_simple()
{
   gStyle->SetOptStat(0);
   auto C = new TCanvas("C", "A ratio example");
   auto h1 = new TH1D("h1", "TRatioPlot Example; x; y", 50, 0, 10);
   auto h2 = new TH1D("h2", "h2", 50, 0, 10);
   auto f1 = new TF1("f1", "exp(- x/[0] )");
   f1->SetParameter(0, 3);
   h1->FillRandom("f1", 1900);
   h2->FillRandom("f1", 2000);
   h1->Sumw2();
   h2->Scale(1.9 / 2.);
   h2->SetLineColor(kRed);

   // Create and draw the ratio plot
   auto rp = new TRatioPlot(h1, h2);
   C->SetTicks(0, 1);
   rp->Draw();
   rp->GetLowYaxis()->SetNdivisions(505);

   // Add a legend to the ratio plot
   rp->GetUpperPad()->cd();
   TLegend *legend = new TLegend(0.3, 0.7, 0.7, 0.85);
   legend->AddEntry("h1", "First histogram", "l");
   legend->AddEntry("h2", "Second histogram", "le");
   legend->Draw();
}
