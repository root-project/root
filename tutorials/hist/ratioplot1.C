/// \file
/// \ingroup tutorial_hist
/// \notebook
/// Example creating a simple ratio plot of two histograms using the `pois` division option.
/// Two histograms are set up and filled with random numbers. The constructor of `TRatioPlot`
/// takes the to histograms, name and title for the object, drawing options for the histograms (`hist` and `E` in this case)
/// and a drawing option for the output graph.
///
/// \macro_image
/// \macro_code
///
/// \author Paul Gessinger

void ratioplot1() {
   gStyle->SetOptStat(0);
   auto c1 = new TCanvas("c1", "A ratio example");
   auto h1 = new TH1D("h1", "h1", 50, 0, 10);
   auto h2 = new TH1D("h2", "h2", 50, 0, 10);
   auto f1 = new TF1("f1", "exp(- x/[0] )");
   f1->SetParameter(0, 3);
   h1->FillRandom("f1", 1900);
   h2->FillRandom("f1", 2000);
   h1->Sumw2();
   h2->Scale(1.9 / 2.);
   h1->GetXaxis()->SetTitle("x");
   h1->GetYaxis()->SetTitle("y");
   auto rp = new TRatioPlot(h1, h2);
   c1->SetTicks(0, 1);
   rp->Draw();
   rp->GetLowYaxis()->SetNdivisions(505);
}
