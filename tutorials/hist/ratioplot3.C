/// \file
/// \ingroup tutorial_hist
/// \notebook
/// Example which shows how you can get the graph of the lower plot and set the y axis range for it.
///
/// Since the lower plot is not created until `TRatioPlot::Draw` is called, you can only use the method
/// afterwards.
///
/// \macro_image
/// \macro_code
///
/// \author Paul Gessinger

void ratioplot3()  {
    gStyle->SetOptStat(0);
    auto c1 = new TCanvas("c1", "fit residual simple");
    c1->SetLogy();
    auto h1 = new TH1D("h1", "h1", 50, -5, 5);
    h1->FillRandom("gaus", 2000);
    h1->Fit("gaus", "0");
    h1->SetMinimum(0.001);
    h1->GetXaxis()->SetTitle("x");
    h1->GetYaxis()->SetTitle("y");
    auto rp1 = new TRatioPlot(h1);
    rp1->Draw();
    rp1->GetLowerRefGraph()->SetMinimum(-2);
    rp1->GetLowerRefGraph()->SetMaximum(2);
}
