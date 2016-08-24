 {
    gStyle->SetOptStat(0);
    auto c1 = new TCanvas("c1", "fit residual simple");
    auto h1 = new TH1D("h1", "h1", 50, -5, 5);
    h1->FillRandom("gaus", 2000);
    h1->Fit("gaus");
    c1->Clear();
    auto rp1 = new TRatioPlot(h1, "rp1", "rp1", "nogrid");
    rp1->SetConfidenceIntervalColors(kBlue, kRed);
    rp1->Draw();
    c1->Update();
    return c1;
 }
