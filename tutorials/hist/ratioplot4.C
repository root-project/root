 {
    gStyle->SetOptStat(0);
    auto c1 = new TCanvas("c1", "fit residual simple");
    auto h1 = new TH1D("h1", "h1", 50, -5, 5);
    h1->FillRandom("gaus", 2000);
    h1->Fit("gaus");
    c1->Clear();
    auto rp1 = new TRatioPlot(h1, "rp1", "rp1");
    std::vector<double> lines = {-3, -2, -1, 0, 1, 2, 3};
    rp1->SetGridlines(lines);
    rp1->Draw();
    rp1->GetLowerRefGraph()->SetMinimum(-4);
    rp1->GetLowerRefGraph()->SetMaximum(4);
    c1->Update();
    return c1;
 }
