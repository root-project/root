void histMax() {
    TCanvas *c1 = new TCanvas();
    TH1F *hist = new TH1F("h1", "Histogram", 100, 0, 100);

    for (int i = 0; i < 1000; ++i) {
        hist->Fill(gRandom->Gaus(50, 10));
    }

    Int_t maxBin = hist->GetMaximumBin();

    cout << "The maximum bin content is at: " <<  maxBin << endl;

    Int_t maxBinContent = hist->GetBinContent(maxBin);

    TArrow *arr = new TArrow(0, 90, maxBin, maxBinContent);

    arr->Draw();

    hist->Draw();
}