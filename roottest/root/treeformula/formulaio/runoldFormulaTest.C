{
TFile *f = new TFile("hsimple.31000.root");
TH1F *hpx = (TH1F*)f->Get("hpx");
hpx->Print();
}
