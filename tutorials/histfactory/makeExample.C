makeExample(){
  TFile* example = new TFile("example.root","RECREATE");
  TH1F* signal = new TH1F("signal","signal histogram (pb)", 2,1,2);
  TH1F* background1 = new TH1F("background1","background 1 histogram (pb)", 2,1,2);
  TH1F* background2 = new TH1F("background2","background 2 histogram (pb)", 2,1,2);

  signal->SetBinContent(1,20);
  signal->SetBinContent(2,10);

  background1->SetBinContent(1,100);
  background2->SetBinContent(2,100);
  example->Write();
  example->Close();
}
