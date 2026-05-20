void write() {
  TFile * file = new TFile("temp.root", "recreate");

  //TF1 * fxn = new TF1("fxn", "[0]*TMath::BreitWigner(x, [1], [2])", -10, 10);
  TFormula * fxn = new TFormula("fxn", "[0]*TMath::BreitWigner(x, [1], [2])");
  // TF1 * fxn = new TF1("fxn", "pol2", -10, 10);

  fxn->SetParameter(0, 10);
  fxn->SetParameter(1, 1.5);
  fxn->SetParameter(2, 1);
  fxn->SetParName(0, "a");
  fxn->SetParName(1, "b");
  fxn->SetParName(2, "c");
  cout << "fxn->GetParameter(0) = " << fxn->GetParameter(0) << endl;
  cout << "fxn->GetParameter(1) = " << fxn->GetParameter(1) << endl;
  cout << "fxn->GetParameter(2) = " << fxn->GetParameter(2) << endl;
  cout << "fxn->GetParName(0) = " << fxn->GetParName(0) << endl;
  cout << "fxn->GetParName(1) = " << fxn->GetParName(1) << endl;
  cout << "fxn->GetParName(2) = " << fxn->GetParName(2) << endl;
  fxn->Write();
  file->Close();
}

void read() {
  TFile * file = new TFile("temp.root");
  TFormula * fxn = (TFormula*) file->Get("fxn");
  cout << "fxn->GetParameter(0) = " << fxn->GetParameter(0) << endl;
  cout << "fxn->GetParameter(1) = " << fxn->GetParameter(1) << endl;
  cout << "fxn->GetParameter(2) = " << fxn->GetParameter(2) << endl;
  cout << "fxn->GetParName(0) = " << fxn->GetParName(0) << endl;
  cout << "fxn->GetParName(1) = " << fxn->GetParName(1) << endl;
  cout << "fxn->GetParName(2) = " << fxn->GetParName(2) << endl;
  file->Close();
}

void readV3file() {
   new TCanvas();
   TFile *_file0 = TFile::Open("result_30gev_sep05.root");
//#ifdef ClingWorkAroundMissingDynamicScope
   TObjArray *ptspec_chisq_deut;
   _file0->GetObject("ptspec_chisq_deut",ptspec_chisq_deut);
//#endif
   for(int i=0;i<ptspec_chisq_deut->GetEntries();++i) {
      TH1D *h = (TH1D*)ptspec_chisq_deut->At(105);
      TF1 *f = (TF1*)h->GetListOfFunctions()->At(0);
      h->Draw();
   }
}

void runformio() {
// Fill out the code of the actual test
   write();
   read();
   //readV3file();
}
