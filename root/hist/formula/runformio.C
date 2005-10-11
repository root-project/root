void write() {
  TFile * file = new TFile("temp.root", "recreate");

  TF1 * fxn = new TF1("fxn", "[0]*TMath::BreitWigner(x, [1], [2])", -10, 10);
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
  TF1 * fxn = (TF1*) file->Get("fxn");
  cout << "fxn->GetParameter(0) = " << fxn->GetParameter(0) << endl;
  cout << "fxn->GetParameter(1) = " << fxn->GetParameter(1) << endl;
  cout << "fxn->GetParameter(2) = " << fxn->GetParameter(2) << endl;
  cout << "fxn->GetParName(0) = " << fxn->GetParName(0) << endl;
  cout << "fxn->GetParName(1) = " << fxn->GetParName(1) << endl;
  cout << "fxn->GetParName(2) = " << fxn->GetParName(2) << endl;
  file->Close();
}

void runformio() {
// Fill out the code of the actual test
   write();
   read();
}
