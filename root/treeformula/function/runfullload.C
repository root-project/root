//
{
  gROOT->ProcessLine(".L all.C+");
  TFile* tf = new TFile("test.root");
  new TCanvas;
  tree->Draw("B.fA.tv.fZ","B.fA.val==1");
  gPad->Modified();
  gPad->Update();
  cout << "Direct access: " << htemp->GetMean() << endl;
  new TCanvas;
  tree->Draw("B.fA.tv.Z()","B.fA.val==1");
  gPad->Modified();
  gPad->Update();
  cout << "Function access: " << htemp->GetMean() << endl;
  tree->Scan("B.fA.tv.fZ:B.fA.GetV().fZ");
  return 0;
}

