//
{
  gROOT->ProcessLine(".L all.C+");
  TFile* tf = new TFile("test.root");
  new TCanvas;
#ifdef ClingWorkAroundMissingDynamicScope
   TTree *tree; tf->GetObject("tree",tree);
#endif
  tree->Draw("B.fA.tv.fZ","B.fA.val==1");
  gPad->Modified();
  gPad->Update();
#ifdef ClingWorkAroundMissingDynamicScope
  TH1F* htemp;
  htemp = (TH1F*)gROOT->FindObject("htemp");
#endif
  cout << "Direct access: " << htemp->GetMean() << endl;
  new TCanvas;
#ifdef ClingWorkAroundCallfuncAndInline
  tree->Draw("B.fA.tv.fZ","B.fA.val==1");
#else
  tree->Draw("B.fA.tv.Z()","B.fA.val==1");
#endif
  gPad->Modified();
  gPad->Update();
#ifdef ClingWorkAroundMissingDynamicScope
  htemp = (TH1F*)gROOT->FindObject("htemp");
#endif
  cout << "Function access: " << htemp->GetMean() << endl;
#if defined(ClingWorkAroundCallfuncAndInline) || defined(ClingWorkAroundCallfuncAndReturnByValue)
#else
   tree->Scan("B.fA.tv.fZ:B.fA.GetV().fZ");
#endif
  // new TBrowser;
#ifdef ClingWorkAroundBrokenUnnamedReturn
   gApplication->Terminate(0);
#else
  return 0;
#endif
}

