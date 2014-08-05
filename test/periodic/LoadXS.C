void LoadXS()
{
  gSystem->Load("libNdb");
  NdbMTReactionXS   pb(1,"Total Cross Section");

  pb.LoadENDF("8200.endf");

  const Int_t npoint=1000;

  Double_t e=1.0E-3;

  const Double_t fact = TMath::Exp(TMath::Log(2.0E7/e)/(Double_t)(npoint-1));

  Float_t *x=new Float_t[npoint];
  Float_t *y=new Float_t[npoint];


  for (Int_t i=0; i<npoint; ++i) {
    x[i]=e;
    y[i]=pb.Interpolate(e);
    e*=fact;
  }

  c1 = new TCanvas("c1","Lead Cross section",200,10,700,500);

  c1->SetFillColor(42);
  c1->SetGridx();
  c1->SetGridy();
  c1->GetFrame()->SetFillColor(21);
  c1->GetFrame()->SetBorderSize(12);
  c1->SetLogx();
  c1->SetLogy();

  TGraph *gr = new TGraph(npoint,x,y);
  gr->SetFillColor(19);
  gr->SetLineColor(2);
  gr->SetLineWidth(2);
  gr->SetMarkerColor(4);
  gr->SetMarkerStyle(21);
  gr->SetMarkerSize(0.2);
  gr->SetTitle("Lead Total Cross section");
  gr->Draw("AWLP");

  //Add axis titles.
  //A graph is drawn using the services of the TH1F histogram class.
  //The histogram is created by TGraph::Paint.
  //TGraph::Paint is called by TCanvas::Update. This function is called by default
  //when typing <CR> at the keyboard. In a macro, one must force TCanvas::Update.

  c1->Update();
  gr->GetHistogram()->SetXTitle("Energy in eV");
  gr->GetHistogram()->SetYTitle("Cross Section in Barns");
}
