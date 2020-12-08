/// \file
/// \ingroup tutorial_math
/// \notebook
/// Tutorial illustrating the multivariate gaussian random number generation
///
/// \macro_image
/// \macro_code
///
/// \author Jorge Lopez

void multivarGaus() {
  ROOT::Math::GSLRandomEngine rnd;
  rnd.Initialize();

  const int dim = 3;
  double pars[dim] = {0, 0, 0.5};
  double genpars[dim] = {0, 0, 0};
  double cov[dim * dim] = {1.0, -0.2, 0.0, -0.2, 1.0, 0.5, 0.0, 0.5, 0.75};

  TH1F* hX = new TH1F("hX", "hX;x;Counts", 100, -5, 5);
  TH1F* hY = new TH1F("hY", "hY;y;Counts", 100, -5, 5);
  TH1F* hZ = new TH1F("hZ", "hZ;z;Counts", 100, -5, 5);

  TH2F* hXY = new TH2F("hXY", "hXY;x;y;Counts", 100, -5, 5, 100, -5, 5);
  TH2F* hXZ = new TH2F("hXZ", "hXZ;x;z;Counts", 100, -5, 5, 100, -5, 5);
  TH2F* hYZ = new TH2F("hYZ", "hYZ;y;z;Counts", 100, -5, 5, 100, -5, 5);

  const int MAX = 10000;
  for (int evnts = 0; evnts < MAX; ++evnts) {
    rnd.GaussianND(dim, pars, cov, genpars);
    auto x = genpars[0];
    auto y = genpars[1];
    auto z = genpars[2];
    hX->Fill(x);
    hY->Fill(y);
    hZ->Fill(z);
    hXY->Fill(x, y);
    hXZ->Fill(x, z);
    hYZ->Fill(y, z);
  }

  TCanvas* c = new TCanvas("c", "Multivariate gaussian random numbers");
  c->Divide(3, 2);
  c->cd(1);
  hX->Draw();
  c->cd(2);
  hY->Draw();
  c->cd(3);
  hZ->Draw();
  c->cd(4);
  hXY->Draw("COL");
  c->cd(5);
  hXZ->Draw("COL");
  c->cd(6);
  hYZ->Draw("COL");
}
