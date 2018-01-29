/// \file
/// \ingroup tutorial_gl
/// Display a 3D histogram using GL (box option).
///
/// \macro_image(nobatch)
/// \macro_code
///
/// \author  Timur Pocheptsov

void glh3c()
{
  gStyle->SetCanvasPreferGL(kTRUE);


  TGLTH3Composition * comp = new TGLTH3Composition;
  TH3F * h1 = new TH3F("h1", "h1", 10, -1., 1., 10, -1., 1., 10, -1., 1.);
  TF3 * g3 = new TF3("gaus3","xyzgaus");
  g3->SetParameters(1,0,1,0,1,0,1);
  h1->FillRandom("gaus3");
  h1->SetFillColor(kRed);
  TH3F * h2 = new TH3F("h2", "h2", 10, -1., 1., 10, -1., 1., 10, -1., 1.);
  TF3 * l3 = new TF3("landau3","landau(x,[0],[1],[2])*gaus(y,1,[3],[4])*gaus(z,1,[3],[4])");
  l3->SetParameters(1,0,1,0.,0.5);
  h2->FillRandom("landau3");
  h2->SetFillColor(kGreen);
  TH3F * h3 = new TH3F("h3", "h3", 10, -1., 1., 10, -1., 1., 10, -1., 1.);
  TF3 * gx = new TF3("gaus1","gaus(x)");
  gx->SetParameters(1,0,1);
  h3->FillRandom("gaus1");
  h3->SetFillColor(kBlue);

  comp->AddTH3(h1);
  comp->AddTH3(h2, TGLTH3Composition::kSphere);
  comp->AddTH3(h3);

  comp->Draw();

  TPaveLabel *title = new TPaveLabel(0.04, 0.86, 0.96, 0.98,
                                     "TH3 composition.");
  title->SetFillColor(32);
  title->Draw();
}
