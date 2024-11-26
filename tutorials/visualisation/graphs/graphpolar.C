/// \file
/// \ingroup tutorial_graphs
/// \notebook
/// Create and draw a polar graph.
///
/// \macro_image
/// \macro_code
///
/// \author Olivier Couet

void graphpolar()
{
   // Illustrates how to use TGraphPolar

   auto CPol = new TCanvas("CPol", "TGraphPolar Examples", 1200, 600);
   CPol->Divide(2, 1);
   CPol->cd(1);

   Double_t xmin = 0;
   Double_t xmax = TMath::Pi() * 2;

   Double_t x[1000];
   Double_t y[1000];
   Double_t xval1[20];
   Double_t yval1[20];

   TF1 fplot("fplot", "cos(2*x)*cos(20*x)", xmin, xmax);

   for (Int_t ipt = 0; ipt < 1000; ipt++) {
      x[ipt] = ipt * (xmax - xmin) / 1000 + xmin;
      y[ipt] = fplot.Eval(x[ipt]);
   }

   auto grP = new TGraphPolar(1000, x, y);
   grP->SetLineColor(2);
   grP->SetLineWidth(2);
   grP->SetFillStyle(3012);
   grP->SetFillColor(2);
   grP->Draw("ARFL"); // AR - radian coordinates, F - fill, L - line

   for (Int_t ipt = 0; ipt < 20; ipt++) {
      xval1[ipt] = x[1000 / 20 * ipt];
      yval1[ipt] = y[1000 / 20 * ipt];
   }

   auto grP1 = new TGraphPolar(20, xval1, yval1);
   grP1->SetMarkerStyle(29);
   grP1->SetMarkerSize(2);
   grP1->SetMarkerColor(4);
   grP1->SetLineColor(4);
   grP1->Draw("CP"); // C - curve, P - markers

   // Update, otherwise GetPolargram returns 0
   CPol->Update();
   if (grP->GetPolargram()) {
      grP->GetPolargram()->SetTextColor(8);
      grP->GetPolargram()->SetNdivPolar(703);
   }

   CPol->cd(2);
   Double_t x2[30];
   Double_t y2[30];
   Double_t ex[30];
   Double_t ey[30];
   for (Int_t ipt = 0; ipt < 30; ipt++) {
      x2[ipt] = x[1000 / 30 * ipt];
      y2[ipt] = 1.2 + 0.4 * sin(TMath::Pi() * 2 * ipt / 30);
      ex[ipt] = 0.2 + 0.1 * cos(2 * TMath::Pi() / 30 * ipt);
      ey[ipt] = 0.2;
   }

   auto grPE = new TGraphPolar(30, x2, y2, ex, ey);
   grPE->SetMarkerStyle(22);
   grPE->SetMarkerSize(1.5);
   grPE->SetMarkerColor(5);
   grPE->SetLineColor(6);
   grPE->SetLineWidth(2);
   grPE->Draw("AREP"); // AR - radian, E - errors, P - markers

   // Update, otherwise GetPolargram returns 0
   CPol->Update();
   if (grPE->GetPolargram())
      grPE->GetPolargram()->SetTextSize(0.03);
}
