/// \file
/// \ingroup tutorial_hist
/// \notebook
/// \preview Example of macro illustrating how to superimpose two histograms
/// with different scales in the "same" pad.
///
/// \macro_image
/// \macro_code
///
/// \date November 2024
/// \author Rene Brun

void hist010_TH1_two_scales()
{
   TCanvas *c1 = new TCanvas("c1", "hists with different scales", 600, 400);

   // create/fill draw h1
   gStyle->SetOptStat(kFALSE);
   TH1D *h1 = new TH1D("h1", "my histogram", 100, -3, 3);
   TRandom3 rng;
   for (int i = 0; i < 10000; i++)
      h1->Fill(rng.Gaus(0, 1));
   h1->Draw();
   c1->Update();

   // create hint1 filled with the bins integral of h1
   TH1D *hint1 = new TH1D("hint1", "h1 bins integral", 100, -3, 3);
   double sum = 0;
   for (int i = 1; i <= 100; i++) {
      sum += h1->GetBinContent(i);
      hint1->SetBinContent(i, sum);
   }

   // scale hint1 to the pad coordinates
   double rightmax = 1.1 * hint1->GetMaximum();
   double scale = gPad->GetUymax() / rightmax;
   hint1->SetLineColor(kRed);
   hint1->Scale(scale);
   hint1->Draw("same");

   // draw an axis on the right side
   TGaxis *axis =
      new TGaxis(gPad->GetUxmax(), gPad->GetUymin(), gPad->GetUxmax(), gPad->GetUymax(), 0, rightmax, 510, "+L");
   axis->SetLineColor(kRed);
   axis->SetLabelColor(kRed);
   axis->Draw();
}
