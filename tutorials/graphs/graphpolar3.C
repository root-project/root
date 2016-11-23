/// \file
/// \ingroup tutorial_graphs
/// \notebook
/// Create and draw a polar graph with PI axis using a TF1.
///
/// \macro_image
/// \macro_code
///
/// \author Olivier Couet

void graphpolar3()
{
   TCanvas * CPol = new TCanvas("CPol","TGraphPolar Examples",500,500);

   Double_t rmin=0;
   Double_t rmax=TMath::Pi()*2;
   Double_t r[1000];
   Double_t theta[1000];

   TF1 * fp1 = new TF1("fplot","cos(x)",rmin,rmax);
   for (Int_t ipt = 0; ipt < 1000; ipt++) {
      r[ipt] = ipt*(rmax-rmin)/1000+rmin;
      theta[ipt] = fp1->Eval(r[ipt]);
   }
   TGraphPolar * grP1 = new TGraphPolar(1000,r,theta);
   grP1->SetTitle("");
   grP1->SetLineColor(2);
   grP1->Draw("AOL");
}
