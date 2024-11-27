/// \file
/// \ingroup tutorial_graphs
/// \notebook
/// Create and draw a polar graph with errors and polar axis in radians (PI fractions).
/// See the [TGraphPolar documentation](https://root.cern/doc/master/classTGraphPolar.html)
///
/// Since TGraphPolar is a TGraphErrors, it is painted with
/// [TGraphPainter](https://root.cern/doc/master/classTGraphPainter.html) options.
///
/// With GetPolargram we retrieve the polar axis to format it; see the
/// [TGraphPolargram documentation](https://root.cern/doc/master/classTGraphPolargram.html)
///
/// \macro_image
/// \macro_code
/// \author Olivier Couet

void gr013_polar2()
{
   TCanvas * CPol = new TCanvas("CPol","TGraphPolar Example",500,500);

   Double_t theta[8];
   Double_t radius[8];
   Double_t etheta[8];
   Double_t eradius[8];

   for (int i=0; i<8; i++) {
      theta[i]   = (i+1)*(TMath::Pi()/4.);
      radius[i]  = (i+1)*0.05;
      etheta[i]  = TMath::Pi()/8.;
      eradius[i] = 0.05;
   }

   TGraphPolar * grP1 = new TGraphPolar(8, theta, radius, etheta, eradius);
   grP1->SetTitle("");

   grP1->SetMarkerStyle(20);
   grP1->SetMarkerSize(2.);
   grP1->SetMarkerColor(4);
   grP1->SetLineColor(2);
   grP1->SetLineWidth(3);
   // Draw with polymarker and errors
   grP1->Draw("PE");

   // To format the polar axis, we retrieve the TGraphPolargram.
   // First update the canvas, otherwise GetPolargram returns 0
   CPol->Update();
   if (grP1->GetPolargram())
      grP1->GetPolargram()->SetToRadian(); // tell ROOT that the theta values are in radians
}
