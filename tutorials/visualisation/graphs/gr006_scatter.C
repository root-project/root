/// \file
/// \ingroup tutorial_graphs
/// \notebook
/// \preview Draw a scatter plot for 4 variables, mapped to: x, y, marker colour and marker size.
///
/// TScatter is available since ROOT v.6.30. See the [TScatter documentation](https://root.cern/doc/master/classTScatter.html)
///
/// \macro_image
/// \macro_code
/// \author Olivier Couet

void gr006_scatter()
{
   auto canvas = new TCanvas();
   canvas->SetRightMargin(0.14);
   gStyle->SetPalette(kBird, 0, 0.6); // define a transparent palette

   const int n = 175;
   double x[n];
   double y[n];
   double c[n];
   double s[n];

   // Define four random data sets
   auto r  = new TRandom();
   for (int i=0; i<n; i++) {
      x[i] = 100*r->Rndm(i);
      y[i] = 200*r->Rndm(i);
      c[i] = 300*r->Rndm(i);
      s[i] = 400*r->Rndm(i);
   }

   auto scatter = new TScatter(n, x, y, c, s);
   scatter->SetMarkerStyle(20);
   scatter->SetTitle("Scatter plot title;X title;Y title;Z title");
   scatter->GetXaxis()->SetRangeUser(20.,90.);
   scatter->GetYaxis()->SetRangeUser(55.,90.);
   scatter->GetZaxis()->SetRangeUser(10.,200.);
   scatter->Draw("A");
}
