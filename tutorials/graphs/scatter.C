/// \file
/// \ingroup tutorial_graphs
/// \notebook
/// Draw a scatter plot.
///
/// \macro_image
/// \macro_code
///
/// \author Olivier Couet

void scatter()
{
   auto canvas = new TCanvas();
   gStyle->SetPalette(kBird, 0, 0.6); // define a transparent palette

   const int n = 100;
   double x[n];
   double y[n];
   double c[n];
   double s[n];

   // Define four random data set
   auto r  = new TRandom();
   for (int i=0; i<n; i++) {
      x[i] = 100*r->Rndm(i);
      y[i] = 200*r->Rndm(i);
      c[i] = 300*r->Rndm(i);
      s[i] = 400*r->Rndm(i);
   }

   auto scatter = new TScatter(n, x, y, c, s);
   scatter->SetMarkerStyle(20);
   scatter->SetMarkerColor(kRed);
   scatter->SetTitle("Scatter plot;X;Y");
   scatter->Draw("A");
}
