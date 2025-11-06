/// \file
/// \ingroup tutorial_graphs
/// \notebook
/// \preview Draw a 2D scatter plot.
///
/// \macro_image
/// \macro_code
///
/// \author Olivier Couet

void gr019_scatter2d()
{
   auto canvas = new TCanvas("canvas","canvas");
   canvas->SetRightMargin(0.14);
   gStyle->SetPalette(kBird, 0, 0.6); // define a transparent palette

   const int n = 50;
   double x1[n];
   double y1[n];
   double z1[n];
   double c1[n];
   double s1[n];
   double x2[n];
   double y2[n];
   double z2[n];
   double c2[n];
   double s2[n];

   // Define four random data set
   auto r  = new TRandom();
   for (int i=0; i<n; i++) {
      x1[i] =   100*r->Rndm(i);
      y1[i] =   200*r->Rndm(i);
      z1[i] =    10*r->Rndm(i);
      c1[i] = 10000*r->Rndm(i);
      s1[i] = 10000*r->Rndm(i);
      x2[i] =   100*r->Rndm(i);
      y2[i] =   200*r->Rndm(i);
      z2[i] =    10*r->Rndm(i);
      c2[i] =  5000*r->Rndm(i);
      s2[i] =   100*r->Rndm(i);
   }
   c1[0] = 1;

   auto scatter1 = new TScatter2D(n, x1, y1, z1, c1, s1);
   scatter1->SetTitle("Scatter plot title;X title;Y title;Z title;C title");
   scatter1->SetMarkerStyle(20);

   auto scatter2 = new TScatter2D(n, x2, y2, z2, c2, s2);
   scatter2->SetMarkerStyle(21);

   canvas->SetLogx();
   scatter1->Draw("logc");
   scatter2->Draw("SAME");
}
