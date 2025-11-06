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
   double x1[n];
   double y1[n];
   double c1[n];
   double s1[n];
   double x2[n];
   double y2[n];
   double c2[n];
   double s2[n];

   // Define four random data sets
   auto r  = new TRandom();
   for (int i=0; i<n; i++) {
      x1[i] = 100*r->Rndm(i);
      y1[i] = 200*r->Rndm(i);
      c1[i] = 300*r->Rndm(i);
      s1[i] = 400*r->Rndm(i);
      x2[i] = 100*r->Rndm(i);
      y2[i] = 200*r->Rndm(i);
      c2[i] = 100*r->Rndm(i);
      s2[i] = 200*r->Rndm(i);
   }

   auto scatter1 = new TScatter(n, x1, y1, c1, s1);
   scatter1->SetMarkerStyle(20);
   scatter1->SetTitle("Scatter plot title;X title;Y title;Z title");
   scatter1->GetXaxis()->SetRangeUser(20.,90.);
   scatter1->GetYaxis()->SetRangeUser(55.,90.);
   scatter1->GetZaxis()->SetRangeUser(10.,200.);
   // an alternative way to zoom the Z-axis:
   // scatter->GetHistogram()->SetMinimum(10);
   // scatter->GetHistogram()->SetMaximum(200);
   scatter1->Draw("A");

   auto scatter2 = new TScatter(n, x2, y2, c2, s2);
   scatter2->SetMarkerStyle(21);
   scatter2->Draw();
}
