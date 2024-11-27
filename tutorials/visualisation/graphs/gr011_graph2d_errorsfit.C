/// \file
/// \ingroup tutorial_graphs
/// \notebook
/// Create, draw and fit a TGraph2DErrors. See the [TGraph2DErrors documentation](https://root.cern/doc/master/classTGraph2DErrors.html)
///
/// \macro_image
/// \macro_code
/// \author Olivier Couet

#include <TMath.h>
#include <TGraph2DErrors.h>
#include <TRandom.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TF2.h>

void gr011_graph2d_errorsfit()
{
   TCanvas *c1 = new TCanvas("c1");

   Double_t rnd, x, y, z, ex, ey, ez;
   Double_t e = 0.3;
   Int_t nd = 500;

   // To generate some random data to put into the graph
   TRandom r;
   TF2  *f2 = new TF2("f2","1000*(([0]*sin(x)/x)*([1]*sin(y)/y))+200",-6,6,-6,6);
   f2->SetParameters(1,1);

   TGraph2DErrors *dte = new TGraph2DErrors(nd);

   // Fill the 2D graph. It was created only specifying the number of points, so all
   // elements are empty. We now "fill" the values and errors with SetPoint and SetPointError.
   // Note that the first point has index zero
   Double_t zmax = 0;
   for (Int_t i=0; i<nd; i++) {
      f2->GetRandom2(x,y);
      rnd = r.Uniform(-e,e); // Generate a random number in [-e,e]
      z = f2->Eval(x,y)*(1+rnd);
      if (z>zmax) zmax = z;
      dte->SetPoint(i,x,y,z);
      ex = 0.05*r.Rndm();
      ey = 0.05*r.Rndm();
      ez = TMath::Abs(z*rnd);
      dte->SetPointError(i,ex,ey,ez);
   }
   // If the fit is not needed, just draw dte here and skip the lines below
   // dte->Draw("A p0");

   // To do the fit we use a function, in this example the same f2 from above
   f2->SetParameters(0.5,1.5);
   dte->Fit(f2);
   TF2 *fit2 = (TF2*)dte->FindObject("f2");
   fit2->SetTitle("Minuit fit result on the Graph2DErrors points");
   fit2->SetMaximum(zmax);
   gStyle->SetHistTopMargin(0);
   fit2->SetLineColor(1);
   fit2->SetLineWidth(1);
   fit2->Draw("surf1");
   dte->Draw("same p0");
}
