/// \file
/// \ingroup tutorial_fit
/// \notebook
/// Generate points distributed with some errors around a circle
/// Fit a circle through the points and draw
/// To run the script, do, eg
///
/// ~~~{.cpp}
///   root > .x fitCircle.C   (10000 points by default)
///   root > .x fitCircle.C(100);  (with only 100 points
///   root > .x fitCircle.C++(100000);  with ACLIC
/// ~~~
///
/// \macro_image
/// \macro_output
/// \macro_code
///
/// \author Rene Brun

#include "TCanvas.h"
#include "TRandom3.h"
#include "TGraph.h"
#include "TMath.h"
#include "TArc.h"
#include "Fit/Fitter.h"
#include <Math/Functor.h>

//____________________________________________________________________
void fitCircle(int n=10000) {
   //generates n points around a circle and fit them
   TCanvas *c1 = new TCanvas("c1","c1",600,600);
   c1->SetGrid();
   TGraph* gr = new TGraph(n);
   if (n> 999) gr->SetMarkerStyle(1);
   else        gr->SetMarkerStyle(3);
   TRandom3 r;
   double x,y;
   for (int i=0;i<n;i++) {
      r.Circle(x,y,r.Gaus(4,0.3));
      gr->SetPoint(i,x,y);
   }
   c1->DrawFrame(-5,-5,5,5);
   gr->Draw("p");


   auto chi2Function = [&](const double *par) {
      //minimisation function computing the sum of squares of residuals
      // looping at the graph points
      int np = gr->GetN();
      double f = 0;
      double *x = gr->GetX();
      double *y = gr->GetY();
      for (int i=0;i<np;i++) {
         double u = x[i] - par[0];
         double v = y[i] - par[1];
         double dr = par[2] - std::sqrt(u*u+v*v);
         f += dr*dr;
      }
      return f;
   };

   // wrap chi2 function in a function object for the fit
   // 3 is the number of fit parameters (size of array par)
   ROOT::Math::Functor fcn(chi2Function,3);
   ROOT::Fit::Fitter  fitter;


   double pStart[3] = {0,0,1};
   fitter.SetFCN(fcn, pStart);
   fitter.Config().ParSettings(0).SetName("x0");
   fitter.Config().ParSettings(1).SetName("y0");
   fitter.Config().ParSettings(2).SetName("R");

   // do the fit
   bool ok = fitter.FitFCN();
   if (!ok) {
      Error("line3Dfit","Line3D Fit failed");
   }

   const ROOT::Fit::FitResult & result = fitter.Result();
   result.Print(std::cout);

   //Draw the circle on top of the points
   TArc *arc = new TArc(result.Parameter(0),result.Parameter(1),result.Parameter(2));
   arc->SetLineColor(kRed);
   arc->SetLineWidth(4);
   arc->SetFillStyle(0);
   arc->Draw();
}
