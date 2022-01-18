/// \file
/// \ingroup tutorial_fit
/// \notebook
/// Fitting a 2-D histogram
/// This tutorial illustrates :
///  - how to create a 2-d function
///  - fill a 2-d histogram randomly from this function
///  - fit the histogram
///  - display the fitted function on top of the histogram
///
/// This example can be executed via the interpreter or ACLIC
///
/// ~~~{.cpp}
///   root > .x fit2.C
///   root > .x fit2.C++
/// ~~~
///
/// \macro_image
/// \macro_output
/// \macro_code
///
/// \author Rene Brun

#include "TF2.h"
#include "TH2.h"
#include "TMath.h"

double g2(double *x, double *par) {
   double r1 = double((x[0]-par[1])/par[2]);
   double r2 = double((x[1]-par[3])/par[4]);
   return par[0]*TMath::Exp(-0.5*(r1*r1+r2*r2));
}
double fun2(double *x, double *par) {
   double *p1 = &par[0];
   double *p2 = &par[5];
   double *p3 = &par[10];
   double result = g2(x,p1) + g2(x,p2) + g2(x,p3);
   return result;
}

void fit2() {
   const int npar = 15;
   double f2params[npar] =
      {100,-3,3,-3,3,160,0,0.8,0,0.9,40,4,0.7,4,0.7};
   TF2 *f2 = new TF2("f2",fun2,-10,10,-10,10, npar);
   f2->SetParameters(f2params);

   //Create an histogram and fill it randomly with f2
   TH2F *h2 = new TH2F("h2","from f2",40,-10,10,40,-10,10);
   int nentries = 100000;
   h2->FillRandom("f2",nentries);
   //Fit h2 with original function f2
   float ratio = 4*nentries/100000;
   f2params[ 0] *= ratio;
   f2params[ 5] *= ratio;
   f2params[10] *= ratio;
   f2->SetParameters(f2params);
   h2->Fit("f2");
   f2->Draw("cont1 same");
}
