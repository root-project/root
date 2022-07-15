/// \file
/// \ingroup tutorial_fit
/// \notebook -js
/// Example of fitting with a linear function, using TLinearFitter
/// This example is for a TGraphErrors, but it can also be used
/// when fitting a histogram, a TGraph2D or a TMultiGraph
///
/// \macro_image
/// \macro_output
/// \macro_code
///
/// \author Anna Kreshuk

#include "TGraphErrors.h"
#include "TF1.h"
#include "TRandom.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TMath.h"


void makePoints(int n, double *x, double *y, double *e, int p);

void fitLinear()
{
   int n = 40;
   double *x = new double[n];
   double *y = new double[n];
   double *e = new double[n];
   TCanvas *myc = new TCanvas("myc",
      "Fitting 3 TGraphErrors with linear functions");
   myc->SetGrid();

   //Generate points along a 3rd degree polynomial:
   makePoints(n, x, y, e, 3);
   TGraphErrors *gre3 = new TGraphErrors(n, x, y, 0, e);
   gre3->Draw("a*");
   //Fit the graph with the predefined "pol3" function
   gre3->Fit("pol3");
   //Access the fit results
   TF1 *f3 = gre3->GetFunction("pol3");
   f3->SetLineWidth(1);

   //Generate points along a sin(x)+sin(2x) function
   makePoints(n, x, y, e, 2);
   TGraphErrors *gre2=new TGraphErrors(n, x, y, 0, e);
   gre2->Draw("*same");
   gre2->SetMarkerColor(kBlue);
   gre2->SetLineColor(kBlue);
   //The fitting function can be predefined and passed to the Fit function
   //The "++" mean that the linear fitter should be used, and the following
   //formula is equivalent to "[0]*sin(x) + [1]*sin(2*x)"
   //A function, defined this way, is in no way different from any other TF1,
   //it can be evaluated, drawn, you can get its parameters, etc.
   //The fit result (parameter values, parameter errors, chisquare, etc) are
   //written into the fitting function.
   TF1 *f2 = new TF1("f2", "sin(x) ++ sin(2*x)", -2, 2);
   gre2->Fit(f2);
   f2 = gre2->GetFunction("f2");
   f2->SetLineColor(kBlue);
   f2->SetLineWidth(1);

   //Generate points along a -2+exp(-x) function
   makePoints(n, x, y, e, 4);
   TGraphErrors *gre4=new TGraphErrors(n, x, y, 0, e);
   gre4->Draw("*same");
   gre4->SetMarkerColor(kRed);
   gre4->SetLineColor(kRed);
   //If you don't want to define the function, you can just pass the string
   //with the the formula:
   gre4->Fit("1 ++ exp(-x)");
   //Access the fit results:
   TF1 *f4 = gre4->GetFunction("1 ++ exp(-x)");
   f4->SetName("f4");
   f4->SetLineColor(kRed);
   f4->SetLineWidth(1);

   TLegend *leg = new TLegend(0.3, 0.7, 0.65, 0.9);
   leg->AddEntry(gre3, " -7 + 2*x*x + x*x*x", "p");
   leg->AddEntry(gre2, "sin(x) + sin(2*x)", "p");
   leg->AddEntry(gre4, "-2 + exp(-x)", "p");
   leg->Draw();

}

void makePoints(int n, double *x, double *y, double *e, int p)
{
  int i;
  TRandom r;

  if (p==2) {
    for (i=0; i<n; i++) {
      x[i] = r.Uniform(-2, 2);
      y[i]=TMath::Sin(x[i]) + TMath::Sin(2*x[i]) + r.Gaus()*0.1;
      e[i] = 0.1;
    }
  }
  if (p==3) {
    for (i=0; i<n; i++) {
      x[i] = r.Uniform(-2, 2);
      y[i] = -7 + 2*x[i]*x[i] + x[i]*x[i]*x[i]+ r.Gaus()*0.1;
      e[i] = 0.1;
    }
  }
  if (p==4) {
    for (i=0; i<n; i++) {
      x[i] = r.Uniform(-2, 2);
      y[i]=-2 + TMath::Exp(-x[i]) + r.Gaus()*0.1;
      e[i] = 0.1;
    }
  }
}
