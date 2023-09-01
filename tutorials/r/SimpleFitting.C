/// \file
/// \ingroup tutorial_r
///
/// Create an exponential fitting
/// The idea is to create a set of numbers x,y with  the function x^3 and some noise from ROOT,
/// fit the function to get the exponent (which must be near 3) and plot the points with noise,
/// the known function and the fitted function
///
/// \macro_code
///
/// \author Omar Zapata

#include<TRInterface.h>
#include<TRandom.h>

TCanvas *SimpleFitting(){
   TCanvas *c1 = new TCanvas("c1","Curve Fitting",700,500);
   c1->SetGrid();

   // draw a frame to define the range
   TMultiGraph *mg = new TMultiGraph();

   // create the first plot (points with gaussian noise)
   const Int_t n = 24;
   Double_t x1[n] ;
   Double_t y1[n] ;
   //Generate the points along a X^3 with noise
   TRandom rg;
   rg.SetSeed(520);
   for (Int_t i = 0; i < n; i++) {
      x1[i] = rg.Uniform(0, 1);
      y1[i] = TMath::Power(x1[i], 3) + rg.Gaus() * 0.06;
   }

   TGraph *gr1 = new TGraph(n,x1,y1);
   gr1->SetMarkerColor(kBlue);
   gr1->SetMarkerStyle(8);
   gr1->SetMarkerSize(1);
   mg->Add(gr1);

      // create the second plot
   TF1 *f_known=new TF1("f_known","pow(x,3)",0,1);
   TGraph *gr2 = new TGraph(f_known);
   gr2->SetMarkerColor(kRed);
   gr2->SetMarkerStyle(8);
   gr2->SetMarkerSize(1);
   mg->Add(gr2);
   //passing data to Rfot fitting
   ROOT::R::TRInterface &r=ROOT::R::TRInterface::Instance();
   r["x"]<<TVectorD(n, x1);
   r["y"]<<TVectorD(n, y1);
   //creating a R data frame
   r<<"ds<-data.frame(x=x,y=y)";
   //fitting x and y to X^power using Nonlinear Least Squares
   r<<"m <- nls(y ~ I(x^power),data = ds, start = list(power = 1),trace = T)";
   //getting the exponent
   Double_t power;
   r["summary(m)$coefficients[1]"]>>power;

   TF1 *f_fitted=new TF1("f_fitted","pow(x,[0])",0,1);
   f_fitted->SetParameter(0,power);
   //plotting the fitted function
   TGraph *gr3 = new TGraph(f_fitted);
   gr3->SetMarkerColor(kGreen);
   gr3->SetMarkerStyle(8);
   gr3->SetMarkerSize(1);

   mg->Add(gr3);
   mg->Draw("ap");

   //displaying basic results
   TPaveText *pt = new TPaveText(0.1,0.6,0.5,0.9,"brNDC");
   pt->SetFillColor(18);
   pt->SetTextAlign(12);
   pt->AddText("Fitting x^power ");
   pt->AddText(" \"Blue\"   Points with gaussian noise to be fitted");
   pt->AddText(" \"Red\"    Known function x^3");
   TString fmsg;
   fmsg.Form(" \"Green\"  Fitted function with power=%.4lf",power);
   pt->AddText(fmsg);
   pt->Draw();
   c1->Update();
   return c1;
}
