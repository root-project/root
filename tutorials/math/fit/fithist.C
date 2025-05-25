/// \file
/// \ingroup tutorial_fit
/// \notebook -js
/// Example of fit where the model is histogram + function
///
/// \macro_image
/// \macro_output
/// \macro_code
///
/// \author Rene Brun

#include <TF1.h>
#include <TFile.h>
#include <TH1F.h>

TH1F *background;
void histgen() {
   //generate the histogram background and save it to a file
   //background taken as linearly decreasing

   TF1 f1("f1","pol1",0,10);
   f1.SetParameters(5,-0.5);
   TH1F h("background","linear background",100,0,10);
   h.FillRandom("f1",10000);
   TFile f("background.root","recreate");
   //save the background histogram
   h.Write();
   //superimpose a Gaussian signal to the background histogram
   TF1 f2("f2","gaus",0,10);
   f2.SetParameters(1,6,0.5);
   h.FillRandom("f2",2000);
   h.SetName("result");
   h.Write();
}

double ftotal(double *x, double *par) {
   double xx = x[0];
   int bin = background->GetXaxis()->FindBin(xx);
   double br = par[3]*background->GetBinContent(bin);
   double arg = (xx-par[1])/par[2];
   double sr = par[0]*TMath::Exp(-0.5*arg*arg);
   return sr + br;
}
void fithist() {
   //fit function ftotal to signal + background

   histgen();

   TFile *f = new TFile("background.root");
   background = (TH1F*)f->Get("background"); //pointer used in ftotal
   TH1F *result = (TH1F*)f->Get("result");

   TF1 *ftot = new TF1("ftot",ftotal,0,10,4);
   double norm = result->GetMaximum();
   ftot->SetParameters(0.5*norm,5,.2,norm);
   ftot->SetParLimits(0,.3*norm,norm);

   result->Fit("ftot","b");
}
