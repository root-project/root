//+ Fitting 1-D histograms with minuit2
// @(#)root/minuit2:$Id$
// Author: L. Moneta    10/2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 ROOT Foundation,  CERN/PH-SFT                   *
 *                                                                    *
 **********************************************************************/

#include "TH1.h"
#include "TF1.h"
#include "TCanvas.h"
#include "TStopwatch.h"
#include "TSystem.h"
#include "TRandom3.h"
#include "TVirtualFitter.h"
#include "TPaveLabel.h"
#include "TStyle.h"
#include "TMath.h"
#include "TROOT.h"
#include "TFrame.h"

//#include "Fit/FitConfig.h"


TF1 *fitFcn;
TH1 *histo;

// Quadratic background function
Double_t background(Double_t *x, Double_t *par) {
   return par[0] + par[1]*x[0] + par[2]*x[0]*x[0];
}

// Lorenzian Peak function
Double_t lorentzianPeak(Double_t *x, Double_t *par) {
   return (0.5*par[0]*par[1]/TMath::Pi()) / 
   TMath::Max( 1.e-10,(x[0]-par[2])*(x[0]-par[2]) + .25*par[1]*par[1]);
}

// Sum of background and peak function
Double_t fitFunction(Double_t *x, Double_t *par) {
  return background(x,par) + lorentzianPeak(x,&par[3]);
}

void DoFit(const char* fitter, TVirtualPad *pad, Int_t npass) {   
   printf("\n*********************************************************************************\n");
   printf("\t %s \n",fitter);
   printf("*********************************************************************************\n");

   gRandom = new TRandom3();
   TStopwatch timer;
   //   timer.Start();
   TVirtualFitter::SetDefaultFitter(fitter);
   //ROOT::Fit::FitConfig::SetDefaultMinimizer(fitter);
   pad->SetGrid();
   pad->SetLogy();
   fitFcn->SetParameters(1,1,1,6,.03,1);
   fitFcn->Update();
   std::string title = std::string(fitter) + " fit bench";
   histo = new TH1D(fitter,title.c_str(),200,0,3);
         
   timer.Start();
   for (Int_t pass=0;pass<npass;pass++) {
      if (pass%100 == 0) printf("pass : %d\n",pass);
      fitFcn->SetParameters(1,1,1,6,.03,1);
      for (Int_t i=0;i<5000;i++) {
         histo->Fill(fitFcn->GetRandom());
      }
      histo->Fit(fitFcn,"Q0");
   }

   histo->Fit(fitFcn,"EV");
   timer.Stop();

   (histo->GetFunction("fitFcn"))->SetLineColor(kRed+3);
   gPad->SetFillColor(kYellow-10);


   Double_t cputime = timer.CpuTime();
   printf("%s, npass=%d  : RT=%7.3f s, Cpu=%7.3f s\n",fitter,npass,timer.RealTime(),cputime);
   TPaveLabel *p = new TPaveLabel(0.45,0.7,0.88,0.8,Form("%s CPU= %g s",fitter,cputime),"brNDC");
   p->Draw();
   p->SetTextColor(kRed+3);
   p->SetFillColor(kYellow-8);
   pad->Update();
}

void minuit2FitBench(Int_t npass=20) {
   TH1::AddDirectory(kFALSE);
   TCanvas *c1 = new TCanvas("FitBench","Fitting Demo",10,10,900,900);
   c1->Divide(2,2);
   c1->SetFillColor(kYellow-9);
   // create a TF1 with the range from 0 to 3 and 6 parameters
   fitFcn = new TF1("fitFcn",fitFunction,0,3,6);
   fitFcn->SetNpx(200);
   gStyle->SetOptFit();
   gStyle->SetStatY(0.6);
    
   //with Minuit
   c1->cd(1);
   DoFit("Minuit",gPad,npass);
   
   //with Fumili
   c1->cd(2);
   DoFit("Fumili",gPad,npass);

   //with Minuit2
   c1->cd(3);
   DoFit("Minuit2",gPad,npass);
   
   //with Fumili2
   c1->cd(4);
   DoFit("Fumili2",gPad,npass);
   
   c1->SaveAs("FitBench.root");
}
