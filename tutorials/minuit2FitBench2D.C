// @(#)root/minuit2:$Id$
// Author: L. Moneta    10/2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 ROOT Foundation,  CERN/PH-SFT                   *
 *                                                                    *
 **********************************************************************/

#include "TH1.h"
#include "TF1.h"
#include "TH2D.h"
#include "TF2.h"
#include "TCanvas.h"
#include "TStopwatch.h"
#include "TSystem.h"
#include "TRandom3.h"
#include "TVirtualFitter.h"
#include "TPaveLabel.h"
#include "TStyle.h"


TF2 *fitFcn;
TH2D *histo;

// Quadratic background function
Double_t gaus2D(Double_t *x, Double_t *par) {
   double t1 =   x[0] - par[1];
   double t2 =   x[1] - par[2];
   return par[0]* exp( - 0.5 * (  t1*t1/( par[3]*par[3]) + t2*t2  /( par[4]*par[4] )  ) ) ;    
}

// Sum of background and peak function
Double_t fitFunction(Double_t *x, Double_t *par) {
  return gaus2D(x,par);
}

void fillHisto(int n =10000) { 

  gRandom = new TRandom3();
  for (int i = 0; i < n; ++i) { 
     double x = gRandom->Gaus(2,3);
     double y = gRandom->Gaus(-1,4);
     histo->Fill(x,y,1.);
  }
}

void DoFit(const char* fitter, TVirtualPad *pad, Int_t npass) {   
   TStopwatch timer;
   TVirtualFitter::SetDefaultFitter(fitter);
   pad->SetGrid();
   fitFcn->SetParameters(100,0,0,2,7);
   fitFcn->Update();
         
   timer.Start();
   histo->Fit("fitFcn","0");
   timer.Stop();

   histo->Draw();
   Double_t cputime = timer.CpuTime();
   printf("%s, npass=%d  : RT=%7.3f s, Cpu=%7.3f s\n",fitter,npass,timer.RealTime(),cputime);
   TPaveLabel *p = new TPaveLabel(0.5,0.7,0.85,0.8,Form("%s CPU= %g s",fitter,cputime),"brNDC");
   p->Draw();
   pad->Update();
}

void minuit2FitBench2D(int n = 100000) {
   TH1::AddDirectory(kFALSE);
   TCanvas *c1 = new TCanvas("c1","Fitting Demo",10,10,900,900);
   c1->Divide(2,2);
   // create a TF1 with the range from 0 to 3 and 6 parameters
   fitFcn = new TF2("fitFcn",fitFunction,-10,10,-10,10,5);
   //fitFcn->SetNpx(200);
   gStyle->SetOptFit();
   gStyle->SetStatY(0.6);
    

   histo = new TH2D("h2","2D Gauss",100,-10,10,100,-10,10);
   fillHisto(n);

   int npass=0;

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
}
