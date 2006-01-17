// @(#)root/test:$name:  $:$id: stressSpectrum.cxx,v 1.15 2002/10/25 10:47:51 rdm exp $
// Author: Rene Brun 17/01/2006

/////////////////////////////////////////////////////////////////
//
//    TSPectrum test suite
//    ====================
//
// This stress program tests many elements of the TSpectrum, TSpectrum2 classes.
//
// To run in batch, do
//   stressSpectrum        : run 100 experiments with graphics (default)
//   stressSpectrum 1000   : run 1000 experiments with graphics
//   stressSpectrum -b 100 : run 100 experiments in batch mode
//
// To run interactively, do
// root -b
//  Root > .x stressSpectrum.cxx      : run 100 experiments with graphics (default)
//  Root > .x stressSpectrum.cxx(20)  : run 20 experiments
//  Root > .x stressSpectrum.cxx+(30) : run 30 experiments via ACLIC
//
// Several tests are run sequentially. Each test will produce one line (Test OK or Test FAILED) .
// At the end of the test a table is printed showing the global results
// Real Time and Cpu Time.
// One single number (ROOTMARKS) is also calculated showing the relative
// performance of your machine compared to a reference machine
// a Pentium IV 3.0 Ghz) with 512 MBytes of memory
// and 120 GBytes IDE disk.
//
// An example of output when all the tests run OK is shown below:
//
//////////////////////////////////////////////////////////////////////////
//                                                                      //
//****************************************************************************
//*  Starting  stress S P E C T R U M                                        *
//****************************************************************************
//Peak1 ; found = 64.31/ 68.94, good = 67.26/ 67.11, ghost = 6.79/ 8.56,--- OK
//****************************************************************************
//stressSpectrum: Real Time =  19.37 seconds Cpu Time =  19.37 seconds
//****************************************************************************
//*  ROOTMARKS = 413.1   *  Root5.09/01   20051216/1229//Peak1 ; found = 34.22/ 68.94, good = 69.20/ 67.11, ghost =10.60/ 8.56,--- OK
//****************************************************************************

#include "TApplication.h"
#include "TBenchmark.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TF1.h"
#include "TRandom.h"
#include "TSpectrum.h"
#include "TStyle.h"
#include "Riostream.h"
   
Int_t npeaks;
Double_t fpeaks(Double_t *x, Double_t *par) {
   Double_t result = par[0] + par[1]*x[0];
   for (Int_t p=0;p<npeaks;p++) {
      Double_t norm  = par[3*p+2];
      Double_t mean  = par[3*p+3];
      Double_t sigma = par[3*p+4];
      result += norm*TMath::Gaus(x[0],mean,sigma);
   }
   return result;
}
void findPeaks(Int_t pmin, Int_t pmax, Int_t &nfound, Int_t &ngood, Int_t &nghost) {
   npeaks = (Int_t)gRandom->Uniform(pmin,pmax);
   Int_t nbins = 500;
   Double_t dxbins = 2;
   TH1F *h = new TH1F("h","test",nbins,0,nbins*dxbins);
   //generate n peaks at random
   Double_t par[3000];
   par[0] = 0.8;
   par[1] = -0.6/1000;
   Int_t p,pf;
   for (p=0;p<npeaks;p++) {
      par[3*p+2] = 1;
      par[3*p+3] = 10+gRandom->Rndm()*(nbins-20)*dxbins;
      par[3*p+4] = 3+2*gRandom->Rndm();
   }
   TF1 *f = new TF1("f",fpeaks,0,nbins*dxbins,2+3*npeaks);
   f->SetNpx(1000);
   f->SetParameters(par);
   h->FillRandom("f",200000);
   TSpectrum *s = new TSpectrum(4*npeaks);
   nfound = s->Search(h,2,"goff");
   //Search found peaks
   ngood = 0;
   Float_t *xpeaks = s->GetPositionX();
   for (p=0;p<npeaks;p++) {
      for (Int_t pf=0;pf<nfound;pf++) {
         Double_t dx = TMath::Abs(xpeaks[pf] - par[3*p+3]);
         if (dx <dxbins) ngood++;
      }
   }
   //Search ghost peaks
   nghost = 0;
   for (pf=0;pf<nfound;pf++) {
      Int_t nf=0;
      for (Int_t p=0;p<npeaks;p++) {
         Double_t dx = TMath::Abs(xpeaks[pf] - par[3*p+3]);
         if (dx <dxbins) nf++;
      }
      if (nf == 0) nghost++;
   }
   delete f;
   delete h;
   delete s;
}
#ifndef __CINT__
void stressSpectrum(Int_t ntimes) {
#else
void stressSpectrum(Int_t ntimes=100) {
#endif
   cout << "****************************************************************************" <<endl;
   cout << "*  Starting  stress S P E C T R U M                                        *" <<endl;
   cout << "****************************************************************************" <<endl;
   gBenchmark->Start("stressSpectrum");
   Int_t pmin = 5;
   Int_t pmax = 55;
   TCanvas *c1 = new TCanvas("c1","Spectrum results",10,10,800,800);
   c1->Divide(2,2);
   gStyle->SetOptFit();
   TH1F *hpeaks = new TH1F("hpeaks","Number of peaks",pmax-pmin,pmin,pmax);
   TH1F *hfound = new TH1F("hfound","% peak founds",100,0,100);
   TH1F *hgood  = new TH1F("hgood", "% good peaks",100,0,100);
   TH1F *hghost = new TH1F("hghost","% ghost peaks",100,0,100);
   Int_t nfound,ngood,nghost;
   Double_t norm;
   for (Int_t i=0;i<ntimes;i++) {
      findPeaks(pmin,pmax,nfound,ngood,nghost);
      hpeaks->Fill(npeaks);
      hfound->Fill(100*Double_t(nfound)/Double_t(npeaks));
      hgood->Fill(100*Double_t(ngood)/Double_t(npeaks));
      hghost->Fill(100*Double_t(nghost)/Double_t(npeaks));
      //printf("npeaks = %d, nfound = %d, ngood = %d, nghost = %d\n",npeaks,nfound,ngood,nghost);
   }
   c1->cd(1);
   hpeaks->Fit("pol1","lq");
   c1->cd(2);
   hfound->Fit("gaus","lq");
   c1->cd(3);
   hgood->Fit("gaus","lq");
   c1->cd(4);
   hghost->Fit("gaus","lq","",0,30);
   c1->cd();
   Double_t p1  = hfound->GetFunction("gaus")->GetParameter(1);
   Double_t ep1 = hfound->GetFunction("gaus")->GetParError(1);
   Double_t p2  = hgood->GetFunction("gaus")->GetParameter(1);
   Double_t ep2 = hgood->GetFunction("gaus")->GetParError(1);
   Double_t p3  = hghost->GetFunction("gaus")->GetParameter(1);
   Double_t ep3 = hghost->GetFunction("gaus")->GetParError(1);
   Double_t p1ref = 68.94; //ref numbers obtained with ntimes=1000
   Double_t p2ref = 67.11;
   Double_t p3ref =  8.56;
   Double_t reftime100 = 10.0; //pcbrun compiled
      
   //printf("p1=%g+-%g, p2=%g+-%g, p3=%g+-%g\n",p1,ep1,p2,ep2,p3,ep3);

   gBenchmark->Stop ("stressSpectrum");
   Double_t ct = gBenchmark->GetCpuTime("stressSpectrum");
   const Double_t rootmarks = 800*reftime100*ntimes/(100*ct);
   char sok[20];
   if (TMath::Abs(p1ref-p1) < 2*ep1 && TMath::Abs(p2ref-p2) < 2*ep2  && TMath::Abs(p3ref-p3) < 2*ep3 ) {
      sprintf(sok,"OK");
   } else {
      sprintf(sok,"failed");
   }
   printf("Peak1 ; found =%6.2f/%6.2f, good =%6.2f/%6.2f, ghost =%5.2f/%5.2f,--- %s\n",
          p1,p1ref,p2,p2ref,p3,p3ref,sok);
   printf("****************************************************************************\n");

   gBenchmark->Print("stressSpectrum");
   printf("****************************************************************************\n");
   printf("*  ROOTMARKS =%6.1f   *  Root%-8s  %d/%d\n",rootmarks,gROOT->GetVersion(),
         gROOT->GetVersionDate(),gROOT->GetVersionTime());
   printf("****************************************************************************\n");
}
#ifndef __CINT__

int main(int argc, char **argv)
{
   TApplication theApp("App", &argc, argv);
   gBenchmark = new TBenchmark();
   Int_t ntimes = 100;
   if (argc > 1)  ntimes = atoi(argv[1]);
   stressSpectrum(ntimes);
   return 0;
}

#endif
