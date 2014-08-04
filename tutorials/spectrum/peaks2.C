// Example to illustrate the 2-d peak finder (class TSpectrum2).
// This script generates a random number of 2-d gaussian peaks
// The position of the peaks is found via TSpectrum2
// To execute this example, do
//  root > .x peaks2.C  (generate up to 50 peaks by default)
//  root > .x peaks2.C(10) (generate up to 10 peaks)
//  root > .x peaks2.C+(200) (generate up to 200 peaks via ACLIC)
//
// The script will iterate generating a new histogram having
// between 5 and the maximun number of peaks specified.
// Double Click on the bottom right corner of the pad to go to a new spectrum
// To Quit, select the "quit" item in the canvas "File" menu
//
//Author: Rene Brun

#include "TSpectrum2.h"
#include "TCanvas.h"
#include "TRandom.h"
#include "TH2.h"
#include "TF2.h"
#include "TMath.h"
#include "TROOT.h"

TSpectrum2 *s;
TH2F *h2 = 0;
Int_t npeaks = 30;
Double_t fpeaks2(Double_t *x, Double_t *par) {
   Double_t result = 0.1;
   for (Int_t p=0;p<npeaks;p++) {
      Double_t norm   = par[5*p+0];
      Double_t mean1  = par[5*p+1];
      Double_t sigma1 = par[5*p+2];
      Double_t mean2  = par[5*p+3];
      Double_t sigma2 = par[5*p+4];
      result += norm*TMath::Gaus(x[0],mean1,sigma1)*TMath::Gaus(x[1],mean2,sigma2);
   }
   return result;
}
void findPeak2() {
   printf("Generating histogram with %d peaks\n",npeaks);
   Int_t nbinsx = 200;
   Int_t nbinsy = 200;
   Double_t xmin   = 0;
   Double_t xmax   = (Double_t)nbinsx;
   Double_t ymin   = 0;
   Double_t ymax   = (Double_t)nbinsy;
   Double_t dx = (xmax-xmin)/nbinsx;
   Double_t dy = (ymax-ymin)/nbinsy;
   delete h2;
   h2 = new TH2F("h2","test",nbinsx,xmin,xmax,nbinsy,ymin,ymax);
   h2->SetStats(0);
   //generate n peaks at random
   Double_t par[3000];
   Int_t p;
   for (p=0;p<npeaks;p++) {
      par[5*p+0] = gRandom->Uniform(0.2,1);
      par[5*p+1] = gRandom->Uniform(xmin,xmax);
      par[5*p+2] = gRandom->Uniform(dx,5*dx);
      par[5*p+3] = gRandom->Uniform(ymin,ymax);
      par[5*p+4] = gRandom->Uniform(dy,5*dy);
   }
   TF2 *f2 = new TF2("f2",fpeaks2,xmin,xmax,ymin,ymax,5*npeaks);
   f2->SetNpx(100);
   f2->SetNpy(100);
   f2->SetParameters(par);
   TCanvas *c1 = (TCanvas*)gROOT->GetListOfCanvases()->FindObject("c1");
   if (!c1) c1 = new TCanvas("c1","c1",10,10,1000,700);
   h2->FillRandom("f2",500000);

   //now the real stuff: Finding the peaks
   Int_t nfound = s->Search(h2,2,"col");

   //searching good and ghost peaks (approximation)
   Int_t pf,ngood = 0;
   Double_t *xpeaks = s->GetPositionX();
   Double_t *ypeaks = s->GetPositionY();
   for (p=0;p<npeaks;p++) {
      for (pf=0;pf<nfound;pf++) {
         Double_t diffx = TMath::Abs(xpeaks[pf] - par[5*p+1]);
         Double_t diffy = TMath::Abs(ypeaks[pf] - par[5*p+3]);
         if (diffx < 2*dx && diffy < 2*dy) ngood++;
      }
   }
   if (ngood > nfound) ngood = nfound;
   //Search ghost peaks (approximation)
   Int_t nghost = 0;
   for (pf=0;pf<nfound;pf++) {
      Int_t nf=0;
      for (p=0;p<npeaks;p++) {
         Double_t diffx = TMath::Abs(xpeaks[pf] - par[5*p+1]);
         Double_t diffy = TMath::Abs(ypeaks[pf] - par[5*p+3]);
         if (diffx < 2*dx && diffy < 2*dy) nf++;
      }
      if (nf == 0) nghost++;
   }
   c1->Update();

   s->Print();
   printf("Gener=%d, Found=%d, Good=%d, Ghost=%d\n",npeaks,nfound,ngood,nghost);
   printf("\nDouble click in the bottom right corner of the pad to continue\n");
   c1->WaitPrimitive();
}
void peaks2(Int_t maxpeaks=50) {
   s = new TSpectrum2(2*maxpeaks);
   for (int i=0; i<10; ++i) {
      npeaks = (Int_t)gRandom->Uniform(5,maxpeaks);
      findPeak2();
   }
}


