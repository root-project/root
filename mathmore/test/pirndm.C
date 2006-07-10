//Test program for random number generators (spped and quality)
//The program get n random number pairs x and y i [0,1]
//It counts the ratio of pairs in the circle of diameter 1
//compared to the total number of pairs.
//This ratio must be Pi/4
//The test shows the graph of the difference of this ratio compared to PI.
//To test only the speed, call pirndm(1) (default)
//To test quality and speed call pirndm(50)
//    root pirndm.C+
//or
//    root "pirndm.C+(10)"
//
//Author: Rene Brun
      
#include "TROOT.h"
#include "TStopwatch.h"
#include "TMath.h"
#include "TRandom1.h"
#include "TRandom2.h"
#include "TRandom3.h"
#include "TCanvas.h"
#include "TH2.h"
#include "TGraph.h"
#include "TSystem.h"
#include "TLegend.h"
#include "TPaveLabel.h"
#include "TSystem.h"
#ifdef USE_MATHMORE
#include "Math/Random.h"
#include "Math/GSLRndmEngines.h"
#endif
#include <vector>
#include <iostream>
   
TLegend *legend = 0;
TCanvas *c1 = 0;
TStopwatch timer;
Double_t cputot = 0;

std::vector<TH2D *>  vh2; 

   
//____________________________________________________________________
template<class Random> 
void piRandom(const char *name, Random *r, Long64_t n, Int_t color) {

   TH2D * h2 = new TH2D("h2",name,300,0,0.000003,300,0.,1.);

   timer.Start();
   TGraph *gr = new TGraph();
   gr->SetMarkerStyle(20);
   gr->SetMarkerSize(0.9);
   gr->SetMarkerColor(color);
   gr->SetLineColor(color);
   gr->SetLineWidth(2);
   
   Int_t k = 0;   
   Double_t diffpi;
   Long64_t npi   = 0;
   Double_t pi = TMath::Pi();
   const Int_t NR = 20000;
   const Int_t NR2 = NR/2;
   Double_t rn[NR];
   Long64_t i = 0;
   //double r1,r2; 
   while (i<=n) {
      i += NR2;
      r->RndmArray(NR,rn);
      for (Int_t j=0;j<NR;j+=2) {
          if (rn[j]*rn[j]+rn[j+1]*rn[j+1] <= 1) npi++;
	  if (rn[j] < 0.001) h2->Fill(rn[j],rn[j+1]);
      }
//       r1 = r->Rndm();
//       r2 = r->Rndm();
      //  if (r1*r1+r2*r2 <= 1) npi++;
      if (i && i % (n/10) == 0) {
	  gSystem->ProcessEvents();
         Double_t norm = 4./Double_t(i);
         diffpi   = norm*npi - pi;
         gr->SetPoint(k,i,diffpi);
         if (k ==0) gr->Draw("lp");
         else {
            c1->Modified();
            c1->Update();
         }
         k++;
      }
   }
   timer.Stop();
   Double_t cpu = timer.CpuTime();
   cputot += cpu;
   Double_t nanos = 1.e9*cpu/Double_t(2*n);
   legend->AddEntry(gr,Form("%-14s: %6.1f ns/call",name,nanos),"lp");
   c1->Modified();
   c1->Update();
   printf("RANDOM = %s : RT=%7.3f s, Cpu=%7.3f s\n",name,timer.RealTime(),cpu);

//    TCanvas * c2 = new TCanvas(); 
//    h2->Draw();
//    c2->Update(); 

//    c1->SetSelected(c1);
   vh2.push_back(h2);

}

//________________________________________________________________________
void ErrorBand(Long64_t n) {
   Int_t np = 40;
   TGraph *g = new TGraph(2*np+2);
   Double_t xmax = Double_t(n)/Double_t(np);
   for (Int_t i=1;i<=np;i++) {
      Double_t x = i*xmax;
      //Double_t e = 1./TMath::Sqrt(x);
      Double_t e = TMath::Sqrt( 2 * TMath::Pi() * (4 - TMath::Pi() )/x );
      g->SetPoint(i,x,e);
      g->SetPoint(2*np-i+1,x,-e);
   }
   Double_t x0 = 0.1*xmax;
   Double_t e0 = 1./TMath::Sqrt(x0);
   g->SetPoint(0,x0,e0);
   g->SetPoint(2*np+1,0,-e0);
   g->SetPoint(2*np+2,0,e0);
   g->SetFillColor(1);
   g->SetFillStyle(3002);
   g->Draw("f");
}

         
//________________________________________________________________________
void pirndm(Long64_t n1=1, unsigned int seed = 0) {
   Long64_t n = n1*20000000;
   c1 = new TCanvas("c1");
   c1->SetLeftMargin(0.12);
   c1->SetFrameFillColor(41);
   c1->SetFrameBorderSize(6);
   c1->SetGrid();
   Double_t dy = 10/TMath::Sqrt(n);
   //   Double_t dy = 1.5e-3;
   //if (n1 < 4) dy *=2;
   TH2F *frame = new TH2F("h","",100,0,1.1*n,100,-dy,dy);
   frame->GetXaxis()->SetTitle("Number of TRandom calls");
   frame->GetYaxis()->SetTitle("Difference with #pi");
   frame->GetYaxis()->SetTitleOffset(1.3);
   frame->GetYaxis()->SetDecimals();
   frame->SetStats(0);
   frame->Draw();
   legend = new TLegend(0.6,0.7,0.88,0.88);
   legend->Draw();


   ErrorBand(n);
   std::cout << "seed is " << seed << std::endl;
   
   piRandom("TRandom",new TRandom(seed),n,kYellow);
   piRandom("TRandom2",new TRandom2(seed),n,kBlue);
   piRandom("TRandom3",new TRandom3(seed),n,kRed);
   piRandom("TRandom1",new TRandom1(seed),n,kGreen);

#ifdef USE_MATHMORE
#define GSL2
#ifdef GSL1
   piRandom("TRandom()",new TRandom(),n,kYellow);
   piRandom("TRandom3(0)",new TRandom3(0),n,kBlack);
   piRandom("MT",new ROOT::Math::Random<ROOT::Math::GSLRngMT>(),n,kRed);
   piRandom("Taus",new ROOT::Math::Random<ROOT::Math::GSLRngTaus>(),n,kMagenta);
   piRandom("GFSR4",new ROOT::Math::Random<ROOT::Math::GSLRngGFSR4>(),n,kGreen);
   piRandom("RanLux",new ROOT::Math::Random<ROOT::Math::GSLRngRanLux>(),n,kBlue);
#endif
#ifdef GSL2
   piRandom("TRandom(1)",new TRandom(1),n,kYellow);
   piRandom("TRandom(2^30)",new TRandom(1073741824),n,kRed);
   piRandom("TRandom(256)",new TRandom(256),n,kMagenta);
   piRandom("TRandom(2)",new TRandom(2),n,kBlue);
   piRandom("Rand",new ROOT::Math::Random<ROOT::Math::GSLRngRand>(),n,kGreen);
   //   piRandom("RANMAR",new ROOT::Math::Random<ROOT::Math::GSLRngRanMar>(),n,kBlue);
   //piRandom("MINSTD",new ROOT::Math::Random<ROOT::Math::GSLRngMinStd>(),n,kBlack);
   piRandom("new TRandom2",new TRandom2(),n,kCyan);
   //piRandom("TRandom3",new TRandom3(),n,kCyan);
//    piRandom("CMRG",new ROOT::Math::Random<ROOT::Math::GSLRngCMRG>(),n,kRed);
//    piRandom("MRG",new ROOT::Math::Random<ROOT::Math::GSLRngMRG>(),n,kMagenta);
#else
   piRandom("new TRandom2",new TRandom2(std::rand()),n,kYellow);
   piRandom("new TRandom2",new TRandom2(std::rand()),n,kRed);
   piRandom("new TRandom2",new TRandom2(std::rand()),n,kMagenta);
   piRandom("new TRandom2",new TRandom2(std::rand()),n,kBlack);
   piRandom("new TRandom2",new TRandom2(std::rand()),n,kBlue);
   piRandom("new TRandom2",new TRandom2(std::rand()),n,kRed);
   piRandom("new TRandom2",new TRandom2(std::rand()),n,kGreen);
#endif
#endif
      
   // reftime calculated on MACOS Intel dualcore 2GHz
   // time for TRandom + TRandom2 + TRandom3 + TRandom1 for n = 10**7 (n1=5000)
   Double_t reftime = (4629.530 + 5358.100  + 5785.240 + 26012.17)/5000.;
   const Double_t rootmarks = 900*Double_t(n1)*reftime/cputot;
   TPaveLabel *pl = new TPaveLabel(0.2,0.92,0.8,0.98,Form("cpu time = %6.1fs - rootmarks = %6.1f",cputot,rootmarks),"brNDC");
   pl->Draw();  
   printf("******************************************************************\n");
   printf("*  ROOTMARKS =%6.1f   *  Root%-8s  %d/%d\n",rootmarks,gROOT->GetVersion(),
         gROOT->GetVersionDate(),gROOT->GetVersionTime());
   printf("******************************************************************\n");

   printf("Time at the end of job = %f seconds\n",cputot);
   c1->Print("pirndm.root");
   c1->Print("pirndm.gif");


   // draw 2D histos 
   TCanvas * c2 = new TCanvas();
   int nx = 0; 
   int ny = vh2.size(); 
   for ( nx = 1; nx < ny; nx++) { 
     double r = double(vh2.size())/nx; 
     ny = int(r - 0.01) + 1; 
   }
   nx--;
   std::cout << nx << "  " << ny << "  " << vh2.size() << std::endl;
   
   c2->Divide(ny,nx); 
   for (unsigned int i = 0; i < vh2.size(); ++i) { 
     c2->cd (i+1); 
     vh2[i]->Draw(); 
   }
   c2->Update(); 
   
}
