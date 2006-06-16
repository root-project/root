#include "Math/Random.h"
#include "Math/GSLRndmEngines.h"
#include "TStopwatch.h"
#include "TRandom1.h"
#include "TRandom2.h"
#include "TRandom3.h"
#include <iostream>
#include <cmath>
#include <typeinfo>
#include "TH1D.h"
#include "TStyle.h"
#include "TF1.h"
#include "TPaveLabel.h"

#include "TCanvas.h"

#ifndef PI
#define PI       3.14159265358979323846264338328      /* pi */
#endif

#define NLOOP 1000;
//#define NEVT 20000000;
#define NEVT 10000;
 
using namespace ROOT::Math;

template <class R> 
void generate( R & r, TH1D * h) { 

  TStopwatch w; 

  r.SetSeed(0);
  //r.SetSeed(int(std::pow(2.0,28)));
  int m = NLOOP;
  int n = NEVT;
  for (int j = 0; j < m; ++j) { 

    //std::cout << r.GetSeed() << "   "; 

    w.Start();
//     if ( n < 40000000) iseed = std::rand();
//     iseed = 0;
    //TRandom3 r3(0);
    //r.SetSeed( 0 ); // generate random seeds
    //TRandom3 r3(0); 
    //r.SetSeed (static_cast<UInt_t> (4294967296.*r3.Rndm()) );

  // estimate PI
    double n1=0; 
    double rn[2000];
    double x; 
    double y; 
    for (int ievt = 0; ievt < n; ievt+=1000 ) { 
      r.RndmArray(2000,rn);
      for (int i=0; i < 1000; i++) { 
	x=rn[2*i];
	y=rn[2*i+1];
	if ( ( x*x + y*y ) <= 1.0 ) n1++;
      }
    }
    double piEstimate = 4.0 * double(n1)/double(n);
    double delta = piEstimate-PI; 
    h->Fill(delta); 
  }

  w.Stop();
  std::cout << std::endl; 
  std::cout << "Random:  " << typeid(r).name() 
	    << "\n\tTime = " << w.RealTime() << "  " << w.CpuTime() << std::endl;   
  std::cout << "Time/call:  " << w.CpuTime()/(2*n)*1.0E9 << std::endl; 
}

int piRandom() {

  TRandom                  r0;
  TRandom1                  r1;
  TRandom2                 r2; 
  TRandom3                 r3; 
  //Random<GSLRngRand>       r1;
//   Random<GSLRngTaus>       r2;
//   Random<GSLRngRanLux>     r3;

  double n = NEVT; 
  int nloop = NLOOP;
  double dy = 15/std::sqrt(n);

  TH1D * h0 = new TH1D("h0","TRandom delta",100,-dy,dy);
  TH1D * h1 = new TH1D("h1","TRandom1 delta",100,-dy,dy);
  TH1D * h2 = new TH1D("h2","TRandom2 delta",100,-dy,dy);
  TH1D * h3 = new TH1D("h3","TRandom3 delta",100,-dy,dy);

  double sigma = std::sqrt( PI * (4 - PI)/n );
  std::cout << "**********************************************************" << std::endl; 
  std::cout << " Generate " << int(n) << " for " << nloop      << " times " << std::endl; 
  std::cout << "**********************************************************" << std::endl; 
  std::cout << "\tExpected Sigma = " << sigma << std::endl; 

#define INTERACTIVE
#ifdef INTERACTIVE

  double del, err;
  TCanvas * c1 = new TCanvas("c1_piRandom","PI Residuals");
  gStyle->SetOptFit(1111);
  gStyle->SetOptLogy();
  c1->Divide(2,2);
  c1->cd(1);
  generate(r0,h0);
  h0->Fit("gaus");
  h0->Draw();
  TF1 * fg = (TF1*) h0->FindObject("gaus");
  if (fg) { 
    del = (fg->GetParameter(2)-sigma); 
    err = fg->GetParError(2);
  }
  else { del = -999; err = 1; }

  char text[20];
  sprintf(text,"Spread %8.4f",del/err);
  TPaveLabel * pl0 = new TPaveLabel(0.6,0.3,0.9,0.4,text,"brNDC");  
  pl0->Draw();
   


  
  c1->cd(2);
  generate(r1,h1);
  h1->Fit("gaus"); 
  h1->Draw();
  fg = (TF1*) h1->FindObject("gaus");
  if (fg) { 
    del = (fg->GetParameter(2)-sigma); 
    err = fg->GetParError(2);
  }
  else { del = -999; err = 1; }

  sprintf(text,"Spread %8.4f",del/err);
  TPaveLabel * pl1 = new TPaveLabel(0.6,0.3,0.9,0.4,text,"brNDC");  
  pl1->Draw();


  c1->cd(3);
  generate(r2,h2);
  h2->Fit("gaus"); 
  h2->Draw();
  fg = (TF1*) h2->FindObject("gaus");
  if (fg) { 
    del = (fg->GetParameter(2)-sigma); 
    err = fg->GetParError(2);
  }
  else { del = -999; err = 1; }

  sprintf(text,"Spread %8.4f",del/err);
  TPaveLabel * pl2 = new TPaveLabel(0.6,0.3,0.9,0.4,text,"brNDC");  
  pl2->Draw();


  c1->cd(4);
  generate(r3,h3);
  h3->Fit("gaus"); 
  h3->Draw();
  fg = (TF1*) h3->FindObject("gaus");
  if (fg) { 
    del = (fg->GetParameter(2)-sigma); 
    err = fg->GetParError(2);
  }
  else { del = -999; err = 1; }

  sprintf(text,"Spread %8.4f",del/err);
  TPaveLabel * pl3 = new TPaveLabel(0.6,0.3,0.9,0.4,text,"brNDC");  
  pl3->Draw();
  
#else 
  generate(r0,h0);
  generate(r1,h1);
  generate(r2,h2);
  generate(r3,h3);
#endif

  return 0;

}

int main() { 

  piRandom();
  return 0; 
}
