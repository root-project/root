#include "TStopwatch.h"
#include "TF1.h"
#include "TFormula.h"
#include "TVectorF.h"
#include "TRandom.h"
#include "TGraph.h"
#include <vector>
using namespace std;

/*
  to do benchmark
  .L TestSpeed.C++O
  //Measure time spent for caluclation
   TestNumeric();  >numeric.txt
   TestLogical();  > logical.txt
   TestFit(1);      >fit.txt
   // test correctnes of computation using invariants 
   TestCorrect(1); //test invariants - correctnes of TFormula  - everything should be OK at  output
                   // allowed numerical instability 10^-11
                   // to be noticed - TMath function using old TFormula - slower and not precise
		   // rounding error at 10^-9 level
*/

//
//  to test  and benchmarke calculation
//  call TestLogical();
//  to test  numerical calculation 
//  call TestNumeric();
//  call TestWrite() and TestRead to check IO functionality

void TestSpeed(const char *formula = "x<y&&x<z&&x<10",Int_t n=1)
{
  //
  // speed test 0
  //
  TFormula f("f",formula);
  TStopwatch timer;
  Double_t sum=0;
  //
  vector<Double_t> x(110000);
  vector<Double_t> par(110000);
  for (Int_t i=0;i<110000;i++){x[i]=gRandom->Rndm(); par[i]=gRandom->Rndm();}
  //
  printf("%s\n",formula);
  timer.Start();
  for (Int_t i=0;i<n;i++){
    for (Int_t j=0;j<100000;j++)
      sum+=f.EvalPar(&x[j],&par[j]);
  }
  timer.Stop();
  timer.Print();   
} 

void TestInvariant(const char *formula ,Double_t invariant, Int_t n=1)
{
  TStopwatch timer;
  TFormula f("f",formula);
  //
  vector<Double_t>  x(110000);
  vector<Double_t> par(110000);
  for (Int_t i=0;i<110000;i++){x[i]=gRandom->Rndm(); par[i]=gRandom->Rndm();}
  //
  printf("%s\n",formula);
  Bool_t isOK =kTRUE;
  timer.Start();
  for (Int_t i=0;i<n;i++){
    for (Int_t j=0;j<100000;j++)
      if (TMath::Abs(f.EvalPar(&x[j],&par[j])-invariant)>0.00000000001){
	isOK=kFALSE;
      }
  }
  timer.Stop();
  timer.Print();  
  if (isOK) printf("Calculation  status  Invariant=%f - OK\n",invariant);
  if (!isOK) printf("Calculation status  Invariant=%f - BUG\n",invariant);
}



void TestFitG(TGraph *gr, TF1*f, Int_t n){
  TStopwatch timer;
  printf("%s\n",f->GetTitle());
  timer.Start();
  for (Int_t i=0;i<n;i++){
    gr->Fit(f,"q");
  }
  timer.Stop();
  timer.Print(); 
  //f->Print();
}

void TestFit(Int_t n,Int_t n2=10000){
  Double_t *x = new Double_t[n2];
  Double_t *y = new Double_t[n2];				
  Double_t par[] = {1,0,1,1};
  TF1 *f = 0;
  Int_t norm = n*100000/n2; //
  TGraph *gr=0;
  for (Int_t i=0;i<n2;i++){x[i]=gRandom->Rndm();}
  //
  //
  for (Int_t i=0;i<n2;i++){y[i]=x[i]+0.1*gRandom->Gaus();}
  gr = new TGraph(n2,x,y);
  f = new TF1("f","pol1",-10,10);
  TestFitG(gr,f,norm);
  //
  for (Int_t i=0;i<n2;i++){x[i]=gRandom->Rndm()*10.-5.; y[i] = TMath::BreitWigner(x[i],0,1)+gRandom->Rndm();} 
  gr = new TGraph(n2/50,x,y);
  f = new TF1("f","[0]*TMath::BreitWigner(x,[1],[2])",-10,10);
  f->SetParameters(par);
  TestFitG(gr,f,norm);
  for (Int_t i=0;i<n2;i++){x[i]=gRandom->Rndm()*10.-5.; y[i] = 1+TMath::BreitWigner(x[i],0,1)+gRandom->Rndm();} 
  gr = new TGraph(n2/50,x,y);
  f = new TF1("f","[0]*TMath::BreitWigner(x,[1],[2])+[3]",-10,10);
  f->SetParameters(par);
  TestFitG(gr,f,norm);
  //
  for (Int_t i=0;i<n2;i++){x[i]=gRandom->Rndm()*10.-5.; y[i] = TMath::Landau(x[i],0,1)+gRandom->Rndm();} 
  gr = new TGraph(n2/50,x,y);
  f = new TF1("f","[0]*TMath::Landau(x,[1],[2])",-10,10); 
  f->SetParameters(par);
  TestFitG(gr,f,norm);
  f = new TF1("f","landau",-10,10);
  f->SetParameters(par);
  TestFitG(gr,f,norm);  

  for (Int_t i=0;i<n2;i++){x[i]=gRandom->Rndm()*10.-5.; y[i] = TMath::Gaus(x[i],0,1)+gRandom->Rndm();} 
  gr = new TGraph(n2/50,x,y);
  f = new TF1("f","[0]*TMath::Gaus(x,[1],[2])",-10,10);
  f->SetParameters(par);
  TestFitG(gr,f,norm);
  f = new TF1("f","gaus",-10,10);
  f->SetParameters(par);
  TestFitG(gr,f,norm);  
  for (Int_t i=0;i<n2;i++){x[i]=gRandom->Rndm()*10.-5.; y[i] = 2+TMath::Gaus(x[i],0,1)+gRandom->Rndm();} 
  f = new TF1("f","gaus+[3]",-10,10);
  f->SetParameters(par);
  TestFitG(gr,f,norm);

  delete [] x;
  delete [] y;
}

void TestCorrect(Int_t n=1){
  //
  // test correctness of formulas using invariants
  //
  // logical
  //
  TestInvariant("x<y||x>y||x==y",1,n);
  TestInvariant("x>y&&x<y",0,n);
  TestInvariant("x>y&&y>z&&z>x",0,n);
  TestInvariant("x*2<x&&x<0",0,n);
  TestInvariant("x*2>x||x<0",1,n);
  //
  // numeric
  //
  TestInvariant("x*(y-1)-x*y+x",0,n);
  TestInvariant("TMath::Sin(x)-sin(x)",0,n);
  TestInvariant("TMath::Abs(y)-abs(y)",0,n);
  TestInvariant("cos(x)*cos(x)+sin(x)*sin(x)",1,n);
  TestInvariant("tan(x)-sin(x)/cos(x)",0,n);  
  TestInvariant("pol1-[0]-[1]*x",0,n);
  TestInvariant("Gaus(x,0,1)-exp(-0.5*x*x)",0,n);
  TestInvariant("Gaus(x,y,1)-exp(-0.5*(x-y)*(x-y))",0,n);
  TestInvariant("cos(TMath::Pi()*x)*cos(TMath::Pi()*x)+sin(TMath::Pi()*x)*sin(TMath::Pi()*x)",1,n);
  //TestInvariant("int(10**15*x)%2+(int(10**15*x)+1)%2",1,n);
  TestInvariant("(10**15*x)%2+((10**15*x)+1)%2",1,n);
}

void TestNumeric(Int_t n=50){
  //
  TestSpeed("1",n);
  TestSpeed("x",n);
  TestSpeed("x*x",n);
  TestSpeed("x+x*y",n);
  TestSpeed("x*x+cos(x)",n);
  TestSpeed("atan2(x,y)",n);  
  TestSpeed("x+y*z*t+t*y",n);
  TestSpeed("xpol1",n);
  TestSpeed("Pol1(0,0)",n);
  TestSpeed("xpol2",n);
  TestSpeed("Pol2(0,0)",n);
  TestSpeed("xpol3",n);
  TestSpeed("Pol2(0,0)",n);
  TestSpeed("xpol3",n);
  TestSpeed("Pol3(0,0)",n);
  TestSpeed("gaus",n);
  TestSpeed("Gaus(x,0,1)",n);  
  //TMath function
  TestSpeed("TMath::BreitWigner(x,0,1)",TMath::Max(Int_t(n/50.),1));
  TestSpeed("TMath::Sin(x)",TMath::Max(Int_t(n/50.),1));
  TestSpeed("TMath::ATan2(x,y)",TMath::Max(Int_t(n/50.),1));
  TestSpeed("cos(TMath::Pi()*x)",TMath::Max(Int_t(n/10.),1));
  TestSpeed("cos(pi*x)",TMath::Max(Int_t(n/10.),1));  
}

void TestRead(const char *fname, const char *oname)
{
  TFile fr(fname);
  TF1 *f1 = (TF1*)fr.Get(oname);
  if (f1) f1->Clone()->Draw();
}

void TestWrite(const char *fname)
{
  TFile fr(fname,"new");
  TF1 *f = 0;
  f = new TF1("f1","sin(x)"); f->Write();delete f;
  f = new TF1("f2","sin(x)*x"); f->Write();delete f;
  f = new TF1("f3","sin(x)+x"); f->Write();delete f;
  f = new TF1("f4","tan(x)/x"); f->Write();delete f;
  f = new TF1("f5","x*x*x"); f->Write();delete f;
  f = new TF1("f6","xpol1"); f->SetParameters(1,2,3,4,5); f->Write();delete f;
  f = new TF1("f7","xpol2"); f->SetParameters(1,2,3,4,5); f->Write();delete f;
  f = new TF1("f8","xpol3"); f->SetParameters(1,2,3,4,5); f->Write();delete f;
  fr.Close();
}

void TestLogical(Int_t n=50){
  //
  TestSpeed("cos(x)<10",n);
  TestSpeed("abs(x-y)<z||x<y&&z>y||abs(z-0.5)<0.5",n);
  TestSpeed("x<z",n); 
  TestSpeed("x<z||x<y",n);
  TestSpeed("x<z&&x<y&&z>y&&z<0.5",n);
  TestSpeed("x<z||x<y&&z>y",n);
  TestSpeed("x<z||x<y&&z>y||z<0.5",n);
  TestSpeed("x<z||x<y&&z>y||z<0.5||x<0.3&&y>0.3",n);
  TestSpeed("x+y<z||x<y&&z>y||z<0.5||x<0.3&&y+x>0.3",n);
}


