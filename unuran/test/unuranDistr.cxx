// test using 1D Distribution object interface
// and compare results and CPU performances using TF1::GetRandom


#include "TStopwatch.h"
#include "TUnuran.h"

#include "TH1.h"

#include "TRandom.h"
#include "TSystem.h"
#include "TStyle.h"

//#include "TApplication.h"
#include "TCanvas.h"

#include "Math/ProbFunc.h"
#include "Math/DistFunc.h"

#include <iostream> 

using std::cout; 
using std::endl; 

double norm(double *x, double *p) { 
   return ROOT::Math::normal_pdf(x[0],p[0]); 
}

double cdf(double *x, double *p) { 
   return ROOT::Math::normal_quant(x[0],p[0]); 
}


void unuranDistr() { 

   //gRandom->SetSeed(0);

   gSystem->Load("libMathCore");
   gSystem->Load("libUnuran");

   // simple test of unuran

   TH1D * h1 = new TH1D("h1","gaussian distribution",100,-10,10);
   TH1D * h2 = new TH1D("h2","gaussian distribution",100,-10,10);

   TH1D * h1u = new TH1D("h1u","test gaussian dist",100,0,1);
   TH1D * h2u = new TH1D("h2u","test gaussian dist",100,0,1);

   

   TF1 * f = new TF1("n",norm,-10,10,1); 
   double par[1] = {1}; 
   f->SetParameters(par); 

   TF1 * fc = new TF1("c",cdf,-10,0,1); 
   fc->SetParameters(par); 

   cout << fc->Eval(-11) << "  " <<    fc->Eval(1) << endl;

   TUnuranDistr dist(f,fc); 

//    double m = dist.Mode();
//    std::cout << " mode " << dist.Mode() << std::endl; 
//    std::cout << " f(mode) " << dist(m) << std::endl; 
//    std::cout << " f(mode) " << dist.Cdf(m) << std::endl; 
//    std::cout << " f(mode) " << dist.Derivative(m) << std::endl


   TUnuran unr(gRandom,2); 

   std::string method = "method=arou";
   //std::string method = "method=auto";
   // "method=hinv"

   if (! unr.Init(dist,method) ) { 
      cout << "Error initializing unuran" << endl;
      return;
   } 

   TCanvas * c1 = new TCanvas("c1_unuranMulti","Multidimensional distribution",10,10,1000,1000); 
   c1->Divide(2,2);
   gStyle->SetOptFit();


   TStopwatch w; 
   w.Start(); 

   int n = 1000000;
   for (int i = 0; i < n; ++i) 
      unr.Sample(); 

   w.Stop(); 
   cout << "Time using Unuran =\t\t " << w.CpuTime() << endl;

   w.Start();
   for (int i = 0; i < n; ++i) 
      f->GetRandom(); 

   w.Stop(); 
   cout << "Time using GetRandom()  =\t " << w.CpuTime() << endl;


   // test the quality
   for (int i = 0; i < n; ++i) {
      double x = unr.Sample();
      h1->Fill(  x ); 
      h1u->Fill( fc->Eval( x ) ); 
   }


   c1->cd(1);
   h1->Draw();
   h1->Fit("gaus");
   c1->cd(2);
   h1u->Draw("Error");
   h1u->Fit("pol0");
   std::cout << "\nFit result on data cdf with unuran: " << std::endl;
   TF1 * f1 = h1u->GetFunction("pol0");
   std::cout << "Fit chi2 = " << f1->GetChisquare() << " ndf = " << f1->GetNDF() << std::endl;
   std::cout << "Fit Prob = " << f1->GetProb() << std::endl;
   

   for (int i = 0; i < n; ++i) {
      double x = f->GetRandom();
      h2->Fill(  x ); 
      h2u->Fill( fc->Eval( x ) ); 
   }



   c1->cd(3);
   h2->Draw("same");
   h2->Fit("gaus");
   c1->cd(4);
   h2u->Draw("Error");
   std::cout << "\nFit result on data cdf with GetRandom: " << std::endl;
   h2u->Fit("pol0");
   f1 = h2u->GetFunction("pol0");
   std::cout << "Fit chi2 = " << f1->GetChisquare() << " ndf = " << f1->GetNDF() << std::endl;
   std::cout << "Fit Prob = " << f1->GetProb() << std::endl;

   std::cout << " chi2 test h1 vs h2 " << std::endl;
   h1->Chi2Test(h2,"UUP");
   
//    TApplication theApp("App",0,0);
//    theApp.Run();

}

#ifndef __CINT__
//int main(int argc, char **argv)
int main()
{
   unuranDistr();
   return 0;
}
#endif
