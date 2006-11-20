// Example macro to show unuran capabilities 
// The results are compared with what is obtained using TRandom or TF1::GetRandom
// The macro is divided in 3 parts: 
//    - testStringAPI         :  show how to use string API of UNURAN to generate Gaussian random numbers
//    - testDistr1D           :  show how to pass a 1D distribution object to UNURAN to generate numbers 
//                               according to the given distribution object
//    - testDistrMultiDIm     :  show how to pass a multidimensional distribution object to UNURAN 
//                               
//
// To execute the macro type in: 
//
// root[0]: .x  gSystem->Load("libMathCore");
// root[0]: .x  gSystem->Load("libUnuran");
// root[0]: .x  unuranDemo.C+


#include "TStopwatch.h"
#include "TUnuran.h"

#include "TH1.h"
#include "TH3.h"
#include "TF3.h"

#include "TRandom2.h"
#include "TSystem.h"
#include "TStyle.h"

#include "TApplication.h"
#include "TCanvas.h"

#include "Math/ProbFunc.h"
#include "Math/DistFunc.h"

#include <iostream> 
#include <cassert>

using std::cout; 
using std::endl; 

// number of distribution generated points
#define NGEN 1000000  

int izone = 0; 
TCanvas * c1 = 0; 

// test using UNURAN string interface
void testStringAPI() {

   TH1D * h1 = new TH1D("h1G","gaussian distribution from Unuran",100,-10,10);
   TH1D * h2 = new TH1D("h2G","gaussian distribution from TRandom",100,-10,10);

   TUnuran unr; 
   if (! unr.Init( "normal()", "method=arou") ) {
      cout << "Error initializing unuran" << endl;
      return;
   }

   int n = NGEN;
   TStopwatch w; 
   w.Start(); 

   for (int i = 0; i < n; ++i) {
       double x = unr.Sample(); 
       h1->Fill(  x ); 
   }

   w.Stop(); 
   cout << "Time using Unuran =\t\t " << w.CpuTime() << endl;

   // use TRandom::Gaus
   w.Start();
   for (int i = 0; i < n; ++i) {
      double x = gRandom->Gaus(0,1); 
       h2->Fill(  x ); 
   }

   w.Stop(); 
   cout << "Time using TRandom::Gaus  =\t " << w.CpuTime() << endl;

   assert(c1 != 0);
   c1->cd(++izone);
   h1->Draw();
   c1->cd(++izone);
   h2->Draw();

}



double distr(double *x, double *p) { 
   return ROOT::Math::breitwigner_pdf(x[0],p[0],p[1]); 
}

double cdf(double *x, double *p) { 
   return ROOT::Math::breitwigner_quant(x[0],p[0],p[1]); 
}
 
// test of unuran passing as input a distribution object( a BreitWigner) distribution 
void testDistr1D() { 

   TH1D * h1 = new TH1D("h1BW","Breit-Wigner distribution from Unuran",100,-10,10);
   TH1D * h2 = new TH1D("h2BW","Breit-Wigner distribution from GetRandom",100,-10,10);


   

   TF1 * f = new TF1("distrFunc",distr,-10,10,2); 
   double par[2] = {1,0};  // values are gamma and mean 
   f->SetParameters(par); 

   TF1 * fc = new TF1("cdfFunc",cdf,-10,10,2); 
   fc->SetParameters(par); 

   // create Unuran 1D distribution object 
   TUnuranDistr dist(f,fc); 
   // to use a different random number engine do: 
   TRandom2 * random = new TRandom2();
   int logLevel = 0;
   TUnuran unr(random,logLevel); 

   // select unuran method for generating the random numbers
   std::string method = "method=arou";
   //std::string method = "method=auto";
   // "method=hinv"

   if (!unr.Init(dist,method) ) { 
      cout << "Error initializing unuran" << endl;
      return;
   } 



   TStopwatch w; 
   w.Start(); 

   int n = NGEN;
   for (int i = 0; i < n; ++i) {
      double x = unr.Sample(); 
      h1->Fill(  x ); 
   }

   w.Stop(); 
   cout << "Time using Unuran =\t\t " << w.CpuTime() << endl;

   w.Start();
   for (int i = 0; i < n; ++i) {
      double x = f->GetRandom(); 
      h2->Fill(  x ); 
   }

   w.Stop(); 
   cout << "Time using GetRandom()  =\t " << w.CpuTime() << endl;

   c1->cd(++izone);
   h1->Draw();

   c1->cd(++izone);
   h2->Draw();
   
   std::cout << " chi2 test of UNURAN vs GetRandom generated histograms:  " << std::endl;
   h1->Chi2Test(h2,"UUP");

}

// 3D gaussian distribution
double gaus3d(double *x, double *p) { 

   double sigma_x = p[0]; 
   double sigma_y = p[1];
   double sigma_z = p[2];
   double rho = p[2]; 
   double u = x[0] / sigma_x ;
   double v = x[1] / sigma_y ;
   double w = x[2] / sigma_z ;
   double c = 1 - rho*rho ;
   double result = (1 / (2 * TMath::Pi() * sigma_x * sigma_y * sigma_z * sqrt(c))) 
      * exp (-(u * u - 2 * rho * u * v + v * v + w*w) / (2 * c));
   return result;
}

// test of unuran passing as input a mluti-dimension distribution object 
void testDistrMultiDim() { 
   

   TH3D * h1 = new TH3D("h13D","gaussian 3D distribution from Unuran",50,-10,10,50,-10,10,50,-10,10);
   TH3D * h2 = new TH3D("h23D","gaussian 3D distribution from GetRandom",50,-10,10,50,-10,10,50,-10,10);


   TF3 * f = new TF3("g3d",gaus3d,-10,10,-10,10,-10,10,3); 
   double par[3] = {2,2,0.5}; 
   f->SetParameters(par); 



   TUnuranDistrMulti dist(f); 
   TUnuran unr(gRandom);
   //std::string method = "method=vnrou";

   std::string method = "method=hitro;use_boundingrectangle=false "; 
   if ( !  unr.Init(dist,method,0) ) { 
      cout << "Error initializing unuran" << endl;
      return;
   } 


   TStopwatch w; 
   w.Start(); 

   double x[3]; 
   for (int i = 0; i < NGEN; ++i) {  
      unr.SampleMulti(x);
      h1->Fill(x[0],x[1],x[2]);
   }

   w.Stop(); 
   cout << "Time using Unuran       =\t " << w.CpuTime() << endl;

   assert(c1 != 0);
   c1->cd(++izone); 
   h1->Draw();


   // need to set a reasanable number of points in TF1 to get accettable quality from GetRandom to 
   int np = 200;
   f->SetNpx(200);
   f->SetNpy(200);
   f->SetNpz(200);

   w.Start();
   for (int i = 0; i < NGEN; ++i) { 
      f->GetRandom3(x[0],x[1],x[2]);
      h2->Fill(x[0],x[1],x[2]); 
   }

   w.Stop(); 
   cout << "Time using TF1::GetRandom  =\t " << w.CpuTime() << endl;

   
   c1->cd(++izone);
   h2->Draw();

   std::cout << " chi2 test of UNURAN vs GetRandom generated histograms:  " << std::endl;
   h1->Chi2Test(h2,"UUP");

}
void unuranDemo() { 

   //gRandom->SetSeed(0);

   // load libraries
   gSystem->Load("libMathCore");
   gSystem->Load("libUnuran");

   // create canvas 

   c1 = new TCanvas("c1_unuranMulti","Multidimensional distribution",10,10,1000,1000); 
   c1->Divide(2,3);
   gStyle->SetOptFit();

   testStringAPI(); 
   testDistr1D();
   testDistrMultiDim();


}
