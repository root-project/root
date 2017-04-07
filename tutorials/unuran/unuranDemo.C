/// \file
/// \ingroup tutorial_unuran
/// Example macro to show unuran capabilities
/// The results are compared with what is obtained using TRandom or TF1::GetRandom
/// The macro is divided in 3 parts:
///    - testStringAPI         :  show how to use string API of UNURAN to generate Gaussian random numbers
///    - testDistr1D           :  show how to pass a 1D distribution object to UNURAN to generate numbers
///                               according to the given distribution object
///    - testDistrMultiDIm     :  show how to pass a multidimensional distribution object to UNURAN
///
///
/// To execute the macro type in:
///
/// ~~~{.cpp}
/// root[0]: gSystem->Load("libMathCore");
/// root[0]: gSystem->Load("libUnuran");
/// root[0]: .x  unuranDemo.C+
/// ~~~
///
/// \macro_code
///
/// \author Lorenzo Moneta


#include "TStopwatch.h"

#include "TUnuran.h"
#include "TUnuranContDist.h"
#include "TUnuranMultiContDist.h"
#include "TUnuranDiscrDist.h"
#include "TUnuranEmpDist.h"

#include "TH1.h"
#include "TH3.h"
#include "TF3.h"
#include "TMath.h"

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

   cout << "\nTest using UNURAN string API \n\n";


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
   cout << "Time using Unuran method " << unr.MethodName() << "\t=\t " << w.CpuTime() << endl;


   // use TRandom::Gaus
   w.Start();
   for (int i = 0; i < n; ++i) {
      double x = gRandom->Gaus(0,1);
       h2->Fill(  x );
   }

   w.Stop();
   cout << "Time using TRandom::Gaus  \t=\t " << w.CpuTime() << endl;

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
   return ROOT::Math::breitwigner_cdf(x[0],p[0],p[1]);
}

// test of unuran passing as input a distribution object( a BreitWigner) distribution
void testDistr1D() {

   cout << "\nTest 1D Continous distributions\n\n";

   TH1D * h1 = new TH1D("h1BW","Breit-Wigner distribution from Unuran",100,-10,10);
   TH1D * h2 = new TH1D("h2BW","Breit-Wigner distribution from GetRandom",100,-10,10);



   TF1 * f = new TF1("distrFunc",distr,-10,10,2);
   double par[2] = {1,0};  // values are gamma and mean
   f->SetParameters(par);

   TF1 * fc = new TF1("cdfFunc",cdf,-10,10,2);
   fc->SetParameters(par);

   // create Unuran 1D distribution object
   TUnuranContDist dist(f);
   // to use a different random number engine do:
   TRandom2 * random = new TRandom2();
   int logLevel = 2;
   TUnuran unr(random,logLevel);

   // select unuran method for generating the random numbers
   std::string method = "tdr";
   //std::string method = "method=auto";
   // "method=hinv"
   // set the cdf for some methods like hinv that requires it
   // dist.SetCdf(fc);

   //cout << "unuran method is " << method << endl;

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
   cout << "Time using Unuran method " << unr.MethodName() << "\t=\t " << w.CpuTime() << endl;

   w.Start();
   for (int i = 0; i < n; ++i) {
      double x = f->GetRandom();
      h2->Fill(  x );
   }

   w.Stop();
   cout << "Time using TF1::GetRandom()  \t=\t " << w.CpuTime() << endl;

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

// test of unuran passing as input a multi-dimension distribution object
void testDistrMultiDim() {

   cout << "\nTest Multidimensional distributions\n\n";

   TH3D * h1 = new TH3D("h13D","gaussian 3D distribution from Unuran",50,-10,10,50,-10,10,50,-10,10);
   TH3D * h2 = new TH3D("h23D","gaussian 3D distribution from GetRandom",50,-10,10,50,-10,10,50,-10,10);



   TF3 * f = new TF3("g3d",gaus3d,-10,10,-10,10,-10,10,3);
   double par[3] = {2,2,0.5};
   f->SetParameters(par);



   TUnuranMultiContDist dist(f);
   TUnuran unr(gRandom);
   //std::string method = "method=vnrou";

   //std::string method = "method=hitro;use_boundingrectangle=false ";
   std::string method = "hitro";
   if ( !  unr.Init(dist,method) ) {
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
   cout << "Time using Unuran method " << unr.MethodName() << "\t=\t\t " << w.CpuTime() << endl;

   assert(c1 != 0);
   c1->cd(++izone);
   h1->Draw();


   // need to set a reasonable number of points in TF1 to get acceptable quality from GetRandom to
   int np = 200;
   f->SetNpx(np);
   f->SetNpy(np);
   f->SetNpz(np);

   w.Start();
   for (int i = 0; i < NGEN; ++i) {
      f->GetRandom3(x[0],x[1],x[2]);
      h2->Fill(x[0],x[1],x[2]);
   }

   w.Stop();
   cout << "Time using TF1::GetRandom  \t\t=\t " << w.CpuTime() << endl;


   c1->cd(++izone);
   h2->Draw();

   std::cout << " chi2 test of UNURAN vs GetRandom generated histograms:  " << std::endl;
   h1->Chi2Test(h2,"UUP");

}
//_____________________________________________
//
// example of discrete distributions

double poisson(double * x, double * p) {
   return ROOT::Math::poisson_pdf(int(x[0]),p[0]);
}

void testDiscDistr() {

   cout << "\nTest Discrete distributions\n\n";

   TH1D * h1 = new TH1D("h1PS","Unuran Poisson prob",20,0,20);
   TH1D * h2 = new TH1D("h2PS","Poisson dist from TRandom",20,0,20);

   double mu = 5;

   TF1 * f = new TF1("fps",poisson,1,0,1);
   f->SetParameter(0,mu);

   TUnuranDiscrDist dist2 = TUnuranDiscrDist(f);
   TUnuran unr;

   // dari method (needs also the mode and pmf sum)
   dist2.SetMode(int(mu) );
   dist2.SetProbSum(1.0);
   bool ret = unr.Init(dist2,"dari");
   if (!ret) return;

   TStopwatch w;
   w.Start();

   int n = NGEN;
   for (int i = 0; i < n; ++i) {
      int k = unr.SampleDiscr();
      h1->Fill( double(k) );
   }

   w.Stop();
   cout << "Time using Unuran method " << unr.MethodName() << "\t=\t\t " << w.CpuTime() << endl;

   w.Start();
   for (int i = 0; i < n; ++i) {
      h2->Fill(  gRandom->Poisson(mu) );
   }
   cout << "Time using TRandom::Poisson " << "\t=\t\t " << w.CpuTime() << endl;

   c1->cd(++izone);
   h1->SetMarkerStyle(20);
   h1->Draw("E");
   h2->Draw("same");

   std::cout << " chi2 test of UNURAN vs TRandom generated histograms:  " << std::endl;
   h1->Chi2Test(h2,"UUP");

}

//_____________________________________________
//
// example of empirical distributions

void testEmpDistr() {


   cout << "\nTest Empirical distributions using smoothing\n\n";

   // start with a set of data
   // for example 1000 two-gaussian data
   const int Ndata = 1000;
   double x[Ndata];
   for (int i = 0; i < Ndata; ++i) {
      if (i < 0.5*Ndata )
         x[i] = gRandom->Gaus(-1.,1.);
      else
         x[i] = gRandom->Gaus(1.,3.);
   }

   TH1D * h0 = new TH1D("h0Ref","Starting data",100,-10,10);
   TH1D * h1 = new TH1D("h1Unr","Unuran unbin Generated data",100,-10,10);
   TH1D * h1b = new TH1D("h1bUnr","Unuran bin Generated data",100,-10,10);
   TH1D * h2 = new TH1D("h2GR","Data from TH1::GetRandom",100,-10,10);

   h0->FillN(Ndata,x,0,1); // fill histogram with starting data

   TUnuran unr;
   TUnuranEmpDist dist(x,x+Ndata,1);


   TStopwatch w;
   int n = NGEN;

   w.Start();
   if (!unr.Init(dist)) return;
   for (int i = 0; i < n; ++i) {
      h1->Fill( unr.Sample() );
   }

   w.Stop();
   cout << "Time using Unuran unbin  " << unr.MethodName() << "\t=\t\t " << w.CpuTime() << endl;

   TUnuranEmpDist binDist(h0);

   w.Start();
   if (!unr.Init(binDist)) return;
   for (int i = 0; i < n; ++i) {
      h1b->Fill( unr.Sample() );
   }
   w.Stop();
   cout << "Time using Unuran bin  " << unr.MethodName() << "\t=\t\t " << w.CpuTime() << endl;

   w.Start();
   for (int i = 0; i < n; ++i) {
      h2->Fill(  h0->GetRandom() );
   }
   cout << "Time using TH1::GetRandom " << "\t=\t\t " << w.CpuTime() << endl;

   c1->cd(++izone);

   h2->Draw();
   h1->SetLineColor(kRed);
   h1->Draw("same");
   h1b->SetLineColor(kBlue);
   h1b->Draw("same");


}



void unuranDemo() {

   //gRandom->SetSeed(0);

   // load libraries
   gSystem->Load("libMathCore");
   gSystem->Load("libUnuran");

   // create canvas

   c1 = new TCanvas("c1_unuranMulti","Multidimensional distribution",10,10,1000,1000);
   c1->Divide(2,4);
   gStyle->SetOptFit();

   testStringAPI();
   c1->Update();
   testDistr1D();
   c1->Update();
   testDistrMultiDim();
   c1->Update();
   testDiscDistr();
   c1->Update();
   testEmpDistr();
   c1->Update();


}
