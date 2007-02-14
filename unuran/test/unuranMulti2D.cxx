// test using Multi-dim (2D)  Distribution object interface
// and compare results and CPU performances using TF2::GetRandom

#include "TStopwatch.h"
#include "TUnuran.h"

#include "TH2.h"
#include "TF2.h"
#include "TCanvas.h"
#include "TMath.h"

#include "TRandom.h"
#include "TSystem.h"
//#include "TApplication.h"

// #include "Math/ProbFunc.h"
// #include "Math/DistFunc.h"

#define _USE_MATH_DEFINES // for Windows
#include <cmath>
#include <iostream> 



using std::cout; 
using std::endl; 



double gaus2d(double *x, double *p) { 

   double sigma_x = p[0]; 
   double sigma_y = p[1];
   double rho = p[2]; 
   double u = x[0] / sigma_x ;
   double v = x[1] / sigma_y ;
   double c = 1 - rho*rho ;
   double result = (1 / (2 * TMath::Pi() * sigma_x * sigma_y * sqrt(c))) 
      * exp (-(u * u - 2 * rho * u * v + v * v ) / (2 * c));
   return result;
}



void unuranMulti2D() { 

   gRandom->SetSeed(0);

   gSystem->Load("libMathCore");
   gSystem->Load("libUnuran");

   // simple test of unuran

   

   TH2D * h1 = new TH2D("h1","gaussian 2D distribution",100,-10,10,100,-10,10);
   TH2D * h2 = new TH2D("h2","gaussian 2D distribution",100,-10,10,100,-10,10);

   

   TF2 * f = new TF2("g2d",gaus2d,-10,10,-10,10,3); 
   double par[3] = {1,1,0.5}; 
   f->SetParameters(par); 

    f->SetNpx(100);
    f->SetNpy(100);
   std::cout << " n function points = " << f->GetNpx() << "  "  << f->GetNpy() << std::endl;

   TCanvas * c1 = new TCanvas("c1_unuranMulti","Multidimensional distribution",10,10,500,500); 
   c1->Divide(1,2);

//    cout << fc->Eval(-11) << "  " <<    fc->Eval(1) << endl;

   TUnuranDistrMulti dist(f); 
   TUnuran unr; 
   std::string method = "method=hitro";
   if ( !  unr.Init(dist,method) ) { 
      cout << "Error initializing unuran" << endl;
      return;
   } 
  

   TStopwatch w; 
   w.Start(); 

   double x[2]; 
   for (int i = 0; i < 1000000; ++i) {
      unr.SampleMulti(x);
//      cout << x[0] << " " << x[1] << endl;
      h1->Fill(x[0],x[1]);
   }

   w.Stop(); 
   cout << "Time using Unuran =\t\t " << w.CpuTime() << endl;
   c1->cd(1); 
   h1->Draw("col");


   w.Start();
   for (int i = 0; i < 1000000; ++i) { 
      f->GetRandom2(x[0],x[1]);
      h2->Fill(x[0],x[1]); 
   }

   w.Stop(); 
   cout << "Time using GetRandom()  =\t " << w.CpuTime() << endl;

   
   c1->cd(2);
   h2->Draw("col");
   //  h2->Fit("gaus");

   std::cout << " chi2 test of histogram generated with Unuran vs histogram generated with TF1::GetRandom " << std::endl;
   h1->Chi2Test(h2,"UUP");

}

#ifndef __CINT__
//int main(int argc, char **argv)
int main( )
{
//   TApplication theApp("App", &argc, argv);
   unuranMulti2D();
//   theApp.Run();
   return 0;
}
#endif
