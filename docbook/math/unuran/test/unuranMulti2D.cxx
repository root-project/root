// test using Multi-dim (2D)  Distribution object interface
// and compare results and CPU performances using TF2::GetRandom
//
// run within ROOT (.x unuranMulti2D.cxx+) or pass any extra parameter in the command line to get  
// a graphics output 

#include "TStopwatch.h"
#include "TUnuran.h"
#include "TUnuranMultiContDist.h"

#include "TH2.h"
#include "TF2.h"
#include "TCanvas.h"
#include "TMath.h"

#include "TRandom3.h"
#include "TSystem.h"
#include "TApplication.h"
#include "TError.h"

#include "Math/Functor.h"

// #include "Math/ProbFunc.h"
// #include "Math/DistFunc.h"

#define _USE_MATH_DEFINES // for Windows
#include <cmath>
#include <iostream> 

//#define DEBUG 

using std::cout; 
using std::endl; 

int n = 1000000;

bool useRandomSeed = false;   // to use a random seed different every time

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

double log_gaus2d(double *x, double *p) { 

   return std::log( gaus2d(x,p) ); // should re-implement it
}



int testUnuran(TUnuran & unr, const std::string & method, const TUnuranMultiContDist & dist, TH2 * h1, const TH2 * href ) { 

#ifdef DEBUG
   n = 1000;
#endif


   // init unuran 
   bool ret =   unr.Init(dist,method); 
   if (!ret) { 
      std::cerr << "Error initializing unuran with method " << unr.MethodName() << endl; 
      return -1;
   } 

   h1->Reset();

   // test first the time
   TStopwatch w; 

   w.Start(); 
   double x[2]; 
   for (int i = 0; i < n; ++i) {
      unr.SampleMulti(x);
      h1->Fill(x[0],x[1]);
      if (method == "gibbs" && i < 100) 
         std::cout << x[0] << " , " << x[1] << std::endl; 
   }

   w.Stop(); 
   double time = w.CpuTime()*1.E9/n; 

   double prob = href->Chi2Test(h1,"UU");
   double ksprob = href->KolmogorovTest(h1);
   cout << "Time using Unuran  " << unr.MethodName() << "   \t=\t " << time << "\tns/call \t\tChi2 Prob = "
        << prob << "\tKS Prob = " << ksprob << std::endl;
   if (prob < 1E-06) { 
      std::cout << "Chi2 Test failed ! " << std::endl;
      href->Chi2Test(h1,"UUP"); // print all chi2 test info
      return 1;
   }
   return 0; 
}  

int testGetRandom(TF2 * f, TH1 * h1, const TH2 * href = 0) { 


   // test first the time
   h1->Reset();
   
   TStopwatch w; 
   w.Start();
   double x[2] = {0,0};
   for (int i = 0; i < n; ++i) {
      f->GetRandom2(x[0],x[1]);
      h1->Fill(x[0],x[1]); 
   }

   w.Stop(); 
   double time = w.CpuTime()*1.E9/n; 


   if (href != 0) { 
      double prob = href->Chi2Test(h1,"UU");
      std::cout << "Time using TF1::GetRandom()    \t=\t " << time << "\tns/call \t\tChi2 Prob = "<< prob << std::endl;
      if (prob < 1E-06) { 
         std::cout << "Chi2 Test failed ! " << std::endl;
         href->Chi2Test(h1,"UUP"); // print all chi2 test info
         return 2;
      }
   }
   else 
      std::cout << "Time using TF1::GetRandom()    \t=\t " << time << "\tns/call\n";
   return 0; 
}  


int unuranMulti2D() { 

   // check if using a random seed
   if (useRandomSeed) gRandom->SetSeed(0);

   // switch off printing of  info messages from chi2 test
   gErrorIgnoreLevel = 1001; 


   gSystem->Load("libMathCore");
   gSystem->Load("libUnuran");

   // simple test of unuran

   

   TH2D * h1 = new TH2D("h1","UNURAN gaussian 2D distribution",100,-10,10,100,-10,10);
   TH2D * h2 = new TH2D("h2","TF1::GetRandom gaussian 2D distribution",100,-10,10,100,-10,10);

   TH2D * h3 = new TH2D("h3","UNURAN truncated gaussian 2D distribution",100,-1,1,100,-1,1);
   TH2D * h4 = new TH2D("h4","TF1::GetRandom truncated gaussian 2D distribution",100,-1,1,100,-1,1);
   

   TF2 * f = new TF2("g2d",gaus2d,-10,10,-10,10,3); 
   double par[3] = {1,1,0.5}; 
   f->SetParameters(par); 

   TF2 * flog = new TF2("logg2d",log_gaus2d,-10,10,-10,10,3); 
   flog->SetParameters(par); 


   f->SetNpx(100);
   f->SetNpy(100);
   std::cout << " Nimber of function points in TF1, Npx = " << f->GetNpx() << " Npy =  "  << f->GetNpy() << std::endl;


   std::cout << "Test using an undefined domain :\n\n";

   // test first with getrandom
   testGetRandom(f,h2);

   // create multi-dim distribution
   TUnuranMultiContDist dist(f); 

   // use directly function interfaces
   //ROOT::Math::Functor f2( *f, 2);
   //TUnuranMultiContDist dist(f2); 

   TUnuran unr(gRandom,2);  // 2 is debug level 
   
   int iret = 0; 
   TH2 * href = h2; 

   //vnrou method (requires only pdf) 
   std::string method = "vnrou";
   iret |= testUnuran(unr, method, dist, h1, href);


   //hitro method (requires only pdf)
   method = "hitro";
   iret |= testUnuran(unr, method, dist, h1, href);
   
   //gibbs requires log of pdf and derivative
//#define USE_GIBBS   
#ifdef USE_GIBBS
   method = "gibbs";
   // need to create a new  multi-dim distribution with log of pdf
   TUnuranMultiContDist logdist(flog,0,true); 
   iret |= testUnuran(unr, method, logdist, h1, href);
#endif

   // test setting the mode
   cout << "\nTest setting the mode in Unuran distribution:\n\n"; 
   double m[2] = {0,0};
   dist.SetMode(m);

   method = "vnrou";
   iret |= testUnuran(unr, method, dist, h1, href);

   method = "hitro";
   iret |= testUnuran(unr, method, dist, h1, href);

#ifdef USE_GIBBS
   method = "gibbs";
   logdist.SetMode(m);
   iret |= testUnuran(unr, method, logdist, h1, href);
#endif


//    std::cout << " chi2 test of histogram generated with Unuran vs histogram generated with TF1::GetRandom " << std::endl;
//    h1->Chi2Test(h2,"UUP");

//#ifdef LATER

   double xmin[2] = { -1, -1 }; 
   double xmax[2] = { 1, 1 }; 

   f->SetRange(xmin[0],xmin[1],xmax[0],xmax[1]);
   // change function domain (not yet implemented in unuran for multidim functions)
   dist.SetDomain(xmin,xmax);

   const double *xlow = dist.GetLowerDomain(); 
   const double *xup = dist.GetUpperDomain(); 
   cout << "\nTest truncated distribution in domain [ " << xlow[0] << " : " << xup[0] 
        << " , " << xlow[1] << " : " << xup[1] << " ] :\n\n"; 
   

   testGetRandom(f,h4);

   method = "vnrou";
   iret |= testUnuran(unr, method, dist, h3, h4);
   method = "hitro";
   iret |= testUnuran(unr, method, dist, h3, h4);
#ifdef USE_GIBBS
   logdist.SetDomain(xmin,xmax);
   method = "gibbs";      
   iret |= testUnuran(unr, method, logdist, h3, h4);
#endif

//#ifdef LATER
   TCanvas * c1 = new TCanvas("c1_unuran2D","Multidimensional distribution",10,10,900,900); 
   c1->Divide(2,2);
#ifdef NO_TRUNC
   TCanvas * c1 = new TCanvas("c1_unuran2D","Multidimensional distribution",10,10,500,500); 
   c1->Divide(1,2);
#endif


   c1->cd(1); 
   h1->Draw("col");

   
   c1->cd(2);
   h2->Draw("col");

   c1->cd(3); 
   h3->Draw("col");
   c1->cd(4); 
   h4->Draw("col");
//#endif

   if (iret != 0) 
      std::cerr <<"\n\nUnuRan 2D Continous Distribution Test:\t  Failed !!!!!!!\n" << std::endl;
   else 
      std::cerr << "\n\nUnuRan 2D Continous Distribution Test:\t OK\n" << std::endl;

   return iret; 

}

#ifndef __CINT__
int main(int argc, char **argv)
{
   int iret = 0; 
   if (argc > 1) { 
      TApplication theApp("App",&argc,argv);
      iret =  unuranMulti2D();
      theApp.Run();
   } 
   else 
      iret =  unuranMulti2D();
   
   return iret; 
}
#endif
