// test using Multi-dim   Distribution object interface
// and compare results and CPU performances using TF3::GetRandom in case of 3D 
// and test also case of dim = 10 and 100
//
// run within ROOT (.x unuranMultiDim.cxx+) or pass any extra parameter in the command line to get  
// a graphics output 


#include "TStopwatch.h"
#include "TUnuran.h"
#include "TUnuranMultiContDist.h"

#include "TH3.h"
#include "TF3.h"
#include "TCanvas.h"
#include "TMath.h"

#include "TRandom3.h"
#include "TSystem.h"
#include "TApplication.h"
#include "TError.h"

// #include "Math/ProbFunc.h"
// #include "Math/DistFunc.h"


#include <iostream> 
#include <cmath>
#include <cassert>

#include <vector>

using std::cout; 
using std::endl; 

int n;

bool useRandomSeed = false;   // to use a random seed different every time

double gaus3d(double *x, double *p) { 

   double sigma_x = p[0]; 
   double sigma_y = p[1];
   double sigma_z = p[2];
   double rho = p[3]; 
   double u = x[0] / sigma_x ;
   double v = x[1] / sigma_y ;
   double w = x[2] / sigma_z ;
   double c = 1 - rho*rho ;
   double result = (1 / (2 * TMath::Pi() * sigma_x * sigma_y * sigma_z * sqrt(c))) 
      * exp (-(u * u - 2 * rho * u * v + v * v + w*w) / (2 * c));
   return result;
}
double log_gaus3d(double *x, double *p) { 
   return std::log(gaus3d(x,p) );
}

double gaus10d(double * x, double *) {
  int i;
  double tmp = 0.;
  for (i=0; i<10; i++)
    tmp -= x[i] * x[i];
  return exp(tmp/2.);
} 

double gaus100d(double * x, double *) {
  int i;
  double tmp = 0.;
  for (i=0; i<100; i++)
    tmp -= x[i] * x[i];
  return exp(tmp/2.);
} 


int testUnuran(TUnuran & unr, const std::string & method, const TUnuranMultiContDist & dist, TH3 * h1, const TH3 * href = 0, const int dim = 3 ) { 


   assert (dim >=3);
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
   std::vector<double> x(dim); 
   for (int i = 0; i < n; ++i) {
      unr.SampleMulti( &x[0]);
      h1->Fill(x[0],x[1],x[2]);
   }

   w.Stop(); 
   double time = w.CpuTime()*1.E9/n; 

   cout << "Time using Unuran  " << unr.MethodName() << "   \t=\t " << time << "\tns/call\t";
   if (href != 0)  { 
      double prob = href->Chi2Test(h1,"UU");
      double ksprob = href->KolmogorovTest(h1);
      std::cout << "\tChi2 Prob = "<< prob << "\tKS Prob = " << ksprob << std::endl;
      // use lower value since hitro is not very precise 
      // use ks for hitro since chi2 gives too big error  
      if (unr.MethodName() == "hitro") prob = ksprob; 
      if (prob < 1.E-6 ) { 
         std::cout << "\nChi2 Test failed ! " << std::endl;
         href->Chi2Test(h1,"UUP"); // print all chi2 test info
         return 1;
      }
   }
   else 
      std::cout << std::endl;

   return 0; 
}  

int testGetRandom(TF3 * f, TH3 * h1, const TH3 * href = 0) { 


   // test first the time
   h1->Reset();
 
   TStopwatch w; 
   w.Start();
   double x[3] = {0,0,0};
   for (int i = 0; i < n; ++i) {
      f->GetRandom3(x[0],x[1],x[2]);
      h1->Fill(x[0],x[1],x[2]); 
   }

   w.Stop(); 
   double time = w.CpuTime()*1.E9/n; 


   if (href != 0) { 
      double prob = href->Chi2Test(h1,"UU");
      double ksprob = href->KolmogorovTest(h1);
      std::cout << "Time using TF1::GetRandom()    \t=\t " << time << "\tns/call \t\tChi2 Prob = "<< prob 
                << "\tKS Prob = " << ksprob << std::endl;
      if (prob < 1E-06) { 
         std::cout << "\tChi2 Test failed ! " << std::endl;
         href->Chi2Test(h1,"UUP"); // print all chi2 test info
         return 2;
      }
   }
   else 
      std::cout << "Time using TF1::GetRandom()    \t=\t " << time << "\tns/call\n";
   return 0; 
}  


int  unuranMultiDim() { 

   // switch off printing of  info messages from chi2 test
   gErrorIgnoreLevel = 1001; 

   // check if using a random seed
   if (useRandomSeed) gRandom->SetSeed(0);

   gSystem->Load("libMathCore");
   gSystem->Load("libUnuran");

   // simple test of unuran
   n   = 100000;

   

   TH3D * h1 = new TH3D("h1","UNURAN gaussian 3D distribution",50,-10,10,50,-10,10,50,-10,10);
   TH3D * h2 = new TH3D("h2","TF1::GetRandom gaussian 3D distribution",50,-10,10,50,-10,10,50,-10,10);

 
   TH3D * h3 = new TH3D("h3","UNURAN truncated gaussian 3D distribution",50,-1,1,50,-1,1,50,-1,1);
   TH3D * h4 = new TH3D("h4","TF1::GetRandom truncated gaussian 3D distribution",50,-1,1,50,-1,1,50,-1,1);
  

   TF3 * f = new TF3("g3d",gaus3d,-10,10,-10,10,-10,10,4); 
   double par[4] = {2,2,2,0.5}; 
   f->SetParameters(par); 
   TF3 * flog = new TF3("logg3d",log_gaus3d,-10,10,-10,10,-10,10,4); 
   flog->SetParameters(par); 



   std::cout << "Test using an undefined domain :\n\n";

   std::cout << "Function Points used in GetRandom: [ " << f->GetNpx() << " , "  
             << f->GetNpy() <<  " , "  << f->GetNpz() << " ]" << std::endl;
   testGetRandom(f,h1);

   // test first with getrandom
   // need to have a larger value to get good quality
   int np = 100;
   f->SetNpx(np);   f->SetNpy(np);   f->SetNpz(np);
   std::cout << "Function Points used in GetRandom: [ " << f->GetNpx() << " , "  
             << f->GetNpy() <<  " , "  << f->GetNpz() << " ]" << std::endl;

   testGetRandom(f,h2,h1);

   *h1 = *h2; 
   np = 200;
   f->SetNpx(np);   f->SetNpy(np);   f->SetNpz(np);
   std::cout << "Function Points used in GetRandom: [ " << f->GetNpx() << " , "  
             << f->GetNpy() <<  " , "  << f->GetNpz() << " ]" << std::endl;

   testGetRandom(f,h2,h1);


   // create multi-dim distribution
   TUnuranMultiContDist dist(f); 


   TUnuran unr(gRandom,2);  // 2 is debug level 
   
   int iret = 0; 
   TH3 * href = new TH3D(*h2); 

   //vnrou method (requires only pdf) 
   std::string method = "vnrou";
   iret |= testUnuran(unr, method, dist, h1, href);


   //hitro method (requires only pdf)
   method = "hitro";
   iret |= testUnuran(unr, method, dist, h1, href);
   
   //gibbs requires log of pdf and derivative
#ifdef USE_GIBBS
   method = "gibbs";
   // need to create a new  multi-dim distribution with log of pdf
   TUnuranMultiContDist logdist(flog,0,true); 
   iret |= testUnuran(unr, method, logdist, h1, href);
#endif

   // test setting the mode
   cout << "\nTest setting the mode in Unuran distribution:\n\n"; 
   double m[3] = {0,0,0};
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

   cout << "\nTest truncated distribution:\n\n"; 

   double xmin[3] = { -1, -1, -1 }; 
   double xmax[3] = {  1,  1,  1 }; 

   f->SetRange(xmin[0],xmin[1],xmin[2],xmax[0],xmax[1],xmax[2]);
   // change function domain (not yet implemented in unuran for multidim functions)
   dist.SetDomain(xmin,xmax);

   np = 30;
   f->SetNpx(np);   f->SetNpy(np);   f->SetNpz(np);
   std::cout << "Function Points used in GetRandom: [ " << f->GetNpx() << " , "  
             << f->GetNpy() <<  " , "  << f->GetNpz() << " ]" << std::endl;
   testGetRandom(f,h3);

   href = h3;
   np = 100;
   f->SetNpx(np);   f->SetNpy(np);   f->SetNpz(np);
   std::cout << "Function Points used in GetRandom: [ " << f->GetNpx() << " , "  
             << f->GetNpy() <<  " , "  << f->GetNpz() << " ]" << std::endl;
   testGetRandom(f,h4,href);
   href = h4;

   method = "vnrou";
   iret |= testUnuran(unr, method, dist, h3, href);
   method = "hitro";
   iret |= testUnuran(unr, method, dist, h3, href);


#ifdef USE_GIBBS
   method = "gibbs";      
   logdist.SetDomain(xmin,xmax);
   iret |= testUnuran(unr, method, logdist, h3, href);
#endif


   TCanvas * c1 = new TCanvas("c1_unuranMulti","Multidimensional distribution",10,10,900,900); 
   c1->Divide(2,2);
   
   c1->cd(1);   h1->Draw();
   c1->cd(2);   h2->Draw();
   c1->cd(3);   h3->Draw();
   c1->cd(4);   h4->Draw();

   // make a ref histo for 10 dim using first 3 dim
   c1->Update();


 
   TH3D * hrefN = new TH3D("hrefN","UNURAN gaussian ref N-Dim distribution (first 3 dim)",30,-3,3,30,-3,3,30,-3,3);
   TH3D * h10v    = new TH3D("h10v","UNURAN gaussian N-Dim distribution (first 3 dim)",30,-3,3,30,-3,3,30,-3,3);
   TH3D * h10h    = new TH3D("h10h","UNURAN gaussian N-Dim distribution (first 3 dim)",30,-3,3,30,-3,3,30,-3,3);
   TH3D * h100    = new TH3D("h100","UNURAN gaussian N-Dim distribution (first 3 dim)",30,-3,3,30,-3,3,30,-3,3);

   int scale = 5;

   double par2[4] = {1,1,1,0.}; 
   f->SetParameters(par2); 
   TUnuranMultiContDist dist3(f); 
   method = "vnrou";
   n/= scale;
   iret |= testUnuran(unr, method, dist3, hrefN);

   cout << "\nTest 10 dimension:       (be patient , it takes time....)\n\n"; 

   TF1 * f10 = new TF1("g10d",gaus10d,-10,10,0); 
   TUnuranMultiContDist dist10(f10,10); 


   TCanvas * c2 = new TCanvas("c2_unuranMulti","Multidimensional distribution",100,10,900,900); 
   c2->Divide(2,2);
   
   c2->cd(1);   hrefN->Draw();
   
   //n/= scale;
   method = "vnrou";
   iret |= testUnuran(unr, method, dist10, h10v, hrefN,10);
   c2->cd(2);   h10v->Draw();
   c2->Update();

   //n*=scale;
   method = "hitro;thinning=20";
   iret |= testUnuran(unr, method, dist10, h10h, hrefN,10);
   c2->cd(3);   h10h->Draw();
   c2->Update();


   // 100 dim
   cout << "\nTest 100 dimension: (  be patient , it takes time....)\n\n"; 
   TF1 * f100 = new TF1("g100d",gaus100d,-10,10,0); 
   TUnuranMultiContDist dist100(f100,100); 
   
   //   scale = 5;
   //  n/=scale;
   std::cout << "number of events to be generated  = " << n << endl;
   method = "hitro;thinning=200";
   iret |= testUnuran(unr, method, dist100, h100, hrefN,100);
//    n/= 100;
//    method = "vnrou";
//    iret |= testUnuran(unr, method, dist100, hN, hrefN,100);


   c2->cd(4);   h100->Draw();
   c2->Update();


   if (iret != 0) 
      std::cerr <<"\n\nUnuRan MultiDim Continous Distribution Test:\t  Failed !!!!!!!\n" << std::endl;
   else 
      std::cerr << "\n\nUnuRan MultiDim Continous Distribution Test:\t OK\n" << std::endl;
   return iret; 

}

#ifndef __CINT__
int main(int argc, char **argv)
{
   int iret = 0; 
   if (argc > 1) { 
      TApplication theApp("App",&argc,argv);
      iret =  unuranMultiDim();
      theApp.Run();
   } 
   else 
      iret =  unuranMultiDim();
   
   return iret; 
}
#endif

