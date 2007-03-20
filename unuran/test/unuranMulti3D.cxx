// test using Multi-dim (3D)  Distribution object interface
// and compare results and CPU performances using TF3::GetRandom


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

// #include "Math/ProbFunc.h"
// #include "Math/DistFunc.h"


#include <iostream> 
#include <cmath>

using std::cout; 
using std::endl; 

int n = 100000;

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
double log_gaus3d(double *x, double *p) { 
   return std::log(gaus3d(x,p) );
}


int testUnuran(TUnuran & unr, const std::string & method, const TUnuranMultiContDist & dist, TH3 * h1, const TH3 * href ) { 


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
   double x[3]; 
   for (int i = 0; i < n; ++i) {
      unr.SampleMulti(x);
      h1->Fill(x[0],x[1],x[2]);
   }

   w.Stop(); 
   double time = w.CpuTime()*1.E9/n; 

   double prob = href->Chi2Test(h1,"UU");
   cout << "Time using Unuran  " << unr.MethodName() << "   \t=\t " << time << "\tns/call \t\tChi2 Prob = "<< prob << endl;
   if (prob < 1E-06) { 
      std::cout << "Chi2 Test failed ! " << std::endl;
      href->Chi2Test(h1,"UUP"); // print all chi2 test info
      return 1;
   }
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


int  unuranMulti3D() { 


   gSystem->Load("libMathCore");
   gSystem->Load("libUnuran");

   // simple test of unuran

   

   TH3D * h1 = new TH3D("h1","UNURAN gaussian 3D distribution",50,-10,10,50,-10,10,50,-10,10);
   TH3D * h2 = new TH3D("h2","TF1::GetRandom gaussian 3D distribution",50,-10,10,50,-10,10,50,-10,10);

 
   TH3D * h3 = new TH3D("h3","UNURAN truncated gaussian 3D distribution",50,-1,1,50,-1,1,50,-1,1);
   TH3D * h4 = new TH3D("h4","TF1::GetRandom truncated gaussian 3D distribution",50,-1,1,50,-1,1,50,-1,1);
  

   TF3 * f = new TF3("g3d",gaus3d,-10,10,-10,10,-10,10,3); 
   double par[3] = {2,2,0.5}; 
   f->SetParameters(par); 
   TF3 * flog = new TF3("logg3d",log_gaus3d,-10,10,-10,10,-10,10,3); 
   flog->SetParameters(par); 


   TRandom3 r; 
   r.SetSeed(0);
   gRandom->SetSeed(0);


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

   h1 = h2; 
   np = 200;
   f->SetNpx(np);   f->SetNpy(np);   f->SetNpz(np);
   std::cout << "Function Points used in GetRandom: [ " << f->GetNpx() << " , "  
             << f->GetNpy() <<  " , "  << f->GetNpz() << " ]" << std::endl;

   testGetRandom(f,h2,h1);


   // create multi-dim distribution
   TUnuranMultiContDist dist(f); 


   TUnuran unr(&r,2);  // 2 is debug level 
   
   int iret = 0; 
   TH3 * href = h2; 

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
   TUnuranMultiContDist logdist(flog,true); 
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

   return iret; 

}

#ifndef __CINT__
int main(int argc, char **argv)
{
   int iret = 0; 
   if (argc > 1) { 
      TApplication theApp("App",&argc,argv);
      iret =  unuranMulti3D();
      theApp.Run();
   } 
   else 
      iret =  unuranMulti3D();
   
   if (iret != 0) 
      std::cerr <<"\n\nUnuRan 3D Continous Distribution Test:\t  Failed !!!!!!!\n" << std::endl;
   else 
      std::cout << "\n\nUnuRan 3D Continous Distribution Test:\t OK\n" << std::endl;
   return iret; 
}
#endif

