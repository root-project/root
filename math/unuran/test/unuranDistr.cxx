// test using 1D Distribution object interface
// and compare results and CPU performances using TF1::GetRandom
//
// run within ROOT (.x unuranDistr.cxx+) or pass any extra parameter in the command line to get  
// a graphics output 


#include "TStopwatch.h"

#include "TUnuran.h"
#include "TUnuranContDist.h"

#include "TH1.h"
#include "TF1.h"

#include "TRandom3.h"
#include "TSystem.h"
#include "TStyle.h"

#include "TApplication.h"
#include "TCanvas.h"

#include "Math/DistFunc.h"
#include <cmath>
#include <cassert>

#include "TError.h"

#include <iostream> 

using std::cout; 
using std::endl; 

int n = 5000000;

bool useRandomSeed = false;   // to use a random seed different every time

double par[1] = {1}; // function parameters

double norm(double *x, double *p) { 
   return ROOT::Math::normal_pdf(x[0],p[0]); 
}

double cdf(double *x, double *p) { 
   return ROOT::Math::normal_cdf(x[0],p[0]); 
}
double cdf_trunc(double *x, double *p) { 
   double a1 = ROOT::Math::normal_cdf(p[1],p[0]); 
   double a2 = ROOT::Math::normal_cdf(p[2],p[0]); 
   
   return ( ROOT::Math::normal_cdf(x[0],p[0]) - a1 )/(a2-a1); 
}

class DistTest { 
public: 
   DistTest(TF1 * f) : 
      fCdf(f), 
      fHref(0)
   { 
      // generate reference histo for distribution using cdf 

      // make ref histo (uniform histo between 0,1
      fHref = new TH1D("Href","uniform ref histo",100,0,1);
      for (int i = 0; i < n; ++i) 
         fHref->Fill(gRandom->Rndm() );
   }

   void SetCdf(TF1 *f) { 
      fCdf = f; 
   }

   int testUnuran(TUnuran & unr) { 

      assert(fHref != 0);
      assert(fCdf != 0);

      // test first the time
      TStopwatch w; 

      w.Start(); 
      for (int i = 0; i < n; ++i) 
         unr.Sample(); 

      w.Stop(); 
      double time = w.CpuTime()*1.E9/n; 


      TH1D htmp("htmp","gaussian generated cdf",100,0,1.);   
      // test quality (use cdf to avoid zero bins)
      int n2 = n/100;
      double x;
      for (int i = 0; i<n2; ++i) { 
         x =  unr.Sample(); 
         htmp.Fill( fCdf->Eval(x) ); 
      } 
      double prob = fHref->Chi2Test(&htmp,"UU");
      cout << "Time using Unuran  " << unr.MethodName() << "   \t=\t " << time << "\tns/call \t\tChi2 Prob = "<< prob << endl;
      if (prob < 1E-06) { 
         std::cout << "Chi2 Test failed ! " << std::endl;
         fHref->Chi2Test(&htmp,"UUP"); // print all chi2 test info
         return 1;
      }
      return 0; 
   }  

   int testGetRandom(TF1 * f) { 

      assert(fHref != 0);
      assert(fCdf != 0);

      // test first the time

      TStopwatch w; 
      w.Start();
      for (int i = 0; i < n; ++i) {
         f->GetRandom(); 
      }

      w.Stop(); 
      double time = w.CpuTime()*1.E9/n; 


      TH1D htmp("htmp","gaussian generated cdf",100,0,1.);   
      // test quality (use cdf to avoid zero bins)
      int n2 = n/100;
      for (int i = 0; i<n2; ++i) { 
         double x =  f->GetRandom(); 
         htmp.Fill( fCdf->Eval(x) ); 
      } 
      double prob = fHref->Chi2Test(&htmp,"UU");
      cout << "Time using TF1::GetRandom()    \t=\t " << time << "\tns/call \t\tChi2 Prob = "<< prob << endl;
      if (prob < 1E-06) { 
         std::cout << "Chi2 Test failed ! " << std::endl;
         fHref->Chi2Test(&htmp,"UUP"); // print all chi2 test info
         return 2;
      }
      return 0; 
   }  

private: 

   TF1 * fCdf; // cumulative distribution
   TH1 * fHref; // reference histo

}; 

int unuranDistr() { 

   int iret = 0;

   // switch off printing of  info messages from chi2 test
   gErrorIgnoreLevel = 1001; 

   // check if using a random seed
   if (useRandomSeed) gRandom->SetSeed(0);


   gSystem->Load("libUnuran");

   // simple test of unuran


   TH1D * h1 = new TH1D("h1","gaussian distribution",100,-10,10);
   TH1D * h2 = new TH1D("h2","gaussian distribution",100,-10,10);

   TH1D * h1u = new TH1D("h1u","test gaussian dist",100,0,1);
   TH1D * h2u = new TH1D("h2u","test gaussian dist",100,0,1);

   

   TF1 * f = new TF1("n",norm,-10,10,1); 
   f->SetParameters(par); 

   TF1 * fc = new TF1("c",cdf,0,1,1); 
   fc->SetParameters(par); 

      
   // tester class
   DistTest t(fc);


   // create unuran 1D distribution
   TUnuranContDist dist(f); 

   // create unuran class 
   TUnuran unr(gRandom,2); 

   // test all unuran methods
   bool ret = false; 


   // arou: needs pdf and dpdf (estimated numerically in dist class)
   ret = unr.Init(dist,"arou"); 
   if (!ret) { 
      std::cerr << "Error initializing unuran with method " << unr.MethodName() << endl; 
      iret = -1;
   } 
   else 
      iret |= t.testUnuran(unr); 

   // nrou (needs only pdf , mode is an option) 
   ret = unr.Init(dist,"nrou"); 
   if (!ret) { 
      std::cerr << "Error initializing unuran with method " << unr.MethodName()  << endl; 
      iret =  -2;
   } 
   else 
      iret |= t.testUnuran(unr); 


   // tdr: needs pdf and dpdf (estimated numerically in dist class)
   ret = unr.Init(dist,"tdr"); 
   if (!ret) { 
      std::cerr << "Error initializing unuran with method " << unr.MethodName()  << endl; 
      iret =  -3;
   } 
   else 
      iret |= t.testUnuran(unr); 


   dist.SetCdf(fc);

   // hinv (needs cdf , pdf and dpdf are  optionally)
   ret = unr.Init(dist,"hinv"); 
   if (!ret) { 
      std::cerr << "Error initializing unuran with method " << unr.MethodName()  << endl; 
      iret =  -4;
   } 
   else 
      iret |= t.testUnuran(unr); 
   

   // ninv (needs cdf, pdf is an option) 
   ret = unr.Init(dist,"ninv"); 
   n/= 10; // method is too slow
   if (!ret) { 
      std::cerr << "Error initializing unuran with method " << unr.MethodName()  << endl; 
      iret =  -5;
   } 
   else 
      iret |= t.testUnuran(unr); 


   dist.SetMode( 0.0 );
   dist.SetPdfArea(1. );

   // srou (need pdf mode sand area)
   ret = unr.Init(dist,"srou"); 
   if (!ret) { 
      std::cerr << "Error initializing unuran with method " << unr.MethodName()  << endl; 
      iret =  -6;
   } 
   else 
      iret |= t.testUnuran(unr); 

   // srou (need pdf mode sand area)
   ret = unr.Init(dist,"ssr"); 
   if (!ret) { 
      std::cerr << "Error initializing unuran with method " << unr.MethodName()  << endl; 
      iret =  -7;
   } 
   else 
      iret |= t.testUnuran(unr); 

   n*= 10;
   ret = unr.Init(dist,"utdr"); 
   if (!ret) { 
      std::cerr << "Error initializing unuran with method " << unr.MethodName()  << endl; 
      iret =  -8;
   } 
   else 
      iret |= t.testUnuran(unr); 


   // test with TF1::GetRandom
   std::cout << "\n";
   iret |= t.testGetRandom(f);

   std::cout << "\nTest truncated distribution:\n\n";
   
   dist.SetDomain(-1.,1.);
   // change cdf for tester
   TF1 * fc2 = new TF1("fc2",cdf_trunc,-1,1,3);
   fc2->SetParameters(par[0],-1,1.);
   t.SetCdf(fc2);

   ret = unr.Init(dist,"auto"); 
   if (!ret) { 
      std::cerr << "Error initializing unuran with method " << unr.MethodName()  << endl; 
      iret =  -10;
   } 
   else 
      iret |= t.testUnuran(unr); 

   f->SetRange(-1,1);
   iret |= t.testGetRandom(f);



   TCanvas * c1 = new TCanvas("c1_unuran1D","Onedimensional distribution",10,10,1000,1000); 
   c1->Divide(2,2);
   gStyle->SetOptFit();

   // remove the domain
   dist.SetDomain(0,0);
   //f->SetRange(1,-1);
   f->SetRange(-10,10); // set verly low and large values is enough
   // show now some plots 
   ret = unr.Init(dist,"auto"); 
   if (!ret) { 
      std::cerr << "Error initializing unuran with method " << unr.MethodName()  << endl; 
      iret =  -20;
   } 
   int n2 = n/10;
   for (int i = 0; i < n2; ++i) {
      double x1 = unr.Sample();
      h1->Fill(  x1 ); 
      h1u->Fill( fc->Eval( x1 ) ); 
      double x2 = f->GetRandom();
      h2->Fill(  x2 ); 
      h2u->Fill( fc->Eval( x2 ) ); 
   }
  
   c1->cd(1);
   h1->Draw();
   h1->Fit("gaus","Q");
   std::cout << "\nFit result on data with unuran: " << std::endl;
   TF1 * f1 = h1->GetFunction("gaus");
   std::cout << "Gaus Fit chi2 = " << f1->GetChisquare() << " ndf = " << f1->GetNDF() << std::endl;
   std::cout << "Gaus Fit Prob = " << f1->GetProb() << std::endl;
   if (  f1->GetProb() < 1.E-6) iret = 11; 
   c1->cd(2);
   h1u->Draw("Error");
   h1u->Fit("pol0","Q");
   TF1 * f1u = h1u->GetFunction("pol0");
   std::cout << "CDF Fit chi2 = " << f1u->GetChisquare() << " ndf = " << f1u->GetNDF() << std::endl;
   std::cout << "CDF Fit Prob = " << f1u->GetProb() << std::endl;
   if (  f1u->GetProb() < 1.E-6) iret = 12; 
   



   c1->cd(3);
   h2->Draw("same");
   h2->Fit("gaus","Q");
   std::cout << "\nFit result on data  with GetRandom: " << std::endl;
   f1 = h2->GetFunction("gaus");
   std::cout << "Gaus Fit chi2 = " << f1->GetChisquare() << " ndf = " << f1->GetNDF() << std::endl;
   std::cout << "Gaus Fit Prob = " << f1->GetProb() << std::endl;
   if (  f1->GetProb() < 1.E-6) iret = 13;    
   c1->cd(4);

   h2u->Draw("Error");
   h2u->Fit("pol0","Q");
   f1 = h2u->GetFunction("pol0");
   std::cout << "Fit chi2 = " << f1->GetChisquare() << " ndf = " << f1->GetNDF() << std::endl;
   std::cout << "Fit Prob = " << f1->GetProb() << std::endl;
   if (  f1->GetProb() < 1.E-6) iret = 14;    


   std::cout << "Chi2 test Gaussian histograms :\t";
   h1->Chi2Test(h2,"UUP");
   std::cout << "Chi2 test Uniform histograms  :\t";
   h1u->Chi2Test(h2u,"UUP");

   if (iret != 0) 
      std::cerr <<"\n\nUnuRan Continous Distribution Test:\t  Failed !!!!!!!\n" << std::endl;
   else 
      std::cerr << "\n\nUnuRan  Continous Distribution Test:\t OK\n" << std::endl;
   return iret; 
   

   return iret; 

}

#ifndef __CINT__
int main(int argc, char **argv)
{
   int iret = 0; 
   if (argc > 1) { 
      TApplication theApp("App",&argc,argv);
      iret =  unuranDistr();
      theApp.Run();
   } 
   else 
      iret =  unuranDistr();
   
}
#endif
