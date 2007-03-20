#include "TUnuran.h"
#include "TUnuranDiscrDist.h"

#include "TH1.h"
#include "TMath.h"
#include "TF1.h"
#include "TRandom3.h"
#include "TApplication.h"
#include "TCanvas.h"
#include "TStopwatch.h"

#include "Math/DistFunc.h"

#include <iostream>

//#define DEBUG
#ifdef DEBUG
int n = 100;
#else
int n = 1000000;
#endif

double pmf(double * x, double * p) { 

   double y = ROOT::Math::poisson_pdf(int(x[0]),p[0]);
//   std::cout << x[0] << " f(x) = " << y << std::endl;
   return y; 
}

int testUnuran(TUnuran & unr, TH1 * h1, const TH1 * href, bool weightHist = false ) { 


   // test first the time
   TStopwatch w; 

   w.Start(); 
   for (int i = 0; i < n; ++i) 
      unr.SampleDiscr(); 

   w.Stop(); 
   double time = w.CpuTime()*1.E9/n; 


   // test quality (use cdf to avoid zero bins)
   h1->Reset();
   int n2 = n/10;
   int x;
   for (int i = 0; i<n2; ++i) { 
      x =  unr.SampleDiscr(); 
      h1->Fill( double( x) ); 
   } 
   double prob; 
   if (weightHist) 
      prob = href->Chi2Test(h1,"WW");
   else
      prob = href->Chi2Test(h1,"UU");
   std::cout << "Time using Unuran  " << unr.MethodName() << "   \t=\t " << time << "\tns/call \t\tChi2 Prob = "<< prob << std::endl;
   if (prob < 1E-06) { 
      std::cout << "Chi2 Test failed for method " << unr.MethodName() << std::endl;
      if (weightHist) 
         href->Chi2Test(h1,"WWP"); // print all chi2 test info
      else 
         href->Chi2Test(h1,"UUP"); // print all chi2 test info

      return 1;
   }
   return 0; 
}  


int unuranDiscrete() {

   int iret = 0;

   TH1D * h0 = new TH1D("h0","reference prob",10,-0.5,9.5);
   TH1D * h1 = new TH1D("h1","UNURAN gen prob",10,-0.5,9.5);

   TH1D * h2 = new TH1D("h2","reference Poisson prob",20,0,20);
   TH1D * h3 = new TH1D("h3","UNURAN gen prob",20,0,20);

   double p[10] = {1.,2.,3.,5.,3.,2.,1.,0.5,0.3,0.5 };
   for (int i = 0; i< 10; ++i) { 
      h0->SetBinContent(i+1,p[i]);
      h0->SetBinError(i,0.0);
   }
   double sum = h0->GetSumOfWeights();
   //std::cout << " prob sum = " << sum << std::endl; 
      
   TRandom3 r; 
   r.SetSeed(0);

   TUnuran unr(&r,2); 

   TUnuranDiscrDist dist(p,p+10);

//    TUnuranDiscrDist fDDist = dist; 
//    std::cout << fDDist.ProbVec().size() << std::endl;

   std::cout << "Test  generation with a PV :\n\n";


   bool ret = unr.Init(dist,"method=dau");
   if (!ret) iret = -1;
   iret |= testUnuran(unr,h1,h0,true);

   ret = unr.Init(dist,"method=dgt");
   if (!ret) iret = -1;
   iret |= testUnuran(unr,h1,h0,true);

   // dss require the sum
   dist.SetProbSum(sum);
   ret = unr.Init(dist,"method=dss");
   if (!ret) iret = -1;
   iret |= testUnuran(unr,h1,h0,true);

   

   TCanvas * c1 = new TCanvas("c1_unuranDiscr","Discrete distribution",10,10,800,800); 
   c1->Divide(1,2);


   c1->cd(1);
   h1->Sumw2();
   h1->Scale( h0->GetSumOfWeights()/(h1->GetSumOfWeights() ) ); 
   h0->Draw();
   h1->Draw("Esame");

   // test with a function 

   std::cout << "\nTest  generation with a Probability function :\n\n";

   TF1 * f = new TF1("f",pmf,1,0,1);

   const double mu = 5; 
   f->SetParameter(0,mu);

   // make reference histo
   TStopwatch w; 
   w.Start(); 
   int n2 = n/10;
   for (int i = 0; i< n2; ++i) { 
      h2->Fill ( gRandom->Poisson(mu) );
   }
   double time = w.CpuTime()*1.E9/n2;
   std::cout << "Time using TRandom::Poisson = \t\t\t " << time << "\tns/call\n";


   TUnuranDiscrDist dist2 = TUnuranDiscrDist(f);

   // dari method (needs mode)
   dist2.SetMode(int(mu) );
   ret = unr.Init(dist2,"method=dari");
   if (!ret) iret = -1;

   iret |= testUnuran(unr,h3,h2);

   // dsrou method (needs mode and sum)

   dist2.SetProbSum(1);
   ret = unr.Init(dist2,"method=dsrou");
   if (!ret) iret = -1;

   iret |= testUnuran(unr,h3,h2);

   c1->cd(2);
   h2->Draw();
   h3->Draw("Esame");

   return iret; 
}


#ifndef __CINT__
int main(int argc, char **argv)
{
   int iret = 0; 
   if (argc > 1) { 
      TApplication theApp("App",&argc,argv);
      iret =  unuranDiscrete();
      theApp.Run();
   } 
   else 
      iret =  unuranDiscrete();
   
   if (iret != 0) 
      std::cerr <<"\n\nUnuRan Discrete Distribution Test:\t  Failed !!!!!!!\n" << std::endl;
   else 
      std::cout << "\n\nUnuRan Discrete Distribution Test:\t OK\n" << std::endl;
   return iret; 
}
#endif
