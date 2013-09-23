// test program for quasi random numbers

#include "Math/QuasiRandom.h"
#include "Math/Random.h"
#include "TH2.h"
#include "TCanvas.h"
#include "TApplication.h"
//#include "TSopwatch.h"

#include <iostream>

using namespace ROOT::Math;

int testQuasiRandom() { 

   const int n = 10000;

   TH2D * h0 = new TH2D("h0","Pseudo-random Sequence",200,0,1,200,0,1);
   TH2D * h1 = new TH2D("h1","Sobol Sequence",200,0,1,200,0,1);
   TH2D * h2 = new TH2D("h2","Niederrer Sequence",200,0,1,200,0,1);

   RandomMT         r0; 
   QuasiRandomSobol r1(2); 
   QuasiRandomNiederreiter r2(2); 
   

   // generate n sequences

   double x[2];
   for (unsigned int i = 0; i < n; ++i)  { 
      r0.RndmArray(2,x);
      h0->Fill(x[0],x[1]);      
   }


   for (unsigned int i = 0; i < n; ++i)  { 
      r1.Next(x); 
      h1->Fill(x[0],x[1]);      
   }

   double vx[2*n];
   r2.RndmArray(n,vx);
   for (unsigned int i = 0; i < n; ++i)  {
      h2->Fill(vx[2*i],vx[2*i+1]);      
   }


   TCanvas * c1 = new TCanvas("c1","Random sequence",600,1200); 
   c1->Divide(1,3);

   c1->cd(1);

   h0->Draw("COLZ");
   
   c1->cd(2);


   // check uniformity 
   h1->Draw("COLZ");

   c1->cd(3);

   h2->Draw("COLZ");

   gPad->Update();


   // test number of empty bins

   int nzerobins0 = 0;
   int nzerobins1 = 0;
   int nzerobins2 = 0;
   for (int i = 1; i <= h1->GetNbinsX(); ++i) { 
      for (int j = 1; j <= h1->GetNbinsY(); ++j) { 
         if (h0->GetBinContent(i,j) == 0 ) nzerobins0++;
         if (h1->GetBinContent(i,j) == 0 ) nzerobins1++;
         if (h2->GetBinContent(i,j) == 0 ) nzerobins2++;
      }
   }

   std::cout << "number of empty bins for pseudo-random = " << nzerobins0 << std::endl;
   std::cout << "number of empty bins for Sobol         = " << nzerobins1 << std::endl;
   std::cout << "number of empty bins for Niederreiter  = " << nzerobins2 << std::endl;

   int iret = 0; 
   if (nzerobins1 >= nzerobins0 ) iret += 1; 
   if (nzerobins2 >= nzerobins0 ) iret += 2; 
   return iret; 
}


int main(int argc, char **argv) 
{ 

   std::cout << "***************************************************\n"; 
   std::cout << " TEST QUASI-RANDOM generators "<< std::endl;
   std::cout << "***************************************************\n\n"; 

   int iret = 0;
   if (argc > 1) { 
      TApplication theApp("App",&argc,argv);
      iret = testQuasiRandom();
      theApp.Run();
   } 
   else 
      iret = testQuasiRandom();

   if (iret != 0) std::cerr << "TEST QUASI-RANDOM FAILED " << std::endl; 

   return iret;




}
