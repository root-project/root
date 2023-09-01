/// \file
/// \ingroup tutorial_math
/// \notebook -js
/// Example of generating quasi-random numbers
///
/// \macro_image
/// \macro_output
/// \macro_code
///
/// \author Lorenzo Moneta

#include "Math/QuasiRandom.h"
#include "Math/Random.h"
#include "TH2.h"
#include "TCanvas.h"
#include "TStopwatch.h"

#include <iostream>

using namespace ROOT::Math;

int quasirandom(int n = 10000, int skip = 0) {

   TH2D * h0 = new TH2D("h0","Pseudo-random Sequence",200,0,1,200,0,1);
   TH2D * h1 = new TH2D("h1","Sobol Sequence",200,0,1,200,0,1);
   TH2D * h2 = new TH2D("h2","Niederrer Sequence",200,0,1,200,0,1);

   RandomMT         r0;
   // quasi random numbers need to be created giving the dimension of the sequence
   // in this case we generate 2-d sequence

   QuasiRandomSobol r1(2);
   QuasiRandomNiederreiter r2(2);

   // generate n random points

   double x[2];
   TStopwatch w; w.Start();
   for (int i = 0; i < n; ++i)  {
      r0.RndmArray(2,x);
      h0->Fill(x[0],x[1]);
   }
   std::cout << "Time for gRandom ";
   w.Print();

   w.Start();
   if( skip>0) r1.Skip(skip);
   for (int i = 0; i < n; ++i)  {
      r1.Next(x);
      h1->Fill(x[0],x[1]);
   }
   std::cout << "Time for Sobol ";
   w.Print();

   w.Start();
   if( skip>0) r2.Skip(skip);
   for (int i = 0; i < n; ++i)  {
      r2.Next(x);
      h2->Fill(x[0],x[1]);
   }
   std::cout << "Time for Niederreiter ";
   w.Print();

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
   std::cout << "number of empty bins for " << r1.Name() << "\t= " << nzerobins1 << std::endl;
   std::cout << "number of empty bins for " << r2.Name() << "\t= " << nzerobins2 << std::endl;

   int iret = 0;
   if (nzerobins1 >= nzerobins0 ) iret += 1;
   if (nzerobins2 >= nzerobins0 ) iret += 2;
   return iret;

}
