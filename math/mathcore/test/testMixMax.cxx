#include "TRandomGen.h"
#include "TH1.h"
#include "TStopwatch.h"
#include "TFitResult.h"
#include <iostream>

bool testMixMax(int nbin = 1.E6 ) {


   //int nbin = 1.E8; 
   TH1F h1("h1","h1",nbin, 0, 1);
   TStopwatch w; 

   TRandomMixMax r; 

   w.Start(); 
   for (int i = 0; i < 20*nbin; ++i) {

      double x = r.Rndm(); 
      h1.Fill(x); 
   }
   w.Print();

   auto res = h1.Fit("pol0", "L S");

   double chi2 = 2*res->MinFcnValue() / double(nbin-1); 

   std::cout << " chi2/ndf " << chi2 << std::endl;

   double chi2_std = sqrt( 2. * (nbin-1) ); 

   if (chi2 > double(nbin-1) + 100 * chi2_std ) {
      std::cout << "ERROR: Chi2 test failed for MixMax - " << chi2 << " larger than " << nbin-1 + 100 * chi2_std << std::endl;
      return false;
   }
   return true; 

}

int main() {
   bool ret = testMixMax();
   if (ret)
      std::cout << "Test OK" << std::endl;
   else
      std::cout << "Test Failed " << std::endl;
}
   
