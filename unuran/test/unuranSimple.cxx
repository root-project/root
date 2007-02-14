
#include "TStopwatch.h"
#include "TUnuran.h"

#include "TH1.h"
#include "TF1.h"

#include "TRandom.h"
#include "TSystem.h"
#include "TRoot.h"
//#include "TApplication.h"
//#include "TRint.h"
#include "TVirtualFitter.h"
#include "TFitter.h"
#include "Math/DistFunc.h"

#include <iostream> 

using std::cout; 
using std::endl; 

void unuranSimple() { 

   // simple test of unuran


   TUnuran unr; 
   if (! unr.Init( "normal()", "method=arou") ) {
      cout << "Error initializing unuran" << endl;
      return;
   }

   int n = 1000000;
   TStopwatch w; 
   w.Start(); 

   for (int i = 0; i < n; ++i) 
      unr.Sample(); 

   w.Stop(); 
   cout << "Time using Unuran =\t\t " << w.CpuTime() << endl;

   w.Start();
   for (int i = 0; i < n; ++i) 
      gRandom->Gaus(0,1); 

   w.Stop(); 
   cout << "Time using TRandom::Gaus  =\t " << w.CpuTime() << endl;

   // test the quality by looking at the cdf

   TH1D * h1 = new TH1D("h1","cdf on the data ",100,0,1);
   for (int i = 0; i < n; ++i) {
      double x = unr.Sample();
      // x = gRandom->Gaus(0,1); 
      h1->Fill( ROOT::Math::normal_cdf( x , 1.0) ); 
   }
//    gSystem->Load("libMinuit2");
//    TVirtualFitter::SetDefaultFitter("Minuit2");
   std::cout << "Result of fitting a cdf to a constant function : " << std::endl;
   // need TRint();
   TFitter * fitter = new TFitter(); 
   TVirtualFitter::SetFitter(fitter);
   h1->Fit("pol0","Q");
   TF1 * f = (TF1*)gROOT->GetFunction("pol0");
   std::cout << "Fit chi2 = " << f->GetChisquare() << " ndf = " << f->GetNDF() << std::endl;
   std::cout << "Fit Prob = " << f->GetProb() << std::endl;

   h1->Draw();

   //delete fitter; 


}

#ifndef __CINT__
int main()
{
//   TApplication theApp("App", &argc, argv);
  //TRint theApp("TRint",&argc, argv);
  //theApp.Run();
   unuranSimple();
   return 0;
}
#endif

