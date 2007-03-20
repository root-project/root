
#include "TStopwatch.h"
#include "TUnuran.h"

#include "TH1.h"
#include "TF1.h"

#include "TRandom.h"
#include "TSystem.h"
#include "TApplication.h"
//#include "TRint.h"
#include "TVirtualFitter.h"
#include "TFitter.h"
#include "Math/DistFunc.h"


#include "Math/Random.h"
#include "Math/GSLRndmEngines.h"

#include <iostream> 

#ifdef HAVE_CLHEP
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Random/MTwistEngine.h"
#endif


using std::cout; 
using std::endl; 

void unuranSimple() { 

   // simple test of unuran


   TUnuran unr; 
   if (! unr.Init( "normal()", "method=arou") ) {
      cout << "Error initializing unuran" << endl;
      return;
   }

   int n = 100000000;
   TStopwatch w;
   double time; 
   w.Start(); 

   for (int i = 0; i < n; ++i) 
      unr.Sample(); 

   w.Stop(); 
   time = w.CpuTime()*1.E9/n;
   cout << "Time using Unuran =\t\t " <<   time << "\tns/call" << endl;

   w.Start();
   for (int i = 0; i < n; ++i) 
      gRandom->Gaus(0,1); 
   w.Stop(); 
   time = w.CpuTime()*1.E9/n;
   cout << "Time using TRandom::Gaus  =\t " <<   time << "\tns/call" << endl;

   // using Rannor
   w.Start();
   double x1,x2; 
   for (int i = 0; i < n/2; ++i) 
      gRandom->Rannor(x1,x2); 
   w.Stop(); 
   time = w.CpuTime()*1.E9/n;
   cout << "Time using TRandom::Rannor  =\t " <<   time << "\tns/call" << endl;

   // using GSL Ziggurat
   ROOT::Math::Random<ROOT::Math::GSLRngMT>     rgsl;
   w.Start();
   for (int i = 0; i < n; ++i) 
      rgsl.Gaus(0,1); 
   w.Stop(); 
   time = w.CpuTime()*1.E9/n;
   cout << "Time using GSL::Gaus  =\t\t " <<   time << "\tns/call" << endl;



//    w.Start();
//    for (int i = 0; i < n; ++i) 
//       rgsl.GausRatio(0,1); 
//    w.Stop(); 
//    time = w.CpuTime()*1.E9/n;
//    cout << "Time using GSL::GausRatio  =\t\t " <<   time << "\tns/call" << endl;

#ifdef HAVE_CLHEP
   MTwistEngine eng(111);
   RandGauss rg(eng);
   w.Start();
   for (int i = 0; i < n; ++i) 
      rg.shoot(0,1); 
   w.Stop(); 
   time = w.CpuTime()*1.E9/n;
   cout << "Time using CLHEP::Gaus  =\t " <<   time << "\tns/call" << endl;

   RandFlat rf(eng);
   w.Start();
   for (int i = 0; i < n; ++i) 
      rf.shoot(); 
   w.Stop(); 
   time = w.CpuTime()*1.E9/n;
   cout << "Time using CLHEP::Flat  =\t " <<   time << "\tns/call" << endl;
#endif

   w.Start();
   for (int i = 0; i < n; ++i) 
      gRandom->Rndm(); 
   w.Stop(); 
   time = w.CpuTime()*1.E9/n;
   cout << "Time using TRandom::Rndm  =\t " <<   time << "\tns/call" << endl;

   // test the quality by looking at the cdf

   TH1D * h1 = new TH1D("h1","cdf on the data ",100,0,1);
   for (int i = 0; i < 1000000; ++i) {
      double x = unr.Sample();
      h1->Fill( ROOT::Math::normal_cdf( x , 1.0) ); 
   }
//    gSystem->Load("libMinuit2");
//    TVirtualFitter::SetDefaultFitter("Minuit2");
   std::cout << "Result of fitting a cdf to a constant function : " << std::endl;
   // need TRint();
   TFitter * fitter = new TFitter(); 
   TVirtualFitter::SetFitter(fitter);
   h1->Fit("pol0","Q");
   TF1 * f = h1->GetFunction("pol0");

   std::cout << "Fit chi2 = " << f->GetChisquare() << " ndf = " << f->GetNDF() << std::endl;
   std::cout << "Fit Prob = " << f->GetProb() << std::endl;

   h1->Draw("E");

   //delete fitter; 


}

#ifndef __CINT__
int main(int argc, char **argv)
{
   if (argc > 1) { 
      TApplication theApp("App",&argc,argv);
      unuranSimple();
      theApp.Run();
   } 
   else 
      unuranSimple();

   return 0;
}
#endif

