// test unuran using the string interface to generate numbers according to the normal distributions
// compare CPU performancecwith TRandom::Gaus and opitonally GSL (using MathMore ) and CLHEP for 
// generating normal distributed random numbers 
//
// run within ROOT (.x unuranSimple.cxx+) or pass any extra parameter in the command line to get  
// a graphics output 
//
#include "TStopwatch.h"
#include "TUnuran.h"

#include "TH1.h"
#include "TF1.h"

#include "TRandom.h"
#include "TRandom1.h"
#include "TRandom2.h"
#include "TSystem.h"
#include "TApplication.h"
#include "TCanvas.h"
//#include "TRint.h"
#include "TVirtualFitter.h"
#include "TFitter.h"
#include "Math/DistFunc.h"

#include <iostream> 

#ifdef HAVE_MATHMORE
#include "Math/Random.h"
#include "Math/GSLRndmEngines.h"
#endif



#ifdef HAVE_CLHEP
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Random/MTwistEngine.h"
#include "CLHEP/Random/JamesRandom.h"
#include "CLHEP/Random/RanluxEngine.h"
#include "CLHEP/Random/Ranlux64Engine.h"
#include "CLHEP/Random/RanecuEngine.h"
#include "CLHEP/Random/Hurd160Engine.h"
#include "CLHEP/Random/Hurd288Engine.h"
#include "CLHEP/Random/RanshiEngine.h"
#include "CLHEP/Random/DualRand.h"
#include "CLHEP/Random/TripleRand.h"

using namespace CLHEP;
#endif


using std::cout; 
using std::endl; 

int unuranSimple( ) { 

   // simple test of unuran


   std::cout << "Test Generation of Gaussian Numbers \n\n";

   TUnuran unr; 
   if (! unr.Init( "normal()", "method=arou") ) {
      cout << "Error initializing unuran" << endl;
      return -1;
   }

    // default is 10**7 but one should use 10**8 for serious timing
   int n = 10000000; 

   TStopwatch w;
   double time; 
   w.Start(); 

   for (int i = 0; i < n; ++i) 
      unr.Sample(); 

   w.Stop(); 
   time = w.CpuTime()*1.E9/n;
   cout << "Time using Unuran method arou =\t " <<   time << "\tns/call" << endl;


   if (! unr.Init( "normal()", "method=tdr") ) {
      cout << "Error initializing unuran" << endl;
      return -1;
   }
   w.Start(); 
   for (int i = 0; i < n; ++i) 
      unr.Sample(); 

   w.Stop(); 
   time = w.CpuTime()*1.E9/n;
   cout << "Time using Unuran method tdr =\t " <<   time << "\tns/call" << endl;

   if (! unr.Init( "normal()", "method=hinv") ) {
      cout << "Error initializing unuran" << endl;
      return -1;
   }
   w.Start(); 
   for (int i = 0; i < n; ++i) 
      unr.Sample(); 

   w.Stop(); 
   time = w.CpuTime()*1.E9/n;
   cout << "Time using Unuran method hinv =\t " <<   time << "\tns/call" << endl;

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

#ifdef HAVE_MATHMORE
   // using GSL Ziggurat
   ROOT::Math::Random<ROOT::Math::GSLRngMT>     rgsl;
   w.Start();
   for (int i = 0; i < n; ++i) 
      rgsl.Gaus(0,1); 
   w.Stop(); 
   time = w.CpuTime()*1.E9/n;
   cout << "Time using GSL::Gaus  =\t\t " <<   time << "\tns/call" << endl;

   // using GSL BoxMuller method
   w.Start();
   for (int i = 0; i < n; ++i) 
      rgsl.GausBM(0,1); 
   w.Stop(); 
   time = w.CpuTime()*1.E9/n;
   cout << "Time using GSL::GausBM  =  \t " <<   time << "\tns/call" << endl;

   // using GSL Ratio method
   w.Start();
   for (int i = 0; i < n; ++i) 
      rgsl.GausR(0,1); 
   w.Stop(); 
   time = w.CpuTime()*1.E9/n;
   cout << "Time using GSL::GausR  =\t " <<   time << "\tns/call" << endl;
#endif


   // run unuran standard Normal methods 2 (Polar from Marsaglia)

   if (! unr.Init( "normal()", "method=cstd;variant=2") ) {
      cout << "Error initializing unuran" << endl;
      return -1;
   }
   w.Start(); 
   for (int i = 0; i < n; ++i) 
      unr.Sample(); 

   w.Stop(); 
   time = w.CpuTime()*1.E9/n;
   cout << "Time using Unuran GausPolarR =\t " <<   time << "\tns/call" << endl;

   // run unuran standard Normal method 3 (KM)

   if (! unr.Init( "normal()", "method=cstd;variant=3") ) {
      cout << "Error initializing unuran" << endl;
      return -1;
   }
   w.Start(); 
   for (int i = 0; i < n; ++i) 
      unr.Sample(); 

   w.Stop(); 
   time = w.CpuTime()*1.E9/n;
   cout << "Time using Unuran Gaus  K-R   =\t " <<   time << "\tns/call" << endl;
  
   if (! unr.Init( "normal()", "method=cstd;variant=6") ) {
      cout << "Error initializing unuran" << endl;
      return -1;
   }
   w.Start(); 
   for (int i = 0; i < n; ++i) 
      unr.Sample(); 

   w.Stop(); 
   time = w.CpuTime()*1.E9/n;
   cout << "Time using Unuran Gaus exp6  =\t " <<   time << "\tns/call" << endl;


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
      rg(0,1); 
   w.Stop(); 
   time = w.CpuTime()*1.E9/n;
   cout << "Time using CLHEP::Gaus  =\t " <<   time << "\tns/call" << endl;
#endif

   std::cout << "\nTest uniform generator\n" << std::endl;
   w.Start();
   for (int i = 0; i < n; ++i) 
      gRandom->Rndm(); 
   w.Stop(); 
   time = w.CpuTime()*1.E9/n;
   cout << "Time using gRandom::Rndm  =\t " <<   time << "\tns/call" << endl;

   TRandom1 r1;
   w.Start();
   for (int i = 0; i < n; ++i) 
      r1.Rndm(); 
   w.Stop(); 
   time = w.CpuTime()*1.E9/n;
   cout << "Time using TRandom1::Rndm  =\t " <<   time << "\tns/call" << endl;

   TRandom2 r2;
   w.Start();
   for (int i = 0; i < n; ++i) 
      r2.Rndm(); 
   w.Stop(); 
   time = w.CpuTime()*1.E9/n;
   cout << "Time using TRandom2::Rndm  =\t " <<   time << "\tns/call" << endl;

#ifdef HAVE_CLHEP
   RandFlat rf(eng);
   w.Start();
   for (int i = 0; i < n; ++i) 
      eng.flat(); // use directly the engine (faster!)
   w.Stop(); 
   time = w.CpuTime()*1.E9/n;
   cout << "Time using CLHEP::MT  =\t\t " <<   time << "\tns/call" << endl;

   { 
      RanecuEngine e; 
      RandFlat rf2(e);
      w.Start();
      for (int i = 0; i < n; ++i) 
         e.flat(); 
      w.Stop(); 
      time = w.CpuTime()*1.E9/n;
      cout << "Time using CLHEP::Ranecu  =\t " <<   time << "\tns/call" << endl;
   }
   { 
      Hurd160Engine e; 
      RandFlat rf2(e);
      w.Start();
      for (int i = 0; i < n; ++i) 
         e.flat();
      w.Stop(); 
      time = w.CpuTime()*1.E9/n;
      cout << "Time using CLHEP::Hard160  =\t " <<   time << "\tns/call" << endl;
   }
   { 
      Hurd288Engine e; 
      RandFlat rf2(e);
      w.Start();
      for (int i = 0; i < n; ++i) 
         e.flat();
         //rf2(); 
      w.Stop(); 
      time = w.CpuTime()*1.E9/n;
      cout << "Time using CLHEP::Hard288  =\t " <<   time << "\tns/call" << endl;
   }
   { 
      DualRand e; 
      RandFlat rf2(e);
      w.Start();
      for (int i = 0; i < n; ++i) 
         e.flat();
      //rf2(); 
      w.Stop(); 
      time = w.CpuTime()*1.E9/n;
      cout << "Time using CLHEP::DualRand  =\t " <<   time << "\tns/call" << endl;
   }
   {
      TripleRand e; 
      RandFlat rf2(e);
      w.Start();
      for (int i = 0; i < n; ++i) 
         e.flat();
      //rf2(); 
      w.Stop(); 
      time = w.CpuTime()*1.E9/n;
      cout << "Time using CLHEP::TripleRand  =\t " <<   time << "\tns/call" << endl;
   }
   {
      RanshiEngine e; 
      RandFlat rf2(e);
      w.Start();
      for (int i = 0; i < n; ++i) 
         //rf2(); 
         e.flat();
      w.Stop(); 
      time = w.CpuTime()*1.E9/n;
      cout << "Time using CLHEP::Runshi  =\t " <<   time << "\tns/call" << endl;
   }
   {
      RanluxEngine e; 
      RandFlat rf2(e);
      w.Start();
      for (int i = 0; i < n; ++i) 
         //rf2(); 
         e.flat();
      w.Stop(); 
      time = w.CpuTime()*1.E9/n;
      cout << "Time using CLHEP::RunLux  =\t " <<   time << "\tns/call" << endl;
   }
   {
      Ranlux64Engine e; 
      RandFlat rf2(e);
      w.Start();
      for (int i = 0; i < n; ++i) 
         e.flat(); 
      w.Stop(); 
      time = w.CpuTime()*1.E9/n;
      cout << "Time using CLHEP::RunLux64  =\t " <<   time << "\tns/call" << endl;
   }
   {
      HepJamesRandom e; 
      RandFlat rf2(e);
      w.Start();
      for (int i = 0; i < n; ++i) 
         //rf2(); 
         e.flat(); 
      w.Stop(); 
      time = w.CpuTime()*1.E9/n;
      cout << "Time using CLHEP::HepJames  =\t " <<   time << "\tns/call" << endl;
   }

#endif


   // test the quality by looking at the cdf
   cout <<"\n\nTest quality of Unuran arou" << endl;
   if (! unr.Init( "normal()", "method=arou") ) {
      cout << "Error initializing unuran" << endl;
      return -1;
   }

   TH1D * h1 = new TH1D("h1","cdf on the data ",1000,0,1);
   for (int i = 0; i < 1000000; ++i) {
      double x = unr.Sample();
      h1->Fill( ROOT::Math::normal_cdf( x , 1.0) ); 
   }

   new TCanvas("c1_unuranGaus","unuran Gaus CDF");

   h1->Fit("pol0","Q");
   TF1 * f = h1->GetFunction("pol0");

   std::cout << "CDF Uniform Fit:  chi2 = " << f->GetChisquare() << " ndf = " << f->GetNDF() << std::endl;
   std::cout << "Fit Prob = " << f->GetProb() << std::endl;
   h1->Draw("E");

   if (f->GetProb() < 1.E-4) { 
      std::cerr << "\nERROR: UnuranSimple Test:\t Failed !!!!"; 
      return -1; 
   }
   std::cerr << "\nUnuranSimple Test:\t OK !" << std::endl;

   return 0; 

}

#ifndef __CINT__
int main(int argc, char **argv)
{
   int iret = 0;
   if (argc > 1) { 
      TApplication theApp("App",&argc,argv);
      iret = unuranSimple();
      theApp.Run();
   } 
   else 
      iret = unuranSimple();

   return iret;
}
#endif

