
#include "TStopwatch.h"
#include "TUnuran.h"

#include "TH1.h"

#include "TRandom.h"
#include "TSystem.h"
#include "TApplication.h"

#include <iostream> 

using std::cout; 
using std::endl; 

void unuranSimple() { 

   // simple test of unuran

   TH1D * h1 = new TH1D("h1","gaussian distribution",100,-10,10);
   TH1D * h2 = new TH1D("h2","gaussian distribution",100,-10,10);

   TUnuran unr; 
   if (! unr.Init( "normal()", "method=arou") ) {
      cout << "Error initializing unuran" << endl;
      return;
   }

   int n = 10000000;
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

   // test the quality
   for (int i = 0; i < n; ++i) {
      double x = unr.Sample();
      h1->Fill(  x ); 
      //h1u->Fill( fc->Eval( x ) ); 
   }


   h1->Draw();



}

#ifndef __CINT__
int main(int argc, char **argv)
{
   TApplication theApp("App", &argc, argv);
   unuranSimple();
   theApp.Run();
   return 0;
}
#endif

