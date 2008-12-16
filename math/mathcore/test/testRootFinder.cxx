#include "TF1.h"
#include "Math/Functor.h"
#include "TStopwatch.h"

#include "Math/BrentRootFinder.h"

#include <iostream>
#include <cmath>

const double ERRORLIMIT = 1E-8;
const int iterTest = 10000;
int myfuncCalls = 0;

double myfunc ( double x ) {
   myfuncCalls += 1;
   return x*x - 5; 
}

int printStats(TStopwatch& timer, double root) {

   //std::cout << "Return code:  " << status << std::endl; 
   double difference = root - std::sqrt(5.0);
   std::cout << "Result:       " << root << std::endl; 
   std::cout << "Exact result: " << sqrt(5.0) << " difference: " << difference << std::endl; 
   std::cout << "Time: " << timer.RealTime()/(double) iterTest << std::endl; 
   std::cout << "Number of calls to function: " << myfuncCalls/iterTest << std::endl;

   return difference > ERRORLIMIT;
}

int testRootFinder() {

   TStopwatch timer;
   double root;
   int status = 0;

   ROOT::Math::Functor1D    *func = new ROOT::Math::Functor1D (&myfunc);
   
   ROOT::Math::BrentRootFinder brf;
   timer.Reset(); timer.Start(); myfuncCalls = 0;
   for (int i = 0; i < iterTest; ++i)
   {
      brf.SetFunction( *func, 0, 5 );
      brf.Solve();
      root = brf.Root();
   }
   timer.Stop();
   std::cout << "RootFinder Stats:" << std::endl;
   status += printStats(timer, root);


   TF1* f1 = new TF1("f1", "x*x - 5", 0, 5);
   timer.Reset(); timer.Start(); myfuncCalls = 0;
   for (int i = 0; i < iterTest; ++i)
   {
      brf.SetFunction( *func, 0, 5 ); // Just to make a fair comparision!
      root = f1->GetX(0, 0, 5);
   }
   timer.Stop();
   std::cout << "\nTF1 Stats:" << std::endl;
   status += printStats(timer, root);

   return status;
}

int main() {

   int status = testRootFinder();
   return status;

}
