#include "TF1.h"
#include "Math/Functor.h"
#include "TStopwatch.h"

#include "Math/RootFinder.h"
#include "Math/DistFunc.h"

#include <iostream>
#include <iomanip>
#include <cmath>

/**
   Test of ROOT finder for various function

   case = 0  simple function (polynomial)
   case = 1  function which fails for a bug in BrentMethod::MinimBrent fixed with r = 32544 (5.27.01)
 */

const double ERRORLIMIT = 1E-8;
const int iterTest = 10000;
int myfuncCalls = 0;

int gTestCase = 0; 

double myfunc ( double x ) {
   myfuncCalls ++;
   if (gTestCase == 0) // polynomial
      return x*x - 5;   
   if (gTestCase == 1) // function showing bug in BrentMethod::
      return ROOT::Math::gamma_cdf(x,16,0.4)-0.32; 
   return 0; 
}

double ExactResult() { 
   if (gTestCase == 0) 
      return sqrt(5); 
   if (gTestCase == 1)
#ifdef R__HAS_MATHMORE  
      return ROOT::Math::gamma_quantile(0.32,16,0.4);
#else
      return 5.55680381022934800;
#endif

   return 0; 
}



double myfunc_p (double *x, double *) { return myfunc(x[0]); }

int printStats(TStopwatch& timer, double root) {

   //std::cout << "Return code:  " << status << std::endl; 
   double difference = root - ExactResult();
   int pr = std::cout.precision(16);
   std::cout << "Result:       " << root << std::endl; 
   std::cout << "Exact result: " << ExactResult();
   std::cout.precision(pr);
   std::cout << " difference: " << difference << std::endl; 
   std::cout << "Time: " << timer.RealTime()/(double) iterTest << std::endl; 
   std::cout << "Number of calls to function: " << myfuncCalls/iterTest << std::endl;

   return difference > ERRORLIMIT;
}

int testRootFinder(int testcase = 0) {

   double xmin = 0; 
   double xmax = 10;

   std::cout << "*************************************************************\n";
   gTestCase = testcase; 
   if (gTestCase == 0) 
      std::cout << "Test for simple function f(x) = x^2 - 5  \n" << std::endl; 
   if (gTestCase == 1) {
      std::cout << "\nTest for  f(x) = gamma_cdf  \n" << std::endl; 
      xmin = 3.955687382047723;
      xmax = 9.3423159494328623;
   }

   TStopwatch timer;
   double root;
   int status = 0;


   ROOT::Math::Functor1D    *func = new ROOT::Math::Functor1D (&myfunc);
   
   TF1* f1 = new TF1("f1", myfunc_p, xmin, xmax);
   timer.Reset(); timer.Start(); myfuncCalls = 0;
   for (int i = 0; i < iterTest; ++i)
   {
      //brf.SetFunction( *func, 0, 10 ); // Just to make a fair comparision!
      root = f1->GetX(0, xmin, xmax);
   }
   timer.Stop();
   std::cout << "\nTF1 Stats:" << std::endl;
   status += printStats(timer, root);

   ROOT::Math::RootFinder brf(ROOT::Math::RootFinder::kBRENT);
   timer.Reset(); timer.Start(); myfuncCalls = 0;
   for (int i = 0; i < iterTest; ++i)
   {
      brf.SetFunction( *func, xmin, xmax );
      bool ret = brf.Solve(100,1E-15,1.E-15);
      if (!ret && i == 0) std::cout << "Error returned from RootFinder::Solve BRENT " << std::endl;
      root = brf.Root();
   }
   timer.Stop();
   std::cout << "Brent RootFinder Stats:" << std::endl;
   status += printStats(timer, root);

   ROOT::Math::RootFinder grf(ROOT::Math::RootFinder::kGSL_BRENT);
   timer.Reset(); timer.Start(); myfuncCalls = 0;
   for (int i = 0; i < iterTest; ++i)
   {
      grf.SetFunction( *func, xmin, xmax );
      bool ret = grf.Solve();
      root = grf.Root();
      if (!ret && i == 0) std::cout << "Error returned from RootFinder::Solve GSL_BRENT" << std::endl;
   }
   timer.Stop();
   std::cout << "GSL Brent RootFinder Stats:" << std::endl;
   status += printStats(timer, root);



   return status;
}

int main() {

   int status = 0; 
   status |= testRootFinder(0);  // test pol function
   status |= testRootFinder(1);  // test gamma_cdf
   std::cout << "*************************************************************\n";
   std::cout << "\nTest RootFinder :\t";
   if (status == 0) 
      std::cout << "OK " << std::endl;
  else 
     std::cout << "Failed !" << std::endl;

   return status;
}
