#include "TF1.h"
#include "Math/Functor.h"
#include "TStopwatch.h"
#include "RConfigure.h"

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

const double Y0_P2 = 5.0;   // Y0 for test 1 (parabola)
// these are value which gave problems in 5.26 for gamma_cdf
const double Y0_GAMMA = 0.32;  // Y0 for test 2 (gamma cdf)
const double ALPHA_GAMMA = 16.; // alpha of gamma cdf
const double THETA_GAMMA = 0.4;  // theta of gamma cdf

int gTestCase = 0;

double myfunc ( double x ) {
   myfuncCalls ++;
   if (gTestCase == 0) // polynomial
      return x*x - Y0_P2;
   if (gTestCase == 1) // function showing bug in BrentMethod::
      return ROOT::Math::gamma_cdf(x,ALPHA_GAMMA,THETA_GAMMA)-Y0_GAMMA;
   return 0;
}

double ExactResult() {
   if (gTestCase == 0) {
      return std::sqrt(Y0_P2);
   }
   if (gTestCase == 1)
      //return ROOT::Math::gamma_quantile(Y0_GAMMA,ALPHA_GAMMA,THETA_GAMMA);
      // put the value to avoid direct MathMore dependency
      return 5.55680381022934800;

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

int runTest(int testcase = 0) {

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
   double root = 0.;
   int status = 0;

   double tol = 1.E-14;
   int maxiter = 100;

   ROOT::Math::Functor1D    *func = new ROOT::Math::Functor1D (&myfunc);

   TF1* f1 = new TF1("f1", myfunc_p, xmin, xmax);
   timer.Reset(); timer.Start(); myfuncCalls = 0;
   for (int i = 0; i < iterTest; ++i)
   {
      //brf.SetFunction( *func, 0, 10 ); // Just to make a fair comparision!
      root = f1->GetX(0, xmin, xmax,tol,maxiter);
   }
   timer.Stop();
   std::cout << "\nTF1 Stats:" << std::endl;
   status += printStats(timer, root);

   ROOT::Math::RootFinder brf(ROOT::Math::RootFinder::kBRENT);
   timer.Reset(); timer.Start(); myfuncCalls = 0;
   for (int i = 0; i < iterTest; ++i)
   {
      brf.SetFunction( *func, xmin, xmax );
      bool ret = brf.Solve(maxiter,tol,tol);
      if (!ret && i == 0) std::cout << "Error returned from RootFinder::Solve BRENT " << std::endl;
      root = brf.Root();
   }
   timer.Stop();
   std::cout << "Brent RootFinder Stats:" << std::endl;
   status += printStats(timer, root);

#ifdef R__HAS_MATHMORE
   ROOT::Math::RootFinder grf(ROOT::Math::RootFinder::kGSL_BRENT);
   timer.Reset(); timer.Start(); myfuncCalls = 0;
   for (int i = 0; i < iterTest; ++i)
   {
      grf.SetFunction( *func, xmin, xmax );
      bool ret = grf.Solve(maxiter,tol,tol);
      root = grf.Root();
      if (!ret && i == 0) std::cout << "Error returned from RootFinder::Solve GSL_BRENT" << std::endl;
   }
   timer.Stop();
   std::cout << "GSL Brent RootFinder Stats:" << std::endl;
   status += printStats(timer, root);
#endif

   if (status) std::cout << "Test-case " << testcase << "   FAILED" << std::endl;


   return status;
}

int testRootFinder() {

   int status = 0;
   status |= runTest(0);  // test pol function
   if (status) std::cerr << "Test pol function  FAILED" << std::endl;

   status |= runTest(1);  // test gamma_cdf
   if (status) std::cerr << "Test gamma function  FAILED" << std::endl;

   std::cerr << "*************************************************************\n";
   std::cerr << "\nTest RootFinder :\t";
   if (status == 0)
      std::cerr << "OK " << std::endl;
  else
     std::cerr << "Failed !" << std::endl;

   return status;
}

int main() {
   return testRootFinder();
}
