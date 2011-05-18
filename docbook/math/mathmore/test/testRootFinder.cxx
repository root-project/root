#include "Math/Polynomial.h"
#include "Math/Functor.h"
#include "Math/RootFinder.h"
#include "Math/RootFinderAlgorithms.h"

#ifdef HAVE_ROOTLIBS
#include "TStopwatch.h"
#else 
struct TStopwatch { 
   void Start(){} 
   void Stop(){} 
   void Reset(){} 
   double RealTime() { return 0; }
   double CpuTime() { return 0; }
};
#endif

#include <iostream>


typedef double ( * FP ) ( double, void * ); 

const int iterTest = 10000;
int myfuncCalls = 0;

const double absTol = 1E-3; 
const double relTol = 1E-6; 

double myfunc ( double x) {
  myfuncCalls += 1;
  return x*x - 5; 
}
// required for GSL type func
double myfunc_gsl(double x, void *) { 
  return myfunc(x);
}

double myfunc_deriv ( double x ) { 
  return 2.0*x; 
}
double myfunc_deriv_gsl ( double x, void * /*params*/) { 
  return myfunc_deriv(x);
}

void myfunc_fdf( double x, void * /*params*/, double *y, double *dy) { 
  *y = x*x - 5; 
  *dy = 2.0*x; 
}

template<class RF> 
int findRoot( RF * r ) { 
  //int returnCode = r->Solve( 100, absTol, relTol); 
  bool ret = r->Solve(); 
  int returnCode = !ret; //r->Status();

  return  returnCode; 
}

template<class RF> 
int printStats( RF * r, int returnCode, TStopwatch& timer ) { 
  std::cout << "\nTest " << r->Name() << " algorithm " << std::endl; 

  double root = r->Root();

  std::cout << "Return code:  " << returnCode << std::endl; 
  std::cout << "Result:       " << root << " n iters = " << r->Iterations() << std::endl; 
  std::cout << "Exact result: " << sqrt(5.0) << " difference: " << root - sqrt(5.0) << std::endl; 
  std::cout << "Time: " << timer.RealTime()/(double) iterTest << std::endl;   
  std::cout << "Number of calls to function: " << myfuncCalls/iterTest << std::endl;
 
  if ( fabs(root - sqrt(5.0)) > absTol ) { 
    std::cerr << "Test Root finder with " << r->Name() << "  failed " << std::endl; 
    return 1; 
  }
  return  0; 
}


int testRootFinder() {

  int returnCode;
  int status = 0;
  TStopwatch timer;

  ROOT::Math::Polynomial polyf(2);
  std::vector<double> p(3);
  p[0] = -5; 
  p[1] = 0; 
  p[2] = 1;

  polyf.SetParameters(&p[0]); 

  //ROOT::Math::IGenFunction *func = &polyf;
  ROOT::Math::Functor1D    func (&myfunc);

  ROOT::Math::RootFinder *rf1 = new ROOT::Math::RootFinder(ROOT::Math::RootFinder::kGSL_BISECTION);
  timer.Reset(); timer.Start(); myfuncCalls = 0;
  for (int i = 0; i < iterTest; ++i)
  {
     rf1->SetFunction( func, 0, 5); 
     returnCode = findRoot(rf1);
  }
  timer.Stop();
  status += printStats(rf1, returnCode, timer);


  ROOT::Math::RootFinder *rf2 = new ROOT::Math::RootFinder(ROOT::Math::RootFinder::kGSL_FALSE_POS);
  timer.Reset(); timer.Start(); myfuncCalls = 0;
  for (int i = 0; i < iterTest; ++i)
  {
     rf2->SetFunction( func, 0, 5); 
     returnCode = findRoot(rf2); 
  }
  timer.Stop();
  status += printStats(rf2, returnCode, timer);

  // methods using derivatives 


  ROOT::Math::RootFinder *rf3 = new ROOT::Math::RootFinder(ROOT::Math::RootFinder::kGSL_SECANT);
  ROOT::Math::GradFunctor1D gfunc( &myfunc, &myfunc_deriv); 
  timer.Reset(); timer.Start(); myfuncCalls = 0;
  for (int i = 0; i < iterTest; ++i)
  {
     rf3->SetFunction( gfunc, 1); 
     returnCode = findRoot(rf3); 
  }
  timer.Stop();
  status += printStats(rf3, returnCode, timer);

  
  ROOT::Math::RootFinder *rf4 = new ROOT::Math::RootFinder(ROOT::Math::RootFinder::kGSL_STEFFENSON);
  timer.Reset(); timer.Start(); myfuncCalls = 0;
  for (int i = 0; i < iterTest; ++i)
  {
     rf4->SetFunction( gfunc, 1); 
     returnCode = findRoot(rf4); 
  }
  timer.Stop();
  status += printStats(rf4, returnCode, timer);
  

  ROOT::Math::Roots::Newton *rf5 = new ROOT::Math::Roots::Newton();
  void * ptr2 = 0; 
  timer.Reset(); timer.Start(); myfuncCalls = 0;
  for (int i = 0; i < iterTest; ++i)
  {
     rf5->SetFunction(myfunc_gsl, myfunc_deriv_gsl, myfunc_fdf, ptr2, 5.); 
     returnCode = findRoot(rf5); 
  }
  timer.Stop();
  status += printStats(rf5, returnCode, timer);



  // the following two examples won't work when interpreted CINT
  //const FP funcPtr = &myfunc;
  ROOT::Math::GSLRootFinder::GSLFuncPointer funcPtr = &myfunc_gsl;
  void * ptr1 = 0; 
  ROOT::Math::Roots::Brent *rf6 = new ROOT::Math::Roots::Brent();
  //ROOT::Math::RootFinder<ROOT::Math::Roots::Brent> *rf6 = new ROOT::Math::RootFinder<ROOT::Math::Roots::Brent>;
  timer.Reset(); timer.Start(); myfuncCalls = 0;
  for (int i = 0; i < iterTest; ++i)
  {
     rf6->SetFunction( funcPtr, ptr1, 0.0, 5.0); 
     returnCode = findRoot(rf6); 
  }
  timer.Stop();
  status += printStats(rf6, returnCode, timer);

  return status;

}

int main() {

  int status = 0;

  status += testRootFinder();

  return status;
}
