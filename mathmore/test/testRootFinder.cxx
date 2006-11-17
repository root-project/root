#include "Math/Polynomial.h"
#include "Math/Functor.h"
#include "Math/RootFinder.h"
#include "Math/RootFinderAlgorithms.h"

#include <iostream>


typedef double ( * FP ) ( double, void * ); 


double myfunc ( double x, void * /*params*/) {
  return x*x - 5; 
}

double myfunc_deriv ( double x, void * /*params*/) { 
  return 2.0*x; 
}

void myfunc_fdf( double x, void * /*params*/, double *y, double *dy) { 
  *y = x*x - 5; 
  *dy = 2.0*x; 
}


template<class RF> 
int findRoot( RF * r ) { 

  std::cout << "\nTest " << r->Name() << " algorithm " << std::endl; 

  double absTol = 1E-3; 
  //double relTol = 1E-6; 
  //int status = r->Solve( 100, absTol, relTol); 
  int status = r->Solve(); 
  double root = r->Root();

  std::cout << "Return code:  " << status << std::endl; 
  std::cout << "Result:       " << root << " n iters = " << r->Iterations() << std::endl; 
  std::cout << "Exact result: " << sqrt(5.0) << " difference: " << root - sqrt(5.0) << std::endl; 
 
  if ( fabs(root - sqrt(5.0)) > absTol ) { 
    std::cerr << "Test Root finder with " << r->Name() << "  failed " << std::endl; 
    return 1; 
  }
  return  0; 
}



void testRootFinder() {


  ROOT::Math::Polynomial polyf(2);
  std::vector<double> p(3);
  p[0] = -5; 
  p[1] = 0; 
  p[2] = 1; 

  polyf.SetParameters(&p[0]); 

  ROOT::Math::IGenFunction &  func = polyf;

  ROOT::Math::RootFinder<ROOT::Math::Roots::Bisection> *rf1 = new ROOT::Math::RootFinder<ROOT::Math::Roots::Bisection>;
  rf1->SetFunction( func, 0, 5); 
  findRoot(rf1);


  ROOT::Math::RootFinder<ROOT::Math::Roots::FalsePos> *rf2 = new ROOT::Math::RootFinder<ROOT::Math::Roots::FalsePos>;
  rf2->SetFunction( func, 0, 5); 
  findRoot(rf2); 

  // methods using derivatives 

  ROOT::Math::RootFinder<ROOT::Math::Roots::Secant> *rf3 = new ROOT::Math::RootFinder<ROOT::Math::Roots::Secant>;
  rf3->SetFunction( polyf, 1); 
  findRoot(rf3); 

  
  ROOT::Math::RootFinder<ROOT::Math::Roots::Steffenson> *rf4 = new ROOT::Math::RootFinder<ROOT::Math::Roots::Steffenson>;
  rf4->SetFunction( polyf, 1); 
  findRoot(rf4); 
  

  ROOT::Math::Roots::Newton *rf5 = new ROOT::Math::Roots::Newton();
  void * ptr2 = 0; 
  rf5->SetFunction(myfunc, myfunc_deriv, myfunc_fdf, ptr2, 5.); 
  findRoot(rf5); 



  // the following two examples won't work when interpreted CINT
  //const FP funcPtr = &myfunc;
  ROOT::Math::GSLRootFinder::GSLFuncPointer funcPtr = &myfunc;
  void * ptr1 = 0; 
  ROOT::Math::Roots::Brent *rf6 = new ROOT::Math::Roots::Brent();
  //ROOT::Math::RootFinder<ROOT::Math::Roots::Brent> *rf6 = new ROOT::Math::RootFinder<ROOT::Math::Roots::Brent>;
  rf6->SetFunction( funcPtr, ptr1, 0.0, 5.0); 
  findRoot(rf6); 




}



int main() {

  testRootFinder();
  return 0;

}
