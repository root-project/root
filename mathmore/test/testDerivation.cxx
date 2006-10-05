#include "Math/Polynomial.h"
#include "Math/Derivator.h"
#include "Math/WrappedFunction.h"
//#include "TF1.h"
#include <iostream>



typedef double ( * FP ) ( double, void * ); 
typedef double ( * FP2 ) ( double ); 


double myfunc ( double x, void * ) { 
  
  return std::pow( x, 1.5); 
}

double myfunc2 ( double x) { 
  return std::pow( x, 1.5); 
}



void testDerivation() {




  // Derivative of an IGenFunction
  // Works when compiled c++, compiled ACLiC, interpreted by CINT
  ROOT::Math::Polynomial *f1 = new ROOT::Math::Polynomial(2);

  std::vector<double> p(3);
  p[0] = 2;
  p[1] = 3;
  p[2] = 4;
  f1->SetParameters(p);

  ROOT::Math::Derivator *der = new ROOT::Math::Derivator(*f1);

  double step = 1E-8;
  double x0 = 2;

  der->SetFunction(*f1);
  double result = der->Eval(x0);
  std::cout << "Derivative of function inheriting from IGenFunction f(x) = 2 + 3x + 4x^2 at x = 2" << std::endl;
  std::cout << "Return code:  " << der->Status() << std::endl;
  std::cout << "Result:       " << result << " +/- " << der->Error() << std::endl;
  std::cout << "Exact result: " << f1->Gradient(x0) << std::endl;
  std::cout << "EvalForward:  " << der->EvalForward(*f1, x0) << std::endl;
  std::cout << "EvalBackward: " << der->EvalBackward(x0, step) << std::endl << std::endl;;


  
  
  // Derivative of a free function
  // Works when compiled c++, compiled ACLiC, does not work when interpreted by CINT  
  FP f2 = &myfunc;
  der->SetFunction(f2);

  std::cout << "Derivative of a free function f(x) = x^(3/2) at x = 2" << std::endl;
  std::cout << "EvalCentral:  " << der->EvalCentral(x0) << std::endl;
  std::cout << "EvalForward:  " << der->EvalForward(x0) << std::endl;
  std::cout << "EvalBackward: " << der->EvalBackward(x0) << std::endl;

  std::cout << "Exact result: " << 1.5*sqrt(x0) << std::endl << std::endl;
  

  
  
  // Derivative of a free function wrapped in an IGenFunction
  // Works when compiled c++, compiled ACLiC, does not work when interpreted by CINT  
  ROOT::Math::WrappedFunction<FP2> *f3 = new ROOT::Math::WrappedFunction<FP2>(myfunc2);

  std::cout << "Derivative of a free function wrapped in WrappedFunction f(x) = x^(3/2) at x = 2" << std::endl;
  std::cout << "EvalCentral:  " << der->EvalCentral( *f3, x0) << std::endl;
  der->SetFunction(*f3);
  std::cout << "EvalForward:  " << der->EvalForward(x0) << std::endl;
  std::cout << "EvalBackward: " << der->EvalBackward(x0) << std::endl;
  std::cout << "Exact result: " << 1.5*sqrt(x0) << std::endl << std::endl;
  

  



}


int main() {

  testDerivation();
  return 0;

}
