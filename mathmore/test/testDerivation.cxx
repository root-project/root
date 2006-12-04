#include "Math/Polynomial.h"
#include "Math/Derivator.h"
#include "Math/IFunction.h"
#include "Math/Functor.h"
//#include "Math/WrappedTF1.h"
#include <iostream>
#include <vector>

#ifdef HAVE_ROOTLIBS
#include "TStopwatch.h"
#include "TF1.h"
#endif


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
  f1->SetParameters(&p[0]);

  ROOT::Math::Derivator *der = new ROOT::Math::Derivator(*f1);

  double step = 1E-8;
  double x0 = 2;

  der->SetFunction(*f1);
  double result = der->Eval(x0);
  std::cout << "Derivative of function inheriting from IGenFunction f(x) = 2 + 3x + 4x^2 at x = 2" << std::endl;
  std::cout << "Return code:  " << der->Status() << std::endl;
  std::cout << "Result:       " << result << " +/- " << der->Error() << std::endl;
  std::cout << "Exact result: " << f1->Derivative(x0) << std::endl;
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
  ROOT::Math::Functor1D<ROOT::Math::IGenFunction> *f3 = new ROOT::Math::Functor1D<ROOT::Math::IGenFunction>(myfunc2);

  std::cout << "Derivative of a free function wrapped in a Functor f(x) = x^(3/2) at x = 2" << std::endl;
  std::cout << "EvalCentral:  " << der->EvalCentral( *f3, x0) << std::endl;
  der->SetFunction(*f3);
  std::cout << "EvalForward:  " << der->EvalForward(x0) << std::endl;
  std::cout << "EvalBackward: " << der->EvalBackward(x0) << std::endl;
  std::cout << "Exact result: " << 1.5*sqrt(x0) << std::endl << std::endl;
  
  // Derivative of a multidim TF1 function
  
// #ifdef LATER
//   TF2 * f2d = new TF2("f2d","x*x + y*y",-10,10,-10,10);
//   // find gradient at x={1,1}
//   double vx[2] = {1.,2.}; 
//   ROOT::Math::WrappedTF1 fx(*f2d); 

//   std::cout << "Derivative of a  f(x,y) = x^2 + y^2 at x = 1,y=2" << std::endl;
//   std::cout << "df/dx  = " << der->EvalCentral(fx,1.) << std::endl;
//   WrappedFunc fy(*f2d,0,vx); 
//   std::cout << "df/dy  = " << der->EvalCentral(fy,2.) << std::endl;
// #endif

}


void testDerivPerf() { 

#ifdef HAVE_ROOTLIBS

   std::cout << "\n\n***************************************************************\n";
   std::cout << "Test derivation performances....\n\n";

  ROOT::Math::Polynomial f1(2); 
  double p[3] = {2,3,4};
  f1.SetParameters(p);
  
  TStopwatch timer; 
  int n = 1000000; 
  double x1 = 0; double x2 = 10; 
  double dx = (x2-x1)/double(n); 

  timer.Start(); 
  double s1 = 0; 
  ROOT::Math::Derivator der(f1);
  for (int i = 0; i < n; ++i) { 
     double x = x1 + dx*i; 
     s1+= der.EvalCentral(x);
  }
  timer.Stop(); 
  std::cout << "Time using ROOT::Math::Derivator :\t" << timer.RealTime() << std::endl; 
  int pr = std::cout.precision(18); std::cout << s1 << std::endl; std::cout.precision(pr);

  timer.Start(); 
  s1 = 0; 
  for (int i = 0; i < n; ++i) { 
     ROOT::Math::Derivator der2(f1);
     double x = x1 + dx*i; 
     s1+= der2.EvalForward(x);
  }
  timer.Stop(); 
  std::cout << "Time using ROOT::Math::Derivator(2):\t" << timer.RealTime() << std::endl; 
  pr = std::cout.precision(18); std::cout << s1 << std::endl; std::cout.precision(pr);


  TF1 f2("pol","pol2",0,10);
  f2.SetParameters(p);
  
  timer.Start(); 
  double s2 = 0; 
  for (int i = 0; i < n; ++i) { 
     double x = x1 + dx*i; 
     s2+= f2.Derivative(x);
  }
  timer.Stop(); 
  std::cout << "Time using TF1::Derivative :\t\t" << timer.RealTime() << std::endl; 
  pr = std::cout.precision(18);
  std::cout << s2 << std::endl;
  std::cout.precision(pr);

#endif  


}


int main() {

  testDerivation();
  testDerivPerf();
  return 0;

}
