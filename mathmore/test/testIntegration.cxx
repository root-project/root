#include "Math/Polynomial.h"
#include "Math/Integrator.h"
#include "Math/Functor.h"
//#include "TF1.h"
#include <iostream>




double exactIntegral ( const std::vector<double> & par, double a, double b) { 

  ROOT::Math::Polynomial *func = new ROOT::Math::Polynomial( par.size() +1);

  std::vector<double> p = par;
  p.push_back(0);
  p[0] = 0; 
  for (unsigned int i = 1; i < p.size() ; ++i) { 
    p[i] = par[i-1]/double(i); 
  }
  func->SetParameters(&p.front());

  return (*func)(b)-(*func)(a); 
}

double singularFunction(double x) { 
   return 1./sqrt(x);
}


void testIntegration() {


  ROOT::Math::Polynomial * f = new ROOT::Math::Polynomial(2);

  std::vector<double> p(3);
  p[0] = 4;
  p[1] = 2;
  p[2] = 6;
  f->SetParameters(&p[0]);
  ROOT::Math::IGenFunction &func = *f; 


  double exactresult = exactIntegral(p, 0,3);
  std::cout << "Exact value " << exactresult << std::endl << std::endl; 


  ROOT::Math::Integrator ig(func, 0.001, 0.01, 100 );
  double value = ig.Integral( 0, 3); 
  // or ig.Integral(*f, 0, 10); if new function 

  std::cout.precision(20);

  std::cout << "Adaptive singular integration:" << std::endl;
  std::cout << "Return code " << ig.Status() << std::endl; 
  std::cout << "Result      " << value << " +/- " << ig.Error() << std::endl << std::endl; 


  
  // integrate again ADAPTIve, with different rule 
  ROOT::Math::Integrator ig2(func, ROOT::Math::Integration::ADAPTIVE, ROOT::Math::Integration::GAUSS61, 0.001, 0.01, 100 );
  value = ig2.Integral(0, 3); 
  // or ig2.Integral(*f, 0, 10); if different function

  std::cout << "Adaptive Gauss61 integration:" << std::endl;
  std::cout << "Return code " << ig2.Status() << std::endl; 
  std::cout << "Result      " << value << " +/- " << ig2.Error() << std::endl << std::endl; 

  
  std::cout << "Testing SetFunction member function" << std::endl;
  ROOT::Math::Polynomial *pol = new ROOT::Math::Polynomial(2);

  pol->SetParameters(&p.front());
  ROOT::Math::IGenFunction &func2 = *pol; 
  ig.SetFunction(func2);
  std::cout << "Result      " << ig.Integral( 0, 3) << " +/- " << ig.Error() << std::endl; 
 

  // test error 
  //typedef double ( * FreeFunc ) ( double);

  std::cout << "Testing a singular function" << std::endl;
  //ROOT::Math::WrappedFunction<FreeFunc> wf(&singularFunction); 
  ROOT::Math::Functor1D<ROOT::Math::IGenFunction> wf(&singularFunction);
  
  ig.SetFunction(wf); 
  double r = ig.Integral(0,1); 
  if (ig.Status() != 0) 
     std::cout << "Error integrating a singular function " << std::endl; 
  else 
     std::cout << "Result      " << r << " +/- " << ig.Error() << std::endl; 
  
 
}



int main() {

  testIntegration();
  return 0;

}
