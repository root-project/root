#include "Math/Polynomial.h"
#include "Math/Minimizer1D.h"
#include "Math/WrappedFunction.h"
#include "TF1.h"
#include <iostream>




void testMinimization1D() {


  ROOT::Math::Polynomial *f = new ROOT::Math::Polynomial(2);

  std::vector<double> p(3);
  p[0] = 1;
  p[1] = -4;
  p[2] = 1;
  f->SetParameters(p);



  ROOT::Math::Minimizer1D min;
  min.SetFunction(*f,1,-10,10); 
  int status = min.Minimize(100,0.01,0.01); 
  std::cout << "Return code " << status << std::endl; 

  std::cout.precision(20);

  std::cout << "Found minimum: x = " << min.XMinimum() << "  f(x) = " << min.FValMinimum() << std::endl;


 
}



int main() {

  testMinimization1D();
  return 0;

}
