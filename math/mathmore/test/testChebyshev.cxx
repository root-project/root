
#include "Math/ChebyshevApprox.h"
#include "Math/IFunction.h"
#include "Math/Functor.h"
#include "Math/SpecFunc.h"

//#include "MathCore/GSLIntegrator.h"

#include <iostream>
#include <cmath>


typedef double ( * FP ) ( double, void * );

// function is a step function

double myfunc ( double x, void * /* params */) {
  //double * p = reinterpret_cast<double *>( params);
  if (x < 0.5)
    return 0.25;
  else
    return 0.75;
}

double gamma_func( double x, void *)
{
  return  ROOT::Math::tgamma(x);
}

// gamma function
class GammaFunction : public ROOT::Math::IGenFunction {

public:


  ROOT::Math::IGenFunction * Clone() const override {
    return new GammaFunction();
  }

private:

  double DoEval ( double x) const override {
    return ROOT::Math::tgamma(x);
  }

};


int printCheb( const ROOT::Math::ChebyshevApprox & c, double x0, double x1, FP func = 0 ) {

  double dx = (x1-x0)/10;
  for ( double x = x0; x < x1; x+= dx ) {

    double y = c(x);
    double ey = c.EvalErr(x).second;
    double y10 = c(x,10);
    double ey10 = c.EvalErr(x,10).second;
    double fVal = 0;
    if (func) fVal = func(x,0);
    std::cout << " x = " << x << " true Val = " << fVal << " y = " << y << " +/- " << ey << "    y@10 = " << y10 << " +/- " << ey10 << std::endl;
  }



  return 0;
}



int main() {


  // test with cos(x) + 1.0
  std::cout << "Test Cheb approx to step function :" << std::endl;
  ROOT::Math::ChebyshevApprox c(myfunc, 0, 0., 1.0, 40);
  printCheb(c, 0, 1, myfunc);
  std::cout << "Test integral of step function :" << std::endl;
  ROOT::Math::ChebyshevApprox *  cInteg = c.Integral();
  printCheb(*cInteg, 0., 1);
  delete cInteg;


  std::cout << "Test Cheb approx to Gamma function :" << std::endl;
  GammaFunction gf;
  ROOT::Math::ChebyshevApprox c2(gf, 1.0, 2.0, 40);
  printCheb(c2, 1.0, 2.0,  gamma_func);
  std::cout << "Test derivative of gammma :" << std::endl;
  ROOT::Math::ChebyshevApprox * cDeriv = c2.Deriv();
  printCheb(*cDeriv, 1.0, 2.0);
  delete cDeriv;



  return 0;

}

