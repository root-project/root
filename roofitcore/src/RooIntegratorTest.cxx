#ifndef __CINT__
#include "RooFitCore/RooNumber.hh"
#endif

#include <iomanip.h>

// Test one-dimensional numerical integration

void run1DTests() {

  // create a formula object to integrate numerically
  RooRealVar x("x","A variable",-1,+1,"ps");
  RooFormulaVar f1D("f1D","An integrand","exp(-abs(x))*cos(2*x)",x);

  // create an object that represents its integral over x
  RooRealIntegral If1D("If1D","Integral of f1D dx",f1D,x);

  // run a test suite
  test1D(If1D,x,-1,+1);
  test1D(If1D,x,-1,+5);
  test1D(If1D,x,-5,+1);

  Double_t infy= RooNumber::infinity;
  test1D(If1D,x,-infy,-5);
  test1D(If1D,x,-infy,-1);
  test1D(If1D,x,-infy,0);
  test1D(If1D,x,-infy,+1);
  test1D(If1D,x,-infy,+5);
  test1D(If1D,x,-infy,+infy);
  test1D(If1D,x,-5,+infy);
  test1D(If1D,x,-1,+infy);
  test1D(If1D,x,0,+infy);
  test1D(If1D,x,+1,+infy);
  test1D(If1D,x,+5,+infy);
}

Bool_t test1D(const RooAbsReal &integrator, RooRealVar &dependent, Double_t x1, Double_t x2) {

  dependent.setFitMin(x1);
  dependent.setFitMax(x2);
  Double_t numeric= integrator.getVal();
  Double_t analytic= analyticIntegral(x1,x2);
  Double_t err= (numeric-analytic)/analytic;

  cout << "=== Integral over (" << x1 << "," << x2 << ") ===" << endl;
  cout << setprecision(10) 
       << "Numeric = " << numeric << " , Analytic = " << analytic << " -> Fractional Error = "
       << err << endl
       << setprecision(5);
}

// Calculate the integral of exp(-|x|)*cos(2*x) over
// the interval (x1,x2) using analytic formulas. Either
// x1 or x2 can be infinite.

Double_t analyticIntegral(Double_t x1, Double_t x2) {
  Double_t result(0);
  if(x1 >= 0) {
    if(RooNumber::isInfinite(x2)) {
      result= 0.2 - positiveIntegral(x1) + positiveIntegral(0);
    }
    else {
      result= positiveIntegral(x2)-positiveIntegral(x1);
    }
  }
  else if(x2 <= 0) {
    if(RooNumber::isInfinite(x1)) {
      result= 0.2 - positiveIntegral(-x2) + positiveIntegral(0);
    }
    else {
      result= positiveIntegral(-x1)-positiveIntegral(-x2);
    }
  }
  else { // (x1,x2) spans zero
    result= analyticIntegral(x1,0) + analyticIntegral(0,x2);
  }
}

// Calculate the integral of exp(-x)*cos(2*x) over the
// interval (0,x) using an analytic formula valid for x >= 0.

Double_t positiveIntegral(Double_t x) {
  return exp(-x)/5.*(2*sin(2*x)-cos(2*x));
}
