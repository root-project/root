#ifndef __CINT__
#include "RooFitCore/RooNumber.hh"
#endif

#include <math.h>
#include <iomanip>
using std::cout;
using std::endl;
using std::setprecision;

// Study the performance of the Monte Carlo integration algorithm

enum MCMode { Naive, Stratified, Importance, QuasiRandom };

void runMCStudy(Int_t samples= 100000, Bool_t wide= kFALSE) {
  Double_t exact(1.1763744607);
  RooRealVar x("x","A variable",-1,+1);
  if(wide) {
    x.setFitMin(-5);
    x.setFitMax(+5);
    exact= 1.7723636230009768;
  }
  RooRealVar s("s","A variable",1,2);
  RooFormulaVar f2D("f2D","An integrand","exp(-x*x/(s*s))/s",RooArgSet(x,s));

  RooRealBinding f(f2D,RooArgSet(x,s));
  RooMCIntegrator If(f);

  cout << "=== " << samples << " Samples ===" << endl;
  calculateMCSpread(If,Naive,samples,exact);
  calculateMCSpread(If,Stratified,samples,exact);
  calculateMCSpread(If,Importance,samples,exact);
  calculateMCSpread(If,QuasiRandom,samples,exact);
}

void calculateMCSpread(RooMCIntegrator &integrator, MCMode mode,
		       Int_t samples, Double_t exact, Int_t trials= 100) {

  // Perform a number of statistically independent Monte Carlo integrations of
  // the specified integrand and return a histogram of the resulting fractional errors
  // relative to the specified exact result. Each integration will use approximately
  // the specified number of integrand samples.

  Double_t sum1(0),sum2(0);
  if(mode == Naive) {
    const RooAbsFunc *func= integrator.integrand();
    int dim= func->getDimension();
    Double_t *x= new Double_t[dim];
    Double_t vol= integrator.grid().getVolume();
    for(int trial= 0; trial < trials; trial++) {
      Double_t sum(0);
      for(int sample= 0; sample < samples; sample++) {
	// choose random values for each dependent
	for(int index= 0; index < dim; index++) {
	  x[index]= func->getMinLimit(index) +
	    (func->getMaxLimit(index) - func->getMinLimit(index))*RooRandom::uniform();
	}
	// evaluate the function at this randomly chosen point
	sum+= integrator.integrand(x);
      }
      Double_t result= sum*vol/samples;
      sum1+= result;
      sum2+= result*result;
    }
    delete x;
  }
  else {
    integrator.setGenType(RooMCIntegrator::PseudoRandom);
    if(mode == Stratified) {
      integrator.setAlpha(0);
    }
    else {
      integrator.setAlpha(1.5);
      if(mode == QuasiRandom) integrator.setGenType(RooMCIntegrator::QuasiRandom);
    }
    for(int trial= 0; trial < trials; trial++) {
      Double_t result(0);
      if(mode == Stratified) {
	result= integrator.vegas(RooMCIntegrator::AllStages,samples,1);
      }
      else {
	// first refine the grid with 5 low precision steps
	Int_t presamples= samples/50;
	integrator.vegas(RooMCIntegrator::AllStages,presamples,5);
	// use the remaining samples for a higher precision integration on the refined grid
	result= integrator.vegas(RooMCIntegrator::ReuseGrid,samples-5*presamples,1);
      }
      sum1+= result;
      sum2+= result*result;
    }
  }

  // calculate the mean (relative to the exact result) and rms of this set of trials,
  // both normalized to the exact result.
  Double_t mean= (sum1/trials - exact)/exact;
  Double_t rms= sqrt((sum2 - sum1*sum1/trials)/(trials-1.))/exact;

  cout << "mode-" << mode << " : " << mean << " +/- " << rms << endl;
}

// Test Monte Carlo numerical integration

void runMCTests() {

  // create a formula object to integrate numerically
  Double_t pi= 2*atan2(1,0);
  RooRealVar x("x","A variable",0,pi);
  RooRealVar y("y","A variable",0,pi);
  RooRealVar z("z","A variable",0,pi);
  RooFormulaVar f3D("f3D","An integrand","1/(1-cos(x)*cos(y)*cos(z))",RooArgSet(x,y,z));

  // create an object that represents its integral over x
  RooRealIntegral If3D("If3D","Integral of f3D dx dy dz",f3D,RooArgSet(x,y,z));

  Double_t numeric= If3D.getVal();
  cout << " numeric = " << numeric << endl;
  cout << "analytic = " << 1.3932039296856768591842462603255*pi*pi*pi << endl;
}

// Test one-dimensional numerical integration

void run1DTests() {

  // create a formula object to integrate numerically
  RooRealVar x("x","A variable",-1,+1,"ps");
  RooGenericPdf f1D("f1D","exp(-abs(x))*cos(2*x)",x);

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
