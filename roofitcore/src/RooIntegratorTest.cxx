#ifndef __CINT__
#include "RooFitCore/RooNumber.hh"
#endif

#include <math.h>
#include <iomanip.h>

// Study the performance of the Monte Carlo integration algorithm

void runMCStudy() {
  RooRealVar x("x","A variable",-1,+1);
  RooRealVar s("s","A variable",1,2);
  RooFormulaVar f2D("f2D","An integrand","exp(-x*x/(s*s))/s",RooArgSet(x,s));

  RooRealBinding f(f2D,RooArgSet(x,s));
  RooMCIntegrator If(f);

  gStyle->SetOptStat();
  TCanvas *c= new TCanvas("mcstudy","MC Integration Study",800,700);
  c->Divide(2,2);

  If.setAlpha(0.0);
  c->cd(1);
  calculateMCSpread(If)->DrawClone();

  If.setAlpha(1.5);
  c->cd(2);
  calculateMCSpread(If)->DrawClone();
}

TH1F *calculateMCSpread(RooMCIntegrator &integrator, Int_t trials= 100) {

  Double_t exact= 1.1763744607;

  // create an empty histogram to fill
  TH1F *hist= new TH1F("hist","hist",20,-2e-4,+2e-4);

  for(int trial= 0; trial < trials; trial++) {
    // divide the first 10k calls between 5 iterations to refine the grid
    integrator.vegas(RooMCIntegrator::AllStages,2000,5);
    Double_t ferr= (integrator.vegas(RooMCIntegrator::ReuseGrid,10000,1) - exact)/exact;
    cout << "ferr = " << ferr << endl;
    hist->Fill(ferr);
  }

  return hist;
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
