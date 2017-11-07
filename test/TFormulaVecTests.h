#include <TF1.h>
#include <TF2.h>
#include <iostream>
#include <TString.h>

bool verbose  = true; 
//#define VEC_IN_CTOR

bool CheckValues(const TString & testName, double r, double r0) { 

   bool ret =  TMath::AreEqualAbs(r,r0,1.E-12);
   if (!ret) {
      std::cout << "Test failed for " << testName << "  : " <<  r << "   " << r0 <<  std::endl;
   }
   else
      if (verbose) std::cout << testName << "\t ok\n";

   return ret;
}


typedef double ( * FreeFunc1D) (double );
bool testVec1D(TF1 * f1, const TString & formula, FreeFunc1D func, double x ) {

   auto y = f1->Eval(x);
   auto y0 = func(x);

   bool ret = CheckValues(formula, y, y0);

   // check passing double_v interface
#ifdef R__HAS_VECCORE
   ROOT::Double_v vx = x;
   ROOT::Double_v vy = f1->EvalPar(&vx, nullptr);
   double y2 = vecCore::Get(vy,0);
   ret &= CheckValues(formula+TString("_v"), y2, y0);
#endif

   return ret;
}
bool testVec1D(const TString & formula, FreeFunc1D func, double x = 1.) {

   // test first by vectorizing in the constructor
   auto f1 = new TF1("f",formula,0,1,"VEC");
   bool ret = testVec1D(f1,formula,func,x);
   // test by vectorizing afterwards calling SetVectorized
   auto f2 = new TF1("f2",formula);
   f2->SetVectorized(true);
   //std::cout << "set vectprization" << std::endl;
   //f2->Print("V");
   ret &=  testVec1D(f2,formula,func,x);
   // test by removing vectorization
   f2->SetVectorized(false);
   ret &=  testVec1D(f1,formula,func,x);
   return ret;
}


typedef double ( * FreeFunc2D) (double, double );
bool testVec2D(TF2 * f1, const TString & formula, FreeFunc2D func, double x, double y ) {

   auto r = f1->Eval(x,y);
   auto r0 = func(x,y);

   bool ret = CheckValues(formula,r,r0);

   // check passing double_v interface
#ifdef R__HAS_VECCORE
   ROOT::Double_v vx[2] = { x, y};
   ROOT::Double_v vy = f1->EvalPar(vx, nullptr);
   double r2 = vecCore::Get(vy,0);
   ret &= CheckValues(formula+TString("_v"), r2, r0);
#endif

   return ret;
}

bool testVec2D(const TString & formula, FreeFunc2D func, double x = 1., double y = 1.) {

   auto f1 = new TF2("f",formula,0,1,0,1,"VEC");
   bool ret = testVec2D(f1, formula, func, x, y);

   auto f2 = new TF2("f",formula);
   f2->SetVectorized(true);
   ret &=  testVec2D(f2,formula,func,x,y);
   return ret;
}


double constant_function(double ) { return 3; }


bool testVecFormula() {

   bool ok = true;

   ok &= testVec1D("3+[0]",constant_function, 3.333);
   ok &= testVec1D("3",constant_function, 3.333);
   ok &= testVec1D("sin(x)",std::sin,1);
   ok &= testVec1D("cos(x)",std::cos,1);
   ok &= testVec1D("exp(x)",std::exp,1);
   ok &= testVec1D("log(x)",std::log,2);
   ok &= testVec1D("log10(x)",TMath::Log10,2);
   ok &= testVec1D("tan(x)",std::tan,1);
   //ok &= testVec1D("sinh(x)",std::sinh,1);
   //ok &= testVec1D("cosh(x)",std::cosh,1);
   //ok &= testVec1D("tanh(x)",std::tanh,1);
   ok &= testVec1D("asin(x)",std::asin,.1);
   ok &= testVec1D("acos(x)",std::acos,.1);
   ok &= testVec1D("atan(x)",std::atan,.1);
   ok &= testVec1D("sqrt(x)",std::sqrt,2);
   ok &= testVec1D("abs(x)",std::abs,-1);
   ok &= testVec2D("pow(x,y)",std::pow,2,3);
   ok &= testVec2D("min(x,y)",TMath::Min,2,3);
   ok &= testVec2D("max(x,y)",TMath::Max,2,3);
   ok &= testVec2D("atan2(x,y)",TMath::ATan2,2,3);

   return ok; 
}
