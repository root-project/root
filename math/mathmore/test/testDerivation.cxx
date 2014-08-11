#include "Math/Polynomial.h"
#include "Math/Derivator.h"
#include "Math/IFunction.h"
#include "Math/Functor.h"
#include "Math/WrappedFunction.h"
#include "Math/WrappedParamFunction.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

#ifdef HAVE_ROOTLIBS

#include "TStopwatch.h"
#include "TF1.h"
#include "Math/WrappedTF1.h"
#include "Math/WrappedMultiTF1.h"
#include "Math/DistFunc.h"

#endif

const double ERRORLIMIT = 1E-5;

typedef double ( * FP ) ( double, void * );
typedef double ( * FP2 ) ( double );


double myfunc ( double x, void * ) {

  return std::pow( x, 1.5);
}

double myfunc2 ( double x) {
  return std::pow( x, 1.5);
}

int testDerivation() {

   int status = 0;


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
  status += fabs(result-f1->Derivative(x0)) > ERRORLIMIT;


  // Derivative of a free function
  // Works when compiled c++, compiled ACLiC, does not work when interpreted by CINT
  FP f2 = &myfunc;
  der->SetFunction(f2);

  std::cout << "Derivative of a free function f(x) = x^(3/2) at x = 2" << std::endl;
  std::cout << "EvalCentral:  " << der->EvalCentral(x0) << std::endl;
  std::cout << "EvalForward:  " << der->EvalForward(x0) << std::endl;
  std::cout << "EvalBackward: " << der->EvalBackward(x0) << std::endl;

  std::cout << "Exact result: " << 1.5*sqrt(x0) << std::endl << std::endl;

  status += fabs(der->EvalCentral(x0)-1.5*sqrt(x0)) > ERRORLIMIT;
  status += fabs(der->EvalForward(x0)-1.5*sqrt(x0)) > ERRORLIMIT;
  status += fabs(der->EvalBackward(x0)-1.5*sqrt(x0)) > ERRORLIMIT;



  // Derivative of a free function wrapped in an IGenFunction
  // Works when compiled c++, compiled ACLiC, does not work when interpreted by CINT
  ROOT::Math::Functor1D *f3 = new ROOT::Math::Functor1D (&myfunc2);

  std::cout << "Derivative of a free function wrapped in a Functor f(x) = x^(3/2) at x = 2" << std::endl;
  std::cout << "EvalCentral:  " << der->Eval( *f3, x0) << std::endl;
  der->SetFunction(*f3);
  std::cout << "EvalForward:  " << der->EvalForward(x0) << std::endl;
  std::cout << "EvalBackward: " << der->EvalBackward(x0) << std::endl;
  std::cout << "Exact result: " << 1.5*sqrt(x0) << std::endl << std::endl;

  status += fabs(der->Eval( *f3, x0)-1.5*sqrt(x0)) > ERRORLIMIT;
  status += fabs(der->EvalForward(x0)-1.5*sqrt(x0)) > ERRORLIMIT;
  status += fabs(der->EvalBackward(x0)-1.5*sqrt(x0)) > ERRORLIMIT;


  // tets case when an empty Derivator is used

  ROOT::Math::Derivator der2;
  std::cout << "Tes a derivator without a function" << std::endl;
  std::cout << der2.Eval(1.0) << std::endl;

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

  return status;
}


#ifdef HAVE_ROOTLIBS

void testDerivPerf() {


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

  timer.Start();
  s1 = 0;
  for (int i = 0; i < n; ++i) {
     double x = x1 + dx*i;
     s1+= ROOT::Math::Derivator::Eval(f1,x);
  }
  timer.Stop();
  std::cout << "Time using ROOT::Math::Derivator(3):\t" << timer.RealTime() << std::endl;
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



}

double userFunc(const double *x, const double *) {
   return std::exp(-x[0]);
}
double userFunc1(double x) { return userFunc(&x, 0); }

double userFunc2(const double * x) { return userFunc(x, 0); }

void testDerivPerfUser() {


   std::cout << "\n\n***************************************************************\n";
   std::cout << "Test derivation performances - using a User function\n\n";

  ROOT::Math::WrappedFunction<> f1(userFunc1);

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

  timer.Start();
  s1 = 0;
  for (int i = 0; i < n; ++i) {
     double x = x1 + dx*i;
     s1+= ROOT::Math::Derivator::Eval(f1,x);
  }
  timer.Stop();
  std::cout << "Time using ROOT::Math::Derivator(3):\t" << timer.RealTime() << std::endl;
  pr = std::cout.precision(18); std::cout << s1 << std::endl; std::cout.precision(pr);


  TF1 f2("uf",userFunc,0,10,0);

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

  //typedef double( * FN ) (const double *, const double * );
  ROOT::Math::WrappedMultiFunction<> f3(userFunc2,1);
  timer.Start();
  s1 = 0;
  double xx[1];
  for (int i = 0; i < n; ++i) {
     xx[0] = x1 + dx*i;
     s1+= ROOT::Math::Derivator::Eval(f3,xx,0);
  }
  timer.Stop();
  std::cout << "Time using ROOT::Math::Derivator Multi:\t" << timer.RealTime() << std::endl;
  pr = std::cout.precision(18); std::cout << s1 << std::endl; std::cout.precision(pr);



}


double gausFunc( const double * x, const double * p) {
   return p[0] * ROOT::Math::normal_pdf(x[0], p[2], p[1] );
}


void testDerivPerfParam() {


   std::cout << "\n\n***************************************************************\n";
   std::cout << "Test derivation performances - using a Gaussian Param function\n\n";

   //TF1 gaus("gaus","gaus",-10,10);
   TF1 gaus("gaus",gausFunc,-10,10,3);
  double params[3] = {10,1.,1.};
  gaus.SetParameters(params);

  ROOT::Math::WrappedTF1 f1(gaus);

  TStopwatch timer;
  int n = 300000;
  double x1 = 0; double x2 = 10;
  double dx = (x2-x1)/double(n);

  timer.Start();
  double s1 = 0;
  for (int i = 0; i < n; ++i) {
     double x = x1 + dx*i;
     // param derivatives
     s1 += ROOT::Math::Derivator::Eval(f1,x,params,0);
     s1 += ROOT::Math::Derivator::Eval(f1,x,params,1);
     s1 += ROOT::Math::Derivator::Eval(f1,x,params,2);
  }
  timer.Stop();
  std::cout << "Time using ROOT::Math::Derivator (1D) :\t" << timer.RealTime() << std::endl;
  int pr = std::cout.precision(18); std::cout << s1 << std::endl; std::cout.precision(pr);

  ROOT::Math::WrappedParamFunction<> f2(&gausFunc,1,params,params+3);
  double xx[1];

  timer.Start();
  s1 = 0;
  for (int i = 0; i < n; ++i) {
     xx[0] = x1 + dx*i;
     s1 += ROOT::Math::Derivator::Eval(f2,xx,params,0);
     s1 += ROOT::Math::Derivator::Eval(f2,xx,params,1);
     s1 += ROOT::Math::Derivator::Eval(f2,xx,params,2);
  }
  timer.Stop();
  std::cout << "Time using ROOT::Math::Derivator(ND):\t" << timer.RealTime() << std::endl;
  pr = std::cout.precision(18); std::cout << s1 << std::endl; std::cout.precision(pr);

  // test that func parameters have not been changed
  assert( std::fabs(params[0] - gaus.GetParameter(0)) < 1.E-15);
  assert( std::fabs(params[1] - gaus.GetParameter(1)) < 1.E-15);
  assert( std::fabs(params[2] - gaus.GetParameter(2)) < 1.E-15);

  timer.Start();
  s1 = 0;
  double g[3];
  for (int i = 0; i < n; ++i) {
     xx[0] = x1 + dx*i;
     gaus.GradientPar(xx,g,1E-8);
     s1 += g[0];
     s1 += g[1];
     s1 += g[2];
  }
  timer.Stop();
  std::cout << "Time using TF1::ParamGradient:\t\t" << timer.RealTime() << std::endl;
  pr = std::cout.precision(18); std::cout << s1 << std::endl; std::cout.precision(pr);

}

#endif

int main() {

  int status = 0;

  status += testDerivation();

#ifdef HAVE_ROOTLIBS
  testDerivPerf();
  testDerivPerfUser();
  testDerivPerfParam();
#endif

  return status;

}
