#include <iostream>
//#include "FunctorNew.h"

//#include "Math/IGenFunction.h"
#include "Math/WrappedFunction.h"
#include "Math/WrappedParamFunction.h"
//#include "Fit/WrappedTF1.h"
#include "TStopwatch.h"
#include <cmath>
#include "TRandom2.h"
#include "TF1.h"
#include "TF2.h"
#include "Math/WrappedTF1.h"
#include "Math/WrappedMultiTF1.h"

#ifdef HAVE_ROOFIT
#include "RooRealVar.h"
#include "RooArgList.h"
#include "RooExponential.h"
#endif

#include "Math/IFunctionfwd.h"
#include "Math/IFunction.h"
#include "Math/Functor.h"
#include "Math/ParamFunctor.h"

#include <functional>
#include <vector>

//#define EXPFUNC
#ifndef EXPFUNC

#define NLOOP 100
#define NTIMES 500000
#define FUNC1D x+x;
#define FUNC x[0]+x[1]

#else

#define NLOOP 10
#define NTIMES 500000
#define FUNC1D std::exp(x);
#define FUNC  std::exp( x[0] + x[1] );

#endif

double freeFunction(const double * x ) {
   return FUNC;
   //return ;
}

double freeRootFunc2D(const double *x, const double *){
   return FUNC;
}
double freeRootFunc1D(const double *xx, const double *){
   double x = *xx;
   return FUNC1D;
}
double freeParamFunc1D(double x, double *){
   return FUNC1D;
}

double freeFunction1D(double  x ) {
   return FUNC1D;
}


class MyFunction {


public:
   double operator()(const double *x) const {
      return FUNC;
      //return x[0]*std::exp(x[1]);
   }

   double Derivative(const double * x, int /* icoord */) const { return FUNC; }
   double Eval(const double * x) const { return FUNC; }
};
struct MyDerivFunction {
   double operator()(const double *x, int ) const {
      return FUNC;
   }
};

class MyFunction1D {


public:

   double operator()(double x) const {
      return FUNC1D;
   }

   double operator()(const double * x) const {
      return (*this)(*x);
   }

   double Eval(double x) const { return FUNC1D; }

   double Derivative(double x) const { return FUNC1D; }
};



class DerivFunction : public ROOT::Math::IMultiGenFunction {

public:


   unsigned int NDim() const { return 2; }

   DerivFunction *  Clone() const {
      return new DerivFunction();
   }

private:


   double DoEval(const double *x) const {
      return FUNC;
   }

};


class DerivFunction1D : public ROOT::Math::IGenFunction {

public:

   DerivFunction1D *  Clone() const {
      return new DerivFunction1D();
   }

private:


   double DoEval(double x) const {
      return FUNC1D;
   }

};

struct F1D {
   double Eval(double x)  {
      return FUNC1D;
   }
};


const int Ntimes = NTIMES;

template <class Func>
void TestTime(const Func & f) {
  //double x[Ntimes];
  // use std::vector's to avoid crashes on Windows
   std::vector<double> x(Ntimes);
   TStopwatch w;
   TRandom2 r;
   r.RndmArray(Ntimes,&x[0]);
   w. Start();
   double s=0;
   for (int ipass = 0; ipass <NLOOP; ++ipass) {
      for (int i = 0; i < Ntimes-1; ++i) {
         const double * xx = &x[i];
         double y = f(xx);
         s+= y;
      }
   }
   w.Stop();
   std::cout << "Time for " << typeid(f).name() << "\t:  " << w.RealTime() << "  " << w.CpuTime() << std::endl;
   std::cout << s << std::endl;
}

template <class PFunc>
void TestTimePF( PFunc & f) {
  //double x[Ntimes];
  // use std::vector's to avoid crashes on Windows
   std::vector<double> x(Ntimes);
   TStopwatch w;
   TRandom2 r;
   r.RndmArray(Ntimes,&x[0]);
   w. Start();
   double s=0;
   double * p = 0;
   for (int ipass = 0; ipass <NLOOP; ++ipass) {
      for (int i = 0; i < Ntimes-1; ++i) {
         double y = f(&x[i],p);
         s+= y;
      }
   }
   w.Stop();
   std::cout << "Time for " << typeid(f).name() << "\t:  " << w.RealTime() << "  " << w.CpuTime() << std::endl;
   std::cout << s << std::endl;
}


void TestTimeGF(const ROOT::Math::IGenFunction & f) {
   TestTime(f);
}


void TestTimeTF1(TF1 & f) {
  //double x[Ntimes];
   std::vector<double> x(Ntimes);
   TStopwatch w;
   TRandom2 r;
   r.RndmArray(Ntimes,&x[0]);
   w. Start();
   double s=0;
   for (int ipass = 0; ipass <NLOOP; ++ipass) {
      for (int i = 0; i < Ntimes-1; ++i) {
         double y = f.EvalPar(&x[i],0);
         s+= y;
      }
   }
   w.Stop();
   std::cout << "Time for " << "TF1\t\t" << "\t:  " << w.RealTime() << "  " << w.CpuTime() << std::endl;
   std::cout << s << std::endl;
}

#ifdef HAVE_ROOFIT
void TestTimeRooPdf(RooAbsPdf & f, RooRealVar * vars) {
   //double x[Ntimes];
   std::vector<double> x(Ntimes);
   TStopwatch w;
   TRandom2 r;
   r.RndmArray(Ntimes,&x[0]);
   w. Start();
   double s=0;
//    RooArgSet * varSet = f.getVariables();
//    RooArgList varList(*varSet);
//    delete varSet;
//    RooAbsArg & arg = varList[0];
//    RooRealVar * vars = dynamic_cast<RooRealVar * > (&arg);
//    assert(x != 0);
   for (int ipass = 0; ipass <NLOOP; ++ipass) {
      for (int i = 0; i < Ntimes-1; ++i) {
         vars->setVal(x[i+1]);
         double y = x[i]*f.getVal();
         s+= y;
      }
   }
   w.Stop();
   std::cout << "Time for " << "RooPdf\t\t" << "\t:  " << w.RealTime() << "  " << w.CpuTime() << std::endl;
   std::cout << s << std::endl;
}
#endif


// test all functor constructs
void testMultiDim() {

   // multi-dim test
   std::cout <<"\n**************************************************************\n";
   std::cout <<"Test of Multi-dim functors" << std::endl;
   std::cout <<"***************************************************************\n\n";

   // test directly calling the function object
   MyFunction myf;
   TestTime(myf);

   // test from a free function pointer
   ROOT::Math::Functor f1(&freeFunction,2);
   TestTime(f1);

   // test from function object
   ROOT::Math::Functor f2(myf,2);
   TestTime(f2);

   // test from a member function
   ROOT::Math::Functor f3(&myf,&MyFunction::Eval,2);
   TestTime(f3);

   // test grad functor from an object providing eval and deriv.
   ROOT::Math::GradFunctor  f4(myf,2);
   TestTime(f4);

   // test grad functor from object and member functions
   ROOT::Math::GradFunctor  f5(&myf,&MyFunction::Eval, &MyFunction::Derivative, 2);
   TestTime(f5);

   // test from 2 function objects
   MyDerivFunction myderf;
   ROOT::Math::GradFunctor  f6(myf,myderf, 2);
   TestTime(f6);
}

// test all functor constructs
void testOneDim() {

  // test 1D functors
   std::cout <<"\n**************************************************************\n";
   std::cout <<"Test of 1D functors" << std::endl;
   std::cout <<"***************************************************************\n\n";

   // test dircectly calling function object
   MyFunction1D myf1;
   TestTime(myf1);

   /// test free function
   ROOT::Math::Functor1D  f1(&freeFunction1D);
   TestTime(f1);

   // test from function object
   ROOT::Math::Functor1D  f2(myf1);
   TestTime(f2);

   // test from member function
   ROOT::Math::Functor1D f3(&myf1,&MyFunction1D::Derivative);
   TestTime(f3);

   // testgrad functor

   // from function object implementing both
   ROOT::Math::GradFunctor1D  f4(myf1);
   TestTime(f4);

   // test grad functor from object and member functions
   ROOT::Math::GradFunctor1D  f5(&myf1,&MyFunction1D::Eval, &MyFunction1D::Derivative);
   TestTime(f5);

   // test from 2 function objects
   ROOT::Math::GradFunctor1D  f6(&freeFunction1D,myf1);
   TestTime(f6);


}


void testMore() {


   std::cout <<"\n**************************************************************\n";
   std::cout <<"Extra  functor tests" << std::endl;
   std::cout <<"***************************************************************\n\n";

   ROOT::Math::ParamFunctor fp1(&freeRootFunc2D);
   TestTimePF(fp1);

//    ROOT::Math::ParamFunctor1D fp2(&freeParamFunc1D);
//    TestTimePF(fp2);


   DerivFunction fdf;
   TestTime(fdf);


   //1D

   DerivFunction1D f13;
   TestTime(f13);




   //TestTimeGF(f3);
   ROOT::Math::WrappedFunction<> f5(freeFunction1D);
   TestTime(f5);

   ROOT::Math::WrappedMultiFunction<> f5b(freeFunction,2);
   TestTime(f5b);



   F1D fobj;
   //std::cout << typeid(&F1D::Eval).name() << std::endl;
   ROOT::Math::Functor1D f6(std::bind1st(std::mem_fun(&F1D::Eval), &fobj) );
   TestTime(f6);

   ROOT::Math::WrappedFunction<std::binder1st<std::mem_fun1_t<double, F1D, double> > >  f6a((std::bind1st(std::mem_fun(&F1D::Eval), &fobj)));
   TestTime(f6a);

   //typedef double( * FreeFunc ) (double );
   //ROOT::Math::WrappedMemFunction<F1D,FreeFunc>  f6b(&fobj, &F1D::Eval, );

//    typedef double (F1D::*MemFun)(double);
//    double (F1D::*p1 )(double) = &F1D::Eval;
//    std::cout << typeid(p1).name() << std::endl;
   ROOT::Math::WrappedMemFunction<F1D, double (F1D::*)(double) >  f6b(fobj, &F1D::Eval );
   TestTime(f6b);

   ROOT::Math::Functor1D f6c(&fobj, &F1D::Eval );
   TestTime(f6c);



#ifdef LATER
   FunctorNV<GradFunc, MyFunction> f5(myf);
   TestTime(f5);

   // test of virtuality two times
   Functor<GenFunc> f6(f3);
   TestTime(f6);
#endif

   TF1 tf1("tf1",freeRootFunc2D,0,1,0);
   //TF2 tf1("tf1","x+y",0,1,0,1);
   TestTimeTF1(tf1);

//    ROOT::Fit::WrappedTF1 f7(&tf1);
//    TestTime(f7);

   ROOT::Math::WrappedMultiTF1 f7b(tf1);
   TestTime(f7b);
   TestTimePF(f7b);

   ROOT::Math::WrappedParamFunction<> wf7(&freeRootFunc2D,2,0,0);
   TestTime(wf7);
   TestTimePF(wf7);

   // use the fact that TF1 implements operator(double *, double *)
   ROOT::Math::WrappedParamFunction<TF1*> wf7b(&tf1,2,0,0);
   TestTimePF(wf7b);



   TF1 tf2("tf2",freeRootFunc1D,0,1,0);
   TestTimeTF1(tf2);

   ROOT::Math::WrappedTF1 f7c(tf2);
   TestTime(f7c);


//    double xx[1] = {2};
//    f7(xx);

   ROOT::Math::Functor f8(f7b,f7b.NDim());
   TestTime(f8);

// this does not compile oin Windows, since it does not understand the default arguments
// It does not work for gcc 4.3 either.
// #ifndef _WIN32
//    ROOT::Math::Functor1D f9(&tf1,&TF1::Eval);
//    TestTime(f9);

//    ROOT::Math::Functor f10(&tf1,&TF1::EvalPar,tf1.GetNdim());
//    TestTime(f10);
// #endif



   // test with rootit
#ifdef HAVE_ROOFIT
   RooRealVar x("x","x",0);
   RooRealVar c("c","c",1.,1.,1.);
   RooExponential rooExp("exp","exponential",x,c);
   TestTimeRooPdf(rooExp,&x);
#endif

}

int main() {

   testMultiDim();

   testOneDim();

   testMore();

   return 0;

}
