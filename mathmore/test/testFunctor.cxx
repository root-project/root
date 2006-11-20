#include <iostream> 
//#include "FunctorNew.h"

//#include "Math/IGenFunction.h"
#include "Math/WrappedFunction.h"
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

#include <functional>

#define NLOOP 100
#define NTIMES 1000000
#define FUNC x[0]+x[1]
//#define FUNC x[0]*std::exp(x[1])
//#define FUNC 100 * (x[1]-x[0]*x[0])*(x[1]-x[0]*x[0]) + (1.-x[0])*(1-x[0])

#define FUNC1D x+x; 
//#define FUNC1D std::exp(x); 

double freeFunction(const double * x ) { 
   return FUNC; 
   //return ; 
}

double freeRootFunc2D(double *x, double *){ 
   return FUNC;
}
double freeRootFunc1D(double *xx, double *){ 
   double x = *xx;
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


class MyFunction1D { 


public: 

   double operator()(double x) const { 
      return FUNC1D; 
   } 

   double operator()(const double * x) const { 
      return (*this)(*x); 
   } 
   
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
   double x[Ntimes]; 
   TStopwatch w; 
   TRandom2 r; 
   r.RndmArray(Ntimes,x);
   w. Start(); 
   double s=0;
   for (int ipass = 0; ipass <NLOOP; ++ipass) {  
      for (int i = 0; i < Ntimes-1; ++i) { 
         double y = f(&x[i]); 
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
   double x[Ntimes]; 
   TStopwatch w; 
   TRandom2 r; 
   r.RndmArray(Ntimes,x);
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
   double x[Ntimes]; 
   TStopwatch w; 
   TRandom2 r; 
   r.RndmArray(Ntimes,x);
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


int main() { 


    
   MyFunction myf;
   TestTime(myf);

   MyFunction1D myf1;
   TestTime(myf1);

   ROOT::Math::Functor<ROOT::Math::IMultiGradFunction>  f1(myf,2); 
   TestTime(f1);

   ROOT::Math::Functor<ROOT::Math::IMultiGenFunction> f2(&freeFunction,2); 
   TestTime(f2);


   DerivFunction f3; 
   TestTime(f3);

   ROOT::Math::Functor<ROOT::Math::IMultiGenFunction> f4(&myf,&MyFunction::Eval,2); 
   TestTime(f4);

   //1D

   ROOT::Math::Functor1D<ROOT::Math::IGradFunction>  f11(myf1); 
   TestTime(f11);

   ROOT::Math::Functor1D<ROOT::Math::IGenFunction> f12(&freeFunction1D); 
   TestTime(f12);

   DerivFunction1D f13; 
   TestTime(f13);

   ROOT::Math::Functor1D<ROOT::Math::IGenFunction> f14(&myf1,&MyFunction1D::Derivative); 
   TestTime(f14);
   

   
   //TestTimeGF(f3); 
   typedef double( * FreeFunc ) (double ); 
   ROOT::Math::WrappedFunction<> f5(freeFunction1D);
   TestTime(f5);

   F1D fobj;
   //std::cout << typeid(&F1D::Eval).name() << std::endl;
   ROOT::Math::Functor1D<ROOT::Math::IGenFunction> f6(std::bind1st(std::mem_fun(&F1D::Eval), &fobj) );
   TestTime(f6);

   ROOT::Math::WrappedFunction<std::binder1st<std::mem_fun1_t<double, F1D, double> > >  f6a((std::bind1st(std::mem_fun(&F1D::Eval), &fobj)));
   TestTime(f6a);

   //ROOT::Math::WrappedMemFunction<F1D,FreeFunc>  f6b(&fobj, &F1D::Eval, );
   
//    typedef double (F1D::*MemFun)(double); 
//    double (F1D::*p1 )(double) = &F1D::Eval; 
//    std::cout << typeid(p1).name() << std::endl;   
   ROOT::Math::WrappedMemFunction<F1D, double (F1D::*)(double) >  f6b(fobj, &F1D::Eval );
   TestTime(f6b);

   ROOT::Math::Functor1D<ROOT::Math::IGenFunction> f6c(&fobj, &F1D::Eval );
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

   TF1 tf2("tf2",freeRootFunc1D,0,1,0);
   TestTimeTF1(tf2);

   ROOT::Math::WrappedTF1 f7c(tf2);
   TestTime(f7c);
   
//    double xx[1] = {2};
//    f7(xx);

   ROOT::Math::Functor<ROOT::Math::IMultiGenFunction> f8(f7b,f7b.NDim());
   TestTime(f8);

// this does not compile oin Windows, since it does not understand the default arguments
#ifndef _WIN32
   ROOT::Math::Functor1D<ROOT::Math::IGenFunction> f9(&tf1,&TF1::Eval);
   TestTime(f9);

   ROOT::Math::Functor<ROOT::Math::IMultiGenFunction> f10(&tf1,&TF1::EvalPar,tf1.GetNdim());
   TestTime(f10);
#endif
   


   // test with rootit
#ifdef HAVE_ROOFIT
   RooRealVar x("x","x",0);
   RooRealVar c("c","c",1.,1.,1.);
   RooExponential rooExp("exp","exponential",x,c);
   TestTimeRooPdf(rooExp,&x);
#endif

   return 0;
}
