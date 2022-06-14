// @(#)root/test:$Id$
// Author: Lorenzo Moneta   06/2005
///////////////////////////////////////////////////////////////////////////////////
//
//  MathCore Benchmark test suite
//  ==============================
//
//  This program performs tests :
//     - mathematical functions in particular the statistical functions by estimating
//         pdf, cdf and quantiles. cdf are estimated directly and compared with calculated integral from pdf
//     - physics vectors (2D, 3D and 4D) including I/O for every type and for both double and Double32_t
//     - SMatrix and SVectors including I/O for double and Double32_t types
//     - I/O of complex objects which dictionary has been generated using CINT (default) or Reflex
//           TrackD and TrackD32 which contain  physics vectors of double and Double32_t
//           TrackErrD and TrackErrD32 which contain physics vectors and an SMatrix of double and Double32_t
//           VecTrackD which contains an std::vector<TrackD>
//
//
// the program cun run only in compiled mode.
// To run outside ROOT do:
//
//    > cd $ROOTSYS/test
//    > make stressMathMore
//    > ./stressMathMore
//
//  to run using REflex set before compiling the environment variable useReflex.
//
//    > export useReflex=1
//    > make stressMathMore
//    > ./stressMathMore
//
// to run inside ROOT using ACliC
//  for using CINT you need first to have the library libTrackMathCoreDict.so
//   (type:  make libTrackMathCoreDict.so to make it)
//
//   root> gSystem->Load("libMathCore");
//   root> gSystem->Load("libTree");
//   root> gSystem->Load("libHist");
//   root> .x stressMathCore.cxx+
//

// for using Reflex dictionaries you need first to have the library libTrackMathCoreRflx.so
//   (type:  make libTrackMathCoreRflx.so to make it)
//
//   root> gSystem->Load("libMathCore");
//   root> gSystem->Load("libTree");
//   root> gSystem->Load("libHist");
//   root> gSystem->Load("libReflex");
//   root> gSystem->SetIncludePath("-DUSE_REFLEX");
//   root> .x stressMathCore.cxx+
//
//


#ifndef __CINT__


#include "Math/DistFuncMathCore.h"
//#define USE_MATHMORE
#ifdef USE_MATHMORE
#include "Math/DistFuncMathMore.h"
#endif

#include "Math/IParamFunction.h"
#include "Math/Integrator.h"
#include "Math/IntegratorOptions.h"
#include <iostream>
#include <iomanip>
#include <limits>
#include <cmath>
#include "TBenchmark.h"
#include "TROOT.h"
#include "TRandom3.h"
#include "TSystem.h"
#include "TTree.h"
#include "TFile.h"
#include "TF1.h"
#include "TMath.h"

#include "Math/Vector2D.h"
#include "Math/Vector3D.h"
#include "Math/Vector4D.h"
#include "Math/VectorUtil.h"

#include "Math/SVector.h"
#include "Math/SMatrix.h"

R__ADD_INCLUDE_PATH($ROOTSYS/test)

#include "TrackMathCore.h"
#include "Math/GenVector/RotationZ.h" // Workaround to autoload libGenVector ROOT-7056

using namespace ROOT::Math;

#endif


//#define DEBUG

bool debug = true;  // print out reason of test failures
bool debugTime = false; // print out separate timings for vectors
bool removeFiles = true; // remove Output root files

void PrintTest(std::string name) {
   std::cout << std::left << std::setw(40) << name;
}

void PrintStatus(int iret) {
   if (iret == 0)
      std::cout <<"\t\t................ OK" << std::endl;
   else
      std::cout <<"\t\t............ FAILED " << std::endl;
}


int compare( std::string name, double v1, double v2, double scale = 2.0) {
  //  ntest = ntest + 1;

   //std::cout << std::setw(50) << std::left << name << ":\t";

  // numerical double limit for epsilon
   double eps = scale* std::numeric_limits<double>::epsilon();
   int iret = 0;
   double delta = v2 - v1;
   double d = 0;
   if (delta < 0 ) delta = - delta;
   if (v1 == 0 || v2 == 0) {
      if  (delta > eps ) {
         iret = 1;
      }
   }
   // skip case v1 or v2 is infinity
   else {
      d = v1;

      if ( v1 < 0) d = -d;
      // add also case when delta is small by default
      if ( delta/d  > eps && delta > eps )
         iret =  1;
   }

   if (iret) {
      if (debug) {
         int pr = std::cout.precision (18);
         std::cout << "\nDiscrepancy in " << name.c_str() << "() :\n  " << v1 << " != " << v2 << " discr = " << int(delta/d/eps)
                   << "   (Allowed discrepancy is " << eps  << ")\n\n";
         std::cout.precision (pr);
      //nfail = nfail + 1;
      }
   }
   //else
      //  std::cout <<".";

   return iret;
}

#ifndef __CINT__


// trait class  for distinguishing the number of parameters for the various functions
template<class Func, unsigned int NPAR>
struct Evaluator {
   static double F(Func f,  double x, const double * ) {
      return f(x);
   }
};
template<class Func>
struct Evaluator<Func, 1> {
   static double F(Func f,  double x, const double * p) {
      return f(x,p[0]);
   }
};
template<class Func>
struct Evaluator<Func, 2> {
   static double F(Func f,  double x, const double * p) {
      return f(x,p[0],p[1]);
   }
};
template<class Func>
struct Evaluator<Func, 3> {
   static double F(Func f,  double x, const double * p) {
      return f(x,p[0],p[1],p[2]);
   }
};

// global test variable
int NFuncTest = 100;

// statistical function class
// template on the number of parameters
template<class Func, class FuncQ, int NPAR, int NPARQ=NPAR-1>
class StatFunction : public ROOT::Math::IParamFunction {

public:

   StatFunction(Func pdf, Func cdf, FuncQ quant) : fPdf(pdf), fCdf(cdf), fQuant(quant)
   {
      fScale1 = 1.0E6; //scale for cdf test (integral)
      fScale2 = 10;  //scale for quantile test
      for(int i = 0; i< NPAR; ++i) fParams[i]=0;
   }


   unsigned int NPar() const { return NPAR; }
   const double * Parameters() const { return fParams; }
   ROOT::Math::IGenFunction * Clone() const { return new StatFunction(fPdf,fCdf,fQuant); }
   void SetParameters(const double * p) { std::copy(p,p+NPAR,fParams); }
   void SetParameters(double p0) { *fParams = p0; }
   void SetParameters(double p0, double p1) { *fParams = p0; *(fParams+1) = p1; }
   void SetParameters(double p0, double p1, double p2) { *fParams = p0; *(fParams+1) = p1; *(fParams+2) = p2; }
   static void SetNTest(int n) { NFuncTest = n; }

   double Cdf(double x) const {
      return Evaluator<Func,NPAR>::F(fCdf,x, fParams);
   }
   double Quantile(double x) const {
      double z =  Evaluator<FuncQ,NPARQ>::F(fQuant,x, fParams);
      if ((NPAR - NPARQ) == 1)
         z += fParams[NPAR-1]; // adjust the offset
      return z;
   }

   // test cumulative function
   int Test(double x1, double x2, double xl = 1, double xu = 0, bool cumul = false);

   void ScaleTol1(double s) { fScale1 *= s; }
   void ScaleTol2(double s) { fScale2 *= s; }


private:


   double DoEvalPar(double x, const double * ) const {
      // implement explicitly using cached parameter values
      return Evaluator<Func,NPAR>::F(fPdf,x, fParams);
   }

   Func fPdf;
   Func fCdf;
   FuncQ fQuant;
   double fParams[NPAR];
   double fScale1;
   double fScale2;
};


// test cdf at value f
template<class F1, class F2, int N1, int N2>
int StatFunction<F1,F2,N1,N2>::Test(double xmin, double xmax, double xlow, double xup, bool c) {

   int iret = 0;

   // scan all values from xmin to xmax
   double dx = (xmax-xmin)/NFuncTest;
#ifdef USE_MATHMORE
   for (int i = 0; i < NFuncTest; ++i) {
      double v1 = xmin + dx*i;  // value used  for testing
      double q1 = Cdf(v1);
      //std::cout << "v1 " << v1 << " pdf " << (*this)(v1) << " cdf " << q1 << " quantile " << Quantile(q1) << std::endl;
      // calculate integral of pdf
      Integrator ig(IntegrationOneDim::kADAPTIVESINGULAR, 1.E-12,1.E-12,100000);
      ig.SetFunction(*this);
      double q2 = 0;
      if (!c) {
         // lower intergal (cdf)
         if (xlow >= xup || xlow > xmin)
            q2 = ig.IntegralLow(v1);
         else
            q2 = ig.Integral(xlow,v1);

         // use a larger scale (integral error is 10-9)
         iret |= compare("test _cdf", q1, q2, fScale1 );
         // test the quantile
         double v2 = Quantile(q1);
         iret |= compare("test _quantile", v1, v2, fScale2 );
      }
      else {
         // upper integral (cdf_c)
         if (xlow >= xup || xup < xmax)
            q2 = ig.IntegralUp(v1);
         else
            q2 = ig.Integral(v1,xup);

         iret |= compare("test _cdf_c", q1, q2, fScale1);
         double v2 = Quantile(q1);
         iret |= compare("test _quantile_c", v1, v2, fScale2 );
      }
      if (iret)  {
         std::cout << "Failed test for x = " << v1 << " p = ";
         for (int j = 0; j < N1; ++j) std::cout << fParams[j] << "\t";
         std::cout << std::endl;
         break;
      }
   }
#else
   // use TF1 for the integral
//    std::cout << "xlow-xuo " << xlow << "   " << xup << std::endl;
//    std::cout << "xmin-xmax " << xmin << "   " << xmax << std::endl;
   double x1,x2 = 0;
   if (xlow >= xup) {
      x1 = -100; x2 = 100;
   }
   else if (xup < xmax) {
      x1 = xlow; x2 = 100;
   }
   else {
      x1=xlow;   x2 = xup;
   }
   //std::cout << "x1-x2 " << x1 << "   " << x2 << std::endl;
   TF1 * f = new TF1("ftemp",ParamFunctor(*this),x1,x2,0);
   //ROOT::Math::IntegratorOneDimOptions::SetDefaultIntegrator("Gauss");

   for (int i = 0; i < NFuncTest; ++i) {
      double v1 = xmin + dx*i;  // value used  for testing
      double q1 = Cdf(v1);

      double q2 = 0;
      if (!c) {
         q2 = f->Integral(x1,v1,1.E-12);
         // use a larger scale (integral error is 10-9)
         iret |= compare("test _cdf", q1, q2, fScale1 );
         // test the quantile
         double v2 = Quantile(q1);
         iret |= compare("test _quantile", v1, v2, fScale2 );
      }
      else {
         // upper integral (cdf_c)
         q2 = f->Integral(v1,x2,1.E-12);
         iret |= compare("test _cdf_c", q1, q2, fScale1);
         double v2 = Quantile(q1);
         iret |= compare("test _quantile_c", v1, v2, fScale2 );
      }
      // if (!c)   std::cout << "i = " << i << " x1 = " << x1 << " v  = " << v1 << " pdf " << (*this)(v1) << " cdf " << q1 << " Int(x1,v)= " << q2 << std::endl;
      // else
      //    std::cout << "i = " << i << " x2 = " << x2 << " v  = " << v1 << " pdf " << (*this)(v1) << " cdf_c " << q1 << " Int(v,x2)= " << q2 << std::endl;

   }
   delete f;
#endif

   if (c || iret != 0) PrintStatus(iret);
   return iret;

}

// typedef defining the functions
typedef double ( * F0) ( double);
typedef double ( * F1) ( double, double);
typedef double ( * F2) ( double, double, double);
typedef double ( * F3) ( double, double, double, double);

typedef StatFunction<F2,F2,2,2> Dist_beta;
typedef StatFunction<F2,F1,2>   Dist_breitwigner;
typedef StatFunction<F2,F1,2>   Dist_chisquared;
typedef StatFunction<F3,F2,3>   Dist_fdistribution;
typedef StatFunction<F3,F2,3>   Dist_gamma;
typedef StatFunction<F2,F1,2>   Dist_gaussian;
typedef StatFunction<F3,F2,3>   Dist_lognormal;
typedef StatFunction<F2,F1,2>   Dist_tdistribution;
typedef StatFunction<F2,F1,2>   Dist_exponential;
typedef StatFunction<F2,F1,2>   Dist_landau;
typedef StatFunction<F3,F2,3>   Dist_uniform;



#define CREATE_DIST(name) Dist_ ##name  dist( name ## _pdf, name ## _cdf, name ##_quantile );
#define CREATE_DIST_C(name) Dist_ ##name  distc( name ## _pdf, name ## _cdf_c, name ##_quantile_c );
// #define CREATE_DIST_C(name) Dist_ ##name  distc( name ## _pdf, name ## _cdf_c, 0 );
// #else
// #endif

template<class Distribution>
int TestDist(Distribution & d, double x1, double x2) {
   int ir = 0;
   ir |= d.Test(x1,x2);
   return ir;
}

int testStatFunctions(int nfunc = 100 ) {
   // test statistical functions

   int iret = 0;
   NFuncTest = nfunc;

   {
      PrintTest("Beta distribution");
      CREATE_DIST(beta);
      dist.SetParameters( 2, 2);
      iret |= dist.Test(0.01,0.99,0.,1.);
      CREATE_DIST_C(beta);
      distc.SetParameters( 2, 2);
      iret |= distc.Test(0.01,0.99,0.,1.,true);
   }

   {
      PrintTest("Gamma distribution");
#ifdef USE_MATHMORE // gamma_quantile is in mathmore
      CREATE_DIST(gamma);
      dist.SetParameters( 2, 1);
      iret |= dist.Test(0.05,5, 0.,TMath::Infinity());
#endif
      CREATE_DIST_C(gamma);
      distc.SetParameters( 2, 1);
      iret |= distc.Test(0.05,5, 0.,TMath::Infinity(),true);
   }

   {
      PrintTest("Chisquare distribution");
#ifdef USE_MATHMORE
      CREATE_DIST(chisquared);
      dist.SetParameters( 10, 0);
      dist.ScaleTol2(10);
      iret |= dist.Test(0.05,30, 0.,1.);
#endif
      CREATE_DIST_C(chisquared);
      distc.SetParameters( 10, 0);
      distc.ScaleTol2(10000000);  // t.b.c.
      iret |= distc.Test(0.05,30, 0.,TMath::Infinity(),true);

   }
   {
      PrintTest("Normal distribution ");
      CREATE_DIST(gaussian);
      dist.SetParameters( 2, 1);
      dist.ScaleTol2(100);
      iret |= dist.Test(-3,5);
      CREATE_DIST_C(gaussian);
      distc.SetParameters( 1, 0);
      distc.ScaleTol2(100);
      iret |= distc.Test(-3,5,1,0,true);
   }
   {
      PrintTest("BreitWigner distribution ");
      CREATE_DIST(breitwigner);
      dist.SetParameters( 1);
      dist.ScaleTol1(1E8);
      dist.ScaleTol2(10);
      iret |= dist.Test(-5,5);
      CREATE_DIST_C(breitwigner);
      distc.SetParameters( 1);
      distc.ScaleTol1(1E8);
      distc.ScaleTol2(10);
      iret |= distc.Test(-5,5,1,0,true);
   }
   {
      PrintTest("F    distribution ");
      CREATE_DIST(fdistribution);
      dist.SetParameters( 5, 4);
      dist.ScaleTol1(1000000);
      dist.ScaleTol2(10);
      // if enlarge scale test fails
      iret |= dist.Test(0.05,5,0,1);
      CREATE_DIST_C(fdistribution);
      distc.SetParameters( 5, 4);
      distc.ScaleTol1(100000000);
      distc.ScaleTol2(10);
      // if enlarge scale test fails
      iret |= distc.Test(0.05,5,0,TMath::Infinity(),true);
   }
#ifdef USE_MATHMORE // wait t quantile is in mathcore
   {
      PrintTest("t    distribution ");
      CREATE_DIST(tdistribution);
      dist.SetParameters( 10 );
//       dist.ScaleTol1(1000);
       dist.ScaleTol2(5000);
      iret |= dist.Test(-10,10);
      CREATE_DIST_C(tdistribution);
      distc.SetParameters( 10 );
      distc.ScaleTol2(10000);  // t.b.c.
      iret |= distc.Test(-10,10,1,0,true);
   }
#endif
   {
      PrintTest("lognormal distribution");
      CREATE_DIST(lognormal);
      dist.SetParameters(1,1 );
      dist.ScaleTol1(1000);
      iret |= dist.Test(0.01,5,0,1);
      CREATE_DIST_C(lognormal);
      distc.SetParameters(1,1 );
      distc.ScaleTol1(1000);
      distc.ScaleTol2(1000000); // t.b.c.
      iret |= distc.Test(0.01,5,0,TMath::Infinity(),true);
   }

   {
      PrintTest("Exponential distribution");
      CREATE_DIST(exponential);
      dist.SetParameters( 2);
      dist.ScaleTol2(100);
      iret |= dist.Test(0.,5.,0.,1.);
      CREATE_DIST_C(exponential);
      distc.SetParameters( 2);
      distc.ScaleTol2(100);
      iret |= distc.Test(0.,5.,0.,1.,true);
   }
   {
      PrintTest("Landau distribution");
      CREATE_DIST(landau);
      dist.SetParameters( 2);
       // Landau is not very precise (put prec at 10-6)
      // as indicated in Landau paper (
      dist.ScaleTol1(10000);
      dist.ScaleTol2(1.E10);
      iret |= dist.Test(-1,10,-TMath::Infinity(),TMath::Infinity());
      CREATE_DIST_C(landau);
      distc.SetParameters( 2);
      distc.ScaleTol1(1000);
      distc.ScaleTol2(1E10);
      iret |= distc.Test(-1,10,-TMath::Infinity(), TMath::Infinity(),true);
   }

   {
      PrintTest("Uniform distribution");
      CREATE_DIST(uniform);
      dist.SetParameters( 1, 2);
      iret |= dist.Test(1.,2.,1.,2.);
      CREATE_DIST_C(uniform);
      distc.SetParameters( 1, 2);
      iret |= distc.Test(1.,2.,1.,2.,true);
   }


   return iret;
}

//*******************************************************************************************************************
// GenVector tests
//*******************************************************************************************************************

// trait for getting vector name

template<class V>
struct VecType {
   static std::string name() { return "MathCoreVector";}
};
template<>
struct VecType<XYVector> {
   static std::string name() { return "XYVector";}
   static std::string name32() { return "ROOT::Math::DisplacementVector2D<ROOT::Math::Cartesian2D<Double32_t> >";}
};
template<>
struct VecType<Polar2DVector> {
   static std::string name() { return "Polar2DVector";}
};
template<>
struct VecType<XYZVector> {
   static std::string name() { return "XYZVector";}
   static std::string name32() { return "ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<Double32_t> >";}
};
template<>
struct VecType<Polar3DVector> {
   static std::string name() { return "Polar3DVector";}
};
template<>
struct VecType<RhoEtaPhiVector> {
   static std::string name() { return "RhoEtaPhiVector";}
};
template<>
struct VecType<RhoZPhiVector> {
   static std::string name() { return "RhoZPhiVector";}
};
template<>
struct VecType<PxPyPzEVector> {
   static std::string name() { return "PxPyPzEVector";}
   static std::string name32() { return "ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<Double32_t> >";}
};
template<>
struct VecType<PtEtaPhiEVector> {
   static std::string name() { return "PtEtaPhiEVector";}
};
template<>
struct VecType<PtEtaPhiMVector> {
   static std::string name() { return "PtEtaPhiMVector";}
};
template<>
struct VecType<PxPyPzMVector> {
   static std::string name() { return "PxPyPzMVector";}
};
template<>
struct VecType<SVector<double,3> > {
   static std::string name() { return "SVector3";}
   static std::string name32() { return "ROOT::Math::SVector<Double32_t,3>";}
};
template<>
struct VecType<SVector<double,4> > {
   static std::string name() { return "SVector4";}
   static std::string name32() { return "ROOT::Math::SVector<Double32_t,4>";}
};
template<>
struct VecType<TrackD> {
   static std::string name() { return "TrackD";}
};
template<>
struct VecType<TrackD32> {
   static std::string name() { return "TrackD32";}
};
template<>
struct VecType<TrackErrD> {
   static std::string name() { return "TrackErrD";}
};
template<>
struct VecType<TrackErrD32> {
   static std::string name() { return "TrackErrD32";}
};
template<>
struct VecType<VecTrack<TrackD> > {
   static std::string name() { return "VecTrackD";}
};
template<>
struct VecType<VecTrack<TrackErrD> > {
   static std::string name() { return "VecTrackErrD";}
};


// generic (2 dim)
template<class V, int Dim>
struct VecOp {

   template<class It>
   static V  Create(It &x, It &y, It & , It&  ) {       return V(*x++,*y++);  }
   template<class It>
   static void Set(V & v, It &x, It &y, It &, It&) { v.SetXY(*x++,*y++);  }

   static double Add(const V & v) { return v.X() + v.Y(); }
   static double Delta(const V & v1, const V & v2) { double d = ROOT::Math::VectorUtil::DeltaPhi(v1,v2);  return d*d; } // is v2-v1


};
// specialized for 3D
template<class V>
struct VecOp<V,3> {

   template<class It>
   static V  Create(It &x, It& y, It& z , It&  ) { return V(*x++,*y++,*z++); }
   template<class It>
   static void Set(V & v, It & x, It &y, It &z, It&) { v.SetXYZ(*x++,*y++,*z++); }
   static V  Create(double x, double y, double z , double  ) { return  V(x,y,z); }
   static void Set(V & v, double x, double y, double z, double) { v.SetXYZ(x,y,z); }
   static double Add(const V & v) { return v.X() + v.Y() + v.Z(); }
   static double Delta(const V & v1, const V & v2) { return ROOT::Math::VectorUtil::DeltaR2(v1,v2); }


};

// specialized for 4D
template<class V>
struct VecOp<V,4> {

   template<class It>
   static V  Create(It &x, It &y, It &z , It &t ) { return V(*x++,*y++,*z++,*t++);}
   template<class It>
   static void Set(V & v, It & x, It &y, It &z, It &t) { v.SetXYZT(*x++,*y++,*z++,*t++);  }

   static double Add(const V & v) { return v.X() + v.Y() + v.Z() + v.E(); }
   static double Delta(const V & v1, const V & v2) {
      return ROOT::Math::VectorUtil::DeltaR2(v1,v2) + ROOT::Math::VectorUtil::InvariantMass(v1,v2);  }

};
// specialized for SVector<3>
template<>
struct VecOp<SVector<double,3>,3> {

   typedef SVector<double,3> V;
   template<class It>
   static V  Create(It &x, It &y, It &z , It & ) { return V(*x++,*y++,*z++);}

   static double Add(const V & v) { return v(0) + v(1) + v(2); }
};
// specialized for SVector<4>
template<>
struct VecOp<SVector<double,4>,4> {

   typedef SVector<double,4> V;
   template<class It>
   static V  Create(It &x, It &y, It &z , It &t ) { return V(*x++,*y++,*z++,*t++);}

   static double Add(const V & v) { return v(0) + v(1) + v(2) + v(3); }
};



// internal structure to measure the time

TStopwatch gTimer;
double gTotTime;

struct Timer {
   Timer()  {
      gTimer.Start();
   }
   ~Timer() {
      gTimer.Stop();
      gTotTime += Time();
      if (debugTime) printTime();
   }

   void printTime( std::string s = "") {
      int pr = std::cout.precision(8);
      std::cout << s << "\t" << " time = " << Time() << "\t(sec)\t"
         //    << time.CpuTime()
                << std::endl;
      std::cout.precision(pr);
   }

   double Time()  { return gTimer.RealTime(); }  // use real time

   TStopwatch gTimer;
   double gTotTime;
};



template<int Dim>
class VectorTest {

private:

// global data variables
   std::vector<double> dataX;
   std::vector<double> dataY;
   std::vector<double> dataZ;
   std::vector<double> dataE;

   int nGen;
   int n2Loop ;

   double fSum; // total sum of x,y,z,t (for testing first addition)

public:

   VectorTest(int n1, int n2=0) :
      nGen(n1),
      n2Loop(n2)
   {
      gTotTime  = 0;
   }




   double TotalTime() const { return gTotTime; }  // use real time

   double Sum() const { return fSum; }

   int check(std::string name, double s1, double s2, double scale=1) {
      int iret = 0;
      PrintTest(name);
      iret |= compare(name,s1,s2,scale);
      PrintStatus(iret);
      return iret;
   }

   void print(std::string name) {
      PrintTest(name);
      std::cout <<"\t\t..............\n";
   }


   void genData() {

      // generate for all 4 d data
      TRandom3 r(111); // use a fixed seed to be able to reproduce tests
      fSum = 0;
      for (int i = 0; i < nGen ; ++ i) {

         // generate a 4D vector and stores only the interested dimensions
         double phi = r.Rndm()*3.1415926535897931;
         double eta = r.Uniform(-5.,5.);
         double pt   = r.Exp(10.);
         double m = r.Uniform(0,10.);
         if ( i%50 == 0 )
            m = r.BreitWigner(1.,0.01);
         double E = sqrt( m*m + pt*pt*std::cosh(eta)*std::cosh(eta) );

         // fill vectors
         PtEtaPhiEVector q( pt, eta, phi, E);
         dataX.push_back( q.x() );
         dataY.push_back( q.y() );
         fSum += q.x() + q.y();
         if (Dim >= 3) {
            dataZ.push_back( q.z() );
            fSum += q.z();
         }
         if (Dim >=4 ) {
            dataE.push_back( q.t() );
            fSum += q.t();
         }
      }
      assert( int(dataX.size()) == nGen);
      assert( int(dataY.size()) == nGen);
      if (Dim >= 3) assert( int(dataZ.size()) == nGen);
      if (Dim >=4 ) assert( int(dataE.size()) == nGen);
// // //       dataZ.resize(nGen);
// // //       dataE.resize(nGen);

   }

   // gen data for a Ndim matrix or vector
   void genDataN() {

      // generate for all 4 d data
      TRandom3 r(111); // use a fixed seed to be able to reproduce tests
      fSum = 0;
      dataX.reserve(nGen*Dim);
      for (int i = 0; i < nGen*Dim ; ++ i) {

         // generate random data between [0,1]
         double x = r.Rndm();
         fSum += x;
         dataX.push_back( x );
      }
   }



   typedef std::vector<double>::const_iterator DataIt;

   // test methods
   template <class V>
   void testCreate( std::vector<V > & dataV) {
      Timer tim;
      DataIt x = dataX.begin();
      DataIt y = dataY.begin();
      DataIt z = dataZ.begin();
      DataIt t = dataE.begin();
      while (x != dataX.end() ) {
         dataV.push_back(VecOp<V,Dim>::Create(x,y,z,t) );
         assert(int(dataV.size()) <= nGen);
      }

   }

   template <class V>
   void testCreateAndSet( std::vector<V > & dataV) {
      Timer tim;
      DataIt x = dataX.begin();
      DataIt y = dataY.begin();
      DataIt z = dataZ.begin();
      DataIt t = dataE.begin();
      while (x != dataX.end() ) {
         V  v;
         VecOp<V,Dim>::Set( v, x,y,z,t);
         dataV.push_back(v);
         assert(int(dataV.size()) <= nGen);
      }
   }



   template <class V>
   double testAddition( const std::vector<V > & dataV) {
      V v0;
      Timer t;
      for (int i = 0; i < nGen; ++i) {
         v0 += dataV[i];
      }
      return VecOp<V,Dim>::Add(v0);
   }


   template <class V>
   double testOperations( const std::vector<V > & dataV) {

      double tot = 0;
      Timer t;
      for (int i = 0; i < nGen-1; ++i) {
         const V  & v1 = dataV[i];
         const V  & v2 = dataV[i+1];
         double a = v1.R();
         double b = v2.mag2(); // mag2 is defined for all dimensions;
         double c = 1./v1.Dot(v2);
         V v3 = c * ( v1/a + v2/b );
         tot += VecOp<V,Dim>::Add(v3);
      }
      return tot;
   }

   // mantain loop in gen otherwise is proportional to N**@
   template <class V>
   double testDelta( const std::vector<V > & dataV) {
      double tot = 0;
      Timer t;
      for (int i = 0; i < nGen-1; ++i) {
         const V  & v1 = dataV[i];
         const V  & v2 = dataV[i+1];
         tot += VecOp<V,Dim>::Delta(v1,v2);
      }
      return tot;
   }


//    template <class V>
//    double testDotProduct( const std::vector<V *> & dataV) {
//       //unsigned int n = std::min(n2Loop, dataV.size() );
//       double tot = 0;
//       V v0 = *(dataV[0]);
//       Timer t;
//       for (unsigned int i = 0; i < nGen-1; ++i) {
//          V  & v1 = *(dataV[i]);
//          tot += v0.Dot(v1);
//       }
//       return tot;
//    }




   template <class V1, class V2>
   void testConversion( std::vector<V1 > & dataV1, std::vector<V2 > & dataV2) {

      Timer t;
      for (int i = 0; i < nGen; ++i) {
         dataV2.push_back( V2( dataV1[i] ) );
      }
   }



   // rotation
   template <class V, class R>
   double testRotation( std::vector<V > & dataV ) {

      double sum = 0;
      double rotAngle = 1;
      Timer t;
      for (unsigned int i = 0; i < nGen; ++i) {
         V  & v1 = dataV[i];
         V v2 = v1;
         v2.Rotate(rotAngle);
         sum += VecOp<V,Dim>::Add(v2);
      }
      return sum;
   }

   template<class V>
   double testWrite(const std::vector<V> & dataV, std::string typeName="", bool compress = false) {


      std::string fname = VecType<V>::name() + ".root";
      // replace < character with _
      TFile file(fname.c_str(),"RECREATE","",compress);

      // create tree
      std::string tree_name="Tree with" + VecType<V>::name();

      TTree tree("VectorTree",tree_name.c_str());

      V *v1 = new V();

      //std::cout << "typeID written : " << typeid(*v1).name() << std::endl;

      // need to add namespace to full type name
      if (typeName == "") {
         typeName = "ROOT::Math::" + VecType<V>::name();
      }
      //std::cout << typeName << std::endl;

      TBranch * br = tree.Branch("Vector_branch",typeName.c_str(),&v1);
      if (br == 0) {
         std::cout << "Error creating branch for" << typeName << "\n\t typeid is "
                   << typeid(*v1).name() << std::endl;
         return -1;
      }


      Timer timer;
      for (int i = 0; i < nGen; ++i) {
         *v1 = dataV[i];
         tree.Fill();
      }

#ifdef DEBUG
      tree.Print(); // debug
#endif

      file.Write();
      file.Close();
      return file.GetSize();
   }

   template<class V>
   int testRead(std::vector<V> & dataV) {

      dataV.clear();
      dataV.reserve(nGen);


      std::string fname = VecType<V>::name() + ".root";

      TFile f1(fname.c_str());
      if (f1.IsZombie() ) {
         std::cout << " Error opening file " << fname << std::endl;
         return -1;
      }


      // create tree
      TTree *tree = dynamic_cast<TTree*>(f1.Get("VectorTree"));
      if (tree == 0) {
         std::cout << " Error reading file " << fname << std::endl;
         return -1;
      }

      V *v1 = 0;

      //std::cout << "reading typeID  : " << typeid(*v1).name() << std::endl;

      // cast to void * to avoid a warning
      tree->SetBranchAddress("Vector_branch",(void *) &v1);

      Timer timer;
      int n = (int) tree->GetEntries();
      if (n != nGen) {
         std::cout << "wrong tree entries from file" << fname << std::endl;
         return -1;
      }

      for (int i = 0; i < n; ++i) {
         tree->GetEntry(i);
         dataV.push_back(*v1);
      }

      if (removeFiles) gSystem->Unlink(fname.c_str());


      return 0;
   }



   // test of SVEctor's or SMatrix
   template<class V>
   void testCreateSV( std::vector<V> & dataV) {
      Timer tim;
      //DataIt x = dataX.begin();
      double * x = &dataX.front();
      double * end = x + dataX.size();
      //SVector cannot be created from a generic iterator (should be fixed)
      while (x != end) {
         dataV.push_back( V(x,x+Dim) );
         assert(int(dataV.size()) <= nGen);
         x += Dim;
      }

   }

   template <class V>
   double testAdditionSV( const std::vector<V > & dataV) {
      V v0;
      Timer t;
      for (int i = 0; i < nGen; ++i) {
         v0 += dataV[i];
      }
      double tot = 0;
      typedef typename V::const_iterator It;
      for (It itr = v0.begin(); itr != v0.end(); ++itr)
         tot += *itr;

      return tot;
   }

   template <class V>
   double testAdditionTR( const std::vector<V > & dataV) {
      V v0;
      Timer t;
      for (int i = 0; i < nGen; ++i) {
         v0 += dataV[i];
      }
#ifdef DEBUG
      v0.Print();
#endif
      return v0.Sum();
   }


};


//--------------------------------------------------------------------------------------
// test of all physics vector (GenVector's)
//--------------------------------------------------------------------------------------

template<class V1, class V2, int Dim>
int testVector(int ngen, bool testio=false) {

   int iret = 0;


   VectorTest<Dim> a(ngen);
   a.genData();


   std::vector<V1> v1;
   std::vector<V2> v2;
   v1.reserve(ngen);
   v2.reserve(ngen);

   double s1, s2 = 0;
   double scale = 1;
   double sref1, sref2 = 0;

   a.testCreate(v1);             iret |= a.check(VecType<V1>::name()+" creation",v1.size(),ngen);
   s1 = a.testAddition(v1);    iret |= a.check(VecType<V1>::name()+" addition",s1,a.Sum(),Dim*20);
   sref1 = s1;
   v1.clear();
   assert(v1.size() == 0);
   a.testCreateAndSet(v1);       iret |= a.check(VecType<V1>::name()+" creation",v1.size(),ngen);
   s2 = a.testAddition(v1);    iret |= a.check(VecType<V1>::name()+" setting",s2,s1);

   a.testConversion(v1,v2);      iret |= a.check(VecType<V1>::name()+" -> " + VecType<V2>::name(),v1.size(),v2.size() );
   scale = 1000;
   if (Dim == 2) scale = 1.E12;  // to be understood
   if (Dim == 3) scale = 1.E4;  // problem with RhoEtaPhiVector
   s2 = a.testAddition(v2);    iret |= a.check("Vector conversion",s2,s1,scale);
   sref2 = s2;

   s1 = a.testOperations(v1);  a.print(VecType<V1>::name()+" operations");
   scale = Dim*20;
   if (Dim==3 && VecType<V2>::name() == "RhoEtaPhiVector") scale *= 12; // for problem with RhoEtaPhi
   if (Dim==4 && ( VecType<V2>::name() == "PtEtaPhiMVector" || VecType<V2>::name() == "PxPyPzMVector")) {
#if (defined(__arm__) || defined(__arm64__) || defined(__aarch64__) || defined(__s390x__))
      scale *= 1.E7;
#else
      scale *= 10;
#endif
#if defined(__FAST_MATH__) && defined(__clang__)
      scale *= 1.E6;
#endif
   }
   // for problem with PtEtaPhiE
#if defined (R__LINUX) && !defined(R__B64)
   // problem of precision on linux 32
   if (Dim ==4) scale = 1000000000;
#endif
   if (Dim==4 && VecType<V2>::name() == "PtEtaPhiEVector") scale = 0.01/(std::numeric_limits<double>::epsilon());
   s2 = a.testOperations(v2);  iret |= a.check(VecType<V2>::name()+" operations",s2,s1,scale);

   s1 = a.testDelta(v1);      a.print(VecType<V1>::name()+" delta values");
   scale = Dim*16;
   if (Dim==4) scale *= 100; // for problem with PtEtaPhiE
   s2 = a.testDelta(v2);      iret |= a.check(VecType<V2>::name()+" delta values",s2,s1,scale);


   double fsize = 0;
   int ir = 0;
   if (!testio) return iret;

   double estSize = ngen*8 * Dim  + 10000;  //add extra bytes
   scale = 0.1 / std::numeric_limits<double>::epsilon();
   fsize = a.testWrite(v1);  iret |= a.check(VecType<V1>::name()+" write",fsize,estSize,scale);
   ir = a.testRead(v1);   iret |= a.check(VecType<V1>::name()+" read",ir,0);
   s1 = a.testAddition(v1);       iret |= a.check(VecType<V1>::name()+" after read",s1,sref1);

   // test io vector 2
   fsize = a.testWrite(v2);  iret |= a.check(VecType<V2>::name()+" write",fsize,estSize,scale);
   ir = a.testRead(v2);      iret |= a.check(VecType<V2>::name()+" read",ir,0);
   scale = 100; // gcc 4.3.2 gives and error for RhoEtaPhiVector for 32 bits
   s2 = a.testAddition(v2);       iret |= a.check(VecType<V2>::name()+" after read",s2,sref2,scale);


   // test io of double 32 for vector 1
   if (Dim==2) return iret;

   estSize = ngen*4 * Dim + 10000;  //add extra bytes
   scale = 0.1 / std::numeric_limits<double>::epsilon();

   std::string typeName = VecType<V1>::name32();
   fsize = a.testWrite(v1,typeName);  iret |= a.check(VecType<V1>::name() +"_D32 write",fsize,estSize,scale);
   ir = a.testRead(v1);   iret |= a.check(VecType<V1>::name()+"_D32 read",ir,0);
   s1 = a.testAddition(v1);       iret |= a.check(VecType<V1>::name()+"_D32 after read",s1,sref1,1.E9);


   return iret;

}

//--------------------------------------------------------------------------------------
// test of Svector of dim 3 or 4
//--------------------------------------------------------------------------------------

template<int Dim>
int testVector34(int ngen, bool testio=false) {

   int iret = 0;



   VectorTest<Dim> a(ngen);
   a.genData();

   typedef SVector<double, Dim> SV;
   std::vector<SV> v1;
   v1.reserve(ngen);

   double s1 = 0;
   //double scale = 1;
   double sref1  = 0;

   std::string name = "SVector<double," + Util::ToString(Dim) + ">";
   a.testCreate(v1);             iret |= a.check(name+" creation",v1.size(),ngen);
   s1 = a.testAddition(v1);    iret |= a.check(name+" addition",s1,a.Sum(),Dim*20);
   sref1 = s1;

   // test the io
   double fsize = 0;
   int ir = 0;
   if (!testio) return iret;

   std::string typeName = "ROOT::Math::"+name;

   double estSize = ngen*8 * Dim + 47000; // add extra bytes (why so much ? ) this is for ngen=10000
   double scale = 0.1 / std::numeric_limits<double>::epsilon();
   fsize = a.testWrite(v1,typeName);  iret |= a.check(name+" write",fsize,estSize,scale);
   ir = a.testRead(v1);   iret |= a.check(name+" read",ir,0);
   s1 = a.testAddition(v1);       iret |= a.check(name+" after read",s1,sref1,10);

   //std::cout << "File size = " << fsize << " estimated " << 8 * Dim * ngen << std::endl;

   // test Double32
   estSize = ngen*4 * Dim + 47000;  //add extra bytes
   scale = 0.1 / std::numeric_limits<double>::epsilon();

   typeName = VecType<SV>::name32();
   fsize = a.testWrite(v1,typeName);  iret |= a.check(VecType<SV>::name() +"_D32 write",fsize,estSize,scale);
   ir = a.testRead(v1);   iret |= a.check(VecType<SV>::name()+"_D32 read",ir,0);
   s1 = a.testAddition(v1);       iret |= a.check(VecType<SV>::name()+"_D32 after read",s1,sref1,1.E9);


   return iret;
}

//--------------------------------------------------------------------------------------
// test of generic Svector
//--------------------------------------------------------------------------------------

template<int Dim>
int testSVector(int ngen, bool testio=false) {

   // test the matrix if D2 is not equal to 1
   int iret = 0;


   VectorTest<Dim> a(ngen);
   a.genDataN();

   typedef SVector<double, Dim> SV;
   std::vector<SV> v1;
   v1.reserve(ngen);

   double s1 = 0;
   //double scale = 1;
   double sref1  = 0;

   std::string name = "SVector<double," + Util::ToString(Dim) + ">";

   a.testCreateSV(v1);             iret |= a.check(name+" creation",v1.size(),ngen);
   s1 = a.testAdditionSV(v1);    iret |= a.check(name+" addition",s1,a.Sum(),Dim*4);
   sref1 = s1;

   // test the io
   double fsize = 0;
   int ir = 0;
   if (!testio) return iret;


   std::string typeName = "ROOT::Math::"+name;
   double estSize = ngen*8 * Dim + 10000;
   double scale = 0.1 / std::numeric_limits<double>::epsilon();
   fsize = a.testWrite(v1,typeName);  iret |= a.check(name+" write",fsize,estSize,scale);
   ir = a.testRead(v1);   iret |= a.check(name+" read",ir,0);
   s1 = a.testAdditionSV(v1);
#if(defined __FAST_MATH__  && defined __clang__)
   iret |= a.check(name+" after read",s1,sref1, 10);
#else
   iret |= a.check(name+" after read",s1,sref1);
#endif


   return iret;
}

template<int D1, int D2>
struct RepStd {
   typedef typename ROOT::Math::MatRepStd<double,D1,D2> R;
   static std::string name() {
      return "ROOT::Math::MatRepStd<double," +  Util::ToString(D1) + "," + Util::ToString(D2) + "> ";
   }
   static std::string name32() {
      return "ROOT::Math::MatRepStd<Double32_t," +  Util::ToString(D1) + "," + Util::ToString(D2) + "> ";
   }
   static std::string sname() { return ""; }
};
template<int D1>
struct RepSym {
   typedef typename ROOT::Math::MatRepSym<double,D1> R;
   static std::string name() {
      return "ROOT::Math::MatRepSym<double," +  Util::ToString(D1) + "> ";
   }
   static std::string name32() {
      return "ROOT::Math::MatRepSym<Double32_t," +  Util::ToString(D1) + "> ";
   }
   static std::string sname() { return "MatRepSym"; }
};

//--------------------------------------------------------------------------------------
// test of generic SMatrix
//--------------------------------------------------------------------------------------

template<int D1, int D2,class Rep >
int testSMatrix(int ngen, bool testio=false) {

   // test the matrix if D2 is not equal to 1
   int iret = 0;


   typedef typename Rep::R R;
   typedef SMatrix<double, D1, D2,R> SM;
   // in the case of sym matrices SM::kSize is different than R::kSize
   // need to use the R::kSize for Dim
   const int Dim = R::kSize;

   VectorTest<Dim> a(ngen);
   a.genDataN();

   std::vector<SM> v1;
   v1.reserve(ngen);

   double s1 = 0;
   //double scale = 1;
   double sref1  = 0;

   std::string name0 = "SMatrix<double," + Util::ToString(D1) + "," + Util::ToString(D2);
   std::string name = name0 + "," + Rep::sname() +  ">";

   a.testCreateSV(v1);             iret |= a.check(name+" creation",v1.size(),ngen);
   s1 = a.testAdditionSV(v1);    iret |= a.check(name+" addition",s1,a.Sum(),Dim*4);
   sref1 = s1;

   // test the io
   if (!testio) return iret;
   double fsize = 0;
   int ir = 0;

   // the full name is needed for sym matrices
   std::string typeName = "ROOT::Math::"+name0 + "," + Rep::name()  + ">";

   double estSize = ngen*8 * Dim + 10000;
   double scale = 0.1 / std::numeric_limits<double>::epsilon();
   fsize = a.testWrite(v1,typeName);  iret |= a.check(name+" write",fsize,estSize,scale);
   ir = a.testRead(v1);   iret |= a.check(name+" read",ir,0);
   s1 = a.testAdditionSV(v1);       iret |= a.check(name+" after read",s1,sref1);


   //std::cout << "File size = " << fsize << " estimated " << 8 * Dim * ngen << std::endl;

   // test storing as Double32_t
   // dictionay exist only for square matrices between 3 and 6
   if (D1 != D2) return iret;
   if (D1 < 3 || D1 > 6) return iret;

   double fsize32 = 0;

   name0 = "SMatrix<Double32_t," + Util::ToString(D1) + "," + Util::ToString(D2);
   name = name0 + "," + Rep::sname() +  ">";
   typeName = "ROOT::Math::"+name0+ "," + Rep::name32()  + ">";


   estSize = ngen* 4 * Dim + 60158;
   scale = 0.1 / std::numeric_limits<double>::epsilon();
   fsize32 = a.testWrite(v1,typeName);     iret |= a.check(name+" write",fsize32,estSize,scale);
   ir = a.testRead(v1);   iret |= a.check(name+" read",ir,0);
   //we read back float (scale errors then)
   s1 = a.testAdditionSV(v1);       iret |= a.check(name+" after read",s1,sref1,1.E9);


   return iret;
}



//--------------------------------------------------------------------------------------
// test of a track an object containing vector and matrices
////////////////////////////////////////////////////////////////////////////////

template<class T>
int testTrack(int ngen) {
   int iret = 0;


   const int Dim = T::kSize;
   std::string name = T::Type();

   //std::cout << "Dim is " << Dim << std::endl;

   VectorTest<Dim> a(ngen);
   a.genDataN();

   std::vector<T> v1;
   v1.reserve(ngen);

   double s1 = 0;
   //double scale = 1;
   double sref1  = 0;


   a.testCreateSV(v1);             iret |= a.check(name+" creation",v1.size(),ngen);
   s1 = a.testAdditionTR(v1);    iret |= a.check(name+" addition",s1,a.Sum(),Dim*4);
   sref1 = s1;
   //std::cout << " reference sum is " << sref1 << std::endl;

   double fsize = 0;
   int ir = 0;

   // the full name is needed for sym matrices
   std::string typeName = name;

   int wsize = 8;
   if (T::IsD32() ) wsize = 4;
   double estSize = ngen*wsize * Dim + 10000;
   double scale = 0.2 / std::numeric_limits<double>::epsilon();
   fsize = a.testWrite(v1,typeName);  iret |= a.check(name+" write",fsize,estSize,scale);
   ir = a.testRead(v1);   iret |= a.check(name+" read",ir,0);
   scale = 1;
   if (T::IsD32() ) scale = 1.E9; // use float numbers
   s1 = a.testAdditionTR(v1);       iret |= a.check(name+" after read",s1,sref1,scale);
   if (T::IsD32() ) {
     // check at double precision type must fail otherwise Double's are stored
     bool pdebug = debug; debug = false;
     int ret = compare(" ",s1,sref1);
     iret |= a.check(name+" Double32 test",ret,1);
     debug = pdebug;
   }

   return iret;
}




int testGenVectors(int ngen,bool io) {

   int iret = 0;
   std::cout <<"******************************************************************************\n";
   std::cout << "\tTest of Physics Vector (GenVector package)\n";
   std::cout <<"******************************************************************************\n";
   iret |= testVector<XYVector, Polar2DVector, 2>(ngen,io);
   iret |= testVector<XYZVector, Polar3DVector, 3>(ngen,io);
   iret |= testVector<XYZVector, RhoEtaPhiVector, 3>(ngen,io);
   iret |= testVector<XYZVector, RhoZPhiVector, 3>(ngen,io);
   iret |= testVector<XYZTVector, PtEtaPhiEVector, 4>(ngen,io);
   iret |= testVector<XYZTVector, PtEtaPhiMVector, 4>(ngen,io);
   iret |= testVector<XYZTVector, PxPyPzMVector, 4>(ngen,io);

   return iret;
}

int testSMatrix(int ngen,bool io) {

   int iret = 0;

   std::cout <<"******************************************************************************\n";
   std::cout << "\tTest of SMatrix package\n";
   std::cout <<"******************************************************************************\n";


   iret |= testVector34<3>(ngen,io);
   iret |= testVector34<4>(ngen,io);

   // test now matrices and vectors
   iret |= testSVector<6>(ngen,io);

   // asymetric matrices we have only 4x3 and 3x4
   iret |= testSMatrix<3,4,RepStd<3,4> >(ngen,io);

   iret |= testSMatrix<4,3,RepStd<4,3> >(ngen,io);
   iret |= testSMatrix<3,3,RepStd<3,3> >(ngen,io);
   // sym matrix
   iret |= testSMatrix<5,5,RepSym<5> >(ngen,io);

   return iret;
}


int testCompositeObj(int ngen) {

   std::cout <<"******************************************************************************\n";
   std::cout << "\tTest of a Composite Object (containing Vector's and Matrices)\n";
   std::cout <<"******************************************************************************\n";

   std::cout << "Test Using CINT library\n\n";

   // put path relative to LD_LIBRARY_PATH

   std::unique_ptr<const char> dynPath(gSystem->DynamicPathName("../test/libTrackMathCoreDict",
                                                  /*quiet*/ true));

   int iret = -1;

   if (dynPath)
      iret = gSystem->Load("../test/libTrackMathCoreDict");
   if (iret < 0) {
      // if not assume running from top ROOT dir (case of roottest)
      iret = gSystem->Load("test/libTrackMathCoreDict");
      if (iret < 0) {
         std::cerr <<"Error Loading libTrackMathCoreDict" << std::endl;
         return iret;
      }
   }

   iret = 0;

   iret |= testTrack<TrackD>(ngen);
   iret |= testTrack<TrackD32>(ngen);
   iret |= testTrack<TrackErrD>(ngen);
   iret |= testTrack<TrackErrD32>(ngen);
   iret |= testTrack<VecTrack<TrackD> >(ngen);
   iret |= testTrack<VecTrack<TrackErrD> >(ngen);



   return iret;
}


#endif // endif ifndef __CINT__


int stressMathCore(double nscale = 1) {

   int iret = 0;

#ifdef __CINT__
   std::cout << "Test must be run in compile mode - use ACLIC to compile!!" << std::endl;


   gSystem->Load("libMathCore");
   gSystem->Load("libTree");
   gROOT->ProcessLine(".L stressMathCore.cxx++");
   return stressMathCore();
#endif
//    iret |= gSystem->Load("libMathCore");
//    iret |= gSystem->Load("libMathMore");
//    if (iret !=0) return iret;


   TBenchmark bm;
   bm.Start("stressMathCore");

   const int ntest = 10000;
   int n = int(nscale*ntest);
   //std::cout << "StressMathCore: test number  n = " << n << std::endl;

   iret |= testStatFunctions(n/10);

   bool io = true;

   iret |= ( gSystem->Load("libSmatrix") < 0 );   // iret = 0 or = 1 is fine
   if (iret != 0) {
     std::cerr <<"Error Loading libSmatrix" << std::endl;
     io = false;
   }

   iret |= testGenVectors(n,io);

   iret |= testSMatrix(n,io);

   if (io) iret |= testCompositeObj(n);

   bm.Stop("stressMathCore");
   std::cout <<"******************************************************************************\n";
   bm.Print("stressMathCore");
   const double reftime = 7.1; // needs to be updated // ref time on  pcbrun4
   double rootmarks = 860 * reftime / bm.GetCpuTime("stressMathCore");
   std::cout << " ROOTMARKS = " << rootmarks << " ROOT version: " << gROOT->GetVersion() << "\t"
             << gROOT->GetGitBranch() << "@" << gROOT->GetGitCommit() << std::endl;
   std::cout <<"*******************************************************************************\n";


   if (iret !=0) std::cerr << "stressMathCore Test Failed !!" << std::endl;
   return iret;
}



int main(int argc,const char *argv[]) {
   std::string inclRootSys = ("-I" + TROOT::GetRootSys() + "/test").Data();
   TROOT::AddExtraInterpreterArgs({inclRootSys});

   double nscale = 1;
   if (argc > 1) {
      nscale = atof(argv[1]);
      //nscale = std::pow(10.0,double(scale));
   }
   return stressMathCore(nscale);
}
