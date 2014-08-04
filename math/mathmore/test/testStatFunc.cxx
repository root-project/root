/// test of the statistical functions cdf and quantiles

//#define NO_MATHCORE

#include "Math/DistFuncMathMore.h"
#ifndef NO_MATHCORE
#include "Math/GSLIntegrator.h"
#include "Math/WrappedFunction.h"
#include "Math/DistFuncMathCore.h"
#endif


#include <iostream>
#include <limits>

using namespace ROOT::Math;

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

   if (iret == 0)
      std::cout <<".";
   else {
      int pr = std::cout.precision (18);
      std::cout << "\nDiscrepancy in " << name.c_str() << "() :\n  " << v1 << " != " << v2 << " discr = " << int(delta/d/eps)
                << "   (Allowed discrepancy is " << eps  << ")\n\n";
      std::cout.precision (pr);
      //nfail = nfail + 1;
   }
   return iret;
}



// for functions with one parameters

struct TestFunc1 {

   typedef double ( * Pdf) ( double, double, double);
   typedef double ( * Cdf) ( double, double, double);
   typedef double ( * Quant) ( double, double);

   // wrappers to pdf to be integrated
   struct PdfFunction {
      PdfFunction( Pdf pdf, double p1 ) : fPdf(pdf), fP1(p1) {}
      double operator() ( double x) const { return fPdf(x,fP1,0.0); }

      Pdf fPdf;
      double fP1;
   };


   TestFunc1( double p1, double s = 2) :
      scale(s)
   {
      p[0] = p1;
   }

#ifndef NO_MATHCORE // skip cdf test when Mathcore is missing

   int testCdf( Pdf f_pdf, Cdf f_cdf, bool c = false) {
      int iret = 0;
      double val = 1.0; // value used  for testing
      double q1 = f_cdf(val, p[0],0.0);
      // calculate integral of pdf
      PdfFunction f(f_pdf,p[0]);
      WrappedFunction<PdfFunction> wf(f);
      GSLIntegrator ig(1.E-12,1.E-12,100000);
      ig.SetFunction(wf);
      if (!c) {
         // lower intergal (cdf)
         double q2 = ig.IntegralLow(val);
         // use a larger scale (integral error is 10-9)
         iret |= compare("test _cdf", q1, q2, 1.0E6);
      }
      else {
         // upper integral (cdf_c)
         double q2 = ig.IntegralUp(val);
         iret |= compare("test _cdf_c", q1, q2, 1.0E6);
      }
      return iret;
   }

#endif

   int testQuantile (Cdf f_cdf, Quant f_quantile, bool c= false) {
      int iret = 0;
      double z1,z2,q;
      z1 = 1.0E-6;
      for (int i = 0; i < 10; ++i) {
         q = f_quantile(z1,p[0]);
         z2 = f_cdf(q, p[0],0.);
         if (!c)
            iret |= compare("test quantile", z1, z2, scale);
         else
            iret |= compare("test quantile_c", z1, z2, scale);
         z1 += 0.1;
      }
      return iret;
   }

   double p[1]; // parameters
   double scale;
};

// for functions with two parameters


struct TestFunc2 {

   typedef double ( * Pdf) ( double, double, double, double);
   typedef double ( * Cdf) ( double, double, double, double);
   typedef double ( * Quant) ( double, double, double);


   struct PdfFunction {
      PdfFunction( Pdf pdf, double p1, double p2 ) : fPdf(pdf), fP1(p1), fP2(p2) {}
      double operator() ( double x) const { return fPdf(x,fP1,fP2,0.0); }

      Pdf fPdf;
      double fP1;
      double fP2;
   };

   TestFunc2( double p1, double p2, double s = 2) :
      scale(s)
   {
      p[0] = p1,
      p[1] = p2;
   }

#ifndef NO_MATHCORE // skip cdf test when Mathcore is missing

    int testCdf( Pdf f_pdf, Cdf f_cdf, bool c = false) {
      int iret = 0;
      double val = 1.0; // value used  for testing
      double q1 = f_cdf(val, p[0],p[1],0.0);
      // calculate integral of pdf
      PdfFunction f(f_pdf,p[0],p[1]);
      WrappedFunction<PdfFunction> wf(f);
      GSLIntegrator ig(1.E-12,1.E-12,100000);
      ig.SetFunction(wf);
      if (!c) {
         // lower intergal (cdf)
         double q2 = ig.IntegralLow(val);
         // use a larger scale (integral error is 10-9)
         iret |= compare("test _cdf", q1, q2, 1.0E6);
      }
      else {
         // upper integral (cdf_c)
         double q2 = ig.IntegralUp(val);
         iret |= compare("test _cdf_c", q1, q2, 1.0E6);
      }
      return iret;
   }

#endif

    int testQuantile (Cdf f_cdf, Quant f_quantile, bool c=false) {
      int iret = 0;
      double z1,z2,q;
      z1 = 1.0E-6;
      for (int i = 0; i < 10; ++i) {
         q = f_quantile(z1,p[0],p[1]);
         z2 = f_cdf(q, p[0],p[1],0.0);
         if (!c)
            iret |= compare("test quantile", z1, z2, scale);
         else
            iret |= compare("test quantile_c", z1, z2, scale);
         z1 += 0.1;
      }
      return iret;
   }

   double p[2]; // parameters
   double scale;
};

void printStatus(int iret) {
   if (iret == 0)
      std::cout <<"\t\t\t\t OK" << std::endl;
   else
      std::cout <<"\t\t\t\t FAILED " << std::endl;
}

#ifndef NO_MATHCORE

#define TESTDIST1(name, p1, s) {\
      int ir = 0; \
      TestFunc1 t(p1,s);\
      ir |= t.testCdf( name ## _pdf , name ## _cdf );\
      ir |= t.testCdf( name ##_pdf , name ##_cdf_c,true);\
      ir |= t.testQuantile( name ## _cdf, name ##_quantile);\
      ir |= t.testQuantile( name ##_cdf_c, name ##_quantile_c,true);\
      printStatus(ir);\
      iret |= ir; }

#define TESTDIST2(name, p1, p2, s) {\
      int ir = 0; \
      TestFunc2 t(p1,p2,s);\
      ir |= t.testCdf( name ## _pdf , name ## _cdf );\
      ir |= t.testCdf( name ##_pdf , name ##_cdf_c,true);\
      ir |= t.testQuantile( name ## _cdf, name ##_quantile);\
      ir |= t.testQuantile( name ##_cdf_c, name ##_quantile_c,true);\
      printStatus(ir);\
      iret |= ir; }




#else
// without mathcore pdf are missing so skip cdf test
#define TESTDIST1(name, p1, s) {\
      int ir = 0; \
      TestFunc1 t(p1,s);\
      ir |= t.testQuantile( name ## _cdf, name ##_quantile);\
      ir |= t.testQuantile( name ##_cdf_c, name ##_quantile_c,true);\
      printStatus(ir);\
      iret |= ir; }

#define TESTDIST2(name, p1, p2, s) {\
      int ir = 0; \
      TestFunc2 t(p1,p2,s);\
      ir |= t.testQuantile( name ## _cdf, name ##_quantile);\
      ir |= t.testQuantile( name ##_cdf_c, name ##_quantile_c,true);\
      printStatus(ir);\
      iret |= ir; }

#endif

// wrapper for the beta
double mbeta_pdf(double x, double a, double b, double){
   return beta_pdf(x,a,b);
}
double mbeta_cdf(double x, double a, double b, double){
   return beta_cdf(x,a,b);
}
double mbeta_cdf_c(double x, double a, double b, double){
   return beta_cdf_c(x,a,b);
}
double mbeta_quantile(double x, double a, double b){
   return beta_quantile(x,a,b);
}
double mbeta_quantile_c(double x, double a, double b){
   return beta_quantile_c(x,a,b);
}


#ifndef NO_MATHCORE

int testPoissonCdf(double mu,double tol) {
   int iret = 0;
   for (int i = 0; i < 12; ++i) {
      double q1 = poisson_cdf(i,mu);
      double q1c = 1.- poisson_cdf_c(i,mu);
      double q2 = 0;
      for (int j = 0; j <= i; ++j) {
         q2 += poisson_pdf(j,mu);
      }
      iret |= compare("test cdf",q1,q2,tol);
      iret |= compare("test cdf_c",q1c,q2,tol);
   }
   printStatus(iret);
   return iret;
}

int testBinomialCdf(double p, int n, double tol) {
   int iret = 0;
   for (int i = 0; i < 12; ++i) {
      double q1 = binomial_cdf(i,p,n);
      double q1c = 1.- binomial_cdf_c(i,p,n);
      double q2 = 0;
      for (int j = 0; j <= i; ++j) {
         q2 += binomial_pdf(j,p,n);
      }
      iret |= compare("test cdf",q1,q2,tol);
      iret |= compare("test cdf_c",q1c,q2,tol);
   }
   printStatus(iret);
   return iret;
}

#endif

int testStatFunc() {

   int iret = 0;
   double tol = 2;


#ifndef NO_MATHCORE

   tol = 8;
   std::cout << "Poisson distrib. \t: ";
   double mu = 5;
   iret |= testPoissonCdf(mu,tol);

   tol = 32;
   std::cout << "Binomial distrib. \t: ";
   double p = 0.5; int nt = 9;
   iret |= testBinomialCdf(p,nt,tol);

   tol = 2;
   std::cout << "BreitWigner distrib\t: ";
   TESTDIST1(breitwigner,1.0,tol);

   std::cout << "Cauchy distribution\t: ";
   TESTDIST1(cauchy,2.0,tol);

   std::cout << "Exponential distrib\t: ";
   TESTDIST1(exponential,1.0,tol);

   std::cout << "Gaussian distribution\t: ";
   TESTDIST1(gaussian,1.0,tol);

   std::cout << "Log-normal distribution\t: ";
   TESTDIST2(lognormal,1.0,1.0,tol);

   std::cout << "Normal distribution\t: ";
   TESTDIST1(normal,1.0,tol);

   std::cout << "Uniform distribution\t: ";
   TESTDIST2(uniform,0.0,10.0,tol);



#endif

   std::cout << "Chisquare distribution\t: ";
   tol = 8;
   TESTDIST1(chisquared,9.,tol);

   tol = 2;

   std::cout << "F distribution\t\t: ";
   double n = 5; double m = 6;
   TESTDIST2(fdistribution,n,m,tol);

   std::cout << "Gamma distribution\t: ";
   double a = 1; double b = 2;
   TESTDIST2(gamma,a,b,tol);

   std::cout << "t distribution\t\t: ";
   double nu = 10;
   TESTDIST1(tdistribution,nu,tol);

   std::cout << "Beta distribution\t: ";
   a = 2; b = 1;
   TESTDIST2(mbeta,a,b,tol);



   return iret;
}
int main() {

   return testStatFunc();
}
