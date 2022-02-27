#ifndef ROOT_STATFUNCTION
#define ROOT_STATFUNCTION

#include "TF1.h"
#include "TestHelper.h"
#include "Math/IParamFunction.h"
#include "gtest/gtest.h"

// trait class  for distinguishing the number of parameters for the various functions
template <class Func, unsigned int NPAR>
struct Evaluator {
   static double F(Func f, double x, const double *) { return f(x); }
};
template <class Func>
struct Evaluator<Func, 1> {
   static double F(Func f, double x, const double *p) { return f(x, p[0]); }
};
template <class Func>
struct Evaluator<Func, 2> {
   static double F(Func f, double x, const double *p) { return f(x, p[0], p[1]); }
};
template <class Func>
struct Evaluator<Func, 3> {
   static double F(Func f, double x, const double *p) { return f(x, p[0], p[1], p[2]); }
};

// statistical function class
// template on the number of parameters
template <class Func, class FuncQ, int NPAR, int NPARQ = NPAR - 1>
class StatFunction : public ROOT::Math::IParamFunction {

public:
   StatFunction(Func pdf, Func cdf, FuncQ quant) : fPdf(pdf), fCdf(cdf), fQuant(quant)
   {
      fScale1 = 1.0E6; // scale for cdf test (integral)
      fScale2 = 10;    // scale for quantile test
      for (int i = 0; i < NPAR; ++i) fParams[i] = 0;
   }

   unsigned int NPar() const override { return NPAR; }
   const double *Parameters() const override { return fParams; }
   ROOT::Math::IGenFunction *Clone() const override { return new StatFunction(fPdf, fCdf, fQuant); }
   void SetParameters(const double *p) override { std::copy(p, p + NPAR, fParams); }
   void SetParameters(double p0) { *fParams = p0; }
   void SetParameters(double p0, double p1)
   {
      *fParams = p0;
      *(fParams + 1) = p1;
   }
   void SetParameters(double p0, double p1, double p2)
   {
      *fParams = p0;
      *(fParams + 1) = p1;
      *(fParams + 2) = p2;
   }

   double Cdf(double x) const { return Evaluator<Func, NPAR>::F(fCdf, x, fParams); }
   double Quantile(double x) const
   {
      double z = Evaluator<FuncQ, NPARQ>::F(fQuant, x, fParams);
      if ((NPAR - NPARQ) == 1) z += fParams[NPAR - 1]; // adjust the offset
      return z;
   }

   // test cumulative function
   // test cdf at value f
   void Test(double xmin, double xmax, double xlow = 1, double xup = 0, bool c = false, int NFuncTest = 1000)
   {
      // scan all values from xmin to xmax
      double dx = (xmax - xmin) / NFuncTest;

      // use TF1 for the integral
      double x1, x2 = 0;
      if (xlow >= xup) {
         x1 = -100;
         x2 = 100;
      } else if (xup < xmax) {
         x1 = xlow;
         x2 = 100;
      } else {
         x1 = xlow;
         x2 = xup;
      }

      TF1 f = TF1("ftemp", ROOT::Math::ParamFunctor(*this), x1, x2, 0);

      for (int i = 0; i < NFuncTest; ++i) {
         double v1 = xmin + dx * i; // value used for testing
         double q1 = Cdf(v1);

         double q2 = 0;
         if (!c) {
            q2 = f.Integral(x1, v1, 1.E-12);
            // use a larger scale (integral error is 10-9)
            ASSERT_TRUE(IsNear("test _cdf", q1, q2, fScale1));
            // test the quantile
            double v2 = Quantile(q1);
            ASSERT_TRUE(IsNear("test _quantile", v1, v2, fScale2));
         } else {
            // upper integral (cdf_c)
            q2 = f.Integral(v1, x2, 1.E-12);
            ASSERT_TRUE(IsNear("test _cdf_c", q1, q2, fScale1));
            double v2 = Quantile(q1);
            ASSERT_TRUE(IsNear("test _quantile_c", v1, v2, fScale2));
         }
      }
   }

   void ScaleTol1(double s) { fScale1 *= s; }
   void ScaleTol2(double s) { fScale2 *= s; }

private:
   double DoEvalPar(double x, const double *) const override
   {
      // implement explicitly using cached parameter values
      return Evaluator<Func, NPAR>::F(fPdf, x, fParams);
   }

   Func fPdf;
   Func fCdf;
   FuncQ fQuant;
   double fParams[NPAR];
   double fScale1;
   double fScale2;
};

// typedef defining the functions
typedef double (*F0)(double);
typedef double (*F1)(double, double);
typedef double (*F2)(double, double, double);
typedef double (*F3)(double, double, double, double);

typedef StatFunction<F2, F2, 2, 2> Dist_beta;
typedef StatFunction<F2, F1, 2> Dist_breitwigner;
typedef StatFunction<F2, F1, 2> Dist_chisquared;
typedef StatFunction<F3, F2, 3> Dist_fdistribution;
typedef StatFunction<F3, F2, 3> Dist_gamma;
typedef StatFunction<F2, F1, 2> Dist_gaussian;
typedef StatFunction<F3, F2, 3> Dist_lognormal;
typedef StatFunction<F2, F1, 2> Dist_tdistribution;
typedef StatFunction<F2, F1, 2> Dist_exponential;
typedef StatFunction<F2, F1, 2> Dist_landau;
typedef StatFunction<F3, F2, 3> Dist_uniform;

#define CREATE_DIST(name) Dist_##name dist(name##_pdf, name##_cdf, name##_quantile);
#define CREATE_DIST_C(name) Dist_##name distc(name##_pdf, name##_cdf_c, name##_quantile_c);

#endif // ROOT_STATFUNCTION
