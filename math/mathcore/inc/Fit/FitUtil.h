// @(#)root/mathcore:$Id$
// Author: L. Moneta Tue Nov 28 10:52:47 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class FitUtil

#ifndef ROOT_Fit_FitUtil
#define ROOT_Fit_FitUtil

#include "Math/IParamFunctionfwd.h"
#include "Math/IParamFunction.h"

#include "TROOT.h"

#ifdef R__USE_IMT
#include "ROOT/TThreadExecutor.hxx"
#endif
#include "ROOT/TSequentialExecutor.hxx"

#include "Fit/FitExecutionPolicy.h"

#include "Math/Integrator.h"
#include "Math/IntegratorMultiDim.h"

#include "TError.h"

// using parameter cache is not thread safe but needed for normalizing the functions
#define USE_PARAMCACHE

#ifdef R__HAS_VECCORE
namespace vecCore {
template <class T>
vecCore::Mask<T> Int2Mask(unsigned i)
{
   T x;
   for (unsigned j = 0; j < vecCore::VectorSize<T>(); j++)
      vecCore::Set<T>(x, j, j);
   return vecCore::Mask<T>(x < T(i));
}
}
#endif

namespace ROOT {

   namespace Fit {

/**
   namespace defining utility free functions using in Fit for evaluating the various fit method
   functions (chi2, likelihood, etc..)  given the data and the model function

   @ingroup FitMain
*/
namespace FitUtil {

  typedef  ROOT::Math::IParamMultiFunction IModelFunction;
  typedef  ROOT::Math::IParamMultiGradFunction IGradModelFunction;

  template <class T>
  using IGradModelFunctionTempl = ROOT::Math::IParamMultiGradFunctionTempl<T>;

  template <class T>
  using IModelFunctionTempl = ROOT::Math::IParamMultiFunctionTempl<T>;

  // internal class to evaluate the function or the integral
  // and cached internal integration details
  // if useIntegral is false no allocation is done
  // and this is a dummy class
  // class is templated on any parametric functor implementing operator()(x,p) and NDim()
  // contains a constant pointer to the function

  template <class ParamFunc = ROOT::Math::IParamMultiFunctionTempl<double>>
  class IntegralEvaluator {

  public:
     IntegralEvaluator(const ParamFunc &func, const double *p, bool useIntegral = true)
        : fDim(0), fParams(0), fFunc(0), fIg1Dim(0), fIgNDim(0), fFunc1Dim(0), fFuncNDim(0)
     {
        if (useIntegral) {
           SetFunction(func, p);
        }
     }

     void SetFunction(const ParamFunc &func, const double *p = 0)
     {
        // set the integrand function and create required wrapper
        // to perform integral in (x) of a generic  f(x,p)
        fParams = p;
        fDim = func.NDim();
        // copy the function object to be able to modify the parameters
        // fFunc = dynamic_cast<ROOT::Math::IParamMultiFunction *>( func.Clone() );
        fFunc = &func;
        assert(fFunc != 0);
        // set parameters in function
        // fFunc->SetParameters(p);
        if (fDim == 1) {
           fFunc1Dim =
              new ROOT::Math::WrappedMemFunction<IntegralEvaluator, double (IntegralEvaluator::*)(double) const>(
                 *this, &IntegralEvaluator::F1);
           fIg1Dim = new ROOT::Math::IntegratorOneDim();
           // fIg1Dim->SetFunction( static_cast<const ROOT::Math::IMultiGenFunction & >(*fFunc),false);
           fIg1Dim->SetFunction(static_cast<const ROOT::Math::IGenFunction &>(*fFunc1Dim));
        } else if (fDim > 1) {
           fFuncNDim =
              new ROOT::Math::WrappedMemMultiFunction<IntegralEvaluator, double (IntegralEvaluator::*)(const double *)
                                                                            const>(*this, &IntegralEvaluator::FN, fDim);
           fIgNDim = new ROOT::Math::IntegratorMultiDim();
           fIgNDim->SetFunction(*fFuncNDim);
        } else
           assert(fDim > 0);
     }

     void SetParameters(const double *p)
     {
        // copy just the pointer
        fParams = p;
     }

     ~IntegralEvaluator()
     {
        if (fIg1Dim)
           delete fIg1Dim;
        if (fIgNDim)
           delete fIgNDim;
        if (fFunc1Dim)
           delete fFunc1Dim;
        if (fFuncNDim)
           delete fFuncNDim;
        // if (fFunc) delete fFunc;
     }

     // evaluation of integrand function (one-dim)
     double F1(double x) const
     {
        double xx = x;
        return ExecFunc(fFunc, &xx, fParams);
     }
     // evaluation of integrand function (multi-dim)
     double FN(const double *x) const { return ExecFunc(fFunc, x, fParams); }

     double Integral(const double *x1, const double *x2)
     {
        // return unormalized integral
        return (fIg1Dim) ? fIg1Dim->Integral(*x1, *x2) : fIgNDim->Integral(x1, x2);
     }

     double operator()(const double *x1, const double *x2)
     {
        // return normalized integral, divided by bin volume (dx1*dx...*dxn)
        if (fIg1Dim) {
           double dV = *x2 - *x1;
           return fIg1Dim->Integral(*x1, *x2) / dV;
        } else if (fIgNDim) {
           double dV = 1;
           for (unsigned int i = 0; i < fDim; ++i)
              dV *= (x2[i] - x1[i]);
           return fIgNDim->Integral(x1, x2) / dV;
           //                   std::cout << " do integral btw x " << x1[0] << "  " << x2[0] << " y " << x1[1] << "  "
           //                   << x2[1] << " dV = " << dV << " result = " << result << std::endl; return result;
        } else
           assert(1.); // should never be here
        return 0;
     }

  private:
     template <class T>
     inline double ExecFunc(T *f, const double *x, const double *p) const
     {
        return (*f)(x, p);
     }

#ifdef R__HAS_VECCORE
     inline double ExecFunc(const IModelFunctionTempl<ROOT::Double_v> *f, const double *x, const double *p) const
     {
        if (fDim == 1) {
           ROOT::Double_v xx;
           vecCore::Load<ROOT::Double_v>(xx, x);
           auto res = (*f)(&xx, p);
           return vecCore::Get<ROOT::Double_v>(res, 0);
        } else {
           std::vector<ROOT::Double_v> xx(fDim);
           for (unsigned int i = 0; i < fDim; ++i) {
              vecCore::Load<ROOT::Double_v>(xx[i], x + i);
           }
           auto res = (*f)(xx.data(), p);
           return vecCore::Get<ROOT::Double_v>(res, 0);
        }
     }
#endif

     // objects of this class are not meant to be copied / assigned
     IntegralEvaluator(const IntegralEvaluator &rhs);
     IntegralEvaluator &operator=(const IntegralEvaluator &rhs);

     unsigned int fDim;
     const double *fParams;
     // ROOT::Math::IParamMultiFunction * fFunc;  // copy of function in order to be able to change parameters
     // const ParamFunc * fFunc;       //  reference to a generic parametric function
     const ParamFunc *fFunc;
     ROOT::Math::IntegratorOneDim *fIg1Dim;
     ROOT::Math::IntegratorMultiDim *fIgNDim;
     ROOT::Math::IGenFunction *fFunc1Dim;
     ROOT::Math::IMultiGenFunction *fFuncNDim;
  };

  inline unsigned setAutomaticChunking(unsigned nEvents)
  {
     auto ncpu = ROOT::GetImplicitMTPoolSize();
     if (nEvents / ncpu < 1000)
        return ncpu;
     return nEvents / 1000;
     //   return ((nEvents/ncpu + 1) % 1000) *40 ; //arbitrary formula
   }

   // derivative with respect of the parameter to be integrated
   template <class GradFunc = IGradModelFunction>
   struct ParamDerivFunc {
      ParamDerivFunc(const GradFunc &f) : fFunc(f), fIpar(0) {}
      void SetDerivComponent(unsigned int ipar) { fIpar = ipar; }
      double operator()(const double *x, const double *p) const { return fFunc.ParameterDerivative(x, p, fIpar); }
      unsigned int NDim() const { return fFunc.NDim(); }
      const GradFunc &fFunc;
      unsigned int fIpar;
   };

   // simple gradient calculator using the 2 points rule

   class SimpleGradientCalculator {

   public:
      // construct from function and gradient dimension gdim
      // gdim = npar for parameter gradient
      // gdim = ndim for coordinate gradients
      // construct (the param values will be passed later)
      // one can choose between 2 points rule (1 extra evaluation) istrat=1
      // or two point rule (2 extra evaluation)
      // (found 2 points rule does not work correctly - minuit2FitBench fails)
      SimpleGradientCalculator(int gdim, const IModelFunction &func, double eps = 2.E-8, int istrat = 1)
         : fEps(eps), fPrecision(1.E-8),                                             // sqrt(epsilon)
           fStrategy(istrat), fN(gdim), fFunc(func), fVec(std::vector<double>(gdim)) // this can be probably optimized
      {
      }

      // internal method to calculate single partial derivative
      // assume cached vector fVec is already set
      double DoParameterDerivative(const double *x, const double *p, double f0, int k) const
      {
         double p0 = p[k];
         double h = std::max(fEps * std::abs(p0), 8.0 * fPrecision * (std::abs(p0) + fPrecision));
         fVec[k] += h;
         double deriv = 0;
         // t.b.d : treat case of infinities
         // if (fval > - std::numeric_limits<double>::max() && fval < std::numeric_limits<double>::max() )
         double f1 = fFunc(x, &fVec.front());
         if (fStrategy > 1) {
            fVec[k] = p0 - h;
            double f2 = fFunc(x, &fVec.front());
            deriv = 0.5 * (f2 - f1) / h;
         } else
            deriv = (f1 - f0) / h;

         fVec[k] = p[k]; // restore original p value
         return deriv;
      }
      // number of dimension in x (needed when calculating the integrals)
      unsigned int NDim() const { return fFunc.NDim(); }
      // number of parameters (needed for grad ccalculation)
      unsigned int NPar() const { return fFunc.NPar(); }

      double ParameterDerivative(const double *x, const double *p, int ipar) const
      {
         // fVec are the cached parameter values
         std::copy(p, p + fN, fVec.begin());
         double f0 = fFunc(x, p);
         return DoParameterDerivative(x, p, f0, ipar);
      }

      // calculate all gradient at point (x,p) knnowing already value f0 (we gain a function eval.)
      void ParameterGradient(const double *x, const double *p, double f0, double *g)
      {
         // fVec are the cached parameter values
         std::copy(p, p + fN, fVec.begin());
         for (unsigned int k = 0; k < fN; ++k) {
            g[k] = DoParameterDerivative(x, p, f0, k);
         }
      }

      // calculate gradient w.r coordinate values
      void Gradient(const double *x, const double *p, double f0, double *g)
      {
         // fVec are the cached coordinate values
         std::copy(x, x + fN, fVec.begin());
         for (unsigned int k = 0; k < fN; ++k) {
            double x0 = x[k];
            double h = std::max(fEps * std::abs(x0), 8.0 * fPrecision * (std::abs(x0) + fPrecision));
            fVec[k] += h;
            // t.b.d : treat case of infinities
            // if (fval > - std::numeric_limits<double>::max() && fval < std::numeric_limits<double>::max() )
            double f1 = fFunc(&fVec.front(), p);
            if (fStrategy > 1) {
               fVec[k] = x0 - h;
               double f2 = fFunc(&fVec.front(), p);
               g[k] = 0.5 * (f2 - f1) / h;
            } else
               g[k] = (f1 - f0) / h;

            fVec[k] = x[k]; // restore original x value
         }
      }

   private:
      double fEps;
      double fPrecision;
      int fStrategy;   // strategy in calculation ( =1 use 2 point rule( 1 extra func) , = 2 use r point rule)
      unsigned int fN; // gradient dimension
      const IModelFunction &fFunc;
      mutable std::vector<double> fVec; // cached coordinates (or parameter values in case of gradientpar)
   };

   // function to avoid infinities or nan
   inline double CorrectValue(double rval)
   {
      // avoid infinities or nan in  rval
      if (rval > -std::numeric_limits<double>::max() && rval < std::numeric_limits<double>::max())
         return rval;
      else if (rval < 0)
         // case -inf
         return -std::numeric_limits<double>::max();
      else
         // case + inf or nan
         return +std::numeric_limits<double>::max();
   }

   // calculation of the integral of the gradient functions
   // for a function providing derivative w.r.t parameters
   // x1 and x2 defines the integration interval , p the parameters
   template <class GFunc>
   void CalculateGradientIntegral(const GFunc &gfunc, const double *x1, const double *x2, const double *p, double *g)
   {

      // needs to calculate the integral for each partial derivative
      ParamDerivFunc<GFunc> pfunc(gfunc);
      IntegralEvaluator<ParamDerivFunc<GFunc>> igDerEval(pfunc, p, true);
      // loop on the parameters
      unsigned int npar = gfunc.NPar();
      for (unsigned int k = 0; k < npar; ++k) {
         pfunc.SetDerivComponent(k);
         g[k] = igDerEval(x1, x2);
      }
   }

} // end namespace FitUtil

} // end namespace Fit

} // end namespace ROOT


#endif /* ROOT_Fit_FitUtil */
