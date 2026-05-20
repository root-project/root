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

#include "Fit/BinData.h"
#include "Fit/UnBinData.h"
#include "ROOT/EExecutionPolicy.hxx"

#include "Math/Integrator.h"
#include "Math/IntegratorMultiDim.h"

#include "TError.h"
#include <vector>

// using parameter cache is not thread safe but needed for normalizing the functions
#define USE_PARAMCACHE

//#define DEBUG_FITUTIL

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

  // internal class defining
  template <class T>
  class LikelihoodAux {
  public:
     LikelihoodAux(T logv = {}, T w = {}, T w2 = {}) : logvalue(logv), weight(w), weight2(w2) {}

     LikelihoodAux operator+(const LikelihoodAux &l) const
     {
        return LikelihoodAux<T>(logvalue + l.logvalue, weight + l.weight, weight2 + l.weight2);
     }

     LikelihoodAux &operator+=(const LikelihoodAux &l)
     {
        logvalue += l.logvalue;
        weight += l.weight;
        weight2 += l.weight2;
        return *this;
     }

     T logvalue;
     T weight;
     T weight2;
  };

  template <>
  class LikelihoodAux<double> {
  public:
     LikelihoodAux(double logv = 0.0, double w = 0.0, double w2 = 0.0) : logvalue(logv), weight(w), weight2(w2){};

     LikelihoodAux operator+(const LikelihoodAux &l) const
     {
        return LikelihoodAux<double>(logvalue + l.logvalue, weight + l.weight, weight2 + l.weight2);
     }

     LikelihoodAux &operator+=(const LikelihoodAux &l)
     {
        logvalue += l.logvalue;
        weight += l.weight;
        weight2 += l.weight2;
        return *this;
     }

     double logvalue;
     double weight;
     double weight2;
  };

  // internal class to evaluate the function or the integral
  // and cached internal integration details
  // if useIntegral is false no allocation is done
  // and this is a dummy class
  // class is templated on any parametric functor implementing operator()(x,p) and NDim()
  // contains a constant pointer to the function

  template <class ParamFunc = ROOT::Math::IParamMultiFunctionTempl<double>>
  class IntegralEvaluator {

  public:
     IntegralEvaluator(const ParamFunc &func, const double *p, bool useIntegral = true,
                       ROOT::Math::IntegrationOneDim::Type igType = ROOT::Math::IntegrationOneDim::kDEFAULT)
        : fDim(0), fParams(nullptr), fFunc(nullptr), fIg1Dim(nullptr), fIgNDim(nullptr), fFunc1Dim(nullptr), fFuncNDim(nullptr)
     {
        if (useIntegral) {
           SetFunction(func, p, igType);
        }
     }

     void SetFunction(const ParamFunc &func, const double *p = nullptr,
                      ROOT::Math::IntegrationOneDim::Type igType = ROOT::Math::IntegrationOneDim::kDEFAULT)
     {
        // set the integrand function and create required wrapper
        // to perform integral in (x) of a generic  f(x,p)
        fParams = p;
        fDim = func.NDim();
        // copy the function object to be able to modify the parameters
        // fFunc = dynamic_cast<ROOT::Math::IParamMultiFunction *>( func.Clone() );
        fFunc = &func;
        assert(fFunc != nullptr);
        // set parameters in function
        // fFunc->SetParameters(p);
        if (fDim == 1) {
           fFunc1Dim =
              new ROOT::Math::WrappedMemFunction<IntegralEvaluator, double (IntegralEvaluator::*)(double) const>(
                 *this, &IntegralEvaluator::F1);
           fIg1Dim = new ROOT::Math::IntegratorOneDim(igType);
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
        // return unnormalized integral
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

#ifdef R__HAS_STD_EXPERIMENTAL_SIMD

#if __clang_major__ > 16
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvla-cxx-extension"
#endif

     inline double ExecFunc(const IModelFunctionTempl<ROOT::Double_v> *f, const double *x, const double *p) const
     {
        ROOT::Double_v xx[fDim];
        for (unsigned int i = 0; i < fDim; ++i) {
           xx[i][0] = x[i];
           for (std::size_t j = 1; j < ROOT::Double_v::size(); ++j) {
              xx[i][j] = 0.0;
           }
        }
        auto res = (*f)(xx, p);
        return res[0];
     }

#if __clang_major__ > 16
#pragma clang diagnostic pop
#endif

#endif

     // objects of this class are not meant to be copied / assigned
     IntegralEvaluator(const IntegralEvaluator &rhs) = delete;
     IntegralEvaluator &operator=(const IntegralEvaluator &rhs) = delete;

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

  /** Chi2 Functions */

  /**
      evaluate the Chi2 given a model function and the data at the point x.
      return also nPoints as the effective number of used points in the Chi2 evaluation
  */
  double EvaluateChi2(const IModelFunction &func, const BinData &data, const double *p, unsigned int &nPoints,
                      ::ROOT::EExecutionPolicy executionPolicy, unsigned nChunks = 0);

  /**
      evaluate the effective Chi2 given a model function and the data at the point x.
      The effective chi2 uses the errors on the coordinates : W = 1/(sigma_y**2 + ( sigma_x_i * df/dx_i )**2 )
      return also nPoints as the effective number of used points in the Chi2 evaluation
  */
  double EvaluateChi2Effective(const IModelFunction &func, const BinData &data, const double *x, unsigned int &nPoints);

  /**
      evaluate the Chi2 gradient given a model function and the data at the point p.
      return also nPoints as the effective number of used points in the Chi2 evaluation
  */
  void EvaluateChi2Gradient(const IModelFunction &func, const BinData &data, const double *p, double *grad,
                            unsigned int &nPoints,
                            ::ROOT::EExecutionPolicy executionPolicy = ::ROOT::EExecutionPolicy::kSequential,
                            unsigned nChunks = 0);

  /**
      evaluate the LogL given a model function and the data at the point x.
      return also nPoints as the effective number of used points in the LogL evaluation
  */
  double EvaluateLogL(const IModelFunction &func, const UnBinData &data, const double *p, int iWeight, bool extended,
                      unsigned int &nPoints, ::ROOT::EExecutionPolicy executionPolicy, unsigned nChunks = 0);

  /**
      evaluate the LogL gradient given a model function and the data at the point p.
      return also nPoints as the effective number of used points in the LogL evaluation
  */
  void EvaluateLogLGradient(const IModelFunction &func, const UnBinData &data, const double *p, double *grad,
                            unsigned int &nPoints,
                            ::ROOT::EExecutionPolicy executionPolicy = ::ROOT::EExecutionPolicy::kSequential,
                            unsigned nChunks = 0);

  // #ifdef R__HAS_STD_EXPERIMENTAL_SIMD
  //    template <class NotCompileIfScalarBackend = std::enable_if<!(std::is_same<double, ROOT::Double_v>::value)>>
  //    void EvaluateLogLGradient(const IModelFunctionTempl<ROOT::Double_v> &, const UnBinData &, const double *, double
  //    *, unsigned int & ) {}
  // #endif

  /**
      evaluate the Poisson LogL given a model function and the data at the point p.
      return also nPoints as the effective number of used points in the LogL evaluation
      By default is extended, pass extend to false if want to be not extended (MultiNomial)
  */
  double EvaluatePoissonLogL(const IModelFunction &func, const BinData &data, const double *p, int iWeight,
                             bool extended, unsigned int &nPoints, ::ROOT::EExecutionPolicy executionPolicy,
                             unsigned nChunks = 0);

  /**
      evaluate the Poisson LogL given a model function and the data at the point p.
      return also nPoints as the effective number of used points in the LogL evaluation
  */
  void EvaluatePoissonLogLGradient(const IModelFunction &func, const BinData &data, const double *p, double *grad,
                                   unsigned int &nPoints,
                                   ::ROOT::EExecutionPolicy executionPolicy = ::ROOT::EExecutionPolicy::kSequential,
                                   unsigned nChunks = 0);

  // methods required by dedicate minimizer like Fumili

  /**
      evaluate the residual contribution to the Chi2 given a model function and the BinPoint data
      and if the pointer g is not null evaluate also the gradient of the residual.
      If the function provides parameter derivatives they are used otherwise a simple derivative calculation
      is used
  */
  double EvaluateChi2Residual(const IModelFunction &func, const BinData &data, const double *p, unsigned int ipoint,
                              double *g = nullptr, double * h = nullptr, bool hasGrad = false, bool fullHessian = false);

  /**
      evaluate the pdf contribution to the LogL given a model function and the BinPoint data.
      If the pointer g is not null evaluate also the gradient of the pdf.
      If the function provides parameter derivatives they are used otherwise a simple derivative calculation
      is used
  */
  double
  EvaluatePdf(const IModelFunction &func, const UnBinData &data, const double *p, unsigned int ipoint, double *g = nullptr, double * h = nullptr, bool hasGrad = false, bool fullHessian = false);


   /**
       evaluate the pdf contribution to the Poisson LogL given a model function and the BinPoint data.
       If the pointer g is not null evaluate also the gradient of the Poisson pdf.
       If the function provides parameter derivatives they are used otherwise a simple derivative calculation
       is used
   */
   double EvaluatePoissonBinPdf(const IModelFunction & func, const BinData & data, const double * x, unsigned int ipoint, double * g = nullptr, double * h = nullptr, bool hasGrad = false, bool fullHessian = false);

   unsigned setAutomaticChunking(unsigned nEvents);

   template <class T>
   struct Evaluate {};

#ifdef R__HAS_STD_EXPERIMENTAL_SIMD
   template <>
   struct Evaluate<Double_v> {
      static double EvalChi2(const IModelFunctionTempl<Double_v> &func, const BinData &data, const double *p,
                             unsigned int &nPoints, ::ROOT::EExecutionPolicy executionPolicy, unsigned nChunks = 0);

      static double EvalLogL(const IModelFunctionTempl<Double_v> &func, const UnBinData &data, const double *const p,
                             int iWeight, bool extended, unsigned int &nPoints,
                             ::ROOT::EExecutionPolicy executionPolicy, unsigned nChunks = 0);

      static double EvalPoissonLogL(const IModelFunctionTempl<Double_v> &func, const BinData &data, const double *p,
                                    int iWeight, bool extended, unsigned int, ::ROOT::EExecutionPolicy executionPolicy,
                                    unsigned nChunks = 0);

      static double
      EvalChi2Effective(const IModelFunctionTempl<Double_v> &, const BinData &, const double *, unsigned int &)
      {
         Error("FitUtil::Evaluate<T>::EvalChi2Effective", "The vectorized evaluation of the Chi2 with coordinate errors is still not supported");
         return -1.;
      }

      static void EvalChi2Gradient(const IModelFunctionTempl<Double_v> &f, const BinData &data, const double *p,
                                   double *grad, unsigned int &nPoints,
                                   ::ROOT::EExecutionPolicy executionPolicy = ::ROOT::EExecutionPolicy::kSequential,
                                   unsigned nChunks = 0);

      static double EvalChi2Residual(const IModelFunctionTempl<Double_v> &, const BinData &, const double *,
                                     unsigned int, double *, double *, bool, bool)
      {
         Error("FitUtil::Evaluate<T>::EvalChi2Residual", "The vectorized evaluation of the Chi2 with the ith residual is still not supported");
         return -1.;
      }

      /// evaluate the pdf (Poisson) contribution to the logl (return actually log of pdf)
      /// and its gradient
      static double EvalPoissonBinPdf(const IModelFunctionTempl<Double_v> &, const BinData &, const double *,
                                      unsigned int, double *, double *, bool, bool)
      {
         Error("FitUtil::Evaluate<T>::EvaluatePoissonBinPdf", "The vectorized evaluation of the BinnedLikelihood fit evaluated point by point is still not supported");
         return -1.;
      }

      static double EvalPdf(const IModelFunctionTempl<Double_v> &, const UnBinData &, const double *, unsigned int,
                            double *, double *, bool, bool)
      {
         Error("FitUtil::Evaluate<T>::EvalPdf", "The vectorized evaluation of the LogLikelihood fit evaluated point by point is still not supported");
         return -1.;
      }

      static void
      EvalPoissonLogLGradient(const IModelFunctionTempl<Double_v> &f, const BinData &data, const double *p,
                              double *grad, unsigned int &,
                              ::ROOT::EExecutionPolicy executionPolicy = ::ROOT::EExecutionPolicy::kSequential,
                              unsigned nChunks = 0);

      static void EvalLogLGradient(const IModelFunctionTempl<Double_v> &f, const UnBinData &data, const double *p,
                                   double *grad, unsigned int &,
                                   ::ROOT::EExecutionPolicy executionPolicy = ::ROOT::EExecutionPolicy::kSequential,
                                   unsigned nChunks = 0);
   };
#endif // R__HAS_STD_EXPERIMENTAL_SIMD

   template <>
   struct Evaluate<double> {

      static double EvalChi2(const IModelFunction &func, const BinData &data, const double *p, unsigned int &nPoints,
                             ::ROOT::EExecutionPolicy executionPolicy, unsigned nChunks = 0)
      {
         // evaluate the chi2 given a  function reference, the data and returns the value and also in nPoints
         // the actual number of used points
         // normal chi2 using only error on values (from fitting histogram)
         // optionally the integral of function in the bin is used


         //Info("EvalChi2","Using non-vectorized implementation %d",(int) data.Opt().fIntegral);

         return FitUtil::EvaluateChi2(func, data, p, nPoints, executionPolicy, nChunks);
      }

      static double EvalLogL(const IModelFunctionTempl<double> &func, const UnBinData &data, const double *p,
                             int iWeight, bool extended, unsigned int &nPoints,
                             ::ROOT::EExecutionPolicy executionPolicy, unsigned nChunks = 0)
      {
         return FitUtil::EvaluateLogL(func, data, p, iWeight, extended, nPoints, executionPolicy, nChunks);
      }

      static double EvalPoissonLogL(const IModelFunctionTempl<double> &func, const BinData &data, const double *p,
                                    int iWeight, bool extended, unsigned int &nPoints,
                                    ::ROOT::EExecutionPolicy executionPolicy, unsigned nChunks = 0)
      {
         return FitUtil::EvaluatePoissonLogL(func, data, p, iWeight, extended, nPoints, executionPolicy, nChunks);
      }

      static double EvalChi2Effective(const IModelFunctionTempl<double> &func, const BinData & data, const double * p, unsigned int &nPoints)
      {
         return FitUtil::EvaluateChi2Effective(func, data, p, nPoints);
      }
      static void EvalChi2Gradient(const IModelFunctionTempl<double> &func, const BinData &data, const double *p,
                                   double *g, unsigned int &nPoints,
                                   ::ROOT::EExecutionPolicy executionPolicy = ::ROOT::EExecutionPolicy::kSequential,
                                   unsigned nChunks = 0)
      {
         FitUtil::EvaluateChi2Gradient(func, data, p, g, nPoints, executionPolicy, nChunks);
      }

      static double EvalChi2Residual(const IModelFunctionTempl<double> &func, const BinData & data, const double * p, unsigned int i, double *g, double * h,
                                    bool hasGrad, bool fullHessian)
      {
         return FitUtil::EvaluateChi2Residual(func, data, p, i, g, h, hasGrad, fullHessian);
      }

      /// evaluate the pdf (Poisson) contribution to the logl (return actually log of pdf)
      /// and its gradient
      static double EvalPoissonBinPdf(const IModelFunctionTempl<double> &func, const BinData & data, const double *p, unsigned int i, double *g, double * h, bool hasGrad, bool fullHessian) {
         return FitUtil::EvaluatePoissonBinPdf(func, data, p, i, g, h, hasGrad, fullHessian);
      }

      static double EvalPdf(const IModelFunctionTempl<double> &func, const UnBinData & data, const double *p, unsigned int i, double *g, double * h, bool hasGrad, bool fullHessian) {
         return FitUtil::EvaluatePdf(func, data, p, i, g, h, hasGrad, fullHessian);
      }

      static void
      EvalPoissonLogLGradient(const IModelFunctionTempl<double> &func, const BinData &data, const double *p, double *g,
                              unsigned int &nPoints,
                              ::ROOT::EExecutionPolicy executionPolicy = ::ROOT::EExecutionPolicy::kSequential,
                              unsigned nChunks = 0)
      {
         FitUtil::EvaluatePoissonLogLGradient(func, data, p, g, nPoints, executionPolicy, nChunks);
      }

      static void EvalLogLGradient(const IModelFunctionTempl<double> &func, const UnBinData &data, const double *p,
                                   double *g, unsigned int &nPoints,
                                   ::ROOT::EExecutionPolicy executionPolicy = ::ROOT::EExecutionPolicy::kSequential,
                                   unsigned nChunks = 0)
      {
         FitUtil::EvaluateLogLGradient(func, data, p, g, nPoints, executionPolicy, nChunks);
      }
   };

   } // end namespace FitUtil

   } // end namespace Fit

   } // end namespace ROOT

#endif /* ROOT_Fit_FitUtil */
