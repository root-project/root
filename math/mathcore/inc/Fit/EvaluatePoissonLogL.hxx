// @(#)root/mathcore:$Id$
// Author: L. Moneta Tue Nov 28 10:52:47 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Fit_EvaluatePoissonLogL
#define ROOT_Fit_EvaluatePoissonLogL

#include "Math/IParamFunctionfwd.h"
#include "Math/IParamFunction.h"

#ifdef R__USE_IMT
#include "ROOT/TThreadExecutor.hxx"
#endif
#include "ROOT/TSequentialExecutor.hxx"

#include "Fit/BinData.h"
#include "Fit/FitExecutionPolicy.h"
#include "Fit/FitUtil.h"

#include "Math/Integrator.h"
#include "Math/IntegratorMultiDim.h"

#include "TError.h"

// using parameter cache is not thread safe but needed for normalizing the functions
#define USE_PARAMCACHE

namespace ROOT {

namespace Fit {

/**
   namespace defining utility free functions using in Fit for evaluating the various fit method
   functions (chi2, likelihood, etc..)  given the data and the model function

   @ingroup FitMain
*/
namespace FitUtil {

/**
    evaluate the Poisson LogL given a model function and the data at the point x.
    return also nPoints as the effective number of used points in the LogL evaluation
    By default is extended, pass extedend to false if want to be not extended (MultiNomial)
*/
double EvaluatePoissonLogL(const IModelFunction &func, const BinData &data, const double *x, int iWeight, bool extended,
                           unsigned int &nPoints, ROOT::Fit::ExecutionPolicy executionPolicy, unsigned nChunks = 0);

/**
    evaluate the Poisson LogL given a model function and the data at the point x.
    return also nPoints as the effective number of used points in the LogL evaluation
*/
void EvaluatePoissonLogLGradient(const IModelFunction &func, const BinData &data, const double *x, double *grad,
                                 unsigned int &nPoints,
                                 ROOT::Fit::ExecutionPolicy executionPolicy = ROOT::Fit::ExecutionPolicy::kSerial,
                                 unsigned nChunks = 0);

/**
    evaluate the pdf contribution to the Poisson LogL given a model function and the BinPoint data.
    If the pointer g is not null evaluate also the gradient of the Poisson pdf.
    If the function provides parameter derivatives they are used otherwise a simple derivative calculation
    is used
*/
double EvaluatePoissonBinPdf(const IModelFunction &func, const BinData &data, const double *x, unsigned int ipoint,
                             double *g = 0);

template <class T>
struct PoissonLogL {
#ifdef R__HAS_VECCORE

   static double Eval(const IModelFunctionTempl<T> &func, const BinData &data, const double *p, int iWeight,
                      bool extended, unsigned int, ROOT::Fit::ExecutionPolicy executionPolicy, unsigned nChunks = 0)
   {
      // evaluate the Poisson Log Likelihood
      // for binned likelihood fits
      // this is Sum ( f(x_i)  -  y_i * log( f (x_i) ) )
      // add as well constant term for saturated model to make it like a Chi2/2
      // by default is etended. If extended is false the fit is not extended and
      // the global poisson term is removed (i.e is a binomial fit)
      // (remember that in this case one needs to have a function with a fixed normalization
      // like in a non extended binned fit)
      //
      // if use Weight use a weighted dataset
      // iWeight = 1 ==> logL = Sum( w f(x_i) )
      // case of iWeight==1 is actually identical to weight==0
      // iWeight = 2 ==> logL = Sum( w*w * f(x_i) )
      //

#ifdef USE_PARAMCACHE
      (const_cast<IModelFunctionTempl<T> &>(func)).SetParameters(p);
#endif
      auto vecSize = vecCore::VectorSize<T>();
      // get fit option and check case of using integral of bins
      const DataOptions &fitOpt = data.Opt();
      if (fitOpt.fExpErrors || fitOpt.fIntegral)
         Error("FitUtil::EvaluatePoissonLogL",
               "The vectorized implementation doesn't support Integrals or BinVolume\n. Aborting operation.");
      bool useW2 = (iWeight == 2);

      auto mapFunction = [&](unsigned int i) {
         T y;
         vecCore::Load<T>(y, data.ValuePtr(i * vecSize));
         T fval{};

         if (data.NDim() > 1) {
            std::vector<T> x(data.NDim());
            for (unsigned int j = 0; j < data.NDim(); ++j)
               vecCore::Load<T>(x[j], data.GetCoordComponent(i * vecSize, j));
#ifdef USE_PARAMCACHE
            fval = func(x.data());
#else
            fval = func(x.data(), p);
#endif
            // one -dim case
         } else {
            T x;
            vecCore::Load<T>(x, data.GetCoordComponent(i * vecSize, 0));
#ifdef USE_PARAMCACHE
            fval = func(&x);
#else
            fval = func(&x, p);
#endif
         }

         // EvalLog protects against 0 values of fval but don't want to add in the -log sum
         // negative values of fval
         vecCore::MaskedAssign<T>(fval, fval < 0.0, 0.0);

         T nloglike{}; // negative loglikelihood

         if (useW2) {
            // apply weight correction . Effective weight is error^2/ y
            // and expected events in bins is fval/weight
            // can apply correction only when y is not zero otherwise weight is undefined
            // (in case of weighted likelihood I don't care about the constant term due to
            // the saturated model)
            assert(data.GetErrorType() != ROOT::Fit::BinData::ErrorType::kNoError);
            T error = 0.0;
            vecCore::Load<T>(error, data.ErrorPtr(i * vecSize));
            // for empty bin use the average weight  computed from the total data weight
            auto m = vecCore::Mask_v<T>(y != 0.0);
            auto weight = vecCore::Blend(m, (error * error) / y, T(data.SumOfError2() / data.SumOfContent()));
            if (extended) {
               nloglike = weight * (fval - y);
            }
            vecCore::MaskedAssign<T>(nloglike, y != 0,
                                     nloglike +
                                        weight * y * (ROOT::Math::Util::EvalLog(y) - ROOT::Math::Util::EvalLog(fval)));

         } else {
            // standard case no weights or iWeight=1
            // this is needed for Poisson likelihood (which are extened and not for multinomial)
            // the formula below  include constant term due to likelihood of saturated model (f(x) = y)
            // (same formula as in Baker-Cousins paper, page 439 except a factor of 2
            if (extended)
               nloglike = fval - y;

            vecCore::MaskedAssign<T>(nloglike, y > 0,
                                     nloglike + y * (ROOT::Math::Util::EvalLog(y) - ROOT::Math::Util::EvalLog(fval)));
         }

         return nloglike;
      };

#ifdef R__USE_IMT
      auto redFunction = [](const std::vector<T> &objs) { return std::accumulate(objs.begin(), objs.end(), T{}); };
#else
      (void)nChunks;

      // If IMT is disabled, force the execution policy to the serial case
      if (executionPolicy == ROOT::Fit::ExecutionPolicy::kMultithread) {
         Warning("FitUtil::EvaluateLog<T>::EvalPoissonLogL",
                 "Multithread execution policy requires IMT, which is disabled. Changing "
                 "to ROOT::Fit::ExecutionPolicy::kSerial.");
         executionPolicy = ROOT::Fit::ExecutionPolicy::kSerial;
      }
#endif

      T res{};
      if (executionPolicy == ROOT::Fit::ExecutionPolicy::kSerial) {
         for (unsigned int i = 0; i < (data.Size() / vecSize); i++) {
            res += mapFunction(i);
         }
#ifdef R__USE_IMT
      } else if (executionPolicy == ROOT::Fit::ExecutionPolicy::kMultithread) {
         ROOT::TThreadExecutor pool;
         auto chunks = nChunks != 0 ? nChunks : setAutomaticChunking(data.Size() / vecSize);
         res = pool.MapReduce(mapFunction, ROOT::TSeq<unsigned>(0, data.Size() / vecSize), redFunction, chunks);
#endif
      } else {
         Error("FitUtil::EvaluateLog<T>::EvalPoissonLogL",
               "Execution policy unknown. Avalaible choices:\n ROOT::Fit::ExecutionPolicy::kSerial (default)\n "
               "ROOT::Fit::ExecutionPolicy::kMultithread (requires IMT)\n");
      }

      // Last padded SIMD vector of elements
      if (data.Size() % vecSize != 0)
         vecCore::MaskedAssign(res, ::vecCore::Int2Mask<T>(data.Size() % vecSize),
                               res + mapFunction(data.Size() / vecSize));

      return vecCore::ReduceAdd(res);
   }

   /// evaluate the pdf (Poisson) contribution to the logl (return actually log of pdf)
   /// and its gradient
   static double EvalBinPdf(const IModelFunctionTempl<T> &, const BinData &, const double *, unsigned int, double *)
   {
      Error("FitUtil::EvaluateLog<T>::EvaluatePoissonBinPdf",
            "The vectorized evaluation of the BinnedLikelihood fit evaluated point by point is still not supported");
      return -1.;
   }

   static void
   EvalGradient(const IModelFunctionTempl<T> &f, const BinData &data, const double *p, double *grad, unsigned int &,
                ROOT::Fit::ExecutionPolicy executionPolicy = ROOT::Fit::ExecutionPolicy::kSerial, unsigned nChunks = 0)
   {
      // evaluate the gradient of the Poisson log likelihood function

      const IGradModelFunctionTempl<T> *fg = dynamic_cast<const IGradModelFunctionTempl<T> *>(&f);
      assert(fg != nullptr); // must be called by a grad function

      const IGradModelFunctionTempl<T> &func = *fg;

      (const_cast<IGradModelFunctionTempl<T> &>(func)).SetParameters(p);

      const DataOptions &fitOpt = data.Opt();
      if (fitOpt.fBinVolume || fitOpt.fIntegral || fitOpt.fExpErrors)
         Error("FitUtil::EvaluatePoissonLogLGradient", "The vectorized implementation doesn't support Integrals,"
                                                       "BinVolume or ExpErrors\n. Aborting operation.");

      unsigned int npar = func.NPar();
      auto vecSize = vecCore::VectorSize<T>();
      unsigned initialNPoints = data.Size();
      unsigned numVectors = initialNPoints / vecSize;

      auto mapFunction = [&](const unsigned int i) {
         // set all vector values to zero
         std::vector<T> gradFunc(npar);
         std::vector<T> pointContributionVec(npar);

         T x1, y;

         vecCore::Load<T>(x1, data.GetCoordComponent(i * vecSize, 0));
         vecCore::Load<T>(y, data.ValuePtr(i * vecSize));

         T fval = 0;

         const T *x = nullptr;

         unsigned ndim = data.NDim();
         std::vector<T> xc;
         if (ndim > 1) {
            xc.resize(ndim);
            xc[0] = x1;
            for (unsigned int j = 1; j < ndim; ++j)
               vecCore::Load<T>(xc[j], data.GetCoordComponent(i * vecSize, j));
            x = xc.data();
         } else {
            x = &x1;
         }

         fval = func(x, p);
         func.ParameterGradient(x, p, &gradFunc[0]);

         // correct the gradient
         for (unsigned int ipar = 0; ipar < npar; ++ipar) {
            vecCore::Mask<T> positiveValuesMask = fval > 0;

            // df/dp * (1.  - y/f )
            vecCore::MaskedAssign(pointContributionVec[ipar], positiveValuesMask, gradFunc[ipar] * (1. - y / fval));

            vecCore::Mask<T> validNegativeValuesMask = !positiveValuesMask && gradFunc[ipar] != 0;

            if (!vecCore::MaskEmpty(validNegativeValuesMask)) {
               const T kdmax1 = vecCore::math::Sqrt(vecCore::NumericLimits<T>::Max());
               const T kdmax2 = vecCore::NumericLimits<T>::Max() / (4 * initialNPoints);
               T gg = kdmax1 * gradFunc[ipar];
               pointContributionVec[ipar] =
                  -vecCore::Blend(gg > 0, vecCore::math::Min(gg, kdmax2), vecCore::math::Max(gg, -kdmax2));
            }
         }

#ifdef DEBUG_FITUTIL
         {
            R__LOCKGUARD(gROOTMutex);
            if (i < 5 || (i > data.Size() - 5)) {
               if (data.NDim() > 1)
                  std::cout << i << "  x " << x[0] << " y " << x[1];
               else
                  std::cout << i << "  x " << x[0];
               std::cout << " func " << fval << " gradient ";
               for (unsigned int ii = 0; ii < npar; ++ii)
                  std::cout << "  " << pointContributionVec[ii];
               std::cout << "\n";
            }
         }
#endif

         return pointContributionVec;
      };

      // Vertically reduce the set of vectors by summing its equally-indexed components
      auto redFunction = [&](const std::vector<std::vector<T>> &partialResults) {
         std::vector<T> result(npar);

         for (auto const &pointContributionVec : partialResults) {
            for (unsigned int parameterIndex = 0; parameterIndex < npar; parameterIndex++)
               result[parameterIndex] += pointContributionVec[parameterIndex];
         }

         return result;
      };

      std::vector<T> gVec(npar);

#ifndef R__USE_IMT
      // to fix compiler warning
      (void)nChunks;

      // If IMT is disabled, force the execution policy to the serial case
      if (executionPolicy == ROOT::Fit::ExecutionPolicy::kMultithread) {
         Warning("FitUtil::EvaluatePoissonLogLGradient",
                 "Multithread execution policy requires IMT, which is disabled. Changing "
                 "to ROOT::Fit::ExecutionPolicy::kSerial.");
         executionPolicy = ROOT::Fit::ExecutionPolicy::kSerial;
      }
#endif

      if (executionPolicy == ROOT::Fit::ExecutionPolicy::kSerial) {
         ROOT::TSequentialExecutor pool;
         gVec = pool.MapReduce(mapFunction, ROOT::TSeq<unsigned>(0, numVectors), redFunction);
      }
#ifdef R__USE_IMT
      else if (executionPolicy == ROOT::Fit::ExecutionPolicy::kMultithread) {
         ROOT::TThreadExecutor pool;
         auto chunks = nChunks != 0 ? nChunks : setAutomaticChunking(numVectors);
         gVec = pool.MapReduce(mapFunction, ROOT::TSeq<unsigned>(0, numVectors), redFunction, chunks);
      }
#endif
      else {
         Error("FitUtil::EvaluatePoissonLogLGradient", "Execution policy unknown. Avalaible choices:\n "
                                                       "ROOT::Fit::ExecutionPolicy::kSerial (default)\n "
                                                       "ROOT::Fit::ExecutionPolicy::kMultithread (requires IMT)\n");
      }

      // Compute the contribution from the remaining points
      unsigned int remainingPoints = initialNPoints % vecSize;
      if (remainingPoints > 0) {
         auto remainingPointsContribution = mapFunction(numVectors);
         // Add the contribution from the valid remaining points and store the result in the output variable
         auto remainingMask = ::vecCore::Int2Mask<T>(remainingPoints);
         for (unsigned int param = 0; param < npar; param++) {
            vecCore::MaskedAssign(gVec[param], remainingMask, gVec[param] + remainingPointsContribution[param]);
         }
      }
      // reduce final gradient result from T to double
      for (unsigned int param = 0; param < npar; param++) {
         grad[param] = vecCore::ReduceAdd(gVec[param]);
      }

#ifdef DEBUG_FITUTIL
      std::cout << "***** Final gradient : ";
      for (unsigned int ii = 0; ii < npar; ++ii)
         std::cout << grad[ii] << "   ";
      std::cout << "\n";
#endif
   }
};

template <>
struct PoissonLogL<double> {
#endif

   static double Eval(const IModelFunctionTempl<double> &func, const BinData &data, const double *p, int iWeight,
                      bool extended, unsigned int &nPoints, ::ROOT::Fit::ExecutionPolicy executionPolicy,
                      unsigned nChunks = 0)
   {
      return FitUtil::EvaluatePoissonLogL(func, data, p, iWeight, extended, nPoints, executionPolicy, nChunks);
   }

   /// evaluate the pdf (Poisson) contribution to the logl (return actually log of pdf)
   /// and its gradient
   static double
   EvalBinPdf(const IModelFunctionTempl<double> &func, const BinData &data, const double *p, unsigned int i, double *g)
   {
      return FitUtil::EvaluatePoissonBinPdf(func, data, p, i, g);
   }

   static void EvalGradient(const IModelFunctionTempl<double> &func, const BinData &data, const double *p, double *g,
                            unsigned int &nPoints,
                            ::ROOT::Fit::ExecutionPolicy executionPolicy = ::ROOT::Fit::ExecutionPolicy::kSerial,
                            unsigned nChunks = 0)
   {
      FitUtil::EvaluatePoissonLogLGradient(func, data, p, g, nPoints, executionPolicy, nChunks);
   }
};

} // end namespace FitUtil

} // end namespace Fit

} // end namespace ROOT

#endif /* ROOT_Fit_EvaluateLogL */
