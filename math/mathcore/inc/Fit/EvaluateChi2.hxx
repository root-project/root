// @(#)root/mathcore:$Id$
// Author: L. Moneta Tue Nov 28 10:52:47 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class FitUtil

#ifndef ROOT_Fit_EvaluateChi2
#define ROOT_Fit_EvaluateChi2

#include "Math/IParamFunctionfwd.h"
#include "Math/IParamFunction.h"

#ifdef R__USE_IMT
#include "ROOT/TThreadExecutor.hxx"
#endif
#include "ROOT/TSequentialExecutor.hxx"

#include "Fit/BinData.h"
#include "Fit/FitExecutionPolicy.h"
#include "Fit/FitUtil.h"

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
    evaluate the Chi2 given a model function and the data at the point x.
    return also nPoints as the effective number of used points in the Chi2 evaluation
*/
double EvaluateChi2(const IModelFunction &func, const BinData &data, const double *x, unsigned int &nPoints,
                    ROOT::Fit::ExecutionPolicy executionPolicy, unsigned nChunks = 0);

/**
    evaluate the effective Chi2 given a model function and the data at the point x.
    The effective chi2 uses the errors on the coordinates : W = 1/(sigma_y**2 + ( sigma_x_i * df/dx_i )**2 )
    return also nPoints as the effective number of used points in the Chi2 evaluation
*/
double EvaluateChi2Effective(const IModelFunction &func, const BinData &data, const double *x, unsigned int &nPoints);

/**
    evaluate the Chi2 gradient given a model function and the data at the point x.
    return also nPoints as the effective number of used points in the Chi2 evaluation
*/
void EvaluateChi2Gradient(const IModelFunction &func, const BinData &data, const double *x, double *grad,
                          unsigned int &nPoints,
                          ROOT::Fit::ExecutionPolicy executionPolicy = ROOT::Fit::ExecutionPolicy::kSerial,
                          unsigned nChunks = 0);

// methods required by dedicate minimizer like Fumili

/**
    evaluate the residual contribution to the Chi2 given a model function and the BinPoint data
    and if the pointer g is not null evaluate also the gradient of the residual.
    If the function provides parameter derivatives they are used otherwise a simple derivative calculation
    is used
*/
double EvaluateChi2Residual(const IModelFunction &func, const BinData &data, const double *x, unsigned int ipoint,
                            double *g = 0);

template <class T>
struct Chi2 {
#ifdef R__HAS_VECCORE
   static double Eval(const IModelFunctionTempl<T> &func, const BinData &data, const double *p, unsigned int &nPoints,
                      ROOT::Fit::ExecutionPolicy executionPolicy, unsigned nChunks = 0)
   {
      // evaluate the chi2 given a  vectorized function reference  , the data and returns the value and also in nPoints
      // the actual number of used points
      // normal chi2 using only error on values (from fitting histogram)
      // optionally the integral of function in the bin is used

      // Info("Eval","Using vecorized implementation %d",(int) data.Opt().fIntegral);

      unsigned int n = data.Size();
      nPoints = data.Size(); // npoints

      // set parameters of the function to cache integral value
#ifdef USE_PARAMCACHE
      (const_cast<IModelFunctionTempl<T> &>(func)).SetParameters(p);
#endif
      // do not cache parameter values (it is not thread safe)
      // func.SetParameters(p);

      // get fit option and check case if using integral of bins
      const DataOptions &fitOpt = data.Opt();
      if (fitOpt.fBinVolume || fitOpt.fIntegral || fitOpt.fExpErrors)
         Error(
            "FitUtil::EvaluateChi2",
            "The vectorized implementation doesn't support Integrals, BinVolume or ExpErrors\n. Aborting operation.");

      (const_cast<IModelFunctionTempl<T> &>(func)).SetParameters(p);

      double maxResValue = std::numeric_limits<double>::max() / n;
      std::vector<double> ones{1, 1, 1, 1};
      auto vecSize = vecCore::VectorSize<T>();

      auto mapFunction = [&](unsigned int i) {
         // in case of no error in y invError=1 is returned
         T x1, y, invErrorVec;
         vecCore::Load<T>(x1, data.GetCoordComponent(i * vecSize, 0));
         vecCore::Load<T>(y, data.ValuePtr(i * vecSize));
         const auto invError = data.ErrorPtr(i * vecSize);
         auto invErrorptr = (invError != nullptr) ? invError : &ones.front();
         vecCore::Load<T>(invErrorVec, invErrorptr);

         const T *x;
         std::vector<T> xc;
         if (data.NDim() > 1) {
            xc.resize(data.NDim());
            xc[0] = x1;
            for (unsigned int j = 1; j < data.NDim(); ++j)
               vecCore::Load<T>(xc[j], data.GetCoordComponent(i * vecSize, j));
            x = xc.data();
         } else {
            x = &x1;
         }

         T fval{};

#ifdef USE_PARAMCACHE
         fval = func(x);
#else
         fval = func(x, p);
#endif

         T tmp = (y - fval) * invErrorVec;
         T chi2 = tmp * tmp;

         // avoid inifinity or nan in chi2 values due to wrong function values
         auto m = vecCore::Mask_v<T>(chi2 > maxResValue);

         vecCore::MaskedAssign<T>(chi2, m, maxResValue);

         return chi2;
      };

      auto redFunction = [](const std::vector<T> &objs) { return std::accumulate(objs.begin(), objs.end(), T{}); };

#ifndef R__USE_IMT
      (void)nChunks;

      // If IMT is disabled, force the execution policy to the serial case
      if (executionPolicy == ROOT::Fit::ExecutionPolicy::kMultithread) {
         Warning("FitUtil::EvaluateChi2", "Multithread execution policy requires IMT, which is disabled. Changing "
                                          "to ROOT::Fit::ExecutionPolicy::kSerial.");
         executionPolicy = ROOT::Fit::ExecutionPolicy::kSerial;
      }
#endif

      T res{};
      if (executionPolicy == ROOT::Fit::ExecutionPolicy::kSerial) {
         ROOT::TSequentialExecutor pool;
         res = pool.MapReduce(mapFunction, ROOT::TSeq<unsigned>(0, data.Size() / vecSize), redFunction);
#ifdef R__USE_IMT
      } else if (executionPolicy == ROOT::Fit::ExecutionPolicy::kMultithread) {
         ROOT::TThreadExecutor pool;
         auto chunks = nChunks != 0 ? nChunks : setAutomaticChunking(data.Size() / vecSize);
         res = pool.MapReduce(mapFunction, ROOT::TSeq<unsigned>(0, data.Size() / vecSize), redFunction, chunks);
#endif
      } else {
         Error("FitUtil::EvaluateChi2",
               "Execution policy unknown. Avalaible choices:\n ROOT::Fit::ExecutionPolicy::kSerial (default)\n "
               "ROOT::Fit::ExecutionPolicy::kMultithread (requires IMT)\n");
      }

      // Last SIMD vector of elements (if padding needed)
      if (data.Size() % vecSize != 0)
         vecCore::MaskedAssign(res, vecCore::Int2Mask<T>(data.Size() % vecSize),
                               res + mapFunction(data.Size() / vecSize));

      return vecCore::ReduceAdd(res);
   }

   static double EvalEffective(const IModelFunctionTempl<T> &, const BinData &, const double *, unsigned int &)
   {
      Error("FitUtil::EvaluateChi2<T>::EvalEffective",
            "The vectorized evaluation of the Chi2 with coordinate errors is still not supported");
      return -1.;
   }

   static void EvalGradient(const IModelFunctionTempl<T> &f, const BinData &data, const double *p, double *grad,
                            unsigned int &nPoints,
                            ROOT::Fit::ExecutionPolicy executionPolicy = ROOT::Fit::ExecutionPolicy::kSerial,
                            unsigned nChunks = 0)
   {
      // evaluate the gradient of the chi2 function
      // this function is used when the model function knows how to calculate the derivative and we can
      // avoid that the minimizer re-computes them
      //
      // case of chi2 effective (errors on coordinate) is not supported

      if (data.HaveCoordErrors()) {
         MATH_ERROR_MSG("FitUtil::EvaluateChi2Gradient",
                        "Error on the coordinates are not used in calculating Chi2 gradient");
         return; // it will assert otherwise later in GetPoint
      }

      const IGradModelFunctionTempl<T> *fg = dynamic_cast<const IGradModelFunctionTempl<T> *>(&f);
      assert(fg != nullptr); // must be called by a gradient function

      const IGradModelFunctionTempl<T> &func = *fg;

      const DataOptions &fitOpt = data.Opt();
      if (fitOpt.fBinVolume || fitOpt.fIntegral || fitOpt.fExpErrors)
         Error("FitUtil::EvaluateChi2Gradient", "The vectorized implementation doesn't support Integrals,"
                                                "BinVolume or ExpErrors\n. Aborting operation.");

      unsigned int npar = func.NPar();
      auto vecSize = vecCore::VectorSize<T>();
      unsigned initialNPoints = data.Size();
      unsigned numVectors = initialNPoints / vecSize;

      // numVectors + 1 because of the padded data (call to mapFunction with i = numVectors after the main loop)
      std::vector<vecCore::Mask<T>> validPointsMasks(numVectors + 1);

      auto mapFunction = [&](const unsigned int i) {
         // set all vector values to zero
         std::vector<T> gradFunc(npar);
         std::vector<T> pointContributionVec(npar);

         T x1, y, invError;

         vecCore::Load<T>(x1, data.GetCoordComponent(i * vecSize, 0));
         vecCore::Load<T>(y, data.ValuePtr(i * vecSize));
         const auto invErrorPtr = data.ErrorPtr(i * vecSize);

         if (invErrorPtr == nullptr)
            invError = 1;
         else
            vecCore::Load<T>(invError, invErrorPtr);

         // TODO: Check error options and invert if needed

         T fval = 0;

         const T *x = nullptr;

         unsigned int ndim = data.NDim();
         // need to declare vector outside if statement
         // otherwise pointer will be invalid
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

         validPointsMasks[i] = isFinite(fval);
         if (vecCore::MaskEmpty(validPointsMasks[i])) {
            // Return a zero contribution to all partial derivatives on behalf of the current points
            return pointContributionVec;
         }

         // loop on the parameters
         for (unsigned int ipar = 0; ipar < npar; ++ipar) {
            // avoid singularity in the function (infinity and nan ) in the chi2 sum
            // eventually add possibility of excluding some points (like singularity)
            validPointsMasks[i] = isFinite(gradFunc[ipar]);

            if (vecCore::MaskEmpty(validPointsMasks[i])) {
               break; // exit loop on parameters
            }

            // calculate derivative point contribution (only for valid points)
            vecCore::MaskedAssign(pointContributionVec[ipar], validPointsMasks[i],
                                  -2.0 * (y - fval) * invError * invError * gradFunc[ipar]);
         }

         return pointContributionVec;
      };

      // Reduce the set of vectors by summing its equally-indexed components
      auto redFunction = [&](const std::vector<std::vector<T>> &partialResults) {
         std::vector<T> result(npar);

         for (auto const &pointContributionVec : partialResults) {
            for (unsigned int parameterIndex = 0; parameterIndex < npar; parameterIndex++)
               result[parameterIndex] += pointContributionVec[parameterIndex];
         }

         return result;
      };

      std::vector<T> gVec(npar);
      std::vector<double> g(npar);

#ifndef R__USE_IMT
      // to fix compiler warning
      (void)nChunks;

      // If IMT is disabled, force the execution policy to the serial case
      if (executionPolicy == ROOT::Fit::ExecutionPolicy::kMultithread) {
         Warning("FitUtil::EvaluateChi2Gradient",
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
         Error("FitUtil::EvaluateChi2Gradient",
               "Execution policy unknown. Avalaible choices:\n 0: Serial (default)\n 1: MultiThread (requires IMT)\n");
      }

      // Compute the contribution from the remaining points
      unsigned int remainingPoints = initialNPoints % vecSize;
      if (remainingPoints > 0) {
         auto remainingPointsContribution = mapFunction(numVectors);
         // Add the contribution from the valid remaining points and store the result in the output variable
         auto remainingMask = vecCore::Int2Mask<T>(remainingPoints);
         for (unsigned int param = 0; param < npar; param++) {
            vecCore::MaskedAssign(gVec[param], remainingMask, gVec[param] + remainingPointsContribution[param]);
         }
      }
      // reduce final gradient result from T to double
      for (unsigned int param = 0; param < npar; param++) {
         grad[param] = vecCore::ReduceAdd(gVec[param]);
      }

      // correct the number of points
      nPoints = initialNPoints;

      if (std::any_of(validPointsMasks.begin(), validPointsMasks.end(),
                      [](vecCore::Mask<T> validPoints) { return !vecCore::MaskFull(validPoints); })) {
         T nRejected_v(0);
         T zeros(0.);
         for (auto mask : validPointsMasks) {
            T partial(1.);
            vecCore::MaskedAssign(partial, mask, zeros);
            nRejected_v += partial;
         }

         auto nRejected = vecCore::ReduceAdd(nRejected_v);

         assert(nRejected <= initialNPoints);
         nPoints = initialNPoints - nRejected;

         if (nPoints < npar) {
            MATH_ERROR_MSG("FitUtil::EvaluateChi2Gradient",
                           "Too many points rejected for overflow in gradient calculation");
         }
      }
   }

   static double EvalResidual(const IModelFunctionTempl<T> &, const BinData &, const double *, unsigned int, double *)
   {
      Error("FitUtil::EvaluateChi2<T>::EvalResidual",
            "The vectorized evaluation of the Chi2 with the ith residual is still not supported");
      return -1.;
   }

       // Compute a mask to filter out infinite numbers and NaN values.
      // The argument rval is updated so infinite numbers and NaN values are replaced by
      // maximum finite values (preserving the original sign).
      static vecCore::Mask<T> isFinite(T &rval)
      {
         return rval > -vecCore::NumericLimits<T>::Max() && rval < vecCore::NumericLimits<T>::Max();
      }
};
template <>
struct Chi2<double> {
#endif

   static double Eval(const IModelFunction &func, const BinData &data, const double *p, unsigned int &nPoints,
                      ::ROOT::Fit::ExecutionPolicy executionPolicy, unsigned nChunks = 0)
   {
      // evaluate the chi2 given a  function reference, the data and returns the value and also in nPoints
      // the actual number of used points
      // normal chi2 using only error on values (from fitting histogram)
      // optionally the integral of function in the bin is used

      return FitUtil::EvaluateChi2(func, data, p, nPoints, executionPolicy, nChunks);
   }

   static double
   EvalEffective(const IModelFunctionTempl<double> &func, const BinData &data, const double *p, unsigned int &nPoints)
   {
      return FitUtil::EvaluateChi2Effective(func, data, p, nPoints);
   }
   static void EvalGradient(const IModelFunctionTempl<double> &func, const BinData &data, const double *p, double *g,
                            unsigned int &nPoints,
                            ::ROOT::Fit::ExecutionPolicy executionPolicy = ::ROOT::Fit::ExecutionPolicy::kSerial,
                            unsigned nChunks = 0)
   {
      FitUtil::EvaluateChi2Gradient(func, data, p, g, nPoints, executionPolicy, nChunks);
   }
   static double EvalResidual(const IModelFunctionTempl<double> &func, const BinData &data, const double *p,
                              unsigned int i, double *g = 0)
   {
      return FitUtil::EvaluateChi2Residual(func, data, p, i, g);
   }
};

} // end namespace FitUtil

} // end namespace Fit

} // end namespace ROOT

#endif /* ROOT_Fit_FitUtil */
