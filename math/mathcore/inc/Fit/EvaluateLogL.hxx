// @(#)root/mathcore:$Id$
// Author: L. Moneta Tue Nov 28 10:52:47 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Fit_EvaluateLogL
#define ROOT_Fit_EvaluateLogL

#include "Math/IParamFunctionfwd.h"
#include "Math/IParamFunction.h"

#ifdef R__USE_IMT
#include "ROOT/TThreadExecutor.hxx"
#endif
#include "ROOT/TSequentialExecutor.hxx"

#include "Fit/BinData.h"
#include "Fit/UnBinData.h"
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


  /**
      evaluate the LogL given a model function and the data at the point x.
      return also nPoints as the effective number of used points in the LogL evaluation
  */
  double EvaluateLogL(const IModelFunction &func, const UnBinData &data, const double *p, int iWeight, bool extended,
                      unsigned int &nPoints, ROOT::Fit::ExecutionPolicy executionPolicy, unsigned nChunks = 0);

  /**
      evaluate the LogL gradient given a model function and the data at the point x.
      return also nPoints as the effective number of used points in the LogL evaluation
  */
  void EvaluateLogLGradient(const IModelFunction &func, const UnBinData &data, const double *x, double *grad,
                            unsigned int &nPoints,
                            ROOT::Fit::ExecutionPolicy executionPolicy = ROOT::Fit::ExecutionPolicy::kSerial,
                            unsigned nChunks = 0);

    /**
      evaluate the pdf contribution to the LogL given a model function and the BinPoint data.
      If the pointer g is not null evaluate also the gradient of the pdf.
      If the function provides parameter derivatives they are used otherwise a simple derivative calculation
      is used
  */
  double EvaluatePdf(const IModelFunction &func, const UnBinData &data, const double *x, unsigned int ipoint, double *g = 0);
  
  #ifdef R__HAS_VECCORE
     template <class NotCompileIfScalarBackend = std::enable_if<!(std::is_same<double, ROOT::Double_v>::value)>>
     double EvaluatePdf(const IModelFunctionTempl<ROOT::Double_v> &func, const UnBinData &data, const double *p, unsigned int i, double *) {
        // evaluate the pdf contribution to the generic logl function in case of bin data
        // return actually the log of the pdf and its derivatives
        // func.SetParameters(p);
        const auto x = vecCore::FromPtr<ROOT::Double_v>(data.GetCoordComponent(i, 0));
        auto fval = func(&x, p);
        auto logPdf = ROOT::Math::Util::EvalLog(fval);
        return vecCore::Get<ROOT::Double_v>(logPdf, 0);
     }
  #endif

   template<class T>
   struct LogL {
#ifdef R__HAS_VECCORE

      static double Eval(const IModelFunctionTempl<T> &func, const UnBinData &data, const double *const p,
                             int iWeight, bool extended, unsigned int &nPoints,
                             ROOT::Fit::ExecutionPolicy executionPolicy, unsigned nChunks = 0)
      {
         // evaluate the LogLikelihood
         unsigned int n = data.Size();
         nPoints = data.Size();  // npoints

         //unsigned int nRejected = 0;
         bool normalizeFunc = false;

         // set parameters of the function to cache integral value
#ifdef USE_PARAMCACHE
         (const_cast<IModelFunctionTempl<T> &>(func)).SetParameters(p);
#endif

#ifdef R__USE_IMT
         // in case parameter needs to be propagated to user function use trick to set parameters by calling one time the function
         // this will be done in sequential mode and parameters can be set in a thread safe manner
         if (!normalizeFunc) {
            if (data.NDim() == 1) {
               T x;
               vecCore::Load<T>(x, data.GetCoordComponent(0, 0));
               func( &x, p);
            }
            else {
               std::vector<T> x(data.NDim());
               for (unsigned int j = 0; j < data.NDim(); ++j)
                  vecCore::Load<T>(x[j], data.GetCoordComponent(0, j));
               func( x.data(), p);
            }
         }
#endif

         // this is needed if function must be normalized
         double norm = 1.0;
         if (normalizeFunc) {
            // compute integral of the function
            std::vector<double> xmin(data.NDim());
            std::vector<double> xmax(data.NDim());
            IntegralEvaluator<IModelFunctionTempl<T>> igEval(func, p, true);
            // compute integral in the ranges where is defined
            if (data.Range().Size() > 0) {
               norm = 0;
               for (unsigned int ir = 0; ir < data.Range().Size(); ++ir) {
                  data.Range().GetRange(&xmin[0], &xmax[0], ir);
                  norm += igEval.Integral(xmin.data(), xmax.data());
               }
            } else {
               // use (-inf +inf)
               data.Range().GetRange(&xmin[0], &xmax[0]);
               // check if funcition is zero at +- inf
               T xmin_v, xmax_v;
               vecCore::Load<T>(xmin_v, xmin.data());
               vecCore::Load<T>(xmax_v, xmax.data());
               if (vecCore::ReduceAdd(func(&xmin_v, p)) != 0 || vecCore::ReduceAdd(func(&xmax_v, p)) != 0) {
                  MATH_ERROR_MSG("FitUtil::EvaluateLogLikelihood", "A range has not been set and the function is not zero at +/- inf");
                  return 0;
               }
               norm = igEval.Integral(&xmin[0], &xmax[0]);
            }
         }

         // needed to compute effective global weight in case of extended likelihood

         auto vecSize = vecCore::VectorSize<T>();
         unsigned int numVectors = n / vecSize;

         auto mapFunction = [ &, p](const unsigned i) {
            T W{};
            T W2{};
            T fval{};

            (void)p; /* avoid unused lambda capture warning if PARAMCACHE is disabled */

            T x1;
            vecCore::Load<T>(x1, data.GetCoordComponent(i * vecSize, 0));
            const T *x = nullptr;
            unsigned int ndim = data.NDim();
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

#ifdef USE_PARAMCACHE
            fval = func(x);
#else
            fval = func(x, p);
#endif

#ifdef DEBUG_FITUTIL
            if (i < 5 || (i > numVectors-5) ) {
               if (ndim == 1) std::cout << i << "  x " << x[0]  << " fval = " << fval; 
               else std::cout << i << "  x " << x[0] << " y " << x[1] << " fval = " << fval; 
            }
#endif

            if (normalizeFunc) fval = fval * (1 / norm);

            // function EvalLog protects against negative or too small values of fval
            auto logval =  ROOT::Math::Util::EvalLog(fval);
            if (iWeight > 0) {
               T weight{};
               if (data.WeightsPtr(i) == nullptr)
                  weight = 1;
               else
                  vecCore::Load<T>(weight, data.WeightsPtr(i*vecSize));
               logval *= weight;
               if (iWeight == 2) {
                  logval *= weight; // use square of weights in likelihood
                  if (!extended) {
                     // needed sum of weights and sum of weight square if likelkihood is extended
                     W  = weight;
                     W2 = weight * weight;
                  }
               }
            }
#ifdef DEBUG_FITUTIL
            if (i < 5 || (i > numVectors-5)  )  {
                 std::cout << "   " << fval << "  logfval " << logval << std::endl;
            }
#endif

            return LikelihoodAux<T>(logval, W, W2);
         };

         auto redFunction = [](const std::vector<LikelihoodAux<T>> &objs) {
            return std::accumulate(objs.begin(), objs.end(), LikelihoodAux<T>(),
            [](const LikelihoodAux<T> &l1, const LikelihoodAux<T> &l2) {
               return l1 + l2;
            });
         };

#ifndef R__USE_IMT
         (void)nChunks;

         // If IMT is disabled, force the execution policy to the serial case
         if (executionPolicy == ROOT::Fit::ExecutionPolicy::kMultithread) {
            Warning("FitUtil::EvaluateLogL", "Multithread execution policy requires IMT, which is disabled. Changing "
                                             "to ROOT::Fit::ExecutionPolicy::kSerial.");
            executionPolicy = ROOT::Fit::ExecutionPolicy::kSerial;
         }
#endif

         T logl_v{};
         T sumW_v{};
         T sumW2_v{};
         ROOT::Fit::FitUtil::LikelihoodAux<T> resArray;
         if (executionPolicy == ROOT::Fit::ExecutionPolicy::kSerial) {
            ROOT::TSequentialExecutor pool;
            resArray = pool.MapReduce(mapFunction, ROOT::TSeq<unsigned>(0, data.Size() / vecSize), redFunction);
#ifdef R__USE_IMT
         } else if (executionPolicy == ROOT::Fit::ExecutionPolicy::kMultithread) {
            ROOT::TThreadExecutor pool;
            auto chunks = nChunks != 0 ? nChunks : setAutomaticChunking( numVectors);
            resArray = pool.MapReduce(mapFunction, ROOT::TSeq<unsigned>(0, data.Size() / vecSize), redFunction, chunks);
#endif
         } else {
            Error("FitUtil::EvaluateLogL", "Execution policy unknown. Avalaible choices:\n ROOT::Fit::ExecutionPolicy::kSerial (default)\n ROOT::Fit::ExecutionPolicy::kMultithread (requires IMT)\n");
         }

         logl_v = resArray.logvalue;
         sumW_v = resArray.weight;
         sumW2_v = resArray.weight2;

         // Compute the contribution from the remaining points ( Last padded SIMD vector of elements )
         unsigned int remainingPoints = n % vecSize;
         if (remainingPoints > 0) {
            auto remainingPointsContribution = mapFunction(numVectors);
            // Add the contribution from the valid remaining points and store the result in the output variable
            auto remainingMask = ::vecCore::Int2Mask<T>(remainingPoints);
            vecCore::MaskedAssign(logl_v, remainingMask, logl_v + remainingPointsContribution.logvalue);
            vecCore::MaskedAssign(sumW_v, remainingMask, sumW_v + remainingPointsContribution.weight);
            vecCore::MaskedAssign(sumW2_v, remainingMask, sumW2_v + remainingPointsContribution.weight2);
         }


         //reduce vector type to double.
         double logl  = vecCore::ReduceAdd(logl_v);
         double sumW  = vecCore::ReduceAdd(sumW_v);
         double sumW2 = vecCore::ReduceAdd(sumW2_v);

         if (extended) {
            // add Poisson extended term
            double extendedTerm = 0; // extended term in likelihood
            double nuTot = 0;
            // nuTot is integral of function in the range
            // if function has been normalized integral has been already computed
            if (!normalizeFunc) {
               IntegralEvaluator<IModelFunctionTempl<T>> igEval(func, p, true);
               std::vector<double> xmin(data.NDim());
               std::vector<double> xmax(data.NDim());

               // compute integral in the ranges where is defined
               if (data.Range().Size() > 0) {
                  nuTot = 0;
                  for (unsigned int ir = 0; ir < data.Range().Size(); ++ir) {
                     data.Range().GetRange(&xmin[0], &xmax[0], ir);
                     nuTot += igEval.Integral(xmin.data(), xmax.data());
                  }
               } else {
                  // use (-inf +inf)
                  data.Range().GetRange(&xmin[0], &xmax[0]);
                  // check if funcition is zero at +- inf
                  T xmin_v, xmax_v;
                  vecCore::Load<T>(xmin_v, xmin.data());
                  vecCore::Load<T>(xmax_v, xmax.data());
                  if (vecCore::ReduceAdd(func(&xmin_v, p)) != 0 || vecCore::ReduceAdd(func(&xmax_v, p)) != 0) {
                     MATH_ERROR_MSG("FitUtil::EvaluateLogLikelihood", "A range has not been set and the function is not zero at +/- inf");
                     return 0;
                  }
                  nuTot = igEval.Integral(&xmin[0], &xmax[0]);
               }

               // force to be last parameter value
               //nutot = p[func.NDim()-1];
               if (iWeight != 2)
                  extendedTerm = - nuTot;  // no need to add in this case n log(nu) since is already computed before
               else {
                  // case use weight square in likelihood : compute total effective weight = sw2/sw
                  // ignore for the moment case when sumW is zero
                  extendedTerm = - (sumW2 / sumW) * nuTot;
               }

            } else {
               nuTot = norm;
               extendedTerm = - nuTot + double(n) *  ROOT::Math::Util::EvalLog(nuTot);
               // in case of weights need to use here sum of weights (to be done)
            }

            logl += extendedTerm;
         }

#ifdef DEBUG_FITUTIL
         std::cout << "Evaluated log L for parameters (";
         for (unsigned int ip = 0; ip < func.NPar(); ++ip)
            std::cout << " " << p[ip];
         std::cout << ")  nll = " << -logl << std::endl;
#endif
         
         return -logl;

      }

      static void EvalGradient(const IModelFunctionTempl<T> &f, const UnBinData &data, const double *p,
                                   double *grad, unsigned int &,
                                   ROOT::Fit::ExecutionPolicy executionPolicy = ROOT::Fit::ExecutionPolicy::kSerial,
                                   unsigned nChunks = 0)
      {
         // evaluate the gradient of the log likelihood function

         const IGradModelFunctionTempl<T> *fg = dynamic_cast<const IGradModelFunctionTempl<T> *>(&f);
         assert(fg != nullptr); // must be called by a grad function

         const IGradModelFunctionTempl<T> &func = *fg;


         unsigned int npar = func.NPar();
         auto vecSize = vecCore::VectorSize<T>();
         unsigned initialNPoints = data.Size();
         unsigned numVectors = initialNPoints / vecSize;

#ifdef DEBUG_FITUTIL 
         std::cout << "\n===> Evaluate Gradient for parameters ";
         for (unsigned int ip = 0; ip < npar; ++ip)
            std::cout << "  " << p[ip];
         std::cout << "\n";
#endif 

         (const_cast<IGradModelFunctionTempl<T> &>(func)).SetParameters(p);

         const T kdmax1 = vecCore::math::Sqrt(vecCore::NumericLimits<T>::Max());
         const T kdmax2 = vecCore::NumericLimits<T>::Max() / (4 * initialNPoints);

         auto mapFunction = [&](const unsigned int i) {
            std::vector<T> gradFunc(npar);
            std::vector<T> pointContributionVec(npar);

            T x1;
            vecCore::Load<T>(x1, data.GetCoordComponent(i * vecSize, 0));

            const T *x = nullptr;

            unsigned int ndim = data.NDim();
            std::vector<T> xc(ndim);
            if (ndim > 1) {
               xc.resize(ndim);
               xc[0] = x1;
               for (unsigned int j = 1; j < ndim; ++j)
                  vecCore::Load<T>(xc[j], data.GetCoordComponent(i * vecSize, j));
               x = xc.data();
            } else {
               x = &x1;
            }


            T fval = func(x, p);
            func.ParameterGradient(x, p, &gradFunc[0]);

#ifdef DEBUG_FITUTIL            
            if (i < 5 || (i > numVectors-5) ) {
               if (ndim > 1) std::cout << i << "  x " << x[0] << " y " << x[1] << " gradient " << gradFunc[0] << "  " << gradFunc[1] << "  " << gradFunc[3] << std::endl;
               else std::cout << i << "  x " << x[0] << " gradient " << gradFunc[0] << "  " << gradFunc[1] << "  " << gradFunc[3] << std::endl;
            }
#endif            

            vecCore::Mask<T> positiveValues = fval > 0;

            for (unsigned int kpar = 0; kpar < npar; ++kpar) {
               if (!vecCore::MaskEmpty(positiveValues))
                  vecCore::MaskedAssign<T>(pointContributionVec[kpar], positiveValues, -1. / fval * gradFunc[kpar]);

               vecCore::Mask<T> nonZeroGradientValues = !positiveValues && gradFunc[kpar] != 0;
               if (!vecCore::MaskEmpty(nonZeroGradientValues)) {
                  T gg = kdmax1 * gradFunc[kpar];
                  pointContributionVec[kpar] =
                     vecCore::Blend(nonZeroGradientValues && gg > 0, -vecCore::math::Min(gg, kdmax2),
                                    -vecCore::math::Max(gg, -kdmax2));
               }
               // if func derivative is zero term is also zero so do not add in g[kpar]
            }

            return pointContributionVec;
         };

         // Vertically reduce the set of vectors by summing its equally-indexed components
         auto redFunction = [&](const std::vector<std::vector<T>> &pointContributions) {
            std::vector<T> result(npar);

            for (auto const &pointContributionVec : pointContributions) {
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
            Warning("FitUtil::EvaluateLogLGradient",
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
            Error("FitUtil::EvaluateLogLGradient", "Execution policy unknown. Avalaible choices:\n "
                                                   "ROOT::Fit::ExecutionPolicy::kSerial (default)\n "
                                                   "ROOT::Fit::ExecutionPolicy::kMultithread (requires IMT)\n");
         }

         // Compute the contribution from the remaining points
         unsigned int remainingPoints = initialNPoints % vecSize;
         if (remainingPoints > 0) {
            auto remainingPointsContribution = mapFunction(numVectors);
            // Add the contribution from the valid remaining points and store the result in the output variable
            auto remainingMask = ::vecCore::Int2Mask<T>(initialNPoints % vecSize);
            for (unsigned int param = 0; param < npar; param++) {
               vecCore::MaskedAssign(gVec[param], remainingMask, gVec[param] + remainingPointsContribution[param]);
            }
         }
         // reduce final gradient result from T to double
         for (unsigned int param = 0; param < npar; param++) {
            grad[param] = vecCore::ReduceAdd(gVec[param]);
         }

#ifdef DEBUG_FITUTIL
         std::cout << "Final gradient ";
         for (unsigned int param = 0; param < npar; param++) {
            std::cout << "  " << grad[param];
         }
         std::cout << "\n";
#endif
      }
   };

   template<>
   struct LogL<double>{
#endif

      static double Eval(const IModelFunctionTempl<double> &func, const UnBinData &data, const double *p,
                             int iWeight, bool extended, unsigned int &nPoints,
                             ::ROOT::Fit::ExecutionPolicy executionPolicy, unsigned nChunks = 0)
      {
         return FitUtil::EvaluateLogL(func, data, p, iWeight, extended, nPoints, executionPolicy, nChunks);
      }

      static void EvalGradient(const IModelFunctionTempl<double> &func, const UnBinData &data, const double *p,
                                   double *g, unsigned int &nPoints,
                                   ::ROOT::Fit::ExecutionPolicy executionPolicy = ::ROOT::Fit::ExecutionPolicy::kSerial,
                                   unsigned nChunks = 0)
      {
         FitUtil::EvaluateLogLGradient(func, data, p, g, nPoints, executionPolicy, nChunks);
      }
   };

} // end namespace FitUtil

} // end namespace Fit

} // end namespace ROOT

#if defined (R__HAS_VECCORE) && defined(R__HAS_VC)
//Fixes alignment for structures of SIMD structures
Vc_DECLARE_ALLOCATOR(ROOT::Fit::FitUtil::LikelihoodAux<ROOT::Double_v>);
#endif

#endif /* ROOT_Fit_EvaluateLogL */
