// @(#)root/mathcore:$Id$
// Author: L. Moneta Tue Nov 28 10:52:47 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class FitUtil

#include "Fit/BinData.h"
#include "Fit/UnBinData.h"
#include "Fit/EvaluatePoissonLogL.hxx"
#include "Math/IFunctionfwd.h"
#include "Math/IParamFunction.h"
#include "Math/Integrator.h"
#include "Math/IntegratorMultiDim.h"
#include "Math/WrappedFunction.h"
#include "Math/OneDimFunctionAdapter.h"
#include "Math/RichardsonDerivator.h"

#include "Math/Error.h"
#include "Math/Util.h" // for safe log(x)

#include <limits>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <numeric>
//#include <memory>

#include "TROOT.h"

//#define DEBUG
#ifdef DEBUG
#define NSAMPLE 10
#include <iostream>
#endif

//  need to implement integral option

namespace ROOT {

namespace Fit {

namespace FitUtil {

//______________________________________________________________________________________________________
//
//  Poisson Log Likelihood functions
//_______________________________________________________________________________________________________


// evaluate the Poisson Log Likelihood
// for binned likelihood fits
// this is Sum ( f(x_i)  -  y_i * log( f (x_i) ) )
// add as well constant term for saturated model to make it like a Chi2/2
// by default is etended. If extended is false the fit is not extended and
// the global poisson term is removed (i.e is a binomial fit)
// (remember that in this case one needs to have a function with a fixed normalization
// like in a non extended unbinned fit)
//
// if use Weight use a weighted dataset
// iWeight = 1 ==> logL = Sum( w f(x_i) )
// case of iWeight==1 is actually identical to weight==0
// iWeight = 2 ==> logL = Sum( w*w * f(x_i) )
double PoissonLogL<double>::Eval(const IModelFunctionTempl<double> &func, const BinData &data, const double *p,
                                 int iWeight, bool extended, unsigned int &nPoints,
                                 ::ROOT::Fit::ExecutionPolicy executionPolicy, unsigned nChunks)
{
   unsigned int n = data.Size();

#ifdef USE_PARAMCACHE
   (const_cast<IModelFunction &>(func)).SetParameters(p);
#endif

   nPoints = data.Size(); // npoints

   // get fit option and check case of using integral of bins
   const DataOptions &fitOpt = data.Opt();
   bool useBinIntegral = fitOpt.fIntegral && data.HasBinEdges();
   bool useBinVolume = (fitOpt.fBinVolume && data.HasBinEdges());
   bool useW2 = (iWeight == 2);

   // normalize if needed by a reference volume value
   double wrefVolume = 1.0;
   if (useBinVolume) {
      if (fitOpt.fNormBinVolume)
         wrefVolume /= data.RefVolume();
   }

#ifdef DEBUG
   std::cout << "Evaluate PoissonLogL for params = [ ";
   for (unsigned int j = 0; j < func.NPar(); ++j)
      std::cout << p[j] << " , ";
   std::cout << "]  - data size = " << n << " useBinIntegral " << useBinIntegral << " useBinVolume " << useBinVolume
             << " useW2 " << useW2 << " wrefVolume = " << wrefVolume << std::endl;
#endif

#ifdef USE_PARAMCACHE
   IntegralEvaluator<> igEval(func, 0, useBinIntegral);
#else
   IntegralEvaluator<> igEval(func, p, useBinIntegral);
#endif

   auto mapFunction = [&](const unsigned i) {
      auto x1 = data.GetCoordComponent(i, 0);
      auto y = *data.ValuePtr(i);

      const double *x = nullptr;
      std::vector<double> xc;
      double fval = 0;
      double binVolume = 1.0;

      if (useBinVolume) {
         unsigned int ndim = data.NDim();
         const double *x2 = data.BinUpEdge(i);
         xc.resize(data.NDim());
         for (unsigned int j = 0; j < ndim; ++j) {
            auto xx = *data.GetCoordComponent(i, j);
            binVolume *= std::abs(x2[j] - xx);
            xc[j] = 0.5 * (x2[j] + xx);
         }
         x = xc.data();
         // normalize the bin volume using a reference value
         binVolume *= wrefVolume;
      } else if (data.NDim() > 1) {
         xc.resize(data.NDim());
         xc[0] = *x1;
         for (unsigned int j = 1; j < data.NDim(); ++j) {
            xc[j] = *data.GetCoordComponent(i, j);
         }
         x = xc.data();
      } else {
         x = x1;
      }

      if (!useBinIntegral) {
#ifdef USE_PARAMCACHE
         fval = func(x);
#else
         fval = func(x, p);
#endif
      } else {
         // calculate integral (normalized by bin volume)
         // need to set function and parameters here in case loop is parallelized
         fval = igEval(x, data.BinUpEdge(i));
      }
      if (useBinVolume)
         fval *= binVolume;

#ifdef DEBUG
      int NSAMPLE = 100;
      if (i % NSAMPLE == 0) {
         std::cout << "evt " << i << " x = [ ";
         for (unsigned int j = 0; j < func.NDim(); ++j)
            std::cout << x[j] << " , ";
         std::cout << "]  ";
         if (fitOpt.fIntegral) {
            std::cout << "x2 = [ ";
            for (unsigned int j = 0; j < func.NDim(); ++j)
               std::cout << data.BinUpEdge(i)[j] << " , ";
            std::cout << "] ";
         }
         std::cout << "  y = " << y << " fval = " << fval << std::endl;
      }
#endif

      // EvalLog protects against 0 values of fval but don't want to add in the -log sum
      // negative values of fval
      fval = std::max(fval, 0.0);

      double nloglike = 0; // negative loglikelihood
      if (useW2) {
         // apply weight correction . Effective weight is error^2/ y
         // and expected events in bins is fval/weight
         // can apply correction only when y is not zero otherwise weight is undefined
         // (in case of weighted likelihood I don't care about the constant term due to
         // the saturated model)

         // use for the empty bins the global weight
         double weight = 1.0;
         if (y != 0) {
            double error = data.Error(i);
            weight = (error * error) / y; // this is the bin effective weight
            nloglike += weight * y * (ROOT::Math::Util::EvalLog(y) - ROOT::Math::Util::EvalLog(fval));
         } else {
            // for empty bin use the average weight  computed from the total data weight
            weight = data.SumOfError2() / data.SumOfContent();
         }
         if (extended) {
            nloglike += weight * (fval - y);
         }

      } else {
         // standard case no weights or iWeight=1
         // this is needed for Poisson likelihood (which are extened and not for multinomial)
         // the formula below  include constant term due to likelihood of saturated model (f(x) = y)
         // (same formula as in Baker-Cousins paper, page 439 except a factor of 2
         if (extended)
            nloglike = fval - y;

         if (y > 0) {
            nloglike += y * (ROOT::Math::Util::EvalLog(y) - ROOT::Math::Util::EvalLog(fval));
         }
      }
      return nloglike;
   };

#ifdef R__USE_IMT
   auto redFunction = [](const std::vector<double> &objs) {
      return std::accumulate(objs.begin(), objs.end(), double{});
   };
#else
   (void)nChunks;

   // If IMT is disabled, force the execution policy to the serial case
   if (executionPolicy == ROOT::Fit::ExecutionPolicy::kMultithread) {
      Warning("FitUtil::EvaluatePoissonLogL", "Multithread execution policy requires IMT, which is disabled. Changing "
                                              "to ROOT::Fit::ExecutionPolicy::kSerial.");
      executionPolicy = ROOT::Fit::ExecutionPolicy::kSerial;
   }
#endif

   double res{};
   if (executionPolicy == ROOT::Fit::ExecutionPolicy::kSerial) {
      for (unsigned int i = 0; i < n; ++i) {
         res += mapFunction(i);
      }
#ifdef R__USE_IMT
   } else if (executionPolicy == ROOT::Fit::ExecutionPolicy::kMultithread) {
      ROOT::TThreadExecutor pool;
      auto chunks = nChunks != 0 ? nChunks : setAutomaticChunking(data.Size());
      res = pool.MapReduce(mapFunction, ROOT::TSeq<unsigned>(0, n), redFunction, chunks);
#endif
      //   } else if(executionPolicy == ROOT::Fit::kMultitProcess){
      // ROOT::TProcessExecutor pool;
      // res = pool.MapReduce(mapFunction, ROOT::TSeq<unsigned>(0, n), redFunction);
   } else {
      Error("FitUtil::EvaluatePoissonLogL", "Execution policy unknown. Avalaible choices:\n "
                                            "ROOT::Fit::ExecutionPolicy::kSerial (default)\n "
                                            "ROOT::Fit::ExecutionPolicy::kMultithread (requires IMT)\n");
   }

#ifdef DEBUG
   std::cout << "Loglikelihood  = " << res << std::endl;
#endif

   return res;
}

// Evaluate the pdf (Poisson) contribution to the logl (return actually log of pdf)
// and its gradient
double PoissonLogL<double>::EvalBinPdf(const IModelFunctionTempl<double> &func, const BinData &data, const double *p,
                                       unsigned int i, double *g)
{
   double y = 0;
   const double *x1 = data.GetPoint(i, y);

   const DataOptions &fitOpt = data.Opt();
   bool useBinIntegral = fitOpt.fIntegral && data.HasBinEdges();
   bool useBinVolume = (fitOpt.fBinVolume && data.HasBinEdges());

   IntegralEvaluator<> igEval(func, p, useBinIntegral);
   double fval = 0;
   const double *x2 = 0;
   // calculate the bin volume
   double binVolume = 1;
   std::vector<double> xc;
   if (useBinVolume) {
      unsigned int ndim = data.NDim();
      xc.resize(ndim);
      x2 = data.BinUpEdge(i);
      for (unsigned int j = 0; j < ndim; ++j) {
         binVolume *= std::abs(x2[j] - x1[j]);
         xc[j] = 0.5 * (x2[j] + x1[j]);
      }
      // normalize the bin volume using a reference value
      binVolume /= data.RefVolume();
   }

   const double *x = (useBinVolume) ? &xc.front() : x1;

   if (!useBinIntegral) {
      fval = func(x, p);
   } else {
      // calculate integral normalized (divided by bin volume)
      x2 = data.BinUpEdge(i);
      fval = igEval(x1, x2);
   }
   if (useBinVolume)
      fval *= binVolume;

   // logPdf for Poisson: ignore constant term depending on N
   fval = std::max(fval, 0.0); // avoid negative or too small values
   double logPdf = -fval;
   if (y > 0.0) {
      // include also constants due to saturate model (see Baker-Cousins paper)
      logPdf += y * ROOT::Math::Util::EvalLog(fval / y) + y;
   }
   // need to return the pdf contribution (not the log)

   // double pdfval =  std::exp(logPdf);

   // if (g == 0) return pdfval;
   if (g == 0)
      return logPdf;

   unsigned int npar = func.NPar();
   const IGradModelFunction *gfunc = dynamic_cast<const IGradModelFunction *>(&func);

   // gradient  calculation
   if (gfunc != 0) {
      // case function provides gradient
      if (!useBinIntegral)
         gfunc->ParameterGradient(x, p, g);
      else {
         // needs to calculate the integral for each partial derivative
         CalculateGradientIntegral(*gfunc, x1, x2, p, g);
      }

   } else {
      SimpleGradientCalculator gc(func.NPar(), func);
      if (!useBinIntegral)
         gc.ParameterGradient(x, p, fval, g);
      else {
         // needs to calculate the integral for each partial derivative
         CalculateGradientIntegral(gc, x1, x2, p, g);
      }
   }
   // correct g[] do be derivative of poisson term
   for (unsigned int k = 0; k < npar; ++k) {
      // apply bin volume correction
      if (useBinVolume)
         g[k] *= binVolume;

      // correct for Poisson term
      if (fval > 0)
         g[k] *= (y / fval - 1.); //* pdfval;
      else if (y > 0) {
         const double kdmax1 = std::sqrt(std::numeric_limits<double>::max());
         g[k] *= kdmax1;
      } else // y == 0 cannot have  negative y
         g[k] *= -1;
   }

#ifdef DEBUG
   std::cout << "x = " << x[0] << " logPdf = " << logPdf << " grad";
   for (unsigned int ipar = 0; ipar < npar; ++ipar)
      std::cout << g[ipar] << "\t";
   std::cout << std::endl;
#endif

   //   return pdfval;
   return logPdf;
}

// Evaluate the gradient of the Poisson log likelihood function
void PoissonLogL<double>::EvalGradient(const IModelFunctionTempl<double> &f, const BinData &data, const double *p,
                                       double *grad, unsigned int &, ::ROOT::Fit::ExecutionPolicy executionPolicy,
                                       unsigned nChunks)
{
   const IGradModelFunction *fg = dynamic_cast<const IGradModelFunction *>(&f);
   assert(fg != nullptr); // must be called by a grad function

   const IGradModelFunction &func = *fg;

#ifdef USE_PARAMCACHE
   (const_cast<IGradModelFunction &>(func)).SetParameters(p);
#endif

   const DataOptions &fitOpt = data.Opt();
   bool useBinIntegral = fitOpt.fIntegral && data.HasBinEdges();
   bool useBinVolume = (fitOpt.fBinVolume && data.HasBinEdges());

   double wrefVolume = 1.0;
   if (useBinVolume && fitOpt.fNormBinVolume)
      wrefVolume /= data.RefVolume();

   IntegralEvaluator<> igEval(func, p, useBinIntegral);

   unsigned int npar = func.NPar();
   unsigned initialNPoints = data.Size();

   auto mapFunction = [&](const unsigned int i) {
      // set all vector values to zero
      std::vector<double> gradFunc(npar);
      std::vector<double> pointContribution(npar);

      const auto x1 = data.GetCoordComponent(i, 0);
      const auto y = data.Value(i);
      auto invError = data.Error(i);

      invError = (invError != 0.0) ? 1.0 / invError : 1;

      double fval = 0;

      const double *x = nullptr;
      std::vector<double> xc;

      unsigned ndim = data.NDim();
      double binVolume = 1.0;
      if (useBinVolume) {
         const double *x2 = data.BinUpEdge(i);

         xc.resize(ndim);
         for (unsigned int j = 0; j < ndim; ++j) {
            auto x1_j = *data.GetCoordComponent(i, j);
            binVolume *= std::abs(x2[j] - x1_j);
            xc[j] = 0.5 * (x2[j] + x1_j);
         }

         x = xc.data();

         // normalize the bin volume using a reference value
         binVolume *= wrefVolume;
      } else if (ndim > 1) {
         xc.resize(ndim);
         xc[0] = *x1;
         for (unsigned int j = 1; j < ndim; ++j)
            xc[j] = *data.GetCoordComponent(i, j);
         x = xc.data();
      } else {
         x = x1;
      }

      if (!useBinIntegral) {
         fval = func(x, p);
         func.ParameterGradient(x, p, &gradFunc[0]);
      } else {
         // calculate integral (normalized by bin volume)
         // need to set function and parameters here in case loop is parallelized
         auto x2 = data.BinUpEdge(i);
         fval = igEval(x, x2);
         CalculateGradientIntegral(func, x, x2, p, &gradFunc[0]);
      }
      if (useBinVolume)
         fval *= binVolume;

#ifdef DEBUG
      {
         R__LOCKGUARD(gROOTMutex);
         if (i < 5 || (i > data.Size() - 5)) {
            if (data.NDim() > 1)
               std::cout << i << "  x " << x[0] << " y " << x[1] << " func " << fval << " gradient " << gradFunc[0]
                         << "  " << gradFunc[1] << "  " << gradFunc[3] << std::endl;
            else
               std::cout << i << "  x " << x[0] << " gradient " << gradFunc[0] << "  " << gradFunc[1] << "  "
                         << gradFunc[3] << std::endl;
         }
      }
#endif

      // correct the gradient
      for (unsigned int ipar = 0; ipar < npar; ++ipar) {

         // correct gradient for bin volumes
         if (useBinVolume)
            gradFunc[ipar] *= binVolume;

         // df/dp * (1.  - y/f )
         if (fval > 0)
            pointContribution[ipar] = gradFunc[ipar] * (1. - y / fval);
         else if (gradFunc[ipar] != 0) {
            const double kdmax1 = std::sqrt(std::numeric_limits<double>::max());
            const double kdmax2 = std::numeric_limits<double>::max() / (4 * initialNPoints);
            double gg = kdmax1 * gradFunc[ipar];
            if (gg > 0)
               gg = std::min(gg, kdmax2);
            else
               gg = std::max(gg, -kdmax2);
            pointContribution[ipar] = -gg;
         }
      }

      return pointContribution;
   };

   // Vertically reduce the set of vectors by summing its equally-indexed components
   auto redFunction = [&](const std::vector<std::vector<double>> &pointContributions) {
      std::vector<double> result(npar);

      for (auto const &pointContribution : pointContributions) {
         for (unsigned int parameterIndex = 0; parameterIndex < npar; parameterIndex++)
            result[parameterIndex] += pointContribution[parameterIndex];
      }

      return result;
   };

   std::vector<double> g(npar);

#ifndef R__USE_IMT
   // If IMT is disabled, force the execution policy to the serial case
   if (executionPolicy == ROOT::Fit::ExecutionPolicy::kMultithread) {
      Warning("FitUtil::EvaluatePoissonLogLGradient",
              "Multithread execution policy requires IMT, which is disabled. Changing "
              "to ROOT::Fit::ExecutionPolicy::kSerial.");
      executionPolicy = ROOT::Fit::ExecutionPolicy::kSerial;
   }
#endif

   if (executionPolicy == ROOT::Fit::ExecutionPolicy::kSerial) {
      std::vector<std::vector<double>> allGradients(initialNPoints);
      for (unsigned int i = 0; i < initialNPoints; ++i) {
         allGradients[i] = mapFunction(i);
      }
      g = redFunction(allGradients);
   }
#ifdef R__USE_IMT
   else if (executionPolicy == ROOT::Fit::ExecutionPolicy::kMultithread) {
      ROOT::TThreadExecutor pool;
      auto chunks = nChunks != 0 ? nChunks : setAutomaticChunking(initialNPoints);
      g = pool.MapReduce(mapFunction, ROOT::TSeq<unsigned>(0, initialNPoints), redFunction, chunks);
   }
#endif
   else {
      Error("FitUtil::EvaluatePoissonLogLGradient",
            "Execution policy unknown. Avalaible choices:\n 0: Serial (default)\n 1: MultiThread (requires IMT)\n");
   }

#ifndef R__USE_IMT
   // to fix compiler warning
   (void)nChunks;
#endif

   // copy result
   std::copy(g.begin(), g.end(), grad);

#ifdef DEBUG
   std::cout << "***** Final gradient : ";
   for (unsigned int ii = 0; ii < npar; ++ii)
      std::cout << grad[ii] << "   ";
   std::cout << "\n";
#endif
}

} // end namespace FitUtil

} // end namespace Fit

} // end namespace ROOT
