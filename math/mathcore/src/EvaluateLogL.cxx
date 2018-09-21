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
#include "Fit/EvaluateLogL.hxx"
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
//  Log Likelihood functions
//_______________________________________________________________________________________________________

// utility function used by the likelihoods

double EvaluatePdf(const IModelFunction &func, const UnBinData &data, const double *p, unsigned int i, double *g)
{
   // evaluate the pdf contribution to the generic logl function in case of bin data
   // return actually the log of the pdf and its derivatives

   // func.SetParameters(p);

   const double *x = data.Coords(i);
   double fval = func(x, p);
   double logPdf = ROOT::Math::Util::EvalLog(fval);
   // return
   if (g == 0)
      return logPdf;

   const IGradModelFunction *gfunc = dynamic_cast<const IGradModelFunction *>(&func);

   // gradient  calculation
   if (gfunc != 0) {
      // case function provides gradient
      gfunc->ParameterGradient(x, p, g);
   } else {
      // estimate gradieant numerically with simple 2 point rule
      // should probably calculate gradient of log(pdf) is more stable numerically
      SimpleGradientCalculator gc(func.NPar(), func);
      gc.ParameterGradient(x, p, fval, g);
   }
   // divide gradient by function value since returning the logs
   for (unsigned int ipar = 0; ipar < func.NPar(); ++ipar) {
      g[ipar] /= fval; // this should be checked against infinities
   }

#ifdef DEBUG
   std::cout << x[i] << "\t";
   std::cout << "\tpar = [ " << func.NPar() << " ] =  ";
   for (unsigned int ipar = 0; ipar < func.NPar(); ++ipar)
      std::cout << p[ipar] << "\t";
   std::cout << "\tfval = " << fval;
   std::cout << "\tgrad = [ ";
   for (unsigned int ipar = 0; ipar < func.NPar(); ++ipar)
      std::cout << g[ipar] << "\t";
   std::cout << " ] " << std::endl;
#endif

   return logPdf;
}

double LogL<double>::Eval(const IModelFunctionTempl<double> &func, const UnBinData &data, const double *p, int iWeight,
                          bool extended, unsigned int &nPoints, ::ROOT::Fit::ExecutionPolicy executionPolicy,
                          unsigned nChunks)
{
   // evaluate the LogLikelihood

   unsigned int n = data.Size();

   // unsigned int nRejected = 0;

   bool normalizeFunc = false;

// set parameters of the function to cache integral value
#ifdef USE_PARAMCACHE
   (const_cast<IModelFunctionTempl<double> &>(func)).SetParameters(p);
#endif

   nPoints = data.Size(); // npoints

#ifdef R__USE_IMT
   // in case parameter needs to be propagated to user function use trick to set parameters by calling one time the
   // function
   // this will be done in sequential mode and parameters can be set in a thread safe manner
   if (!normalizeFunc) {
      if (data.NDim() == 1) {
         const double *x = data.GetCoordComponent(0, 0);
         func(x, p);
      } else {
         std::vector<double> x(data.NDim());
         for (unsigned int j = 0; j < data.NDim(); ++j)
            x[j] = *data.GetCoordComponent(0, j);
         func(x.data(), p);
      }
   }
#endif

   double norm = 1.0;
   if (normalizeFunc) {
      // compute integral of the function
      std::vector<double> xmin(data.NDim());
      std::vector<double> xmax(data.NDim());
      IntegralEvaluator<> igEval(func, p, true);
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
         if (func(xmin.data(), p) != 0 || func(xmax.data(), p) != 0) {
            MATH_ERROR_MSG("FitUtil::LogL<double>::Eval",
                           "A range has not been set and the function is not zero at +/- inf");
            return 0;
         }
         norm = igEval.Integral(&xmin[0], &xmax[0]);
      }
   }

   // needed to compue effective global weight in case of extended likelihood

   auto mapFunction = [&](const unsigned i) {
      double W = 0;
      double W2 = 0;
      double fval = 0;

      if (data.NDim() > 1) {
         std::vector<double> x(data.NDim());
         for (unsigned int j = 0; j < data.NDim(); ++j)
            x[j] = *data.GetCoordComponent(i, j);
#ifdef USE_PARAMCACHE
         fval = func(x.data());
#else
         fval = func(x.data(), p);
#endif

         // one -dim case
      } else {
         const auto x = data.GetCoordComponent(i, 0);
#ifdef USE_PARAMCACHE
         fval = func(x);
#else
         fval = func(x, p);
#endif
      }

      if (normalizeFunc)
         fval = fval * (1 / norm);

      // function EvalLog protects against negative or too small values of fval
      double logval = ROOT::Math::Util::EvalLog(fval);
      if (iWeight > 0) {
         double weight = data.Weight(i);
         logval *= weight;
         if (iWeight == 2) {
            logval *= weight; // use square of weights in likelihood
            if (!extended) {
               // needed sum of weights and sum of weight square if likelkihood is extended
               W = weight;
               W2 = weight * weight;
            }
         }
      }
      return LikelihoodAux<double>(logval, W, W2);
   };

#ifdef R__USE_IMT
   auto redFunction = [](const std::vector<LikelihoodAux<double>> &objs) {
      return std::accumulate(objs.begin(), objs.end(), LikelihoodAux<double>(0.0, 0.0, 0.0),
                             [](const LikelihoodAux<double> &l1, const LikelihoodAux<double> &l2) { return l1 + l2; });
   };

#else
   (void)nChunks;

   // If IMT is disabled, force the execution policy to the serial case
   if (executionPolicy == ROOT::Fit::ExecutionPolicy::kMultithread) {
      Warning("FitUtil::LogL<double>::Eval", "Multithread execution policy requires IMT, which is disabled. Changing "
                                             "to ROOT::Fit::ExecutionPolicy::kSerial.");
      executionPolicy = ROOT::Fit::ExecutionPolicy::kSerial;
   }
#endif

   double logl{};
   double sumW{};
   double sumW2{};
   if (executionPolicy == ROOT::Fit::ExecutionPolicy::kSerial) {
      for (unsigned int i = 0; i < n; ++i) {
         auto resArray = mapFunction(i);
         logl += resArray.logvalue;
         sumW += resArray.weight;
         sumW2 += resArray.weight2;
      }
#ifdef R__USE_IMT
   } else if (executionPolicy == ROOT::Fit::ExecutionPolicy::kMultithread) {
      ROOT::TThreadExecutor pool;
      auto chunks = nChunks != 0 ? nChunks : setAutomaticChunking(data.Size());
      auto resArray = pool.MapReduce(mapFunction, ROOT::TSeq<unsigned>(0, n), redFunction, chunks);
      logl = resArray.logvalue;
      sumW = resArray.weight;
      sumW2 = resArray.weight2;
#endif
   } else {
      Error("FitUtil::LogL<double>::Eval", "Execution policy unknown. Avalaible choices:\n "
                                           "ROOT::Fit::ExecutionPolicy::kSerial (default)\n "
                                           "ROOT::Fit::ExecutionPolicy::kMultithread (requires IMT)\n");
   }

   if (extended) {
      // add Poisson extended term
      double extendedTerm = 0; // extended term in likelihood
      double nuTot = 0;
      // nuTot is integral of function in the range
      // if function has been normalized integral has been already computed
      if (!normalizeFunc) {
         IntegralEvaluator<> igEval(func, p, true);
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
            if (func(xmin.data(), p) != 0 || func(xmax.data(), p) != 0) {
               MATH_ERROR_MSG("FitUtil::LogL<double>::Eval",
                              "A range has not been set and the function is not zero at +/- inf");
               return 0;
            }
            nuTot = igEval.Integral(&xmin[0], &xmax[0]);
         }

         // force to be last parameter value
         // nutot = p[func.NDim()-1];
         if (iWeight != 2)
            extendedTerm = -nuTot; // no need to add in this case n log(nu) since is already computed before
         else {
            // case use weight square in likelihood : compute total effective weight = sw2/sw
            // ignore for the moment case when sumW is zero
            extendedTerm = -(sumW2 / sumW) * nuTot;
         }

      } else {
         nuTot = norm;
         extendedTerm = -nuTot + double(n) * ROOT::Math::Util::EvalLog(nuTot);
         // in case of weights need to use here sum of weights (to be done)
      }
      logl += extendedTerm;
   }

#ifdef DEBUG
   std::cout << "Evaluated log L for parameters (";
   for (unsigned int ip = 0; ip < func.NPar(); ++ip)
      std::cout << " " << p[ip];
   std::cout << ")  fval = " << -logl << std::endl;
#endif

   return -logl;
}

void LogL<double>::EvalGradient(const IModelFunctionTempl<double> &f, const UnBinData &data, const double *p,
                                double *grad, unsigned int &, ::ROOT::Fit::ExecutionPolicy executionPolicy,
                                unsigned nChunks)
{
   // evaluate the gradient of the log likelihood function

   const IGradModelFunction *fg = dynamic_cast<const IGradModelFunction *>(&f);
   assert(fg != nullptr); // must be called by a grad function

   const IGradModelFunction &func = *fg;

   unsigned int npar = func.NPar();
   unsigned initialNPoints = data.Size();

   (const_cast<IGradModelFunction &>(func)).SetParameters(p);

#ifdef DEBUG
   std::cout << "\n===> Evaluate Gradient for parameters ";
   for (unsigned int ip = 0; ip < npar; ++ip)
      std::cout << "  " << p[ip];
   std::cout << "\n";
#endif

   const double kdmax1 = std::sqrt(std::numeric_limits<double>::max());
   const double kdmax2 = std::numeric_limits<double>::max() / (4 * initialNPoints);

   auto mapFunction = [&](const unsigned int i) {
      std::vector<double> gradFunc(npar);
      std::vector<double> pointContribution(npar);

      const double *x = nullptr;
      std::vector<double> xc;
      if (data.NDim() > 1) {
         xc.resize(data.NDim());
         for (unsigned int j = 0; j < data.NDim(); ++j)
            xc[j] = *data.GetCoordComponent(i, j);
         x = xc.data();
      } else {
         x = data.GetCoordComponent(i, 0);
      }

      double fval = func(x, p);
      func.ParameterGradient(x, p, &gradFunc[0]);

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

      for (unsigned int kpar = 0; kpar < npar; ++kpar) {
         if (fval > 0)
            pointContribution[kpar] = -1. / fval * gradFunc[kpar];
         else if (gradFunc[kpar] != 0) {
            double gg = kdmax1 * gradFunc[kpar];
            if (gg > 0)
               gg = std::min(gg, kdmax2);
            else
               gg = std::max(gg, -kdmax2);
            pointContribution[kpar] = -gg;
         }
         // if func derivative is zero term is also zero so do not add in g[kpar]
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
      Warning("FitUtil::LogL<double>::EvalGradient",
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
      Error("FitUtil::LogL<double>::EvalGradient", "Execution policy unknown. Avalaible choices:\n "
                                                   "ROOT::Fit::ExecutionPolicy::kSerial (default)\n "
                                                   "ROOT::Fit::ExecutionPolicy::kMultithread (requires IMT)\n");
   }

#ifndef R__USE_IMT
   // to fix compiler warning
   (void)nChunks;
#endif

   // copy result
   std::copy(g.begin(), g.end(), grad);

#ifdef DEBUG
   std::cout << "FitUtil.cxx : Final gradient ";
   for (unsigned int param = 0; param < npar; param++) {
      std::cout << "  " << grad[param];
   }
   std::cout << "\n";
#endif
}

} // end namespace FitUtil

} // end namespace Fit

} // end namespace ROOT
