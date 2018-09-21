// @(#)root/mathcore:$Id$
// Author: L. Moneta Tue Nov 28 10:52:47 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class FitUtil

#include "Fit/EvaluateChi2.hxx"

#include "Fit/BinData.h"

#include "Math/IFunctionfwd.h"
#include "Math/IParamFunction.h"
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

#include "TROOT.h"

//#define DEBUG
#ifdef DEBUG
#define NSAMPLE 10
#include <iostream>
#endif

namespace ROOT {

namespace Fit {

namespace FitUtil {


double Chi2<double>::Eval(const IModelFunction &func, const BinData &data, const double *p, unsigned int &,
                          ::ROOT::Fit::ExecutionPolicy executionPolicy, unsigned nChunks)
{
   // evaluate the chi2 given a  function reference  , the data and returns the value and also in nPoints
   // the actual number of used points
   // normal chi2 using only error on values (from fitting histogram)
   // optionally the integral of function in the bin is used

   unsigned int n = data.Size();

// set parameters of the function to cache integral value
#ifdef USE_PARAMCACHE
   (const_cast<IModelFunction &>(func)).SetParameters(p);
#endif
   // do not cache parameter values (it is not thread safe)
   // func.SetParameters(p);

   // get fit option and check case if using integral of bins
   const DataOptions &fitOpt = data.Opt();
   bool useBinIntegral = fitOpt.fIntegral && data.HasBinEdges();
   bool useBinVolume = (fitOpt.fBinVolume && data.HasBinEdges());
   bool useExpErrors = (fitOpt.fExpErrors);
   bool isWeighted = data.IsWeighted();

#ifdef DEBUG
   std::cout << "\n\nFit data size = " << n << std::endl;
   std::cout << "evaluate chi2 using function " << &func << "  " << p << std::endl;
   std::cout << "use empty bins  " << fitOpt.fUseEmpty << std::endl;
   std::cout << "use integral    " << fitOpt.fIntegral << std::endl;
   std::cout << "use all error=1 " << fitOpt.fErrors1 << std::endl;
   if (isWeighted)
      std::cout << "Weighted data set - sumw =  " << data.SumOfContent() << "  sumw2 = " << data.SumOfError2()
                << std::endl;
#endif

#ifdef USE_PARAMCACHE
   IntegralEvaluator<> igEval(func, 0, useBinIntegral);
#else
   IntegralEvaluator<> igEval(func, p, useBinIntegral);
#endif
   double maxResValue = std::numeric_limits<double>::max() / n;
   double wrefVolume = 1.0;
   if (useBinVolume) {
      if (fitOpt.fNormBinVolume)
         wrefVolume /= data.RefVolume();
   }

   (const_cast<IModelFunction &>(func)).SetParameters(p);

   auto mapFunction = [&](const unsigned i) {
      double chi2{};
      double fval{};

      const auto x1 = data.GetCoordComponent(i, 0);
      const auto y = data.Value(i);
      auto invError = data.InvError(i);

      // invError = (invError!= 0.0) ? 1.0/invError :1;

      const double *x = nullptr;
      std::vector<double> xc;
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
         for (unsigned int j = 1; j < data.NDim(); ++j)
            xc[j] = *data.GetCoordComponent(i, j);
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
         // calculate integral normalized by bin volume
         // need to set function and parameters here in case loop is parallelized
         fval = igEval(x, data.BinUpEdge(i));
      }
      // normalize result if requested according to bin volume
      if (useBinVolume)
         fval *= binVolume;

      // expected errors
      if (useExpErrors) {
         double invWeight = 1.0;
         if (isWeighted) {
            // we need first to check if a weight factor needs to be applied
            // weight = sumw2/sumw = error**2/content
            // invWeight = y * invError * invError;
            // we use always the global weight and not the observed one in the bin
            // for empty bins use global weight (if it is weighted data.SumError2() is not zero)
            invWeight = data.SumOfContent() / data.SumOfError2();
            // if (invError > 0) invWeight = y * invError * invError;
         }

         //  if (invError == 0) invWeight = (data.SumOfError2() > 0) ? data.SumOfContent()/ data.SumOfError2() : 1.0;
         // compute expected error  as f(x) / weight
         double invError2 = (fval > 0) ? invWeight / fval : 0.0;
         invError = std::sqrt(invError2);
         // std::cout << "using Pearson chi2 " << x[0] << "  " << 1./invError2 << "  " << fval << std::endl;
      }

//#define DEBUG
#ifdef DEBUG
      std::cout << x[0] << "  " << y << "  " << 1. / invError << " params : ";
      for (unsigned int ipar = 0; ipar < func.NPar(); ++ipar)
         std::cout << p[ipar] << "\t";
      std::cout << "\tfval = " << fval << " bin volume " << binVolume << " ref " << wrefVolume << std::endl;
#endif
      //#undef DEBUG

      if (invError > 0) {

         double tmp = (y - fval) * invError;
         double resval = tmp * tmp;

         // avoid inifinity or nan in chi2 values due to wrong function values
         if (resval < maxResValue)
            chi2 += resval;
         else {
            // nRejected++;
            chi2 += maxResValue;
         }
      }
      return chi2;
   };

#ifdef R__USE_IMT
   auto redFunction = [](const std::vector<double> &objs) {
      return std::accumulate(objs.begin(), objs.end(), double{});
   };
#else
   (void)nChunks;

   // If IMT is disabled, force the execution policy to the serial case
   if (executionPolicy == ROOT::Fit::ExecutionPolicy::kMultithread) {
      Warning("Chi2<double>::EvaluateChi2", "Multithread execution policy requires IMT, which is disabled. Changing "
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
      Error("Chi2<double>::EvaluateChi2", "Execution policy unknown. Avalaible choices:\n "
                                          "ROOT::Fit::ExecutionPolicy::kSerial (default)\n "
                                          "ROOT::Fit::ExecutionPolicy::kMultithread (requires IMT)\n");
   }

   return res;
}

double Chi2<double>::EvalEffective(const IModelFunctionTempl<double> &func, const BinData &data, const double *p,
                                   unsigned int &nPoints)
{
   // evaluate the chi2 given a  function reference  , the data and returns the value and also in nPoints
   // the actual number of used points
   // method using the error in the coordinates
   // integral of bin does not make sense in this case

   unsigned int n = data.Size();

#ifdef DEBUG
   std::cout << "\n\nFit data size = " << n << std::endl;
   std::cout << "evaluate effective chi2 using function " << &func << "  " << p << std::endl;
#endif

   assert(data.HaveCoordErrors() || data.HaveAsymErrors());

   double chi2 = 0;
   // int nRejected = 0;

   // func.SetParameters(p);

   unsigned int ndim = func.NDim();

   // use Richardson derivator
   ROOT::Math::RichardsonDerivator derivator;

   double maxResValue = std::numeric_limits<double>::max() / n;

   for (unsigned int i = 0; i < n; ++i) {

      double y = 0;
      const double *x = data.GetPoint(i, y);

      double fval = func(x, p);

      double delta_y_func = y - fval;

      double ey = 0;
      const double *ex = 0;
      if (!data.HaveAsymErrors())
         ex = data.GetPointError(i, ey);
      else {
         double eylow, eyhigh = 0;
         ex = data.GetPointError(i, eylow, eyhigh);
         if (delta_y_func < 0)
            ey = eyhigh; // function is higher than points
         else
            ey = eylow;
      }
      double e2 = ey * ey;
      // before calculating the gradient check that all error in x are not zero
      unsigned int j = 0;
      while (j < ndim && ex[j] == 0.) {
         j++;
      }
      // if j is less ndim some elements are not zero
      if (j < ndim) {
         // need an adapter from a multi-dim function to a one-dimensional
         ROOT::Math::OneDimMultiFunctionAdapter<const IModelFunction &> f1D(func, x, 0, p);
         // select optimal step size  (use 10--2 by default as was done in TF1:
         double kEps = 0.01;
         double kPrecision = 1.E-8;
         for (unsigned int icoord = 0; icoord < ndim; ++icoord) {
            // calculate derivative for each coordinate
            if (ex[icoord] > 0) {
               // gradCalc.Gradient(x, p, fval, &grad[0]);
               f1D.SetCoord(icoord);
               // optimal spep size (take ex[] as scale for the points and 1% of it
               double x0 = x[icoord];
               double h = std::max(kEps * std::abs(ex[icoord]), 8.0 * kPrecision * (std::abs(x0) + kPrecision));
               double deriv = derivator.Derivative1(f1D, x[icoord], h);
               double edx = ex[icoord] * deriv;
               e2 += edx * edx;
#ifdef DEBUG
               std::cout << "error for coord " << icoord << " = " << ex[icoord] << " deriv " << deriv << std::endl;
#endif
            }
         }
      }
      double w2 = (e2 > 0) ? 1.0 / e2 : 0;
      double resval = w2 * (y - fval) * (y - fval);

#ifdef DEBUG
      std::cout << x[0] << "  " << y << " ex " << ex[0] << " ey  " << ey << " params : ";
      for (unsigned int ipar = 0; ipar < func.NPar(); ++ipar)
         std::cout << p[ipar] << "\t";
      std::cout << "\tfval = " << fval << "\tresval = " << resval << std::endl;
#endif

      // avoid (infinity and nan ) in the chi2 sum
      // eventually add possibility of excluding some points (like singularity)
      if (resval < maxResValue)
         chi2 += resval;
      else
         chi2 += maxResValue;
      // nRejected++;
   }

   // reset the number of fitting data points
   nPoints = n; // no points are rejected
                // if (nRejected != 0)  nPoints = n - nRejected;

#ifdef DEBUG
   std::cout << "chi2 = " << chi2 << " n = " << nPoints << std::endl;
#endif

   return chi2;
}

void Chi2<double>::EvalGradient(const IModelFunctionTempl<double> &f, const BinData &data, const double *p,
                                double *grad, unsigned int &nPoints, ::ROOT::Fit::ExecutionPolicy executionPolicy,
                                unsigned nChunks)
{
   // evaluate the gradient of the chi2 function
   // this function is used when the model function knows how to calculate the derivative and we can
   // avoid that the minimizer re-computes them
   //
   // case of chi2 effective (errors on coordinate) is not supported

   if (data.HaveCoordErrors()) {
      MATH_ERROR_MSG("Chi2<double>::EvaluateChi2Gradient",
                     "Error on the coordinates are not used in calculating Chi2 gradient");
      return; // it will assert otherwise later in GetPoint
   }

   const IGradModelFunction *fg = dynamic_cast<const IGradModelFunction *>(&f);
   assert(fg != nullptr); // must be called by a gradient function

   const IGradModelFunction &func = *fg;

#ifdef DEBUG
   std::cout << "\n\nFit data size = " << n << std::endl;
   std::cout << "evaluate chi2 using function gradient " << &func << "  " << p << std::endl;
#endif

   const DataOptions &fitOpt = data.Opt();
   bool useBinIntegral = fitOpt.fIntegral && data.HasBinEdges();
   bool useBinVolume = (fitOpt.fBinVolume && data.HasBinEdges());

   double wrefVolume = 1.0;
   if (useBinVolume) {
      if (fitOpt.fNormBinVolume)
         wrefVolume /= data.RefVolume();
   }

   IntegralEvaluator<> igEval(func, p, useBinIntegral);

   unsigned int npar = func.NPar();
   unsigned initialNPoints = data.Size();

   std::vector<bool> isPointRejected(initialNPoints);

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

      unsigned int ndim = data.NDim();
      double binVolume = 1;
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
         auto x2 = data.BinUpEdge(i);
         // calculate normalized integral and gradient (divided by bin volume)
         // need to set function and parameters here in case loop is parallelized
         fval = igEval(x, x2);
         CalculateGradientIntegral(func, x, x2, p, &gradFunc[0]);
      }
      if (useBinVolume)
         fval *= binVolume;

#ifdef DEBUG
      std::cout << x[0] << "  " << y << "  " << 1. / invError << " params : ";
      for (unsigned int ipar = 0; ipar < npar; ++ipar)
         std::cout << p[ipar] << "\t";
      std::cout << "\tfval = " << fval << std::endl;
#endif
      if (!std::isfinite(fval)) {
         isPointRejected[i] = true;
         // Return a zero contribution to all partial derivatives on behalf of the current point
         return pointContribution;
      }

      // loop on the parameters
      unsigned int ipar = 0;
      for (; ipar < npar; ++ipar) {

         // correct gradient for bin volumes
         if (useBinVolume)
            gradFunc[ipar] *= binVolume;

         // avoid singularity in the function (infinity and nan ) in the chi2 sum
         // eventually add possibility of excluding some points (like singularity)
         double dfval = gradFunc[ipar];
         if (!std::isfinite(dfval)) {
            break; // exit loop on parameters
         }

         // calculate derivative point contribution
         pointContribution[ipar] = -2.0 * (y - fval) * invError * invError * gradFunc[ipar];
      }

      if (ipar < npar) {
         // case loop was broken for an overflow in the gradient calculation
         isPointRejected[i] = true;
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
      Warning("Chi2<double>::EvaluateChi2Gradient",
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
   // else if(executionPolicy == ROOT::Fit::kMultiprocess){
   //    ROOT::TProcessExecutor pool;
   //    g = pool.MapReduce(mapFunction, ROOT::TSeq<unsigned>(0, n), redFunction);
   // }
   else {
      Error("Chi2<double>::EvaluateChi2Gradient",
            "Execution policy unknown. Avalaible choices:\n 0: Serial (default)\n 1: MultiThread (requires IMT)\n");
   }

#ifndef R__USE_IMT
   // to fix compiler warning
   (void)nChunks;
#endif

   // correct the number of points
   nPoints = initialNPoints;

   if (std::any_of(isPointRejected.begin(), isPointRejected.end(), [](bool point) { return point; })) {
      unsigned nRejected = std::accumulate(isPointRejected.begin(), isPointRejected.end(), 0);
      assert(nRejected <= initialNPoints);
      nPoints = initialNPoints - nRejected;

      if (nPoints < npar)
         MATH_ERROR_MSG("Chi2<double>::EvaluateChi2Gradient",
                        "Error - too many points rejected for overflow in gradient calculation");
   }

   // copy result
   std::copy(g.begin(), g.end(), grad);
}

////////////////////////////////////////////////////////////////////////////////
/// evaluate the chi2 contribution (residual term) only for data with no coord-errors
/// This function is used in the specialized least square algorithms like FUMILI or L.M.
/// if we have error on the coordinates the method is not yet implemented
///  integral option is also not yet implemented
///  one can use in that case normal chi2 method
double Chi2<double>::EvalResidual(const IModelFunctionTempl<double> &func, const BinData &data, const double *p,
                                  unsigned int i, double *g)
{
   if (data.GetErrorType() == BinData::kCoordError && data.Opt().fCoordErrors) {
      MATH_ERROR_MSG("Chi2<double>::EvaluateChi2Residual",
                     "Error on the coordinates are not used in calculating Chi2 residual");
      return 0; // it will assert otherwise later in GetPoint
   }

   // func.SetParameters(p);

   double y, invError = 0;

   const DataOptions &fitOpt = data.Opt();
   bool useBinIntegral = fitOpt.fIntegral && data.HasBinEdges();
   bool useBinVolume = (fitOpt.fBinVolume && data.HasBinEdges());
   bool useExpErrors = (fitOpt.fExpErrors);

   const double *x1 = data.GetPoint(i, y, invError);

   IntegralEvaluator<> igEval(func, p, useBinIntegral);
   double fval = 0;
   unsigned int ndim = data.NDim();
   double binVolume = 1.0;
   const double *x2 = 0;
   if (useBinVolume || useBinIntegral)
      x2 = data.BinUpEdge(i);

   double *xc = 0;

   if (useBinVolume) {
      xc = new double[ndim];
      for (unsigned int j = 0; j < ndim; ++j) {
         binVolume *= std::abs(x2[j] - x1[j]);
         xc[j] = 0.5 * (x2[j] + x1[j]);
      }
      // normalize the bin volume using a reference value
      binVolume /= data.RefVolume();
   }

   const double *x = (useBinVolume) ? xc : x1;

   if (!useBinIntegral) {
      fval = func(x, p);
   } else {
      // calculate integral (normalized by bin volume)
      // need to set function and parameters here in case loop is parallelized
      fval = igEval(x1, x2);
   }
   // normalize result if requested according to bin volume
   if (useBinVolume)
      fval *= binVolume;

   // expected errors
   if (useExpErrors) {
      // we need first to check if a weight factor needs to be applied
      // weight = sumw2/sumw = error**2/content
      // NOTE: assume histogram is not weighted
      // don't know how to do with bins with weight = 0
      // double invWeight = y * invError * invError;
      // if (invError == 0) invWeight = (data.SumOfError2() > 0) ? data.SumOfContent()/ data.SumOfError2() : 1.0;
      // compute expected error  as f(x) / weight
      double invError2 = (fval > 0) ? 1.0 / fval : 0.0;
      invError = std::sqrt(invError2);
   }

   double resval = (y - fval) * invError;

   // avoid infinities or nan in  resval
   resval = CorrectValue(resval);

   // estimate gradient
   if (g != 0) {

      unsigned int npar = func.NPar();
      const IGradModelFunction *gfunc = dynamic_cast<const IGradModelFunction *>(&func);

      if (gfunc != 0) {
         // case function provides gradient
         if (!useBinIntegral) {
            gfunc->ParameterGradient(x, p, g);
         } else {
            // needs to calculate the integral for each partial derivative
            CalculateGradientIntegral(*gfunc, x1, x2, p, g);
         }
      } else {
         SimpleGradientCalculator gc(npar, func);
         if (!useBinIntegral)
            gc.ParameterGradient(x, p, fval, g);
         else {
            // needs to calculate the integral for each partial derivative
            CalculateGradientIntegral(gc, x1, x2, p, g);
         }
      }
      // mutiply by - 1 * weight
      for (unsigned int k = 0; k < npar; ++k) {
         g[k] *= -invError;
         if (useBinVolume)
            g[k] *= binVolume;
      }
   }

   if (useBinVolume)
      delete[] xc;

   return resval;
}

} // namespace FitUtil

} // namespace Fit

} // end namespace ROOT
