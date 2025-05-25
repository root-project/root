/// \cond ROOFIT_INTERNAL

/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

/**
\file RooRombergIntegrator.cxx
\class RooRombergIntegrator
\ingroup Roofitcore

Adaptive numerical integration algorithm.

It uses Romberg's method with trapezoids or midpoints.
The integrand is approximated by \f$ 1, 2, 4, 8, \ldots, 2^n \f$ trapezoids, and
Richardson series acceleration is applied to the series. For refinement step \f$ n \f$, that is
\f[
  R(n,m) = \frac{1}{4^m - 1} \left( 4^m R(n, m-1) - R(n-1, m-1) \right)
\f]
The integrator will evaluate the first six refinements (i.e. 64 points) in one go,
apply a five-point series acceleration, and successively add more steps until the
desired precision is reached.
**/

#include "Riostream.h"

#include "TClass.h"
#include "RooRombergIntegrator.h"
#include "RooArgSet.h"
#include "RooRealVar.h"
#include "RooNumber.h"
#include "RooNumIntConfig.h"
#include "RooNumIntFactory.h"
#include "RooMsgService.h"

#include <cassert>

namespace {

constexpr int nPoints = 5;

////////////////////////////////////////////////////////////////////////////////
/// Calculate the n-th stage of refinement of the Second Euler-Maclaurin
/// summation rule which has the useful property of not evaluating the
/// integrand at either of its endpoints but requires more function
/// evaluations than the trapezoidal rule. This rule can be used with
/// a suitable change of variables to estimate improper integrals.

double addMidpoints(std::function<double(double)> const &func, double savedResult, int n, double xmin, double xmax)
{
   const double range = xmax - xmin;

   if (n == 1) {
      double xmid = 0.5 * (xmin + xmax);
      return range * func(xmid);
   }

   int it = 1;
   for (int j = 1; j < n - 1; j++) {
      it *= 3;
   }
   double tnm = it;
   double del = range / (3. * tnm);
   double ddel = del + del;
   double x = xmin + 0.5 * del;
   double sum = 0;
   for (int j = 1; j <= it; j++) {
      sum += func(x);
      x += ddel;
      sum += func(x);
      x += del;
   }
   return (savedResult + range * sum / tnm) / 3.;
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate the n-th stage of refinement of the extended trapezoidal
/// summation rule. This is the most efficient rule for a well behaved
/// integrands that can be evaluated over its entire range, including the
/// endpoints.

double addTrapezoids(std::function<double(double)> const &func, double savedResult, int n, double xmin, double xmax)
{
   const double range = xmax - xmin;

   if (n == 1) {
      // use a single trapezoid to cover the full range
      return 0.5 * range * (func(xmin) + func(xmax));
   }

   // break the range down into several trapezoids using 2**(n-2)
   // equally-spaced interior points
   const int nInt = 1 << (n - 2);
   const double del = range / nInt;

   double sum = 0.;
   // TODO Replace by batch computation
   for (int j = 0; j < nInt; ++j) {
      double x = xmin + (0.5 + j) * del;
      sum += func(x);
   }

   return 0.5 * (savedResult + range * sum / nInt);
}

////////////////////////////////////////////////////////////////////////////////
/// Extrapolate result to final value.

std::pair<double, double> extrapolate(int n, double const *h, double const *s, double *c, double *d)
{
   double const *xa = &h[n - nPoints];
   double const *ya = &s[n - nPoints];
   int ns = 1;

   double dif = std::abs(xa[1]);
   for (int i = 1; i <= nPoints; i++) {
      double dift = std::abs(xa[i]);
      if (dift < dif) {
         ns = i;
         dif = dift;
      }
      c[i] = ya[i];
      d[i] = ya[i];
   }
   double extrapError = 0.0;
   double extrapValue = ya[ns--];
   for (int m = 1; m < nPoints; m++) {
      for (int i = 1; i <= nPoints - m; i++) {
         double ho = xa[i];
         double hp = xa[i + m];
         double w = c[i + 1] - d[i];
         double den = ho - hp;
         if (den == 0.0) {
            throw std::runtime_error("RooRombergIntegrator::extrapolate: internal error");
         }
         den = w / den;
         d[i] = hp * den;
         c[i] = ho * den;
      }
      extrapError = 2 * ns < (nPoints - m) ? c[ns + 1] : d[ns--];
      extrapValue += extrapError;
   }
   return {extrapValue, extrapError};
}

} // namespace

namespace RooFit {
namespace Detail {

std::pair<double, int> integrate1d(std::function<double(double)> func, bool doTrapezoid, int maxSteps, int minStepsZero,
                                   int fixSteps, double epsAbs, double epsRel, bool doExtrap, double xmin, double xmax,
                                   std::span<double> hArr, std::span<double> sArr)
{
   assert(int(hArr.size()) == maxSteps + 2);
   assert(int(sArr.size()) == maxSteps + 2);

   const double range = xmax - xmin;

   // Small working arrays can be on the stack
   std::array<double, nPoints + 1> cArr = {};
   std::array<double, nPoints + 1> dArr = {};

   hArr[1] = 1.0;
   double zeroThresh = epsAbs / range;
   for (int j = 1; j <= maxSteps; ++j) {
      // refine our estimate using the appropriate summation rule
      sArr[j] =
         doTrapezoid ? addTrapezoids(func, sArr[j - 1], j, xmin, xmax) : addMidpoints(func, sArr[j - 1], j, xmin, xmax);

      if (j >= minStepsZero) {
         bool allZero(true);
         for (int jj = 0; jj <= j; jj++) {
            if (sArr[j] >= zeroThresh) {
               allZero = false;
            }
         }
         if (allZero) {
            // std::cout << "Roo1DIntegrator(" << this << "): zero convergence at step " << j << ", value = " << 0 <<
            // std::endl ;
            return {0, j};
         }
      }

      if (fixSteps > 0) {

         // Fixed step mode, return result after fixed number of steps
         if (j == fixSteps) {
            // std::cout << "returning result at fixed step " << j << std::endl ;
            return {sArr[j], j};
         }

      } else if (j >= nPoints) {

         double extrapValue = 0.0;
         double extrapError = 0.0;

         // extrapolate the results of recent refinements and check for a stable result
         if (doExtrap) {
            std::tie(extrapValue, extrapError) = extrapolate(j, hArr.data(), sArr.data(), cArr.data(), dArr.data());
         } else {
            extrapValue = sArr[j];
            extrapError = sArr[j] - sArr[j - 1];
         }

         if (std::abs(extrapError) <= epsRel * std::abs(extrapValue)) {
            return {extrapValue, j};
         }
         if (std::abs(extrapError) <= epsAbs) {
            return {extrapValue, j};
         }
      }
      // update the step size for the next refinement of the summation
      hArr[j + 1] = doTrapezoid ? hArr[j] / 4. : hArr[j] / 9.;
   }

   return {sArr[maxSteps], maxSteps};
}

} // namespace Detail
} // namespace RooFit


// Register this class with RooNumIntConfig

////////////////////////////////////////////////////////////////////////////////
/// Register integrator plugins, their parameters and capabilities with RooNumIntFactory.

void RooRombergIntegrator::registerIntegrator(RooNumIntFactory &fact)
{
   RooCategory sumRule("sumRule", "Summation Rule");
   sumRule.defineType("Trapezoid", RooRombergIntegrator::Trapezoid);
   sumRule.defineType("Midpoint", RooRombergIntegrator::Midpoint);
   sumRule.setLabel("Trapezoid");
   RooCategory extrap("extrapolation", "Extrapolation procedure");
   extrap.defineType("None", 0);
   extrap.defineType("Wynn-Epsilon", 1);
   extrap.setLabel("Wynn-Epsilon");
   RooRealVar maxSteps("maxSteps", "Maximum number of steps", 20);
   RooRealVar minSteps("minSteps", "Minimum number of steps", 999);
   RooRealVar fixSteps("fixSteps", "Fixed number of steps", 0);
   RooRealVar numSeg("numSeg", "Number of segments", 3); // only for the segmented integrators

   std::string name = "RooIntegrator1D";

   auto creator = [](const RooAbsFunc &function, const RooNumIntConfig &config) {
      return std::make_unique<RooRombergIntegrator>(function, config, 1, /*doSegmentation=*/false);
   };

   fact.registerPlugin(name, creator, {sumRule, extrap, maxSteps, minSteps, fixSteps},
                       /*canIntegrate1D=*/true,
                       /*canIntegrate2D=*/false,
                       /*canIntegrateND=*/false,
                       /*canIntegrateOpenEnded=*/false);

   RooNumIntConfig::defaultConfig().method1D().setLabel(name);

   auto creator2d = [](const RooAbsFunc &function, const RooNumIntConfig &config) {
      return std::make_unique<RooRombergIntegrator>(function, config, 2, /*doSegmentation=*/false);
   };
   std::string name2d = "RooIntegrator2D";
   fact.registerPlugin(name2d, creator2d, {},
                       /*canIntegrate1D=*/false,
                       /*canIntegrate2D=*/true,
                       /*canIntegrateND=*/false,
                       /*canIntegrateOpenEnded=*/false,
                       /*depName=*/"RooIntegrator1D");
   RooNumIntConfig::defaultConfig().method2D().setLabel(name2d);

   auto creatorSeg = [](const RooAbsFunc &function, const RooNumIntConfig &config) {
      return std::make_unique<RooRombergIntegrator>(function, config, 1, /*doSegmentation=*/true);
   };

   fact.registerPlugin("RooSegmentedIntegrator1D", creatorSeg, {numSeg},
                       /*canIntegrate1D=*/true,
                       /*canIntegrate2D=*/false,
                       /*canIntegrateND=*/false,
                       /*canIntegrateOpenEnded=*/false,
                       /*depName=*/"RooIntegrator1D");

   auto creatorSeg2d = [](const RooAbsFunc &function, const RooNumIntConfig &config) {
      return std::make_unique<RooRombergIntegrator>(function, config, 2, /*doSegmentation=*/true);
   };

   fact.registerPlugin("RooSegmentedIntegrator2D", creatorSeg2d, {},
                       /*canIntegrate1D=*/false,
                       /*canIntegrate2D=*/true,
                       /*canIntegrateND=*/false,
                       /*canIntegrateOpenEnded=*/false,
                       /*depName=*/"RooSegmentedIntegrator1D");
}

////////////////////////////////////////////////////////////////////////////////
/// Construct integrator on given function binding, using specified summation
/// rule, maximum number of steps and conversion tolerance. The integration
/// limits are taken from the function binding.

RooRombergIntegrator::RooRombergIntegrator(const RooAbsFunc &function, SummationRule rule, int maxSteps, double eps)
   : RooAbsIntegrator(function), _useIntegrandLimits(true), _rule(rule), _maxSteps(maxSteps), _epsAbs(eps), _epsRel(eps)
{
   _valid = initialize();
}

////////////////////////////////////////////////////////////////////////////////
/// Construct integrator on given function binding for given range,
/// using specified summation rule, maximum number of steps and
/// conversion tolerance. The integration limits are taken from the
/// function binding.

RooRombergIntegrator::RooRombergIntegrator(const RooAbsFunc &function, double xmin, double xmax, SummationRule rule,
                                           int maxSteps, double eps)
   : RooAbsIntegrator(function),
     _useIntegrandLimits(false),
     _rule(rule),
     _maxSteps(maxSteps),
     _epsAbs(eps),
     _epsRel(eps)
{
   _xmin.push_back(xmin);
   _xmax.push_back(xmax);
   _valid = initialize();
}

////////////////////////////////////////////////////////////////////////////////
/// Construct integrator on given function binding, using specified
/// configuration object. The integration limits are taken from the
/// function binding

RooRombergIntegrator::RooRombergIntegrator(const RooAbsFunc &function, const RooNumIntConfig &config, int nDim,
                                           bool doSegmentation)
   : RooAbsIntegrator(function, config.printEvalCounter()),
     _nDim{nDim},
     _epsAbs(config.epsAbs()),
     _epsRel(config.epsRel())
{
   // Extract parameters from config object
   const RooArgSet &configSet = config.getConfigSection("RooIntegrator1D");
   _rule = (SummationRule)configSet.getCatIndex("sumRule", Trapezoid);
   _maxSteps = (int)configSet.getRealValue("maxSteps", 20);
   _minStepsZero = (int)configSet.getRealValue("minSteps", 999);
   _fixSteps = (int)configSet.getRealValue("fixSteps", 0);
   _doExtrap = (bool)configSet.getCatIndex("extrapolation", 1);
   if (doSegmentation) {
      _nSeg = (int)config.getConfigSection("RooSegmentedIntegrator1D").getRealValue("numSeg", 3);
      _epsAbs /= std::sqrt(_nSeg);
      _epsRel /= std::sqrt(_nSeg);
   }

   if (_fixSteps > _maxSteps) {
      oocoutE(nullptr, Integration) << "RooRombergIntegrator::ctor() ERROR: fixSteps>maxSteps, fixSteps set to maxSteps"
                                    << std::endl;
      _fixSteps = _maxSteps;
   }

   _useIntegrandLimits = true;
   _valid = initialize();
}

////////////////////////////////////////////////////////////////////////////////
/// Construct integrator on given function binding, using specified
/// configuration object and integration range

RooRombergIntegrator::RooRombergIntegrator(const RooAbsFunc &function, double xmin, double xmax,
                                           const RooNumIntConfig &config, int nDim)
   : RooAbsIntegrator(function, config.printEvalCounter()),
     _useIntegrandLimits(false),
     _nDim{nDim},
     _epsAbs(config.epsAbs()),
     _epsRel(config.epsRel())
{
   // Extract parameters from config object
   const RooArgSet &configSet = config.getConfigSection("RooIntegrator1D");
   _rule = (SummationRule)configSet.getCatIndex("sumRule", Trapezoid);
   _maxSteps = (int)configSet.getRealValue("maxSteps", 20);
   _minStepsZero = (int)configSet.getRealValue("minSteps", 999);
   _fixSteps = (int)configSet.getRealValue("fixSteps", 0);
   _doExtrap = (bool)configSet.getCatIndex("extrapolation", 1);

   _xmin.push_back(xmin);
   _xmax.push_back(xmax);
   _valid = initialize();
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize the integrator

bool RooRombergIntegrator::initialize()
{
   // apply defaults if necessary
   if (_maxSteps <= 0) {
      _maxSteps = (_rule == Trapezoid) ? 20 : 14;
   }

   if (_epsRel <= 0)
      _epsRel = 1e-6;
   if (_epsAbs <= 0)
      _epsAbs = 1e-6;

   // check that the integrand is a valid function
   if (!isValid()) {
      oocoutE(nullptr, Integration) << "RooRombergIntegrator::initialize: cannot integrate invalid function"
                                    << std::endl;
      return false;
   }

   // Allocate coordinate buffer size after number of function dimensions
   _x.resize(_function->getDimension());

   // Allocate workspace for numerical integration engine
   _wksp.resize(_nDim * 2 * _maxSteps + 4);

   return checkLimits();
}

////////////////////////////////////////////////////////////////////////////////
/// Change our integration limits. Return true if the new limits are
/// ok, or otherwise false. Always returns false and does nothing
/// if this object was constructed to always use our integrand's limits.

bool RooRombergIntegrator::setLimits(double *xmin, double *xmax)
{
   if (_useIntegrandLimits) {
      oocoutE(nullptr, Integration) << "RooRombergIntegrator::setLimits: cannot override integrand's limits"
                                    << std::endl;
      return false;
   }
   _xmin.resize(_nDim);
   _xmax.resize(_nDim);
   for (int iDim = 0; iDim < _nDim; ++iDim) {
      _xmin[iDim] = xmin[iDim];
      _xmax[iDim] = xmax[iDim];
   }
   return checkLimits();
}

////////////////////////////////////////////////////////////////////////////////
/// Check that our integration range is finite and otherwise return false.
/// Update the limits from the integrand if requested.

bool RooRombergIntegrator::checkLimits() const
{
   if (_useIntegrandLimits) {
      assert(nullptr != integrand() && integrand()->isValid());
      const_cast<std::vector<double> &>(_xmin).resize(_nDim);
      const_cast<std::vector<double> &>(_xmax).resize(_nDim);
      for (int iDim = 0; iDim < _nDim; ++iDim) {
         const_cast<double &>(_xmin[iDim]) = integrand()->getMinLimit(iDim);
         const_cast<double &>(_xmax[iDim]) = integrand()->getMaxLimit(iDim);
      }
   }
   for (int iDim = 0; iDim < _nDim; ++iDim) {
      const double xmin = _xmin[iDim];
      const double xmax = _xmax[iDim];
      const double range = xmax - xmin;
      if (range < 0.) {
         oocoutE(nullptr, Integration) << "RooRombergIntegrator::checkLimits: bad range with min > max (_xmin[" << iDim
                                       << "] = " << xmin << " _xmax[" << iDim << "] = " << xmax << ")" << std::endl;
         return false;
      }
      if (RooNumber::isInfinite(xmin) || RooNumber::isInfinite(xmax)) {
         return false;
      }
   }
   return true;
}

double RooRombergIntegrator::integral(const double *yvec)
{
   // Copy yvec to xvec if provided
   if (yvec) {
      for (unsigned int i = 0; i < _function->getDimension() - 1; i++) {
         _x[i + _nDim] = yvec[i];
      }
   }

   return integral(_nDim - 1, _nSeg, _wksp);
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate numeric integral at given set of function binding parameters.

double RooRombergIntegrator::integral(int iDim, int nSeg, std::span<double> wksp)
{
   assert(isValid());

   const double xmin = _xmin[iDim];
   const double xmax = _xmax[iDim];
   const double range = xmax - xmin;

   if (range == 0.)
      return 0.;

   // In case of segmentation, split this integral up in a loop.
   if (nSeg > 1) {
      const double segSize = (xmax - xmin) / nSeg;
      double result = 0.0;
      for (int iSeg = 0; iSeg < nSeg; iSeg++) {
         _xmin[iDim] = xmin + iSeg * segSize;
         _xmax[iDim] = xmin + (iSeg + 1) * segSize;
         double part = integral(iDim, 1, wksp);
         result += part;
      }

      // reset limits
      _xmin[iDim] = xmin;
      _xmax[iDim] = xmax;

      return result;
   }

   // From the working array
   std::size_t nWorkingArr = _maxSteps + 2;
   double *hArr = wksp.data();
   double *sArr = wksp.data() + nWorkingArr;

   double output = 0.0;
   int steps = 0;

   std::span<double> nextWksp{wksp.data() + 2 * _maxSteps + 4, wksp.data() + wksp.size()};

   auto func = [&](double x) {
      _x[iDim] = x;

      return iDim == 0 ? integrand(_x.data()) : integral(iDim - 1, _nSeg, nextWksp);
   };

   std::tie(output, steps) =
      RooFit::Detail::integrate1d(func, _rule == Trapezoid, _maxSteps, _minStepsZero, _fixSteps, _epsAbs, _epsRel,
                                  _doExtrap, xmin, xmax, {hArr, nWorkingArr}, {sArr, nWorkingArr});

   if (steps == _maxSteps) {

      oocoutW(nullptr, Integration) << "RooRombergIntegrator::integral: integral of " << _function->getName()
                                    << " over range (" << xmin << "," << xmax << ") did not converge after "
                                    << _maxSteps << " steps" << std::endl;
      for (int j = 1; j <= _maxSteps; ++j) {
         ooccoutW(nullptr, Integration) << "   [" << j << "] h = " << hArr[j] << " , s = " << sArr[j] << std::endl;
      }
   }

   return output;
}

/// \endcond
