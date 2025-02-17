/*
 * Project: RooFit
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

/** \class RooLegacyExpPoly
    \ingroup Roofit

RooLegacyExpPoly implements a polynomial PDF of the form \f[ f(x) =
\mathcal{N} \cdot \exp( \sum_{i} a_{i} * x^{i} ) \f] \f$ \mathcal{N}
\f$ is a normalisation constant that is automatically calculated when
the function is used in computations.

The sum can be truncated at the low end. See the main constructor
RooLegacyExpPoly::RooLegacyExpPoly(const char*, const char*, RooAbsReal&, const RooArgList&, int)

\image html RooLegacyExpPoly.png

**/

#include <RooLegacyExpPoly.h>

#include <RooAbsReal.h>
#include <RooArgList.h>
#include <RooMath.h>
#include <RooMsgService.h>
#include <RooRealVar.h>
#include "RooBatchCompute.h"

#include <TMath.h>
#include <TError.h>

#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <sstream>


////////////////////////////////////////////////////////////////////////////////
/// Create a polynomial in the variable `x`.
/// \param[in] name Name of the PDF
/// \param[in] title Title for plotting the PDF
/// \param[in] x The variable of the polynomial
/// \param[in] coefList The coefficients \f$ a_i \f$
/// \param[in] lowestOrder [optional] Truncate the sum such that it skips the lower orders:
/// \f[
///     1. + \sum_{i=0}^{\mathrm{coefList.size()}} a_{i} * x^{(i + \mathrm{lowestOrder})}
/// \f]
///
/// This means that
/// \code{.cpp}
/// RooLegacyExpPoly pol("pol", "pol", x, RooArgList(a, b), lowestOrder = 2)
/// \endcode
/// computes
/// \f[
///   \mathrm{pol}(x) = 1 * x^0 + (0 * x^{\ldots}) + a * x^2 + b * x^3.
/// \f]

RooLegacyExpPoly::RooLegacyExpPoly(const char *name, const char *title, RooAbsReal &x, const RooArgList &coefList, int lowestOrder)
   : RooAbsPdf(name, title),
     _x("x", "Dependent", this, x),
     _coefList("coefList", "List of coefficients", this),
     _lowestOrder(lowestOrder)
{
   // Check lowest order
   if (_lowestOrder < 0) {
      coutE(InputArguments) << "RooLegacyExpPoly::ctor(" << GetName()
                            << ") WARNING: lowestOrder must be >=0, setting value to 0" << std::endl;
      _lowestOrder = 0;
   }

   _coefList.addTyped<RooAbsReal>(coefList);
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooLegacyExpPoly::RooLegacyExpPoly(const RooLegacyExpPoly &other, const char *name)
   : RooAbsPdf(other, name),
     _x("x", this, other._x),
     _coefList("coefList", this, other._coefList),
     _lowestOrder(other._lowestOrder)
{
}

////////////////////////////////////////////////////////////////////////////////

double RooLegacyExpPoly::evaluateLog() const
{
   // Calculate and return value of polynomial

   const unsigned sz = _coefList.size();
   const int lowestOrder = _lowestOrder;
   if (!sz)
      return lowestOrder ? 1. : 0.;
   std::vector<double> coefs;
   coefs.reserve(sz);

   const RooArgSet *nset = _coefList.nset();
   for (auto coef : _coefList) {
      coefs.push_back(static_cast<RooAbsReal *>(coef)->getVal(nset));
   };
   const double x = _x;
   double xpow = std::pow(x, lowestOrder);
   double retval = 0;
   for (size_t i = 0; i < sz; ++i) {
      retval += coefs[i] * xpow;
      xpow *= x;
   }

   if (std::numeric_limits<double>::max_exponent < retval) {
      coutE(InputArguments) << "RooLegacyExpPoly::evaluateLog(" << GetName() << ") ERROR: exponent at " << x
                            << " larger than allowed maximum, result will be infinite! " << retval << " > "
                            << std::numeric_limits<double>::max_exponent << " in " << this->getFormulaExpression(true)
                            << std::endl;
   }
   return retval;
}

////////////////////////////////////////////////////////////////////////////////

/// Compute multiple values of ExpPoly distribution.
void RooLegacyExpPoly::doEval(RooFit::EvalContext &ctx) const
{
   std::vector<std::span<const double>> vars;
   vars.reserve(_coefList.size() + 1);
   vars.push_back(ctx.at(_x));

   std::vector<double> coefVals;
   for (RooAbsArg *coef : _coefList) {
      vars.push_back(ctx.at(coef));
   }

   std::array<double, 2> args{static_cast<double>(_lowestOrder), static_cast<double>(_coefList.size())};

   RooBatchCompute::compute(ctx.config(this), RooBatchCompute::ExpPoly, ctx.output(), vars, args);
}

////////////////////////////////////////////////////////////////////////////////

void RooLegacyExpPoly::adjustLimits()
{
   // Adjust the limits of all the coefficients to reflect the numeric boundaries

   const unsigned sz = _coefList.size();
   double max = std::numeric_limits<double>::max_exponent / sz;
   const int lowestOrder = _lowestOrder;
   std::vector<double> coefs;
   coefs.reserve(sz);

   RooRealVar *x = dynamic_cast<RooRealVar *>(&(*_x));
   if (x) {
      const double xmax = x->getMax();
      double xmaxpow = std::pow(xmax, lowestOrder);
      for (size_t i = 0; i < sz; ++i) {
         double thismax = max / xmaxpow;
         RooRealVar *coef = dynamic_cast<RooRealVar *>(this->_coefList.at(i));
         if (coef) {
            coef->setVal(thismax);
            coef->setMax(thismax);
         }
         xmaxpow *= xmax;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////

double RooLegacyExpPoly::evaluate() const
{
   // Calculate and return value of function

   const double logval = this->evaluateLog();
   const double val = std::exp(logval);
   if (std::isinf(val)) {
      coutE(InputArguments) << "RooLegacyExpPoly::evaluate(" << GetName()
                            << ") ERROR: result of exponentiation is infinite! exponent was " << logval << std::endl;
   }
   return val;
}

////////////////////////////////////////////////////////////////////////////////

double RooLegacyExpPoly::getLogVal(const RooArgSet *nset) const
{
   return RooAbsPdf::getLogVal(nset);
   //  return this->evaluateLog();
}

////////////////////////////////////////////////////////////////////////////////

int RooLegacyExpPoly::getAnalyticalIntegral(RooArgSet &allVars, RooArgSet &analVars, const char * /*rangeName*/) const
{

   if ((_coefList.size() + _lowestOrder < 4) &&
       ((_coefList.size() + _lowestOrder < 3) ||
        (static_cast<RooAbsReal *>(_coefList.at(2 - _lowestOrder))->getVal() <= 0)) &&
       matchArgs(allVars, analVars, _x)) {
      return 0;
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////

#define PI TMath::Pi()

namespace {
double deltaerf(long double x1, long double x2)
{
   // several things happening here
   // 1. use "erf(x) = -erf(-x)" to evaluate the function only ever for the positive side (higher precision)
   // 2. use "erf(x) = 1.-erfc(x)" and, instead of "erf(x1) - erf(x2)", do "(1.-erfc(x1)) - (1.-erfc(x2)) = erfc(x2) -
   // erfc(x1)"
   double y2 = x1 > 0 ? erfc(x1) : -erfc(-x1);
   double y1 = x2 > 0 ? erfc(x2) : -erfc(-x2);
   if (y1 == y2) {
      std::cout << "WARNING in calculation of analytical integral limited by numerical precision" << std::endl;
      std::cout << "x: " << x1 << " , " << x2 << std::endl;
      std::cout << "y: " << y1 << " , " << y2 << std::endl;
   }
   return y1 - y2;
}

double deltaerfi(double x1, double x2)
{
   std::complex<double> u1 = {x1, 0.};
   std::complex<double> u2 = {x2, 0.};

   std::complex<double> y2 = x2 > 0 ? RooMath::faddeeva(u2) : RooMath::faddeeva(-u2);
   std::complex<double> y1 = x1 > 0 ? RooMath::faddeeva(u1) : RooMath::faddeeva(-u1);
   if (y1 == y2) {
      std::cout << "WARNING in calculation of analytical integral limited by numerical precision" << std::endl;
      std::cout << "x: " << x1 << " , " << x2 << std::endl;
      std::cout << "y: " << y1 << " , " << y2 << std::endl;
   }
   return y1.imag() - y2.imag();
}
} // namespace

double RooLegacyExpPoly::analyticalIntegral(int /*code*/, const char *rangeName) const
{
   const double xmin = _x.min(rangeName);
   const double xmax = _x.max(rangeName);
   const unsigned sz = _coefList.size();
   if (!sz)
      return xmax - xmin;

   std::vector<double> coefs;
   coefs.reserve(sz);
   const RooArgSet *nset = _coefList.nset();
   for (auto c : _coefList) {
      coefs.push_back(static_cast<RooAbsReal *>(c)->getVal(nset));
   }

   switch (_coefList.size() + _lowestOrder) {
   case 1: return xmax - xmin;
   case 2: {
      const double a = coefs[1 - _lowestOrder];
      if (a != 0) {
         return 1. / a * (exp(a * xmax) - exp(a * xmin)) * (_lowestOrder == 0 ? exp(coefs[0]) : 1);
      } else {
         return xmax - xmin;
      }
   }
   case 3: {
      const double a = coefs[2 - _lowestOrder];
      const double b = _lowestOrder == 2 ? 0. : coefs[1 - _lowestOrder];
      const double c = _lowestOrder == 0 ? coefs[0] : 0.;
      const double absa = std::abs(a);
      const double sqrta = std::sqrt(absa);
      if (a < 0) {
         double d = ::deltaerf((-b + 2 * absa * xmax) / (2 * sqrta), (-b + 2 * absa * xmin) / (2 * sqrta));
         double retval = exp(b * b / (4 * absa) + c) * std::sqrt(PI) * d / (2 * sqrta);
         return retval;
      } else if (a > 0) {
         double d = ::deltaerfi((b + 2 * absa * xmax) / (2 * sqrta), (b + 2 * absa * xmin) / (2 * sqrta));
         double retval = exp(-b * b / (4 * absa) + c) * std::sqrt(PI) * d / (2 * sqrta);
         return retval;
      } else if (b != 0) {
         return 1. / b * (std::exp(b * xmax) - exp(b * xmin)) * exp(c);
      } else {
         return xmax - xmin;
      }
   }
   }
   return 0.;
}

////////////////////////////////////////////////////////////////////////////////

std::string RooLegacyExpPoly::getFormulaExpression(bool expand) const
{
   std::stringstream ss;
   ss << "exp(";
   int order = _lowestOrder;
   for (auto coef : _coefList) {
      if (order != _lowestOrder)
         ss << "+";
      if (expand) {
         ss << (static_cast<RooAbsReal *>(coef))->getVal();
      } else {
         ss << coef->GetName();
      }
      ss << "*pow(" << _x.GetName() << "," << order << ")";
      ++order;
   }
   ss << ")";
   return ss.str();
}
