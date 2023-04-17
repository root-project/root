/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 * @(#)root/roofit:$Id$
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

/** \class RooPolynomial
    \ingroup Roofit

RooPolynomial implements a polynomial p.d.f of the form
\f[ f(x) = \mathcal{N} \cdot \sum_{i} a_{i} * x^i \f]
By default, the coefficient \f$ a_0 \f$ is chosen to be 1, as polynomial
probability density functions have one degree of freedom
less than polynomial functions due to the normalisation condition. \f$ \mathcal{N} \f$
is a normalisation constant that is automatically calculated when the polynomial is used
in computations.

The sum can be truncated at the low end. See the main constructor
RooPolynomial::RooPolynomial(const char*, const char*, RooAbsReal&, const RooArgList&, Int_t)
**/

#include "RooPolynomial.h"
#include "RooArgList.h"
#include "RooMsgService.h"
#include "RooPolyVar.h"

#include <RooFit/Detail/AnalyticalIntegrals.h>
#include <RooFit/Detail/EvaluateFuncs.h>

#include "TError.h"
#include <vector>

ClassImp(RooPolynomial);

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
/// RooPolynomial pol("pol", "pol", x, RooArgList(a, b), lowestOrder = 2)
/// \endcode
/// computes
/// \f[
///   \mathrm{pol}(x) = 1 * x^0 + (0 * x^{\ldots}) + a * x^2 + b * x^3.
/// \f]


RooPolynomial::RooPolynomial(const char* name, const char* title,
              RooAbsReal& x, const RooArgList& coefList, Int_t lowestOrder) :
  RooAbsPdf(name, title),
  _x("x", "Dependent", this, x),
  _coefList("coefList","List of coefficients",this),
  _lowestOrder(lowestOrder)
{
  // Check lowest order
  if (_lowestOrder<0) {
    coutE(InputArguments) << "RooPolynomial::ctor(" << GetName()
           << ") WARNING: lowestOrder must be >=0, setting value to 0" << std::endl;
    _lowestOrder=0 ;
  }

  for (auto *coef : coefList) {
    if (!dynamic_cast<RooAbsReal*>(coef)) {
      coutE(InputArguments) << "RooPolynomial::ctor(" << GetName() << ") ERROR: coefficient " << coef->GetName()
             << " is not of type RooAbsReal" << std::endl;
      R__ASSERT(0) ;
    }
    _coefList.add(*coef) ;
  }
}

////////////////////////////////////////////////////////////////////////////////

RooPolynomial::RooPolynomial(const char* name, const char* title,
                           RooAbsReal& x) :
  RooAbsPdf(name, title),
  _x("x", "Dependent", this, x),
  _coefList("coefList","List of coefficients",this),
  _lowestOrder(1)
{ }

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooPolynomial::RooPolynomial(const RooPolynomial& other, const char* name) :
  RooAbsPdf(other, name),
  _x("x", this, other._x),
  _coefList("coefList",this,other._coefList),
  _lowestOrder(other._lowestOrder)
{ }

////////////////////////////////////////////////////////////////////////////////

double RooPolynomial::evaluate() const
{
  const unsigned sz = _coefList.getSize();
  if (!sz)
    return _lowestOrder ? 1. : 0.;

  RooPolyVar::fillCoeffValues(_wksp, _coefList);

  return RooFit::Detail::EvaluateFuncs::polynomialEvaluate<true>(_wksp.data(), sz, _lowestOrder, _x);
}

void RooPolynomial::translate(RooFit::Detail::CodeSquashContext &ctx) const
{
  const unsigned sz = _coefList.size();
  if (!sz) {
    ctx.addResult(this, std::to_string((_lowestOrder ? 1. : 0.)));
    return;
  }

  ctx.addResult(
     this, ctx.buildCall("RooFit::Detail::EvaluateFuncs::polynomialEvaluate<true>", _coefList, sz, _lowestOrder, _x));
}

/// Compute multiple values of Polynomial.
void RooPolynomial::computeBatch(cudaStream_t *stream, double *output, size_t nEvents,
                                 RooFit::Detail::DataMap const &dataMap) const
{
   return RooPolyVar::computeBatchImpl(stream, output, nEvents, dataMap, _x.arg(), _coefList, _lowestOrder);
}

////////////////////////////////////////////////////////////////////////////////
/// Advertise to RooFit that this function can be analytically integrated.
Int_t RooPolynomial::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /*rangeName*/) const
{
  if (matchArgs(allVars, analVars, _x)) return 1;
  return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Do the analytical integral according to the code that was returned by getAnalyticalIntegral().
double RooPolynomial::analyticalIntegral(Int_t code, const char* rangeName) const
{
  R__ASSERT(code==1) ;

  const double xmin = _x.min(rangeName), xmax = _x.max(rangeName);
  const unsigned sz = _coefList.getSize();
  if (!sz)
    return _lowestOrder ? xmax - xmin : 0.0;

  RooPolyVar::fillCoeffValues(_wksp, _coefList);

  return RooFit::Detail::AnalyticalIntegrals::polynomialIntegral<true>(_wksp.data(), sz, _lowestOrder, xmin, xmax);
}

std::string RooPolynomial::buildCallToAnalyticIntegral(Int_t /* code */, const char *rangeName,
                                                       RooFit::Detail::CodeSquashContext &ctx) const
{
  const double xmin = _x.min(rangeName), xmax = _x.max(rangeName);
  const unsigned sz = _coefList.getSize();
  if (!sz)
    return std::to_string(_lowestOrder ? xmax - xmin : 0.0);

  std::string integralName = ctx.getTmpVarName();
  std::string integralDecl = "double " + integralName + " = ";
  integralDecl += ctx.buildCall("RooFit::Detail::AnalyticalIntegrals::polynomialIntegral<true>", _coefList, sz,
                                _lowestOrder, xmin, xmax);
  integralDecl += ";\n";
  ctx.addToCodeBody(integralDecl);

  return integralName;
}
