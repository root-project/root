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
\file RooPolyVar.cxx
\class RooPolyVar
\ingroup Roofitcore

A RooAbsReal implementing a polynomial in terms
of a list of RooAbsReal coefficients
\f[f(x) = \sum_{i} a_{i} \cdot x^i \f]
Class RooPolyvar implements analytical integrals of all polynomials
it can define.
**/

#include "RooPolyVar.h"
#include "RooArgList.h"
#include "RooMsgService.h"
#include "RooBatchCompute.h"

#include <RooFit/Detail/MathFuncs.h>

#include "TError.h"

#include <algorithm>
#include <array>
#include <cmath>


////////////////////////////////////////////////////////////////////////////////
/// Construct polynomial in x with coefficients in coefList. If
/// lowestOrder is not zero, then the first element in coefList is
/// interpreted as as the 'lowestOrder' coefficients and all
/// subsequent coefficient elements are shifted by a similar amount.
RooPolyVar::RooPolyVar(const char *name, const char *title, RooAbsReal &x, const RooArgList &coefList,
                       Int_t lowestOrder)
   : RooAbsReal(name, title),
     _x("x", "Dependent", this, x),
     _coefList("coefList", "List of coefficients", this),
     _lowestOrder(lowestOrder)
{
   // Check lowest order
   if (_lowestOrder < 0) {
      coutE(InputArguments) << "RooPolyVar::ctor(" << GetName()
                            << ") WARNING: lowestOrder must be >=0, setting value to 0" << std::endl;
      _lowestOrder = 0;
   }

   _coefList.addTyped<RooAbsReal>(coefList);
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor of flat polynomial function

RooPolyVar::RooPolyVar(const char *name, const char *title, RooAbsReal &x)
   : RooAbsReal(name, title),
     _x("x", "Dependent", this, x),
     _coefList("coefList", "List of coefficients", this),
     _lowestOrder(1)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooPolyVar::RooPolyVar(const RooPolyVar &other, const char *name)
   : RooAbsReal(other, name),
     _x("x", this, other._x),
     _coefList("coefList", this, other._coefList),
     _lowestOrder(other._lowestOrder)
{
}

void RooPolyVar::fillCoeffValues(std::vector<double> &wksp, RooListProxy const &coefList)
{
   wksp.clear();
   wksp.reserve(coefList.size());
   {
      const RooArgSet *nset = coefList.nset();
      for (const auto arg : coefList) {
         const auto c = static_cast<RooAbsReal *>(arg);
         wksp.push_back(c->getVal(nset));
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate and return value of polynomial

double RooPolyVar::evaluate() const
{
   const unsigned sz = _coefList.size();
   if (!sz)
      return _lowestOrder ? 1. : 0.;

   fillCoeffValues(_wksp, _coefList);

   return RooFit::Detail::MathFuncs::polynomial(_wksp.data(), sz, _lowestOrder, _x);
}

void RooPolyVar::doEvalImpl(RooAbsArg const *caller, RooFit::EvalContext &ctx, RooAbsReal const &x,
                            RooArgList const &coefs, int lowestOrder)
{
   std::span<double> output = ctx.output();
   if (coefs.empty()) {
      output[0] = lowestOrder ? 1.0 : 0.0;
      return;
   }

   std::vector<std::span<const double>> vars;
   vars.reserve(coefs.size() + 2);

   // Fill the coefficients for the skipped orders. By a conventions started in
   // RooPolynomial, if the zero-th order is skipped, it implies a coefficient
   // for the constant term of one.
   std::array<double, RooBatchCompute::bufferSize> zeros;
   std::array<double, RooBatchCompute::bufferSize> ones;
   std::fill_n(zeros.data(), zeros.size(), 0.0);
   std::fill_n(ones.data(), ones.size(), 1.0);
   std::span<const double> zerosSpan{zeros.data(), 1};
   std::span<const double> onesSpan{ones.data(), 1};
   for (int i = lowestOrder - 1; i >= 0; --i) {
      vars.push_back(i == 0 ? onesSpan : zerosSpan);
   }

   for (RooAbsArg *coef : coefs) {
      vars.push_back(ctx.at(coef));
   }
   vars.push_back(ctx.at(&x));
   std::array<double, 1> extraArgs{double(vars.size() - 1)};
   RooBatchCompute::compute(ctx.config(caller), RooBatchCompute::Polynomial, ctx.output(), vars, extraArgs);
}

/// Compute multiple values of Polynomial.
void RooPolyVar::doEval(RooFit::EvalContext &ctx) const
{
   doEvalImpl(this, ctx, _x.arg(), _coefList, _lowestOrder);
}

////////////////////////////////////////////////////////////////////////////////
/// Advertise that we can internally integrate over x

Int_t RooPolyVar::getAnalyticalIntegral(RooArgSet &allVars, RooArgSet &analVars, const char * /*rangeName*/) const
{
   return matchArgs(allVars, analVars, _x) ? 1 : 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate and return analytical integral over x

double RooPolyVar::analyticalIntegral(Int_t code, const char *rangeName) const
{
   R__ASSERT(code == 1);

   const double xmin = _x.min(rangeName);
   const double xmax = _x.max(rangeName);
   const unsigned sz = _coefList.size();
   if (!sz)
      return _lowestOrder ? xmax - xmin : 0.0;

   fillCoeffValues(_wksp, _coefList);

   return RooFit::Detail::MathFuncs::polynomialIntegral(_wksp.data(), sz, _lowestOrder, xmin, xmax);
}
