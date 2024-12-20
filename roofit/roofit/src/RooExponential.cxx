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

/** \class RooExponential
    \ingroup Roofit

Exponential PDF. It computes
\f[
  \mathrm{RooExponential}(x, c) = \mathcal{N} \cdot \exp(c\cdot x),
\f]
where \f$ \mathcal{N} \f$ is a normalisation constant that depends on the
range and values of the arguments.
**/

#include "RooExponential.h"

#include "RooRealVar.h"
#include "RooBatchCompute.h"

#include <RooFit/Detail/MathFuncs.h>

#include <algorithm>
#include <cmath>

ClassImp(RooExponential);

////////////////////////////////////////////////////////////////////////////////

RooExponential::RooExponential(const char *name, const char *title, RooAbsReal &variable, RooAbsReal &coefficient,
                               bool negateCoefficient)
   : RooAbsPdf{name, title},
     x{"x", "Dependent", this, variable},
     c{"c", "Exponent", this, coefficient},
     _negateCoefficient{negateCoefficient}
{
}

////////////////////////////////////////////////////////////////////////////////

RooExponential::RooExponential(const RooExponential &other, const char *name)
   : RooAbsPdf{other, name}, x{"x", this, other.x}, c{"c", this, other.c}, _negateCoefficient{other._negateCoefficient}
{
}

////////////////////////////////////////////////////////////////////////////////

double RooExponential::evaluate() const
{
   double coef = c;
   if (_negateCoefficient) {
      coef = -coef;
   }
   return std::exp(coef * x);
}

////////////////////////////////////////////////////////////////////////////////
/// Compute multiple values of Exponential distribution.
void RooExponential::doEval(RooFit::EvalContext &ctx) const
{
   auto computer = _negateCoefficient ? RooBatchCompute::ExponentialNeg : RooBatchCompute::Exponential;
   RooBatchCompute::compute(ctx.config(this), computer, ctx.output(), {ctx.at(x), ctx.at(c)});
}

Int_t RooExponential::getAnalyticalIntegral(RooArgSet &allVars, RooArgSet &analVars, const char * /*rangeName*/) const
{
   if (matchArgs(allVars, analVars, x))
      return 1;
   if (matchArgs(allVars, analVars, c))
      return 2;
   return 0;
}

////////////////////////////////////////////////////////////////////////////////

double RooExponential::analyticalIntegral(Int_t code, const char *rangeName) const
{
   assert(code == 1 || code == 2);

   bool isOverX = code == 1;

   double coef = c;
   if (_negateCoefficient) {
      coef = -coef;
   }

   double constant = isOverX ? coef : x;
   auto &integrand = isOverX ? x : c;

   double min = integrand.min(rangeName);
   double max = integrand.max(rangeName);

   if (!isOverX && _negateCoefficient) {
      std::swap(min, max);
      min = -min;
      max = -max;
   }

   return RooFit::Detail::MathFuncs::exponentialIntegral(min, max, constant);
}
