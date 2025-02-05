/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 * @(#)root/roofit:$Id$
 * Authors:                                                                  *
 *   GR, Gerhard Raven,   UC San Diego, Gerhard.Raven@slac.stanford.edu
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

/** \class RooChebychev
   \ingroup Roofit

Chebychev polynomial p.d.f. of the first kind.

The coefficient that goes with \f$ T_0(x)=1 \f$ (i.e. the constant polynomial) is
implicitly assumed to be 1, and the list of coefficients supplied by callers
starts with the coefficient that goes with \f$ T_1(x)=x \f$ (i.e. the linear term).
**/

#include "RooChebychev.h"
#include "RooRealVar.h"
#include "RooArgList.h"
#include "RooNameReg.h"
#include "RooBatchCompute.h"

#include <RooFit/Detail/MathFuncs.h>

#include <cmath>


////////////////////////////////////////////////////////////////////////////////

RooChebychev::RooChebychev() = default;

////////////////////////////////////////////////////////////////////////////////
/// Constructor

RooChebychev::RooChebychev(const char* name, const char* title,
                           RooAbsReal& x, const RooArgList& coefList):
  RooAbsPdf(name, title),
  _x("x", "Dependent", this, x),
  _coefList("coefficients","List of coefficients",this)
{
  _coefList.addTyped<RooAbsReal>(coefList);
}

////////////////////////////////////////////////////////////////////////////////

RooChebychev::RooChebychev(const RooChebychev& other, const char* name) :
  RooAbsPdf(other, name),
  _x("x", this, other._x),
  _coefList("coefList",this,other._coefList),
  _refRangeName(other._refRangeName)
{
}

////////////////////////////////////////////////////////////////////////////////

void RooChebychev::selectNormalizationRange(const char* rangeName, bool force)
{
  if (rangeName && (force || !_refRangeName)) {
    _refRangeName = const_cast<TNamed*>(RooNameReg::instance().constPtr(rangeName));
  }
  if (!rangeName) {
    _refRangeName = nullptr ;
  }
}

////////////////////////////////////////////////////////////////////////////////

double RooChebychev::evaluate() const
{
   // first bring the range of the variable _x to the normalised range [-1, 1]
   // calculate sum_k c_k T_k(x) where x is given in the normalised range,
   // c_0 = 1, and the higher coefficients are given in _coefList
   double xmax = _x.max(_refRangeName ? _refRangeName->GetName() : nullptr);
   double xmin = _x.min(_refRangeName ? _refRangeName->GetName() : nullptr);

   std::vector<double> coeffs;
   for (auto it : _coefList) {
      coeffs.push_back(static_cast<const RooAbsReal &>(*it).getVal());
   }
   return RooFit::Detail::MathFuncs::chebychev(coeffs.data(), _coefList.size(), _x, xmin, xmax);
}

////////////////////////////////////////////////////////////////////////////////
/// Compute multiple values of Chebychev.
void RooChebychev::doEval(RooFit::EvalContext &ctx) const
{
   std::vector<double> extraArgs;
   extraArgs.reserve(_coefList.size() + 2);
   for (auto *coef : _coefList) {
      extraArgs.push_back(static_cast<const RooAbsReal *>(coef)->getVal());
   }
   extraArgs.push_back(_x.min(_refRangeName ? _refRangeName->GetName() : nullptr));
   extraArgs.push_back(_x.max(_refRangeName ? _refRangeName->GetName() : nullptr));
   RooBatchCompute::compute(ctx.config(this), RooBatchCompute::Chebychev, ctx.output(), {ctx.at(_x)}, extraArgs);
}

////////////////////////////////////////////////////////////////////////////////


Int_t RooChebychev::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /* rangeName */) const
{
  return matchArgs(allVars, analVars, _x) ? 1 : 0;
}

////////////////////////////////////////////////////////////////////////////////

double RooChebychev::analyticalIntegral(Int_t code, const char *rangeName) const
{
   assert(1 == code);
   (void)code;

   double xmax = _x.max(_refRangeName ? _refRangeName->GetName() : nullptr);
   double xmin = _x.min(_refRangeName ? _refRangeName->GetName() : nullptr);
   unsigned int sz = _coefList.size();

   std::vector<double> coeffs;
   for (auto it : _coefList)
      coeffs.push_back(static_cast<const RooAbsReal &>(*it).getVal());

   return RooFit::Detail::MathFuncs::chebychevIntegral(coeffs.data(), sz, xmin, xmax, _x.min(rangeName),
                                                       _x.max(rangeName));
}
