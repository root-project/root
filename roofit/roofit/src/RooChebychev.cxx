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

#include <RooFit/Detail/AnalyticalIntegrals.h>
#include <RooFit/Detail/EvaluateFuncs.h>

#include <cmath>

ClassImp(RooChebychev);

////////////////////////////////////////////////////////////////////////////////

RooChebychev::RooChebychev() : _refRangeName(nullptr)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor

RooChebychev::RooChebychev(const char* name, const char* title,
                           RooAbsReal& x, const RooArgList& coefList):
  RooAbsPdf(name, title),
  _x("x", "Dependent", this, x),
  _coefList("coefficients","List of coefficients",this),
  _refRangeName(nullptr)
{
  for (const auto coef : coefList) {
    if (!dynamic_cast<RooAbsReal*>(coef)) {
      coutE(InputArguments) << "RooChebychev::ctor(" << GetName() <<
       ") ERROR: coefficient " << coef->GetName() <<
       " is not of type RooAbsReal" << std::endl ;
      throw std::invalid_argument("Wrong input arguments for RooChebychev");
    }
    _coefList.add(*coef) ;
  }
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
    _refRangeName = (TNamed*) RooNameReg::instance().constPtr(rangeName) ;
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
  for (auto it : _coefList)
     coeffs.push_back(static_cast<const RooAbsReal &>(*it).getVal());
  return RooFit::Detail::EvaluateFuncs::chebychevEvaluate(coeffs.data(), _coefList.size(), _x, xmin, xmax);
}

void RooChebychev::translate(RooFit::Detail::CodeSquashContext &ctx) const
{
   // first bring the range of the variable _x to the normalised range [-1, 1]
   // calculate sum_k c_k T_k(x) where x is given in the normalised range,
   // c_0 = 1, and the higher coefficients are given in _coefList
   double xmax = _x.max(_refRangeName ? _refRangeName->GetName() : nullptr);
   double xmin = _x.min(_refRangeName ? _refRangeName->GetName() : nullptr);

   ctx.addResult(this,
                 ctx.buildCall("RooFit::Detail::EvaluateFuncs::chebychevEvaluate", _coefList, _coefList.size(), _x, xmin, xmax));
}

////////////////////////////////////////////////////////////////////////////////
/// Compute multiple values of Chebychev.
void RooChebychev::computeBatch(double* output, size_t nEvents, RooFit::Detail::DataMap const& dataMap) const
{
  RooBatchCompute::ArgVector extraArgs;
  for (auto* coef:_coefList)
    extraArgs.push_back( static_cast<const RooAbsReal*>(coef)->getVal() );
  extraArgs.push_back( _x.min(_refRangeName?_refRangeName->GetName() : nullptr) );
  extraArgs.push_back( _x.max(_refRangeName?_refRangeName->GetName() : nullptr) );
  RooBatchCompute::compute(dataMap.config(this), RooBatchCompute::Chebychev, output, nEvents, {dataMap.at(_x)}, extraArgs);
}

////////////////////////////////////////////////////////////////////////////////


Int_t RooChebychev::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /* rangeName */) const
{
  if (matchArgs(allVars, analVars, _x)) return 1;
  return 0;
}

////////////////////////////////////////////////////////////////////////////////

double RooChebychev::analyticalIntegral(Int_t code, const char* rangeName) const
{
  assert(1 == code); (void)code;

  double xmax = _x.max(_refRangeName ? _refRangeName->GetName() : nullptr);
  double xmaxFull = _x.max(rangeName);
  double xmin = _x.min(_refRangeName ? _refRangeName->GetName() : nullptr);
  double xminFull = _x.min(rangeName);
  unsigned int sz = _coefList.size();

  std::vector<double> coeffs;
  for (auto it : _coefList)
     coeffs.push_back(static_cast<const RooAbsReal &>(*it).getVal());

  return RooFit::Detail::AnalyticalIntegrals::chebychevIntegral(coeffs.data(), sz, xmin, xmax, xminFull, xmaxFull);
}

std::string RooChebychev::buildCallToAnalyticIntegral(Int_t /* code */, const char *rangeName,
                                                      RooFit::Detail::CodeSquashContext &ctx) const
{
   double xmax = _x.max(_refRangeName ? _refRangeName->GetName() : nullptr);
   double xmaxFull = _x.max(rangeName);
   double xmin = _x.min(_refRangeName ? _refRangeName->GetName() : nullptr);
   double xminFull = _x.min(rangeName);
   unsigned int sz = _coefList.size();

   return ctx.buildCall("RooFit::Detail::AnalyticalIntegrals::chebychevIntegral", _coefList, sz, xmin, xmax, xminFull, xmaxFull);
}
