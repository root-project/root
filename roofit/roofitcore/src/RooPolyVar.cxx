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

Class RooPolyVar is a RooAbsReal implementing a polynomial in terms
of a list of RooAbsReal coefficients
\f[f(x) = \sum_{i} a_{i} \cdot x^i \f]
Class RooPolyvar implements analytical integrals of all polynomials
it can define.
**/

#include <cmath>

#include "RooPolyVar.h"
#include "RooArgList.h"
#include "RooMsgService.h"
#include "RooBatchCompute.h"

#include <RooFit/Detail/AnalyticalIntegrals.h>
#include <RooFit/Detail/EvaluateFuncs.h>

#include "TError.h"

ClassImp(RooPolyVar);

////////////////////////////////////////////////////////////////////////////////
/// Construct polynomial in x with coefficients in coefList. If
/// lowestOrder is not zero, then the first element in coefList is
/// interpreted as as the 'lowestOrder' coefficients and all
/// subsequent coeffient elements are shifted by a similar amount.
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

   for (RooAbsArg *coef : coefList) {
      if (!dynamic_cast<RooAbsReal *>(coef)) {
         coutE(InputArguments) << "RooPolyVar::ctor(" << GetName() << ") ERROR: coefficient " << coef->GetName()
                               << " is not of type RooAbsReal" << std::endl;
         R__ASSERT(0);
      }
      _coefList.add(*coef);
   }
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
   const unsigned sz = _coefList.getSize();
   if (!sz)
      return _lowestOrder ? 1. : 0.;

   fillCoeffValues(_wksp, _coefList);

   return RooFit::Detail::EvaluateFuncs::polynomialEvaluate(_wksp.data(), sz, _lowestOrder, _x);
}

void RooPolyVar::translate(RooFit::Detail::CodeSquashContext &ctx) const
{
   const unsigned sz = _coefList.size();
   if (!sz) {
      ctx.addResult(this, std::to_string((_lowestOrder ? 1. : 0.)));
      return;
   }

   ctx.addResult(this,
                 ctx.buildCall("RooFit::Detail::EvaluateFuncs::polynomialEvaluate", _coefList, sz, _lowestOrder, _x));
}

void RooPolyVar::computeBatchImpl(cudaStream_t *stream, double *output, size_t nEvents,
                                  RooFit::Detail::DataMap const &dataMap, RooAbsReal const &x, RooArgList const &coefs,
                                  int lowestOrder)
{
   if (coefs.empty()) {
      output[0] = lowestOrder ? 1.0 : 0.0;
      return;
   }

   RooBatchCompute::VarVector vars;
   vars.reserve(coefs.size() + 2);

   // Fill the coefficients for the skipped orders. By a conventions started in
   // RooPolynomial, if the zero-th order is skipped, it implies a coefficient
   // for the constant term of one.
   const double zero = 1.0;
   const double one = 1.0;
   for (int i = lowestOrder - 1; i >= 0; --i) {
      vars.push_back(i == 0 ? RooSpan<const double>{&one, 1} : RooSpan<const double>{&zero, 1});
   }

   for (RooAbsArg *coef : coefs) {
      vars.push_back(dataMap.at(coef));
   }
   vars.push_back(dataMap.at(&x));
   auto dispatch = stream ? RooBatchCompute::dispatchCUDA : RooBatchCompute::dispatchCPU;
   RooBatchCompute::ArgVector extraArgs{double(vars.size() - 1)};
   dispatch->compute(stream, RooBatchCompute::Polynomial, output, nEvents, vars, extraArgs);
}

/// Compute multiple values of Polynomial.
void RooPolyVar::computeBatch(cudaStream_t *stream, double *output, size_t nEvents,
                              RooFit::Detail::DataMap const &dataMap) const
{
   computeBatchImpl(stream, output, nEvents, dataMap, _x.arg(), _coefList, _lowestOrder);
}

////////////////////////////////////////////////////////////////////////////////
/// Advertise that we can internally integrate over x

Int_t RooPolyVar::getAnalyticalIntegral(RooArgSet &allVars, RooArgSet &analVars, const char * /*rangeName*/) const
{
   if (matchArgs(allVars, analVars, _x))
      return 1;
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate and return analytical integral over x

double RooPolyVar::analyticalIntegral(Int_t code, const char *rangeName) const
{
   R__ASSERT(code == 1);

   const double xmin = _x.min(rangeName), xmax = _x.max(rangeName);
   const unsigned sz = _coefList.getSize();
   if (!sz)
      return _lowestOrder ? xmax - xmin : 0.0;

   fillCoeffValues(_wksp, _coefList);

   return RooFit::Detail::AnalyticalIntegrals::polynomialIntegral(_wksp.data(), sz, _lowestOrder, xmin, xmax);
}

std::string RooPolyVar::buildCallToAnalyticIntegral(Int_t /* code */, const char *rangeName,
                                                    RooFit::Detail::CodeSquashContext &ctx) const
{
   const double xmin = _x.min(rangeName), xmax = _x.max(rangeName);
   const unsigned sz = _coefList.getSize();
   if (!sz)
      return std::to_string(_lowestOrder ? xmax - xmin : 0.0);

   return ctx.buildCall("RooFit::Detail::AnalyticalIntegrals::polynomialIntegral", _coefList, sz, _lowestOrder, xmin,
                        xmax);
}
