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

#include "TError.h"

ClassImp(RooPolyVar);

////////////////////////////////////////////////////////////////////////////////
/// Construct polynomial in x with coefficients in coefList. If
/// lowestOrder is not zero, then the first element in coefList is
/// interpreted as as the 'lowestOrder' coefficients and all
/// subsequent coeffient elements are shifted by a similar amount.
RooPolyVar::RooPolyVar(const char* name, const char* title,
              RooAbsReal& x, const RooArgList& coefList, Int_t lowestOrder) :
  RooAbsReal(name, title),
  _x("x", "Dependent", this, x),
  _coefList("coefList","List of coefficients",this),
  _lowestOrder(lowestOrder)
{
  // Check lowest order
  if (_lowestOrder<0) {
    coutE(InputArguments) << "RooPolyVar::ctor(" << GetName()
           << ") WARNING: lowestOrder must be >=0, setting value to 0" << std::endl;
    _lowestOrder=0 ;
  }

  for(RooAbsArg * coef : coefList) {
    if (!dynamic_cast<RooAbsReal*>(coef)) {
      coutE(InputArguments) << "RooPolyVar::ctor(" << GetName() << ") ERROR: coefficient " << coef->GetName()
             << " is not of type RooAbsReal" << std::endl;
      R__ASSERT(0) ;
    }
    _coefList.add(*coef) ;
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Constructor of flat polynomial function

RooPolyVar::RooPolyVar(const char* name, const char* title,
                           RooAbsReal& x) :
  RooAbsReal(name, title),
  _x("x", "Dependent", this, x),
  _coefList("coefList","List of coefficients",this),
  _lowestOrder(1)
{ }



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooPolyVar::RooPolyVar(const RooPolyVar& other, const char* name) :
  RooAbsReal(other, name),
  _x("x", this, other._x),
  _coefList("coefList",this,other._coefList),
  _lowestOrder(other._lowestOrder)
{ }


////////////////////////////////////////////////////////////////////////////////
/// Calculate and return value of polynomial

double RooPolyVar::evaluate() const
{
  const unsigned sz = _coefList.getSize();
  const int lowestOrder = _lowestOrder;
  if (!sz) return lowestOrder ? 1. : 0.;
  _wksp.clear();
  _wksp.reserve(sz);
  {
    const RooArgSet* nset = _coefList.nset();
    for (const auto arg : _coefList) {
      const auto c = static_cast<RooAbsReal*>(arg);
      _wksp.push_back(c->getVal(nset));
    }
  }
  const double x = _x;
  double retVal = _wksp[sz - 1];
  for (unsigned i = sz - 1; i--; ) retVal = _wksp[i] + x * retVal;
  return retVal * std::pow(x, lowestOrder);
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
   for(int i = lowestOrder - 1; i >= 0; --i) {
      vars.push_back(i == 0 ? RooSpan<const double>{&one, 1} : RooSpan<const double>{&zero, 1});
   }

   for (RooAbsArg *coef : coefs) {
      vars.push_back(dataMap.at(coef));
   }
   vars.push_back(dataMap.at(&x));
   auto dispatch = stream ? RooBatchCompute::dispatchCUDA : RooBatchCompute::dispatchCPU;
   dispatch->compute(stream, RooBatchCompute::Polynomial, output, nEvents, vars, {double(vars.size() - 1)});
}

/// Compute multiple values of Polynomial.
void RooPolyVar::computeBatch(cudaStream_t *stream, double *output, size_t nEvents,
                              RooFit::Detail::DataMap const &dataMap) const
{
   computeBatchImpl(stream, output, nEvents, dataMap, _x.arg(), _coefList, _lowestOrder);
}

////////////////////////////////////////////////////////////////////////////////
/// Advertise that we can internally integrate over x

Int_t RooPolyVar::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /*rangeName*/) const
{
  if (matchArgs(allVars, analVars, _x)) return 1;
  return 0;
}



////////////////////////////////////////////////////////////////////////////////
/// Calculate and return analytical integral over x

double RooPolyVar::analyticalIntegral(Int_t code, const char* rangeName) const
{
  R__ASSERT(code==1) ;

  const double xmin = _x.min(rangeName), xmax = _x.max(rangeName);
  const int lowestOrder = _lowestOrder;
  const unsigned sz = _coefList.getSize();
  if (!sz) return xmax - xmin;
  _wksp.clear();
  _wksp.reserve(sz);
  {
    const RooArgSet* nset = _coefList.nset();
    unsigned i = 1 + lowestOrder;
    for(auto * c : static_range_cast<RooAbsReal*>(_coefList)) {
      _wksp.push_back(c->getVal(nset) / double(i));
      ++i;
    }
  }
  double min = _wksp[sz - 1], max = _wksp[sz - 1];
  for (unsigned i = sz - 1; i--; )
    min = _wksp[i] + xmin * min, max = _wksp[i] + xmax * max;
  return max * std::pow(xmax, 1 + lowestOrder) - min * std::pow(xmin, 1 + lowestOrder);
}
