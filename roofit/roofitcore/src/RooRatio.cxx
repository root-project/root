// Author: Rahul Balasubramanian, Nikhef 01 Apr 2021

/*****************************************************************************
 * RooFit
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2019, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

/**
\file RooRatio.cxx
\class RooRatio
\ingroup Roofitcore

A RooRatio represents the ratio of two given RooAbsReal objects.

**/

#include "RooRatio.h"
#include "Riostream.h"
#include "RooMsgService.h"
#include "RooRealVar.h"
#include "RooTrace.h"
#include "RooBatchCompute.h"

#include "TMath.h"

#include <math.h>

ClassImp(RooRatio);

RooRatio::RooRatio(){TRACE_CREATE}

RooRatio::RooRatio(const char *name, const char *title, RooAbsReal &nr,
                   RooAbsReal &dr)
    : RooAbsReal(name, title), _numerator("numerator", "numerator", this, nr),
      _denominator("denominator", "denominator", this, dr){TRACE_CREATE}

      RooRatio::RooRatio(const char *name, const char *title, RooAbsReal &nr,
                         double dr)
    : RooAbsReal(name, title), _numerator("numerator", "numerator", this, nr),
      _denominator("denominator", "denominator", this) {
  auto drvar = new RooRealVar(Form("%s_dr", name), Form("%s_dr", name), dr);
  _denominator.setArg(*drvar);
  addOwnedComponents(RooArgSet(*drvar));
  TRACE_CREATE
}

RooRatio::RooRatio(const char *name, const char *title, double nr,
                   RooAbsReal &dr)
    : RooAbsReal(name, title), _numerator("numerator", "numerator", this),
      _denominator("denominator", "denominator", this, dr) {
  auto nrvar = new RooRealVar(Form("%s_nr", name), Form("%s_nr", name), nr);
  _numerator.setArg(*nrvar);
  addOwnedComponents(RooArgSet(*nrvar));
  TRACE_CREATE
}

RooRatio::RooRatio(const char *name, const char *title, double nr,
                   double dr)
    : RooAbsReal(name, title), _numerator("numerator", "numerator", this),
      _denominator("denominator", "denominator", this) {
  auto nrvar = new RooRealVar(Form("%s_nr", name), Form("%s_nr", name), nr);
  auto drvar = new RooRealVar(Form("%s_dr", name), Form("%s_dr", name), dr);
  _numerator.setArg(*nrvar);
  _denominator.setArg(*drvar);
  addOwnedComponents(RooArgSet(*nrvar, *drvar));
  TRACE_CREATE
}

RooRatio::RooRatio(const char *name, const char *title,
                   const RooArgList &nrlist, const RooArgList &drlist)
    : RooAbsReal(name, title), _numerator("numerator", "numerator", this),
      _denominator("denominator", "denominator", this) {
  auto nrprod =
      new RooProduct(Form("%s_nr", name), Form("%s_nr", name), nrlist);
  auto drprod =
      new RooProduct(Form("%s_dr", name), Form("%s_dr", name), drlist);
  _numerator.setArg(*nrprod);
  _denominator.setArg(*drprod);
  addOwnedComponents(RooArgSet(*nrprod, *drprod));
  TRACE_CREATE
}

RooRatio::~RooRatio(){TRACE_DESTROY}

RooRatio::RooRatio(const RooRatio &other, const char *name)
    : RooAbsReal(other, name), _numerator("numerator", this, other._numerator),
      _denominator("denominator", this, other._denominator){TRACE_CREATE}

      double RooRatio::evaluate() const {

  if (_denominator == 0.0) {
    if (_numerator == 0.0)
      return std::numeric_limits<double>::quiet_NaN();
    else
      return (_numerator > 0.0) ? RooNumber::infinity()
                                : -1.0 * RooNumber::infinity();
  } else
    return _numerator / _denominator;
}

////////////////////////////////////////////////////////////////////////////////
/// Evaluate in batch mode.
void RooRatio::computeBatch(cudaStream_t* stream, double* output, size_t nEvents, RooFit::Detail::DataMap const& dataMap) const
{
  auto dispatch = stream ? RooBatchCompute::dispatchCUDA : RooBatchCompute::dispatchCPU;
  dispatch->compute(stream, RooBatchCompute::Ratio, output, nEvents, {dataMap.at(_numerator), dataMap.at(_denominator)});
}
