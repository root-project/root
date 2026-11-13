/*
 * Project: RooFit
 * Author:
 *   Rahul Balasubramanian, Nikhef 01 Apr 2021
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

/**
\file RooRatio.cxx
\class RooRatio
\ingroup Roofitcore

Represents the ratio of two RooAbsReal objects.

**/

#include <RooRatio.h>

#include <RooBatchCompute.h>
#include <RooMsgService.h>
#include <RooProduct.h>
#include <RooRealVar.h>

#include "RooFit/Detail/MathFuncs.h"

#include <TMath.h>


RooRatio::RooRatio()
{
}

RooRatio::RooRatio(const char *name, const char *title, RooAbsReal &numerator, RooAbsReal &denominator)
   : RooAbsReal(name, title),
     _numerator("numerator", "numerator", this, numerator),
     _denominator("denominator", "denominator", this, denominator)
{
}

RooRatio::RooRatio(const char *name, const char *title, RooAbsReal &numerator, double denominator)
   : RooAbsReal(name, title),
     _numerator("numerator", "numerator", this, numerator),
     _denominator("denominator", "denominator", this)
{
   auto drvar = new RooRealVar(Form("%s_dr", name), Form("%s_dr", name), denominator);
   _denominator.setArg(*drvar);
   addOwnedComponents(RooArgSet(*drvar));
}

RooRatio::RooRatio(const char *name, const char *title, double numerator, RooAbsReal &denominator)
   : RooAbsReal(name, title),
     _numerator("numerator", "numerator", this),
     _denominator("denominator", "denominator", this, denominator)
{
   auto nrvar = new RooRealVar(Form("%s_nr", name), Form("%s_nr", name), numerator);
   _numerator.setArg(*nrvar);
   addOwnedComponents(RooArgSet(*nrvar));
}

RooRatio::RooRatio(const char *name, const char *title, double numerator, double denominator)
   : RooAbsReal(name, title),
     _numerator("numerator", "numerator", this),
     _denominator("denominator", "denominator", this)
{
   auto nrvar = new RooRealVar(Form("%s_nr", name), Form("%s_nr", name), numerator);
   auto drvar = new RooRealVar(Form("%s_dr", name), Form("%s_dr", name), denominator);
   _numerator.setArg(*nrvar);
   _denominator.setArg(*drvar);
   addOwnedComponents(RooArgSet(*nrvar, *drvar));
}

RooRatio::RooRatio(const char *name, const char *title, const RooArgList &num, const RooArgList &denom)
   : RooAbsReal(name, title),
     _numerator("numerator", "numerator", this),
     _denominator("denominator", "denominator", this)
{
   auto nrprod = new RooProduct(Form("%s_nr", name), Form("%s_nr", name), num);
   auto drprod = new RooProduct(Form("%s_dr", name), Form("%s_dr", name), denom);
   _numerator.setArg(*nrprod);
   _denominator.setArg(*drprod);
   addOwnedComponents(RooArgSet(*nrprod, *drprod));
}

RooRatio::~RooRatio()
{
}

RooRatio::RooRatio(const RooRatio &other, const char *name)
   : RooAbsReal(other, name),
     _numerator("numerator", this, other._numerator),
     _denominator("denominator", this, other._denominator)
{
}

double RooRatio::evaluate() const
{
   return RooFit::Detail::MathFuncs::ratio(_numerator, _denominator);
}

////////////////////////////////////////////////////////////////////////////////
/// Evaluate in batch mode.
void RooRatio::doEval(RooFit::EvalContext &ctx) const
{
   RooBatchCompute::compute(ctx.config(this), RooBatchCompute::Ratio, ctx.output(),
                            {ctx.at(_numerator), ctx.at(_denominator)});
}
