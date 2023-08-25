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

#include <RooFit/Detail/AnalyticalIntegrals.h>

#include <cmath>

ClassImp(RooExponential);

////////////////////////////////////////////////////////////////////////////////

RooExponential::RooExponential(const char *name, const char *title,
                RooAbsReal& _x, RooAbsReal& _c) :
  RooAbsPdf(name, title),
  x("x","Dependent",this,_x),
  c("c","Exponent",this,_c)
{
}

////////////////////////////////////////////////////////////////////////////////

RooExponential::RooExponential(const RooExponential& other, const char* name) :
  RooAbsPdf(other, name), x("x",this,other.x), c("c",this,other.c)
{
}

////////////////////////////////////////////////////////////////////////////////

double RooExponential::evaluate() const{
  return std::exp(c*x);
}

////////////////////////////////////////////////////////////////////////////////
/// Compute multiple values of Exponential distribution.
void RooExponential::computeBatch(double* output, size_t nEvents, RooFit::Detail::DataMap const& dataMap) const
{
   RooBatchCompute::compute(dataMap.config(this), RooBatchCompute::Exponential, output, nEvents, {dataMap.at(x),dataMap.at(c)});
}


Int_t RooExponential::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /*rangeName*/) const
{
  if (matchArgs(allVars,analVars,x)) return 1;
  if (matchArgs(allVars,analVars,c)) return 2;
  return 0 ;
}

////////////////////////////////////////////////////////////////////////////////

double RooExponential::analyticalIntegral(Int_t code, const char* rangeName) const
{
  assert(code == 1 || code ==2);

  auto& constant  = code == 1 ? c : x;
  auto& integrand = code == 1 ? x : c;

  return RooFit::Detail::AnalyticalIntegrals::exponentialIntegral(integrand.min(rangeName), integrand.max(rangeName), constant);
}

////////////////////////////////////////////////////////////////////////////////

void RooExponential::translate(RooFit::Detail::CodeSquashContext &ctx) const
{
   // Build a call to the stateless gaussian defined later.
   ctx.addResult(this, "std::exp(" + ctx.getResult(c) + " * " + ctx.getResult(x) + ")");
}

////////////////////////////////////////////////////////////////////////////////

std::string RooExponential::buildCallToAnalyticIntegral(Int_t code, const char *rangeName,
                                                     RooFit::Detail::CodeSquashContext &ctx) const
{
   auto& constant  = code == 1 ? c : x;
   auto& integrand = code == 1 ? x : c;

   return ctx.buildCall("RooFit::Detail::AnalyticalIntegrals::exponentialIntegral",
                        integrand.min(rangeName), integrand.max(rangeName), constant);
}
