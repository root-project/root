/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 * @(#)root/roofit:$Id$
 * Authors:                                                                  *
 *   Abi Soffer, Colorado State University, abi@slac.stanford.edu            *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California,         *
 *                          Colorado State University                        *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

/** \class RooBifurGauss
    \ingroup Roofit

Bifurcated Gaussian p.d.f with different widths on left and right
side of maximum value.
**/

#include <RooBifurGauss.h>

#include "RooBatchCompute.h"

#include <RooFit/Detail/AnalyticalIntegrals.h>
#include <RooFit/Detail/EvaluateFuncs.h>

ClassImp(RooBifurGauss);

////////////////////////////////////////////////////////////////////////////////

RooBifurGauss::RooBifurGauss(const char *name, const char *title, RooAbsReal &_x, RooAbsReal &_mean,
                             RooAbsReal &_sigmaL, RooAbsReal &_sigmaR)
   : RooAbsPdf(name, title),
     x("x", "Dependent", this, _x),
     mean("mean", "Mean", this, _mean),
     sigmaL("sigmaL", "Left Sigma", this, _sigmaL),
     sigmaR("sigmaR", "Right Sigma", this, _sigmaR)

{
}

////////////////////////////////////////////////////////////////////////////////

RooBifurGauss::RooBifurGauss(const RooBifurGauss &other, const char *name)
   : RooAbsPdf(other, name),
     x("x", this, other.x),
     mean("mean", this, other.mean),
     sigmaL("sigmaL", this, other.sigmaL),
     sigmaR("sigmaR", this, other.sigmaR)
{
}

////////////////////////////////////////////////////////////////////////////////

double RooBifurGauss::evaluate() const
{
   return RooFit::Detail::EvaluateFuncs::bifurGaussEvaluate(x, mean, sigmaL, sigmaR);
}

////////////////////////////////////////////////////////////////////////////////

void RooBifurGauss::translate(RooFit::Detail::CodeSquashContext &ctx) const
{
   ctx.addResult(this, ctx.buildCall("RooFit::Detail::EvaluateFuncs::bifurGaussEvaluate", x, mean, sigmaL, sigmaR));
}

////////////////////////////////////////////////////////////////////////////////
/// Compute multiple values of BifurGauss distribution.
void RooBifurGauss::doEval(RooFit::EvalContext & ctx) const
{
   RooBatchCompute::compute(ctx.config(this), RooBatchCompute::BifurGauss, ctx.output(),
          {ctx.at(x),ctx.at(mean),ctx.at(sigmaL),ctx.at(sigmaR)});
}

////////////////////////////////////////////////////////////////////////////////

Int_t RooBifurGauss::getAnalyticalIntegral(RooArgSet &allVars, RooArgSet &analVars, const char * /*rangeName*/) const
{
   if (matchArgs(allVars, analVars, x))
      return 1;
   if (matchArgs(allVars, analVars, mean))
      return 2;
   return 0;
}

////////////////////////////////////////////////////////////////////////////////

double RooBifurGauss::analyticalIntegral(Int_t code, const char *rangeName) const
{
   auto &constant = code == 1 ? mean : x;
   auto &integrand = code == 1 ? x : mean;

   return RooFit::Detail::AnalyticalIntegrals::bifurGaussIntegral(integrand.min(rangeName), integrand.max(rangeName),
                                                                  constant, sigmaL, sigmaR);
}

////////////////////////////////////////////////////////////////////////////////

std::string RooBifurGauss::buildCallToAnalyticIntegral(Int_t code, const char *rangeName,
                                                       RooFit::Detail::CodeSquashContext &ctx) const
{
   auto &constant = code == 1 ? mean : x;
   auto &integrand = code == 1 ? x : mean;

   return ctx.buildCall("RooFit::Detail::AnalyticalIntegrals::bifurGaussIntegral", integrand.min(rangeName),
                        integrand.max(rangeName), constant, sigmaL, sigmaR);
}
