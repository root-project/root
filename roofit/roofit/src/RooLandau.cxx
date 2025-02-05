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

/** \class RooLandau
    \ingroup Roofit

Landau distribution p.d.f
\image html RF_Landau.png "PDF of the Landau distribution."
**/

#include "RooLandau.h"
#include "RooHelpers.h"
#include "RooRandom.h"
#include "RooBatchCompute.h"

#include "RooFit/Detail/MathFuncs.h"

#include "TMath.h"
#include "Math/ProbFunc.h"


////////////////////////////////////////////////////////////////////////////////

RooLandau::RooLandau(const char *name, const char *title, RooAbsReal::Ref _x, RooAbsReal::Ref _mean, RooAbsReal::Ref _sigma) :
  RooAbsPdf(name,title),
  x("x","Dependent",this,_x),
  mean("mean","Mean",this,_mean),
  sigma("sigma","Width",this,_sigma)
{
  RooHelpers::checkRangeOfParameters(this, {&static_cast<RooAbsReal&>(_sigma)}, 0.0);
}

////////////////////////////////////////////////////////////////////////////////

RooLandau::RooLandau(const RooLandau& other, const char* name) :
  RooAbsPdf(other,name),
  x("x",this,other.x),
  mean("mean",this,other.mean),
  sigma("sigma",this,other.sigma)
{
}

////////////////////////////////////////////////////////////////////////////////

double RooLandau::evaluate() const
{
  return RooFit::Detail::MathFuncs::landau(x, mean, sigma);
}

////////////////////////////////////////////////////////////////////////////////
/// Compute multiple values of Landau distribution.
void RooLandau::doEval(RooFit::EvalContext &ctx) const
{
   RooBatchCompute::compute(ctx.config(this), RooBatchCompute::Landau, ctx.output(),
                            {ctx.at(x), ctx.at(mean), ctx.at(sigma)});
}

////////////////////////////////////////////////////////////////////////////////

Int_t RooLandau::getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, bool /*staticInitOK*/) const
{
  return matchArgs(directVars,generateVars,x) ? 1 : 0;
}

Int_t RooLandau::getAnalyticalIntegral(RooArgSet &allVars, RooArgSet &analVars, const char * /*rangeName*/) const
{
   return matchArgs(allVars, analVars, x) ? 1 : 0;
}

Double_t RooLandau::analyticalIntegral(Int_t /*code*/, const char *rangeName) const
{
   // Don't do anything with "code". It can only be "1" anyway (see
   // implementation of getAnalyticalIntegral).

   const double meanVal = mean;
   const double sigmaVal = sigma;

   const double a = ROOT::Math::landau_cdf(x.max(rangeName), sigmaVal, meanVal);
   const double b = ROOT::Math::landau_cdf(x.min(rangeName), sigmaVal, meanVal);
   return sigmaVal * (a - b);
}

////////////////////////////////////////////////////////////////////////////////

void RooLandau::generateEvent(Int_t code)
{
  assert(1 == code); (void)code;
  double xgen ;
  while(true) {
    xgen = RooRandom::randomGenerator()->Landau(mean,sigma);
    if (xgen<x.max() && xgen>x.min()) {
      x = xgen ;
      break;
    }
  }
  return;
}
