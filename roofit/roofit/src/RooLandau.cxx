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

#include "TMath.h"
#include "Math/ProbFunc.h"

ClassImp(RooLandau);

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
  return TMath::Landau(x, mean, sigma);
}

////////////////////////////////////////////////////////////////////////////////

void RooLandau::translate(RooFit::Detail::CodeSquashContext &ctx) const
{
   ctx.addResult(this, ctx.buildCall("TMath::Landau", x, mean, sigma));
}

////////////////////////////////////////////////////////////////////////////////
/// Compute multiple values of Landau distribution.
void RooLandau::computeBatch(cudaStream_t* stream, double* output, size_t nEvents, RooFit::Detail::DataMap const& dataMap) const
{
  auto dispatch = stream ? RooBatchCompute::dispatchCUDA : RooBatchCompute::dispatchCPU;
  dispatch->compute(stream, RooBatchCompute::Landau, output, nEvents,
          {dataMap.at(x), dataMap.at(mean), dataMap.at(sigma)});
}

////////////////////////////////////////////////////////////////////////////////

Int_t RooLandau::getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, bool /*staticInitOK*/) const
{
  if (matchArgs(directVars,generateVars,x)) return 1 ;
  return 0;
}

Int_t RooLandau::getAnalyticalIntegral(RooArgSet &allVars, RooArgSet &analVars, const char * /*rangeName*/) const
{
   if (matchArgs(allVars, analVars, x))
      return 1;
   return 0;
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

std::string RooLandau::buildCallToAnalyticIntegral(Int_t /*code*/, const char *rangeName,
                                                   RooFit::Detail::CodeSquashContext &ctx) const
{
   // Don't do anything with "code". It can only be "1" anyway (see
   // implementation of getAnalyticalIntegral).
   const std::string a = ctx.buildCall("ROOT::Math::landau_cdf", x.max(rangeName), sigma, mean);
   const std::string b = ctx.buildCall("ROOT::Math::landau_cdf", x.min(rangeName), sigma, mean);
   return ctx.getResult(sigma) + " * " + "(" + a + " - " + b + ")";
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
