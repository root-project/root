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

/** \class RooGaussian
    \ingroup Roofit

Plain Gaussian p.d.f
**/

#include "RooGaussian.h"
#include "RooBatchCompute.h"
#include "RooHelpers.h"
#include "RooRandom.h"

#include <RooFit/Detail/MathFuncs.h>

#include <vector>


////////////////////////////////////////////////////////////////////////////////

RooGaussian::RooGaussian(const char *name, const char *title,
          RooAbsReal::Ref _x, RooAbsReal::Ref _mean,
          RooAbsReal::Ref _sigma) :
  RooAbsPdf(name,title),
  x("x","Observable",this,_x),
  mean("mean","Mean",this,_mean),
  sigma("sigma","Width",this,_sigma)
{
  RooHelpers::checkRangeOfParameters(this, {&static_cast<RooAbsReal&>(_sigma)}, 0);
}

////////////////////////////////////////////////////////////////////////////////

RooGaussian::RooGaussian(const RooGaussian& other, const char* name) :
  RooAbsPdf(other,name), x("x",this,other.x), mean("mean",this,other.mean),
  sigma("sigma",this,other.sigma)
{
}

////////////////////////////////////////////////////////////////////////////////

double RooGaussian::evaluate() const
{
   return RooFit::Detail::MathFuncs::gaussian(x, mean, sigma);
}


////////////////////////////////////////////////////////////////////////////////
/// Compute multiple values of Gaussian distribution.
void RooGaussian::doEval(RooFit::EvalContext & ctx) const
{
  RooBatchCompute::compute(ctx.config(this), RooBatchCompute::Gaussian, ctx.output(),
          {ctx.at(x), ctx.at(mean), ctx.at(sigma)});
}

////////////////////////////////////////////////////////////////////////////////

Int_t RooGaussian::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /*rangeName*/) const
{
  if (matchArgs(allVars,analVars,x)) return 1 ;
  if (matchArgs(allVars,analVars,mean)) return 2 ;
  return 0 ;
}

////////////////////////////////////////////////////////////////////////////////

double RooGaussian::analyticalIntegral(Int_t code, const char* rangeName) const
{
   using namespace RooFit::Detail::MathFuncs;

   auto& constant  = code == 1 ? mean : x;
   auto& integrand = code == 1 ? x : mean;

   return gaussianIntegral(integrand.min(rangeName), integrand.max(rangeName), constant, sigma);
}

////////////////////////////////////////////////////////////////////////////////

Int_t RooGaussian::getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, bool /*staticInitOK*/) const
{
  if (matchArgs(directVars,generateVars,x)) return 1 ;
  if (matchArgs(directVars,generateVars,mean)) return 2 ;
  return 0 ;
}

////////////////////////////////////////////////////////////////////////////////

void RooGaussian::generateEvent(Int_t code)
{
  assert(code==1 || code==2) ;
  double xgen ;
  if(code==1){
    while(true) {
      xgen = RooRandom::randomGenerator()->Gaus(mean,sigma);
      if (xgen<x.max() && xgen>x.min()) {
   x = xgen ;
   break;
      }
    }
  } else if(code==2){
    while(true) {
      xgen = RooRandom::randomGenerator()->Gaus(x,sigma);
      if (xgen<mean.max() && xgen>mean.min()) {
   mean = xgen ;
   break;
      }
    }
  } else {
    std::cout << "error in RooGaussian generateEvent"<< std::endl;
  }

  return;
}
