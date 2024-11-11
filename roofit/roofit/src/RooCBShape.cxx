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

/** 
 * \class RooCBShape
 * \ingroup Roofit
 * 
 * \brief The RooCBShape class represents the Crystal Ball distribution.
 * 
 * This class implements the Crystal Ball distribution, which is widely used in high-energy physics for fitting 
 * asymmetric distributions. The Crystal Ball function consists of a Gaussian core and a power-law tail on the left side.
 * 
 * \section CrystalBallDistribution Crystal Ball Distribution
 * 
 * The general equation for the Crystal Ball distribution is:
 * 
 * \f[
 * f(x; \alpha, \beta, \mu, \sigma) = 
 * \begin{cases} 
 * \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) & \text{for} \quad \frac{x-\mu}{\sigma} > -\alpha, \\
 * A \left(B - \frac{x-\mu}{\sigma}\right)^{-n} & \text{for} \quad \frac{x-\mu}{\sigma} \leq -\alpha 
 * \end{cases} 
 * \f]
 * 
 * where:
 * 
 * \f[
 * \begin{aligned}
 * n &= \frac{\alpha^2}{\beta}, \\
 * A &= \left(\frac{n}{|\alpha|}\right)^n \exp\left(-\frac{\alpha^2}{2}\right), \\
 * B &= \frac{n}{|\alpha|} - |\alpha| 
 * \end{aligned}
 * \f]
 * 
 * In this equation:
 * - \f$\mu\f$: The mean or peak of the distribution.
 * - \f$\sigma\f$: The standard deviation, which determines the width of the Gaussian core.
 * - \f$\alpha\f$: Controls the transition between the Gaussian core and the power-law tail.
 * - \f$\beta\f$: Controls the slope of the power-law tail.
 * 
 * The parameters and their effects on the distribution's shape are discussed in detail below.
 * 
 * \subsection ParameterDescriptions Parameter Descriptions
 * 
 * \paragraph{Sigma (\f$\sigma\f$)}
 * The parameter \f$\sigma\f$ determines the scale of the Crystal Ball distribution:
 * 
 * \image html combined_sigmas.png "Effect of varying sigma on the Crystal Ball distribution."
 * 
 * As \f$\sigma\f$ increases, the distribution becomes wider and more spread out, indicating greater variability. 
 * Conversely, as \f$\sigma\f$ decreases, the distribution becomes narrower and more concentrated around the mean \f$\mu\f$.
 * 
 * \paragraph{Alpha (\f$\alpha\f$)}
 * The parameter \f$\alpha\f$ controls the transition point between the Gaussian core and the power-law tail:
 * 
 * \image html combined_alphas.png "Effect of varying alpha on the Crystal Ball distribution."
 * 
 * Higher values of \f$\alpha\f$ result in a steeper left tail, while lower values of \f$\alpha\f$ create a smoother transition.
 * This parameter significantly influences the steepness and shape of the left tail of the distribution.
 * 
 * \paragraph{Beta (\f$\beta\f$)}
 * The parameter \f$\beta\f$ determines the power of the tail:
 * 
 * \image html combined_betas.png "Effect of varying beta on the Crystal Ball distribution."
 * 
 * Higher values of \f$\beta\f$ lead to a fatter tail, indicating a higher probability of extreme values. Lower values 
 * of \f$\beta\f$ result in a thinner tail. This parameter affects the thickness and extent of the tail on the left 
 * side of the distribution.
 * 
 * \note The figures included in this documentation visually represent the effects of varying \f$\sigma\f$, \f$\alpha\f$, and \f$\beta\f$ on the shape of the Crystal Ball distribution.
 */


#include "RooCBShape.h"

#include "RooRealVar.h"
#include "RooMath.h"
#include "RooBatchCompute.h"

#include <RooFit/Detail/MathFuncs.h>

#include "TMath.h"

#include <cmath>

ClassImp(RooCBShape);

////////////////////////////////////////////////////////////////////////////////

RooCBShape::RooCBShape(const char *name, const char *title,
             RooAbsReal& _m, RooAbsReal& _m0, RooAbsReal& _sigma,
             RooAbsReal& _alpha, RooAbsReal& _n) :
  RooAbsPdf(name, title),
  m("m", "Dependent", this, _m),
  m0("m0", "M0", this, _m0),
  sigma("sigma", "Sigma", this, _sigma),
  alpha("alpha", "Alpha", this, _alpha),
  n("n", "Order", this, _n)
{
}

////////////////////////////////////////////////////////////////////////////////

RooCBShape::RooCBShape(const RooCBShape& other, const char* name) :
  RooAbsPdf(other, name), m("m", this, other.m), m0("m0", this, other.m0),
  sigma("sigma", this, other.sigma), alpha("alpha", this, other.alpha),
  n("n", this, other.n)
{
}

////////////////////////////////////////////////////////////////////////////////

double RooCBShape::evaluate() const
{
   return RooFit::Detail::MathFuncs::cbShape(m, m0, sigma, alpha, n);
}

void RooCBShape::translate(RooFit::Detail::CodeSquashContext &ctx) const
{
   ctx.addResult(this, ctx.buildCall("RooFit::Detail::MathFuncs::cbShape", m, m0, sigma, alpha, n));
}

////////////////////////////////////////////////////////////////////////////////
/// Compute multiple values of Crystal ball Shape distribution.
void RooCBShape::doEval(RooFit::EvalContext &ctx) const
{
   RooBatchCompute::compute(ctx.config(this), RooBatchCompute::CBShape, ctx.output(),
                            {ctx.at(m), ctx.at(m0), ctx.at(sigma), ctx.at(alpha), ctx.at(n)});
}

////////////////////////////////////////////////////////////////////////////////

Int_t RooCBShape::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /*rangeName*/) const
{
  if( matchArgs(allVars,analVars,m) )
    return 1 ;

  return 0;
}

////////////////////////////////////////////////////////////////////////////////

double RooCBShape::analyticalIntegral(Int_t /*code*/, const char *rangeName) const
{
   using namespace RooFit::Detail::MathFuncs;
   return cbShapeIntegral(m.min(rangeName), m.max(rangeName), m0, sigma, alpha, n);
}

std::string
RooCBShape::buildCallToAnalyticIntegral(Int_t /*code*/, const char *rangeName, RooFit::Detail::CodeSquashContext &ctx) const
{
   return ctx.buildCall("RooFit::Detail::MathFuncs::cbShapeIntegral",
                        m.min(rangeName), m.max(rangeName), m0, sigma, alpha, n);
}

////////////////////////////////////////////////////////////////////////////////
/// Advertise that we know the maximum of self for given (m0,alpha,n,sigma)

Int_t RooCBShape::getMaxVal(const RooArgSet& vars) const
{
   RooArgSet dummy ;

  if (matchArgs(vars,dummy,m)) {
     return 1 ;
   }
  return 0 ;
}

////////////////////////////////////////////////////////////////////////////////

double RooCBShape::maxVal(Int_t code) const
{
  R__ASSERT(code==1) ;

  // The maximum value for given (m0,alpha,n,sigma)
  // is 1./ Integral in the variable range
  return 1.0/analyticalIntegral(1) ;
}
