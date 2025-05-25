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

/** \class RooArgusBG
    \ingroup Roofit

RooArgusBG is a RooAbsPdf implementation describing the ARGUS background shape.
\f[
  \mathrm{Argus}(m, m_0, c, p) = \mathcal{N} \cdot m \cdot \left[ 1 - \left( \frac{m}{m_0} \right)^2 \right]^p
  \cdot \exp\left[ c \cdot \left(1 - \left(\frac{m}{m_0}\right)^2 \right) \right]
\f]
\image html RooArgusBG.png
*/

#include "RooArgusBG.h"
#include "RooRealVar.h"
#include "RooRealConstant.h"
#include "RooMath.h"
#include "RooBatchCompute.h"

#include "TMath.h"

#include <cmath>


////////////////////////////////////////////////////////////////////////////////
/// Constructor.

RooArgusBG::RooArgusBG(const char *name, const char *title,
             RooAbsReal::Ref _m, RooAbsReal::Ref _m0, RooAbsReal::Ref _c, RooAbsReal::Ref _p) :
  RooAbsPdf(name, title),
  m("m","Mass",this,_m),
  m0("m0","Resonance mass",this,_m0),
  c("c","Slope parameter",this,_c),
  p("p","Power",this,_p)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

RooArgusBG::RooArgusBG(const RooArgusBG& other, const char* name) :
  RooAbsPdf(other,name),
  m("m",this,other.m),
  m0("m0",this,other.m0),
  c("c",this,other.c),
  p("p",this,other.p)
{
}

////////////////////////////////////////////////////////////////////////////////

double RooArgusBG::evaluate() const {
  double t= m/m0;
  if(t >= 1) return 0;

  double u= 1 - t*t;
  return m*std::pow(u,p)*exp(c*u) ;
}

////////////////////////////////////////////////////////////////////////////////

void RooArgusBG::doEval(RooFit::EvalContext & ctx) const
{
  RooBatchCompute::compute(ctx.config(this), RooBatchCompute::ArgusBG, ctx.output(),
          {ctx.at(m), ctx.at(m0), ctx.at(c), ctx.at(p)});
}

////////////////////////////////////////////////////////////////////////////////

Int_t RooArgusBG::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /*rangeName*/) const
{
  if (p.arg().isConstant()) {
    // We can integrate over m if power = 0.5
    if (matchArgs(allVars,analVars,m) && p == 0.5) return 1;
  }
  return 0;

}

////////////////////////////////////////////////////////////////////////////////

double RooArgusBG::analyticalIntegral(Int_t code, const char* rangeName) const
{
  R__ASSERT(code==1);
  // Formula for integration over m when p=0.5
  static const double pi = atan2(0.0,-1.0);
  double min = (m.min(rangeName) < m0) ? m.min(rangeName) : m0;
  double max = (m.max(rangeName) < m0) ? m.max(rangeName) : m0;
  double f1 = (1.-std::pow(min/m0,2));
  double f2 = (1.-std::pow(max/m0,2));
  double aLow;
  double aHigh;
  if ( c < 0. ) {
    aLow  = -0.5*m0*m0*(exp(c*f1)*sqrt(f1)/c + 0.5/std::pow(-c,1.5)*sqrt(pi)*RooMath::erf(sqrt(-c*f1)));
    aHigh = -0.5*m0*m0*(exp(c*f2)*sqrt(f2)/c + 0.5/std::pow(-c,1.5)*sqrt(pi)*RooMath::erf(sqrt(-c*f2)));
  } else if ( c == 0. ) {
    aLow  = -m0*m0/3.*f1*sqrt(f1);
    aHigh = -m0*m0/3.*f1*sqrt(f2);
  } else {
    aLow  = 0.5*m0*m0*exp(c*f1)/(c*sqrt(c)) * (0.5*sqrt(pi)*(RooMath::faddeeva(sqrt(c*f1))).imag() - sqrt(c*f1));
    aHigh = 0.5*m0*m0*exp(c*f2)/(c*sqrt(c)) * (0.5*sqrt(pi)*(RooMath::faddeeva(sqrt(c*f2))).imag() - sqrt(c*f2));
  }
  double area = aHigh - aLow;
  //cout << "c = " << c << "aHigh = " << aHigh << " aLow = " << aLow << " area = " << area << std::endl ;
  return area;

}
