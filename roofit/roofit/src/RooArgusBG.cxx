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

#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include <math.h>

#include "RooArgusBG.h"
#include "RooRealVar.h"
#include "RooRealConstant.h"
#include "RooMath.h"
#include "TMath.h"

#include "TError.h"
#include "BatchHelpers.h"
#include "vdt/exp.h"
#include "vdt/log.h"

using namespace std;

ClassImp(RooArgusBG);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

RooArgusBG::RooArgusBG(const char *name, const char *title,
             RooAbsReal& _m, RooAbsReal& _m0, RooAbsReal& _c) :
  RooAbsPdf(name, title),
  m("m","Mass",this,_m),
  m0("m0","Resonance mass",this,_m0),
  c("c","Slope parameter",this,_c),
  p("p","Power",this,(RooRealVar&)RooRealConstant::value(0.5))
{
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

RooArgusBG::RooArgusBG(const char *name, const char *title,
             RooAbsReal& _m, RooAbsReal& _m0, RooAbsReal& _c, RooAbsReal& _p) :
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

Double_t RooArgusBG::evaluate() const {
  Double_t t= m/m0;
  if(t >= 1) return 0;

  Double_t u= 1 - t*t;
  //cout << "c = " << c << " result = " << m*TMath::Power(u,p)*exp(c*u) << endl ;
  return m*TMath::Power(u,p)*exp(c*u) ;
}

////////////////////////////////////////////////////////////////////////////////

namespace ArgusBatchEvaluate {
//Author: Emmanouil Michalainas, CERN 19 AUGUST 2019  

template<class Tm, class Tm0, class Tc, class Tp>
void compute(  size_t batchSize,
               double * __restrict__ output,
               Tm M, Tm0 M0, Tc C, Tp P)
{
  for (size_t i=0; i<batchSize; i++) {
    const double t = M[i]/M0[i];
    const double u = 1 - t*t;
    output[i] = C[i]*u + P[i]*vdt::fast_log(u);
  }
  for (size_t i=0; i<batchSize; i++) {
    if (M[i] >= M0[i]) output[i] = 0.0;
    else output[i] = M[i]*vdt::fast_exp(output[i]);
  }
}
};

RooSpan<double> RooArgusBG::evaluateBatch(std::size_t begin, std::size_t batchSize) const {
  using namespace BatchHelpers;
  using namespace ArgusBatchEvaluate;
    
  EvaluateInfo info = getInfo( {&m,&m0,&c,&p}, begin, batchSize );
  auto output = _batchData.makeWritableBatchUnInit(begin, info.size);
  auto mData = m.getValBatch(begin, info.size);
  if (info.nBatches == 0) {
    throw std::logic_error("Requested a batch computation, but no batch data available.");
  }
  else if (info.nBatches==1 && !mData.empty()) {
    compute(info.size, output.data(), mData.data(), 
    BracketAdapter<double> (m0), 
    BracketAdapter<double> (c), 
    BracketAdapter<double> (p));
  }
  else {
    compute(info.size, output.data(), 
    BracketAdapterWithMask (m,m.getValBatch(begin,batchSize)), 
    BracketAdapterWithMask (m0,m0.getValBatch(begin,batchSize)), 
    BracketAdapterWithMask (c,c.getValBatch(begin,batchSize)), 
    BracketAdapterWithMask (p,p.getValBatch(begin,batchSize)));
  }
  return output;
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

Double_t RooArgusBG::analyticalIntegral(Int_t code, const char* rangeName) const
{
  R__ASSERT(code==1);
  // Formula for integration over m when p=0.5
  static const Double_t pi = atan2(0.0,-1.0);
  Double_t min = (m.min(rangeName) < m0) ? m.min(rangeName) : m0;
  Double_t max = (m.max(rangeName) < m0) ? m.max(rangeName) : m0;
  Double_t f1 = (1.-TMath::Power(min/m0,2));
  Double_t f2 = (1.-TMath::Power(max/m0,2));
  Double_t aLow, aHigh ;
  if ( c < 0. ) { 
    aLow  = -0.5*m0*m0*(exp(c*f1)*sqrt(f1)/c + 0.5/TMath::Power(-c,1.5)*sqrt(pi)*RooMath::erf(sqrt(-c*f1)));
    aHigh = -0.5*m0*m0*(exp(c*f2)*sqrt(f2)/c + 0.5/TMath::Power(-c,1.5)*sqrt(pi)*RooMath::erf(sqrt(-c*f2)));
  } else if ( c == 0. ) {
    aLow  = -m0*m0/3.*f1*sqrt(f1);
    aHigh = -m0*m0/3.*f1*sqrt(f2);
  } else {
    aLow  = 0.5*m0*m0*exp(c*f1)/(c*sqrt(c)) * (0.5*sqrt(pi)*(RooMath::faddeeva(sqrt(c*f1))).imag() - sqrt(c*f1));
    aHigh = 0.5*m0*m0*exp(c*f2)/(c*sqrt(c)) * (0.5*sqrt(pi)*(RooMath::faddeeva(sqrt(c*f2))).imag() - sqrt(c*f2));
  }
  Double_t area = aHigh - aLow;
  //cout << "c = " << c << "aHigh = " << aHigh << " aLow = " << aLow << " area = " << area << endl ;
  return area;

}
