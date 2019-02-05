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

#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include <math.h>

#include "RooExponential.h"
#include "RooRealVar.h"
#include "BatchHelpers.h"

#ifdef USE_VDT
#include "vdt/exp.h"
#endif

using namespace std;

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
///cout << "exp(x=" << x << ",c=" << c << ")=" << exp(c*x) << endl ;

Double_t RooExponential::evaluate() const{
  return exp(c*x);
}

////////////////////////////////////////////////////////////////////////////////

Int_t RooExponential::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /*rangeName*/) const
{
  if (matchArgs(allVars,analVars,x)) return 1;
  if (matchArgs(allVars,analVars,c)) return 2;
  return 0 ;
}

////////////////////////////////////////////////////////////////////////////////

Double_t RooExponential::analyticalIntegral(Int_t code, const char* rangeName) const
{
  assert(code == 1 || code ==2);

  auto& constant  = code == 1 ? c : x;
  auto& integrand = code == 1 ? x : c;

  if (constant == 0.0) {
    return integrand.max(rangeName) - integrand.min(rangeName);
  }

  return (exp(constant*integrand.max(rangeName)) - exp(constant*integrand.min(rangeName)))
      / constant;
}


namespace {

template<class Tx, class Tc>
void compute(RooSpan<double> output, Tx x, Tc c) {
  const int n = output.size();

  #pragma omp simd
  for (int i = 0; i < n; ++i) {
#ifdef USE_VDT
    output[i] = vdt::fast_exp(x[i]*c[i]);
#else
    output[i] = exp(x[i]*c[i]);
#endif
  }
}

}


void RooExponential::evaluateBatch(RooSpan<double> output,
      const std::vector<RooSpan<const double>>& inputs,
      const RooArgSet& inputVars) const {
  BatchHelpers::LookupBatchData lu(inputs, inputVars, {x, c});
  assert(!lu.testOverlap(output));

  //Now explicitly write down all possible template instantiations of compute() above:
  const bool batchX = lu.isBatch(x);
  const bool batchC = lu.isBatch(c);

  if (batchX && !batchC) {
    compute(output, lu.data(x), BatchHelpers::BracketAdapter<RooRealProxy>(c));
  } else if (!batchX && batchC) {
    compute(output, BatchHelpers::BracketAdapter<RooRealProxy>(x), lu.data(c));
  } else if (!batchX && !batchC) {
    compute(output,
        BatchHelpers::BracketAdapter<RooRealProxy>(x),
        BatchHelpers::BracketAdapter<RooRealProxy>(c));
  } else {
    compute(output, lu.data(x), lu.data(c));
  }
}
