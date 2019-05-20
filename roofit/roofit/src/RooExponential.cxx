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

////////////////////////////////////////////////////////////////////////////////
/// Evaluate the exponential without normalising it on the given batch.
/// \param[in] batchIndex Index of the batch to be computed.
/// \param[in] batchSize Size of each batch. The last batch may be smaller.
/// \return A span with the computed values.

RooSpan<double> RooExponential::evaluateBatch(std::size_t begin, std::size_t end) const {
  auto output = _batchData.makeWritableBatchUnInit(begin, end);

  //Now explicitly write down all possible template instantiations of compute() above:
  auto xData = x.getValBatch(begin, end);
  auto cData = c.getValBatch(begin, end);

  const bool batchX = !xData.empty();
  const bool batchC = !cData.empty();

  if (batchX && !batchC) {
    compute(output, xData, BatchHelpers::BracketAdapter<RooRealProxy>(c));
  } else if (!batchX && batchC) {
    compute(output, BatchHelpers::BracketAdapter<RooRealProxy>(x), cData);
  } else if (!batchX && !batchC) {
    compute(output,
        BatchHelpers::BracketAdapter<RooRealProxy>(x),
        BatchHelpers::BracketAdapter<RooRealProxy>(c));
  } else {
    compute(output, xData, cData);
  }

  return output;
}
