/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 * @(#)root/roofit:$Id$
 * Authors:                                                                  *
 *   AS, Abi Soffer, Colorado State University, abi@slac.stanford.edu        *
 *   TS, Thomas Schietinger, SLAC, schieti@slac.stanford.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          Colorado State University                        *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

/** \class RooBreitWigner
    \ingroup Roofit

Class RooBreitWigner is a RooAbsPdf implementation
that models a non-relativistic Breit-Wigner shape
**/

#include "RooFit.h"

#include "Riostream.h"
#include <math.h>

#include "RooBreitWigner.h"
#include "RooAbsReal.h"
#include "RooRealVar.h"
#include "BatchHelpers.h"
#include "RooFitComputeInterface.h"

using namespace std;

ClassImp(RooBreitWigner);

////////////////////////////////////////////////////////////////////////////////

RooBreitWigner::RooBreitWigner(const char *name, const char *title,
          RooAbsReal& _x, RooAbsReal& _mean,
          RooAbsReal& _width) :
  RooAbsPdf(name,title),
  x("x","Dependent",this,_x),
  mean("mean","Mean",this,_mean),
  width("width","Width",this,_width)
{
}

////////////////////////////////////////////////////////////////////////////////

RooBreitWigner::RooBreitWigner(const RooBreitWigner& other, const char* name) :
  RooAbsPdf(other,name), x("x",this,other.x), mean("mean",this,other.mean),
  width("width",this,other.width)
{
}

////////////////////////////////////////////////////////////////////////////////

Double_t RooBreitWigner::evaluate() const
{
  Double_t arg= x - mean;
  return 1. / (arg*arg + 0.25*width*width);
}

////////////////////////////////////////////////////////////////////////////////

namespace {
//Author: Emmanouil Michalainas, CERN 21 August 2019

template<class Tx, class Tmean, class Twidth>
void compute(	size_t batchSize,
              double * __restrict output,
              Tx X, Tmean M, Twidth W)
{
  for (size_t i=0; i<batchSize; i++) {
    const double arg = X[i]-M[i];
    output[i] = 1 / (arg*arg + 0.25*W[i]*W[i]);
  }
}
};

RooSpan<double> RooBreitWigner::evaluateBatch(std::size_t begin, std::size_t batchSize) const {
  using namespace BatchHelpers;
  EvaluateInfo info = getInfo( {&x, &mean, &width}, begin, batchSize );
  if (info.nBatches == 0) {
    return {};
  }

  auto output = _batchData.makeWritableBatchUnInit(begin, batchSize);
  auto xData = x.getValBatch(begin, info.size);
  
  if (info.nBatches==1 && !xData.empty()) {
    compute(batchSize, output.data(), xData.data(), BracketAdapter<double> (mean), BracketAdapter<double> (width));
  }
  else {
    compute(batchSize, output.data(), 
    BracketAdapterWithMask (x,xData), 
    BracketAdapterWithMask (mean,mean.getValBatch(begin,info.size)), 
    BracketAdapterWithMask (width,width.getValBatch(begin,info.size)) );
  }
  return output;
}

////////////////////////////////////////////////////////////////////////////////

RooSpan<double> RooBreitWigner::evaluateSpan(BatchHelpers::RunContext& evalData, const RooArgSet* normSet) const {
  return RooFitCompute::dispatch->computeBreitWigner(this, evalData, x->getValues(evalData, normSet), mean->getValues(evalData, normSet), width->getValues(evalData, normSet));
}

////////////////////////////////////////////////////////////////////////////////

Int_t RooBreitWigner::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /*rangeName*/) const
{
  if (matchArgs(allVars,analVars,x)) return 1 ;
  return 0 ;
}

////////////////////////////////////////////////////////////////////////////////

Double_t RooBreitWigner::analyticalIntegral(Int_t code, const char* rangeName) const
{
  switch(code) {
  case 1:
    {
      Double_t c = 2./width;
      return c*(atan(c*(x.max(rangeName)-mean)) - atan(c*(x.min(rangeName)-mean)));
    }
  }

  assert(0) ;
  return 0 ;
}
