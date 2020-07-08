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

#include "RooFit.h"
#include "BatchHelpers.h"
#include "RooAbsReal.h"
#include "RooRealVar.h"
#include "RooRandom.h"
#include "RooMath.h"

#include "RooVDTHeaders.h"

using namespace BatchHelpers;
using namespace std;

ClassImp(RooGaussian);

////////////////////////////////////////////////////////////////////////////////

RooGaussian::RooGaussian(const char *name, const char *title,
          RooAbsReal& _x, RooAbsReal& _mean,
          RooAbsReal& _sigma) :
  RooAbsPdf(name,title),
  x("x","Observable",this,_x),
  mean("mean","Mean",this,_mean),
  sigma("sigma","Width",this,_sigma)
{
}

////////////////////////////////////////////////////////////////////////////////

RooGaussian::RooGaussian(const RooGaussian& other, const char* name) :
  RooAbsPdf(other,name), x("x",this,other.x), mean("mean",this,other.mean),
  sigma("sigma",this,other.sigma)
{
}

////////////////////////////////////////////////////////////////////////////////

Double_t RooGaussian::evaluate() const
{
  const double arg = x - mean;
  const double sig = sigma;
  return exp(-0.5*arg*arg/(sig*sig));
}


namespace {

///Actual computations for the batch evaluation of the Gaussian.
///May vectorise over x, mean, sigma, depending on the types of the inputs.
///\note The output and input spans are assumed to be non-overlapping. If they
///overlap, results will likely be garbage.
template<class Tx, class TMean, class TSig>
void compute(RooSpan<double> output, Tx x, TMean mean, TSig sigma) {
  const auto n = output.size();

  for (std::size_t i = 0; i < n; ++i) {
    const double arg = x[i] - mean[i];
    const double halfBySigmaSq = -0.5 / (sigma[i] * sigma[i]);

    output[i] = _rf_fast_exp(arg*arg * halfBySigmaSq);
  }
}

}

////////////////////////////////////////////////////////////////////////////////
/// Compute \f$ \exp(-0.5 \cdot \frac{(x - \mu)^2}{\sigma^2} \f$ in batches.
/// The local proxies {x, mean, sigma} will be searched for batch input data,
/// and if found, the computation will be batched over their
/// values. If batch data are not found for one of the proxies, the proxies value is assumed to
/// be constant over the batch.
/// \param[in] begin Index of the batch to be computed.
/// \param[in] batchSize Size of each batch. The last batch may be smaller.
/// \return A span with the computed values.

RooSpan<double> RooGaussian::evaluateBatch(std::size_t begin, std::size_t batchSize) const {
  auto xData = x.getValBatch(begin, batchSize);
  auto meanData = mean.getValBatch(begin, batchSize);
  auto sigmaData = sigma.getValBatch(begin, batchSize);

  //Now explicitly write down all possible template instantiations of compute() above:
  const bool batchX = !xData.empty();
  const bool batchMean = !meanData.empty();
  const bool batchSigma = !sigmaData.empty();

  if (!(batchX || batchMean || batchSigma)) {
    return {};
  }

  auto output = _batchData.makeWritableBatchUnInit(begin, batchSize);

  if (batchX && !batchMean && !batchSigma) {
    compute(output, xData, BracketAdapter<double>(mean), BracketAdapter<double>(sigma));
  }
  else if (batchX && batchMean && !batchSigma) {
    compute(output, xData, meanData, BracketAdapter<double>(sigma));
  }
  else if (batchX && !batchMean && batchSigma) {
    compute(output, xData, BracketAdapter<double>(mean), sigmaData);
  }
  else if (batchX && batchMean && batchSigma) {
    compute(output, xData, meanData, sigmaData);
  }
  else if (!batchX && batchMean && !batchSigma) {
    compute(output, BracketAdapter<double>(x), meanData, BracketAdapter<double>(sigma));
  }
  else if (!batchX && !batchMean && batchSigma) {
    compute(output, BracketAdapter<double>(x), BracketAdapter<double>(mean), sigmaData);
  }
  else if (!batchX && batchMean && batchSigma) {
    compute(output, BracketAdapter<double>(x), meanData, sigmaData);
  }

  return output;
}

////////////////////////////////////////////////////////////////////////////////

Int_t RooGaussian::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /*rangeName*/) const
{
  if (matchArgs(allVars,analVars,x)) return 1 ;
  if (matchArgs(allVars,analVars,mean)) return 2 ;
  return 0 ;
}

////////////////////////////////////////////////////////////////////////////////

Double_t RooGaussian::analyticalIntegral(Int_t code, const char* rangeName) const
{
  assert(code==1 || code==2);

  //The normalisation constant 1./sqrt(2*pi*sigma^2) is left out in evaluate().
  //Therefore, the integral is scaled up by that amount to make RooFit normalise
  //correctly.
  const double resultScale = sqrt(TMath::TwoPi()) * sigma;

  //Here everything is scaled and shifted into a standard normal distribution:
  const double xscale = TMath::Sqrt2() * sigma;
  double max = 0.;
  double min = 0.;
  if (code == 1){
    max = (x.max(rangeName)-mean)/xscale;
    min = (x.min(rangeName)-mean)/xscale;
  } else { //No == 2 test because of assert
    max = (mean.max(rangeName)-x)/xscale;
    min = (mean.min(rangeName)-x)/xscale;
  }


  //Here we go for maximum precision: We compute all integrals in the UPPER
  //tail of the Gaussian, because erfc has the highest precision there.
  //Therefore, the different cases for range limits in the negative hemisphere are mapped onto
  //the equivalent points in the upper hemisphere using erfc(-x) = 2. - erfc(x)
  const double ecmin = std::erfc(std::abs(min));
  const double ecmax = std::erfc(std::abs(max));


  return resultScale * 0.5 * (
      min*max < 0.0 ? 2.0 - (ecmin + ecmax)
                    : max <= 0. ? ecmax - ecmin : ecmin - ecmax
  );
}

////////////////////////////////////////////////////////////////////////////////

Int_t RooGaussian::getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, Bool_t /*staticInitOK*/) const
{
  if (matchArgs(directVars,generateVars,x)) return 1 ;
  if (matchArgs(directVars,generateVars,mean)) return 2 ;
  return 0 ;
}

////////////////////////////////////////////////////////////////////////////////

void RooGaussian::generateEvent(Int_t code)
{
  assert(code==1 || code==2) ;
  Double_t xgen ;
  if(code==1){
    while(1) {
      xgen = RooRandom::randomGenerator()->Gaus(mean,sigma);
      if (xgen<x.max() && xgen>x.min()) {
   x = xgen ;
   break;
      }
    }
  } else if(code==2){
    while(1) {
      xgen = RooRandom::randomGenerator()->Gaus(x,sigma);
      if (xgen<mean.max() && xgen>mean.min()) {
   mean = xgen ;
   break;
      }
    }
  } else {
    cout << "error in RooGaussian generateEvent"<< endl;
  }

  return;
}
