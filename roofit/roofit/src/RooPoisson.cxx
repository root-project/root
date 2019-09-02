 /*****************************************************************************
  * Project: RooFit                                                           *
  *                                                                           *
  * Simple Poisson PDF
  * author: Kyle Cranmer <cranmer@cern.ch>
  *                                                                           *
  *****************************************************************************/

/** \class RooPoisson
    \ingroup Roofit

Poisson pdf
**/

#include "RooPoisson.h"

#include "RooAbsReal.h"
#include "RooAbsCategory.h"

#include "RooRandom.h"
#include "RooMath.h"
#include "TMath.h"
#include "Math/ProbFuncMathCore.h"

#include "BatchHelpers.h"
#include "RooVDTHeaders.h"

#include <limits>
#include <cmath>

using namespace std;

ClassImp(RooPoisson);

////////////////////////////////////////////////////////////////////////////////
/// Constructor

RooPoisson::RooPoisson(const char *name, const char *title,
             RooAbsReal& _x,
             RooAbsReal& _mean,
             Bool_t noRounding) :
  RooAbsPdf(name,title),
  x("x","x",this,_x),
  mean("mean","mean",this,_mean),
  _noRounding(noRounding),
  _protectNegative(false)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

 RooPoisson::RooPoisson(const RooPoisson& other, const char* name) :
   RooAbsPdf(other,name),
   x("x",this,other.x),
   mean("mean",this,other.mean),
   _noRounding(other._noRounding),
   _protectNegative(other._protectNegative)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Implementation in terms of the TMath::Poisson() function.

Double_t RooPoisson::evaluate() const
{
  Double_t k = _noRounding ? x : floor(x);
  if(_protectNegative && mean<0)
    return 1e-3;
  return TMath::Poisson(k,mean) ;
}



namespace {

template<class Tx, class TMean>
void compute(RooSpan<double> output, Tx x, TMean mean,
    const bool protectNegative, const bool noRounding) {
  const int n = output.size();

  for (int i = 0; i < n; ++i) { //CHECK_VECTORISE
    const double x_i = noRounding ? x[i] : floor(x[i]);
    // The std::lgamma yields different values than in the scalar implementation.
    // Need to check which one is more accurate.
//    output[i] = std::lgamma(x_i + 1.);
    output[i] = TMath::LnGamma(x_i + 1.);
  }


  for (int i = 0; i < n; ++i) { //CHECK_VECTORISE
    const double x_i = noRounding ? x[i] : floor(x[i]);
    const double logMean = vdt::fast_log(mean[i]);
    const double logPoisson = x_i * logMean - mean[i] - output[i];
    output[i] = vdt::fast_exp(logPoisson);

    // Cosmetics
    if (x_i < 0.)
      output[i] = 0.;
    else if (x_i == 0.) {
      output[i] = 1./vdt::fast_exp(mean[i]);
    }
    if (protectNegative && mean[i] < 0.)
      output[i] = 1.E-3;
  }
}

}


////////////////////////////////////////////////////////////////////////////////
/// Compute Poisson values in batches.
RooSpan<double> RooPoisson::evaluateBatch(std::size_t begin, std::size_t batchSize) const {
  auto output = _batchData.makeWritableBatchUnInit(begin, batchSize);
  auto xData = x.getValBatch(begin, batchSize);
  auto meanData = mean.getValBatch(begin, batchSize);

  const bool batchX = !xData.empty();
  const bool batchMean = !meanData.empty();

  if (batchX && batchMean) {
    compute(output, xData, meanData, _protectNegative, _noRounding);
  } else if (batchX && !batchMean) {
    compute(output, xData, BatchHelpers::BracketAdapter<double>(mean), _protectNegative, _noRounding);
  }
  else if (!batchX && batchMean) {
    compute(output, BatchHelpers::BracketAdapter<double>(x), meanData, _protectNegative, _noRounding);
  } else if (!batchX && !batchMean) {
    compute(output, BatchHelpers::BracketAdapter<double>(x), BatchHelpers::BracketAdapter<double>(mean), _protectNegative, _noRounding);
  } else {
    //Should not happen
    assert(false);
  }

  return output;
}


////////////////////////////////////////////////////////////////////////////////

Int_t RooPoisson::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /*rangeName*/) const
{
  if (matchArgs(allVars,analVars,x)) return 1 ;
  if (matchArgs(allVars, analVars, mean)) return 2;
  return 0 ;
}

////////////////////////////////////////////////////////////////////////////////

Double_t RooPoisson::analyticalIntegral(Int_t code, const char* rangeName) const
{
  R__ASSERT(code == 1 || code == 2) ;

  if(_protectNegative && mean<0)
    return exp(-2*mean); // make it fall quickly

  if (code == 1) {
    // Implement integral over x as summation. Add special handling in case
    // range boundaries are not on integer values of x
    const double xmin = std::max(0., x.min(rangeName));
    const double xmax = x.max(rangeName);

    if (xmax < 0. || xmax < xmin) {
      return 0.;
    }
    if (!x.hasMax() || RooNumber::isInfinite(xmax)) {
      //Integrating the full Poisson distribution here 
      return 1.;
    }

    // The range as integers. ixmin is included, ixmax outside.
    const unsigned int ixmin = xmin;
    const unsigned int ixmax = std::min(xmax + 1.,
                                        (double)std::numeric_limits<unsigned int>::max());
    
    // Sum from 0 to just before the bin outside of the range.
    if (ixmin == 0) {
      return ROOT::Math::poisson_cdf(ixmax - 1, mean);
    }
    else {
      // If necessary, subtract from 0 to the beginning of the range
      if (ixmin <= mean) {
        return ROOT::Math::poisson_cdf(ixmax - 1, mean) - ROOT::Math::poisson_cdf(ixmin - 1, mean);
      }
      else {
        //Avoid catastrophic cancellation in the high tails:
        return ROOT::Math::poisson_cdf_c(ixmin - 1, mean) - ROOT::Math::poisson_cdf_c(ixmax - 1, mean);
      }

    }

  } else if(code == 2) {

    // the integral with respect to the mean is the integral of a gamma distribution
    Double_t mean_min = mean.min(rangeName);
    Double_t mean_max = mean.max(rangeName);

    Double_t ix;
    if(_noRounding) ix = x + 1;
    else ix = Int_t(TMath::Floor(x)) + 1.0; // negative ix does not need protection (gamma returns 0.0)

    return ROOT::Math::gamma_cdf(mean_max, ix, 1.0) - ROOT::Math::gamma_cdf(mean_min, ix, 1.0);
  }

  return 0;

}

////////////////////////////////////////////////////////////////////////////////
/// Advertise internal generator in x

Int_t RooPoisson::getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, Bool_t /*staticInitOK*/) const
{
  if (matchArgs(directVars,generateVars,x)) return 1 ;
  return 0 ;
}

////////////////////////////////////////////////////////////////////////////////
/// Implement internal generator using TRandom::Poisson

void RooPoisson::generateEvent(Int_t code)
{
  R__ASSERT(code==1) ;
  Double_t xgen ;
  while(1) {
    xgen = RooRandom::randomGenerator()->Poisson(mean);
    if (xgen<=x.max() && xgen>=x.min()) {
      x = xgen ;
      break;
    }
  }
  return;
}
