/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 * @(#)root/roofit:$Id$
 * Authors:                                                                  *
 *   Kyle Cranmer
 *                                                                           *
 *****************************************************************************/

/** \class RooChiSquarePdf
    \ingroup Roofit

The PDF of the Chi Square distribution for n degrees of freedom.
Oddly, this is hard to find in ROOT (except via relation to GammaDist).
Here we also implement the analytic integral.
**/

#include "RooChiSquarePdf.h"
#include "RooFit.h"
#include "RooAbsReal.h"
#include "RooRealVar.h"
#include "BatchHelpers.h"
#include "RooVDTHeaders.h"

#include "TMath.h"

#include <cmath>
using namespace std;

ClassImp(RooChiSquarePdf);

////////////////////////////////////////////////////////////////////////////////

RooChiSquarePdf::RooChiSquarePdf()
{
}

////////////////////////////////////////////////////////////////////////////////

RooChiSquarePdf::RooChiSquarePdf(const char* name, const char* title,
                           RooAbsReal& x, RooAbsReal& ndof):
  RooAbsPdf(name, title),
  _x("x", "Dependent", this, x),
  _ndof("ndof","ndof", this, ndof)
{
}

////////////////////////////////////////////////////////////////////////////////

RooChiSquarePdf::RooChiSquarePdf(const RooChiSquarePdf& other, const char* name) :
  RooAbsPdf(other, name),
  _x("x", this, other._x),
  _ndof("ndof",this,other._ndof)
{
}

////////////////////////////////////////////////////////////////////////////////

Double_t RooChiSquarePdf::evaluate() const
{
  if(_x <= 0) return 0;

  return  pow(_x,(_ndof/2.)-1.) * exp(-_x/2.) / TMath::Gamma(_ndof/2.) / pow(2.,_ndof/2.);
}

////////////////////////////////////////////////////////////////////////////////

namespace ChiSquarePdfBatchEvaluate {
//Author: Emmanouil Michalainas, CERN 28 Aug 2019

template<class T_x, class T_ndof>
void compute(	size_t batchSize,
              double * __restrict__ output,
              T_x X, T_ndof N)
{
  if ( N.isBatch() ) {
    for (size_t i=0; i<batchSize; i++) {
      if (X[i] > 0) {
        output[i] = 1/std::tgamma(N[i]/2.0);
      }
    }
  }
  else {
    const double gamma = 1/std::tgamma(N[2019]/2.0);
    for (size_t i=0; i<batchSize; i++) {
      output[i] = gamma;
    }
  }
  
  constexpr double ln2 = 0.693147180559945309417232121458;
  const double lnx0 = std::log(X[0]);
  for (size_t i=0; i<batchSize; i++) {
    double lnx;
    if ( X.isBatch() ) lnx = vdt::fast_log(X[i]);
    else lnx = lnx0;
    
    double arg = (N[i]-2)*lnx -X[i] -N[i]*ln2;
    output[i] *= vdt::fast_exp(0.5*arg);
  }
}
};

RooSpan<double> RooChiSquarePdf::evaluateBatch(std::size_t begin, std::size_t batchSize) const {
  using namespace BatchHelpers;
  using namespace ChiSquarePdfBatchEvaluate;
  auto _xData = _x.getValBatch(begin, batchSize);
  auto _ndofData = _ndof.getValBatch(begin, batchSize);

  batchSize = findSize({ _xData, _ndofData });
  auto output = _batchData.makeWritableBatchUnInit(begin, batchSize);

  const bool batch_x = !_xData.empty();
  const bool batch_ndof = !_ndofData.empty();
  if (batch_x && !batch_ndof ) {
    compute(batchSize, output.data(), _xData, BracketAdapter<double>(_ndof));
  }
  else if (!batch_x && batch_ndof ) {
    compute(batchSize, output.data(), BracketAdapter<double>(_x), _ndofData);
  }
  else if (batch_x && batch_ndof ) {
    compute(batchSize, output.data(), _xData, _ndofData);
  }
  else{
    throw std::logic_error("Requested a batch computation, but no batch data available.");
  }

  return output;
}


////////////////////////////////////////////////////////////////////////////////
/// No analytical calculation available (yet) of integrals over subranges

Int_t RooChiSquarePdf::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName) const
{
  if (rangeName && strlen(rangeName)) {
    return 0 ;
  }

  if (matchArgs(allVars, analVars, _x)) return 1;
  return 0;
}

////////////////////////////////////////////////////////////////////////////////

Double_t RooChiSquarePdf::analyticalIntegral(Int_t code, const char* rangeName) const
{
  assert(1 == code); (void)code;
  Double_t xmin = _x.min(rangeName); Double_t xmax = _x.max(rangeName);

  // TMath::Prob needs ndof to be an integer, or it returns 0.
  //  return TMath::Prob(xmin, _ndof) - TMath::Prob(xmax,_ndof);

  // cumulative is known based on lower incomplete gamma function, or regularized gamma function
  // Wikipedia defines lower incomplete gamma function without the normalization 1/Gamma(ndof),
  // but it is included in the ROOT implementation.
  Double_t pmin = TMath::Gamma(_ndof/2,xmin/2);
  Double_t pmax = TMath::Gamma(_ndof/2,xmax/2);

  // only use this if range is appropriate
  return pmax-pmin;
}
