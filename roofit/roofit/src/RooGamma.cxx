 /*****************************************************************************
  * Project: RooFit                                                           *
  *                                                                           *
  * Simple Gamma distribution
  * authors: Stefan A. Schmitz, Gregory Schott
  *                                                                           *
  *****************************************************************************/

/** \class RooGamma
    \ingroup Roofit

Implementation of the Gamma PDF for RooFit/RooStats.
\f[
f(x) = \frac{(x-\mu)^{\gamma-1} \cdot \exp^{(-(x-mu) / \beta}}{\Gamma(\gamma) \cdot \beta^{\gamma}}
\f]
defined for \f$ x \geq 0 \f$ if \f$ \mu = 0 \f$

Notes from Kyle Cranmer:

Wikipedia and several sources refer to the Gamma distribution as

\f[
G(\mu,\alpha,\beta) = \beta^\alpha \mu^{(\alpha-1)} \frac{e^{(-\beta \mu)}}{\Gamma(\alpha)}
\f]

Below is the correspondence:

| Wikipedia       | This Implementation      |
|-----------------|--------------------------|
| \f$ \alpha \f$  | \f$ \gamma \f$           |
| \f$ \beta \f$   | \f$ \frac{1}{\beta} \f$  |
| \f$ \mu \f$     | x                        |
| 0               | \f$ \mu \f$              |


Note, that for a model Pois(N|mu), a uniform prior on mu, and a measurement N
the posterior is in the Wikipedia parameterization Gamma(mu, alpha=N+1, beta=1)
thus for this implementation it is:

`RooGamma(_x=mu,_gamma=N+1,_beta=1,_mu=0)`

Also note, that in this case it is equivalent to
RooPoison(N,mu) and treating the function as a PDF in mu.
**/

#include "RooGamma.h"

#include "RooAbsReal.h"
#include "RooRealVar.h"
#include "RooRandom.h"
#include "RooMath.h"
#include "RooHelpers.h"
#include "BatchHelpers.h"
#include "RooVDTHeaders.h"

#include "TMath.h"
#include <Math/SpecFuncMathCore.h>
#include <Math/PdfFuncMathCore.h>
#include <Math/ProbFuncMathCore.h>

#include <iostream>
#include <cmath>
using namespace std;

ClassImp(RooGamma);

////////////////////////////////////////////////////////////////////////////////

RooGamma::RooGamma(const char *name, const char *title,
          RooAbsReal& _x, RooAbsReal& _gamma,
          RooAbsReal& _beta, RooAbsReal& _mu) :
  RooAbsPdf(name,title),
  x("x","Observable",this,_x),
  gamma("gamma","Mean",this,_gamma),
  beta("beta","Width",this,_beta),
  mu("mu","Para",this,_mu)
{
  RooHelpers::checkRangeOfParameters(this, {&_gamma, &_beta}, 0.);
}

////////////////////////////////////////////////////////////////////////////////

RooGamma::RooGamma(const RooGamma& other, const char* name) :
  RooAbsPdf(other,name), x("x",this,other.x), gamma("mean",this,other.gamma),
  beta("beta",this,other.beta), mu("mu",this,other.mu)
{
}

////////////////////////////////////////////////////////////////////////////////

Double_t RooGamma::evaluate() const
{
  return TMath::GammaDist(x, gamma, mu, beta) ;
}

////////////////////////////////////////////////////////////////////////////////

namespace GammaBatchEvaluate {
//Author: Emmanouil Michalainas, CERN 22 August 2019

template<class Tx, class Tgamma, class Tbeta, class Tmu>
void compute(	size_t batchSize,
              double * __restrict output,
              Tx X, Tgamma G, Tbeta B, Tmu M)
{
  constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
  for (size_t i=0; i<batchSize; i++) {
    if (X[i]<M[i] || G[i] <= 0.0 || B[i] <= 0.0) {
      output[i] = NaN;
    }
    if (X[i] == M[i]) {
      output[i] = (G[i]==1.0)/B[i];
    } 
    else {
      output[i] = 0.0;
    }
  }
  
  if (G.isBatch()) {
    for (size_t i=0; i<batchSize; i++) {
      if (output[i] == 0.0) {
        output[i] = -std::lgamma(G[i]);
      }
    }
  }
  else {
    double gamma = -std::lgamma(G[2019]);
    for (size_t i=0; i<batchSize; i++) {
      if (output[i] == 0.0) {
        output[i] = gamma;
      }
    }
  }
  
  for (size_t i=0; i<batchSize; i++) {
    if (X[i] != M[i]) {
      const double invBeta = 1/B[i];
      double arg = (X[i]-M[i])*invBeta;
      output[i] -= arg;
      arg = _rf_fast_log(arg);
      output[i] += arg*(G[i]-1);
      output[i] = _rf_fast_exp(output[i]);
      output[i] *= invBeta;
    }
  }
}
};

RooSpan<double> RooGamma::evaluateBatch(std::size_t begin, std::size_t batchSize) const {
  using namespace BatchHelpers;
  using namespace GammaBatchEvaluate;

  EvaluateInfo info = getInfo( {&x, &gamma, &beta, &mu}, begin, batchSize );
  auto output = _batchData.makeWritableBatchUnInit(begin, batchSize);

  auto xData = x.getValBatch(begin, info.size);
  if (info.nBatches == 0) {
    throw std::logic_error("Requested a batch computation, but no batch data available.");
  }
  else if (info.nBatches==1 && !xData.empty()) {
    compute(info.size, output.data(), xData.data(),
    BracketAdapter<double> (gamma),
    BracketAdapter<double> (beta),
    BracketAdapter<double> (mu));
  }
  else {
    compute(info.size, output.data(),
    BracketAdapterWithMask (x,x.getValBatch(begin,info.size)),
    BracketAdapterWithMask (gamma,gamma.getValBatch(begin,info.size)),
    BracketAdapterWithMask (beta,beta.getValBatch(begin,info.size)),
    BracketAdapterWithMask (mu,mu.getValBatch(begin,info.size)));
  }
  return output;
}

////////////////////////////////////////////////////////////////////////////////

Int_t RooGamma::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /*rangeName*/) const
{
  if (matchArgs(allVars,analVars,x)) return 1 ;
  return 0 ;
}

////////////////////////////////////////////////////////////////////////////////

Double_t RooGamma::analyticalIntegral(Int_t code, const char* rangeName) const
{
  R__ASSERT(code==1) ;

 //integral of the Gamma distribution via ROOT::Math
  Double_t integral = ROOT::Math::gamma_cdf(x.max(rangeName), gamma, beta, mu) - ROOT::Math::gamma_cdf(x.min(rangeName), gamma, beta, mu);
  return integral ;
}

////////////////////////////////////////////////////////////////////////////////

Int_t RooGamma::getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, Bool_t /*staticInitOK*/) const
{
  if (matchArgs(directVars,generateVars,x)) return 1 ;
  return 0 ;
}

////////////////////////////////////////////////////////////////////////////////
/// algorithm adapted from code example in:
/// Marsaglia, G. and Tsang, W. W.
/// A Simple Method for Generating Gamma Variables
/// ACM Transactions on Mathematical Software, Vol. 26, No. 3, September 2000
///
/// The speed of this algorithm depends on the speed of generating normal variates.
/// The algorithm is limited to \f$ \gamma \geq 0 \f$ !

void RooGamma::generateEvent(Int_t code)
{
  R__ASSERT(code==1) ;


  while(1) {

  double d = 0;
  double c = 0;
  double xgen = 0;
  double v = 0;
  double u = 0;
  d = gamma -1./3.; c = 1./TMath::Sqrt(9.*d);

  while(v <= 0.){
    xgen = RooRandom::randomGenerator()->Gaus(); v = 1. + c*xgen;
  }
  v = v*v*v; u = RooRandom::randomGenerator()->Uniform();
  if( u < 1.-.0331*(xgen*xgen)*(xgen*xgen) ) {
    if ( (((d*v)* beta + mu ) < x.max()) && (((d*v)* beta + mu) > x.min()) ) {
      x = ((d*v)* beta + mu) ;
      break;
    }
  }
  if( TMath::Log(u) < 0.5*xgen*xgen + d*(1.-v + TMath::Log(v)) ) {
    if ( (((d*v)* beta + mu ) < x.max()) && (((d*v)* beta + mu) > x.min()) ) {
      x = ((d*v)* beta + mu) ;
      break;
    }
  }

  }


  return;
}


