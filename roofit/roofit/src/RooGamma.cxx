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

#include "RooRandom.h"
#include "RooHelpers.h"
#include "RooBatchCompute.h"

#include "TMath.h"
#include <Math/ProbFuncMathCore.h>

#include <cmath>


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
  RooAbsPdf(other,name), x("x",this,other.x), gamma("gamma",this,other.gamma),
  beta("beta",this,other.beta), mu("mu",this,other.mu)
{
}

////////////////////////////////////////////////////////////////////////////////

double RooGamma::evaluate() const
{
  return TMath::GammaDist(x, gamma, mu, beta) ;
}

////////////////////////////////////////////////////////////////////////////////
/// Compute multiple values of Gamma PDF.
void RooGamma::doEval(RooFit::EvalContext &ctx) const
{
   RooBatchCompute::compute(ctx.config(this), RooBatchCompute::Gamma, ctx.output(),
                            {ctx.at(x), ctx.at(gamma), ctx.at(beta), ctx.at(mu)});
}

////////////////////////////////////////////////////////////////////////////////

Int_t RooGamma::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /*rangeName*/) const
{
  if (matchArgs(allVars,analVars,x)) return 1 ;
  return 0 ;
}

////////////////////////////////////////////////////////////////////////////////

double RooGamma::analyticalIntegral(Int_t /*code*/, const char *rangeName) const
{
   // integral of the Gamma distribution via ROOT::Math
   return ROOT::Math::gamma_cdf(x.max(rangeName), gamma, beta, mu) -
          ROOT::Math::gamma_cdf(x.min(rangeName), gamma, beta, mu);
}

namespace {

inline double randomGamma(double gamma, double beta, double mu, double xmin, double xmax)
{
   while (true) {

      double d = gamma - 1. / 3.;
      double c = 1. / std::sqrt(9. * d);
      double xgen = 0;
      double v = 0;

      while (v <= 0.) {
         xgen = RooRandom::randomGenerator()->Gaus();
         v = 1. + c * xgen;
      }
      v = v * v * v;
      double u = RooRandom::randomGenerator()->Uniform();
      if (u < 1. - .0331 * (xgen * xgen) * (xgen * xgen)) {
         double x = ((d * v) * beta + mu);
         if ((x < xmax) && (x > xmin)) {
            return x;
         }
      }
      if (std::log(u) < 0.5 * xgen * xgen + d * (1. - v + TMath::Log(v))) {
         double x = ((d * v) * beta + mu);
         if ((x < xmax) && (x > xmin)) {
            return x;
         }
      }
   }
}

} // namespace

////////////////////////////////////////////////////////////////////////////////

Int_t RooGamma::getGenerator(const RooArgSet &directVars, RooArgSet &generateVars, bool /*staticInitOK*/) const
{
   if (matchArgs(directVars, generateVars, x))
      return 1;
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// algorithm adapted from code example in:
/// Marsaglia, G. and Tsang, W. W.
/// A Simple Method for Generating Gamma Variables
/// ACM Transactions on Mathematical Software, Vol. 26, No. 3, September 2000
///
/// The speed of this algorithm depends on the speed of generating normal variates.
/// The algorithm is limited to \f$ \gamma \geq 0 \f$ !

void RooGamma::generateEvent(Int_t /*code*/)
{
   if (gamma >= 1) {
      x = randomGamma(gamma, beta, mu, x.min(), x.max());
      return;
   }

   double xVal = 0.0;
   bool accepted = false;

   while (!accepted) {
      double u = RooRandom::randomGenerator()->Uniform();
      double tmp = randomGamma(1 + gamma, beta, mu, 0, std::numeric_limits<double>::infinity());
      xVal = tmp * std::pow(u, 1.0 / gamma);
      accepted = xVal < x.max() && xVal > x.min();
   }

   x = xVal;
}
