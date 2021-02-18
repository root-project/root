// Author: Jonas Rembser, CERN  02/2021

#include "RooCrystalBall.h"
#include "RooAbsReal.h"
#include "RooHelpers.h"
#include "TError.h"

#include "ROOT/RMakeUnique.hxx"

#include <cmath>
#include <limits>

ClassImp(RooCrystalBall);

////////////////////////////////////////////////////////////////////////////////
/// Creates the fully parametrized crystal ball shape with asymmetric Gaussian core and asymmetric tails

RooCrystalBall::RooCrystalBall(const char *name, const char *title, RooAbsReal &x, RooAbsReal &x0, RooAbsReal &sigmaL,
                               RooAbsReal &sigmaR, RooAbsReal &alphaL, RooAbsReal &nL, RooAbsReal &alphaR,
                               RooAbsReal &nR)
   : RooAbsPdf(name, title), x_("x", "Dependent", this, x), x0_("x0", "X0", this, x0),
     sigmaL_("sigmaL", "Left Sigma", this, sigmaL), sigmaR_("sigmaR", "Right Sigma", this, sigmaR),
     alphaL_{std::make_unique<RooRealProxy>("alphaL", "Left Alpha", this, alphaL)}, nL_{std::make_unique<RooRealProxy>(
                                                                                       "nL", "Left Order", this, nL)},
     alphaR_{std::make_unique<RooRealProxy>("alphaR", "Right Alpha", this, alphaR)}, nR_{std::make_unique<RooRealProxy>(
                                                                                        "nR", "Right Order", this, nR)}
{
   RooHelpers::checkRangeOfParameters(this, {&sigmaL}, 0.0);
   RooHelpers::checkRangeOfParameters(this, {&sigmaR}, 0.0);
   RooHelpers::checkRangeOfParameters(this, {&alphaL}, 0.0);
   RooHelpers::checkRangeOfParameters(this, {&alphaR}, 0.0);
   RooHelpers::checkRangeOfParameters(this, {&nL}, 0.0);
   RooHelpers::checkRangeOfParameters(this, {&nR}, 0.0);
}

////////////////////////////////////////////////////////////////////////////////
/// Create a crystal ball shape with symmetric Gaussian core and asymmetric tails (just like `RooDSCBShape`)

RooCrystalBall::RooCrystalBall(const char *name, const char *title, RooAbsReal &x, RooAbsReal &x0, RooAbsReal &sigma,
                               RooAbsReal &alphaL, RooAbsReal &nL, RooAbsReal &alphaR, RooAbsReal &nR)
   : RooAbsPdf(name, title), x_("x", "Dependent", this, x), x0_("x0", "X0", this, x0),
     sigmaL_("sigmaL", "Left Sigma", this, sigma), sigmaR_("sigmaR", "Right Sigma", this, sigma),
     alphaL_{std::make_unique<RooRealProxy>("alphaL", "Left Alpha", this, alphaL)}, nL_{std::make_unique<RooRealProxy>(
                                                                                       "nL", "Left Order", this, nL)},
     alphaR_{std::make_unique<RooRealProxy>("alphaR", "Right Alpha", this, alphaR)}, nR_{std::make_unique<RooRealProxy>(
                                                                                        "nR", "Right Order", this, nR)}
{
   RooHelpers::checkRangeOfParameters(this, {&sigma}, 0.0);
   RooHelpers::checkRangeOfParameters(this, {&alphaL}, 0.0);
   RooHelpers::checkRangeOfParameters(this, {&alphaR}, 0.0);
   RooHelpers::checkRangeOfParameters(this, {&nL}, 0.0);
   RooHelpers::checkRangeOfParameters(this, {&nR}, 0.0);
}

////////////////////////////////////////////////////////////////////////////////
/// Create a crystal ball shape with symmetric Gaussian core and only a tail on the left (just like `RooCBShape`)

RooCrystalBall::RooCrystalBall(const char *name, const char *title, RooAbsReal &x, RooAbsReal &x0, RooAbsReal &sigma,
                               RooAbsReal &alpha, RooAbsReal &n, TailSide tailSide)
   : RooAbsPdf(name, title), x_("x", "Dependent", this, x), x0_("x0", "X0", this, x0),
     sigmaL_("sigmaL", "Left Sigma", this, sigma), sigmaR_("sigmaR", "Right Sigma", this, sigma)
{
   if (tailSide == TailSide::Left || tailSide == TailSide::Both) {
      alphaL_ = std::make_unique<RooRealProxy>("alphaL", "Left Alpha", this, alpha);
      nL_ = std::make_unique<RooRealProxy>("nL", "Left Order", this, n);
   }
   if (tailSide == TailSide::Right || tailSide == TailSide::Both) {
      alphaR_ = std::make_unique<RooRealProxy>("alphaR", "Right Alpha", this, alpha);
      nR_ = std::make_unique<RooRealProxy>("nR", "Right Order", this, n);
   }

   RooHelpers::checkRangeOfParameters(this, {&sigma}, 0.0);
   RooHelpers::checkRangeOfParameters(this, {&alpha}, 0.0);
   RooHelpers::checkRangeOfParameters(this, {&n}, 0.0);
}

////////////////////////////////////////////////////////////////////////////////

RooCrystalBall::RooCrystalBall(const RooCrystalBall &other, const char *name)
   : RooAbsPdf(other, name), x_("x", this, other.x_), x0_("x0", this, other.x0_),
     sigmaL_("sigmaL", this, other.sigmaL_),
     sigmaR_("sigmaR", this, other.sigmaR_), alphaL_{other.alphaL_
                                                        ? std::make_unique<RooRealProxy>("alphaL", this, *other.alphaL_)
                                                        : nullptr},
     nL_{other.nL_ ? std::make_unique<RooRealProxy>("nL", this, *other.nL_) : nullptr},
     alphaR_{other.alphaR_ ? std::make_unique<RooRealProxy>("alphaR", this, *other.alphaR_) : nullptr},
     nR_{other.nR_ ? std::make_unique<RooRealProxy>("nR", this, *other.nR_) : nullptr}
{
}

////////////////////////////////////////////////////////////////////////////////

namespace {

inline double evaluateCrystalBallTail(double t, double alpha, double n)
{
   double a = std::pow(n / alpha, n) * std::exp(-0.5 * alpha * alpha);
   double b = n / alpha - alpha;

   return a / std::pow(b - t, n);
}

inline double integrateGaussian(double sigmaL, double sigmaR, double tmin, double tmax)
{
   constexpr double sqrtPiOver2 = 1.2533141373;
   constexpr double sqrt2 = 1.4142135624;

   const double sigmaMin = tmin < 0 ? sigmaL : sigmaR;
   const double sigmaMax = tmax < 0 ? sigmaL : sigmaR;

   return sqrtPiOver2 * (sigmaMax * std::erf(tmax / sqrt2) - sigmaMin * std::erf(tmin / sqrt2));
}

inline double integrateTailLogVersion(double sigma, double alpha, double n, double tmin, double tmax)
{
   double a = std::pow(n / alpha, n) * exp(-0.5 * alpha * alpha);
   double b = n / alpha - alpha;

   return a * sigma * (log(b - tmin) - log(b - tmax));
}

inline double integrateTailRegular(double sigma, double alpha, double n, double tmin, double tmax)
{
   double a = std::pow(n / alpha, n) * exp(-0.5 * alpha * alpha);
   double b = n / alpha - alpha;

   return a * sigma / (1.0 - n) * (1.0 / (std::pow(b - tmin, n - 1.0)) - 1.0 / (std::pow(b - tmax, n - 1.0)));
}

} // namespace

////////////////////////////////////////////////////////////////////////////////

Double_t RooCrystalBall::evaluate() const
{
   const double x = x_;
   const double x0 = x0_;
   const double sigmaL = std::abs(sigmaL_);
   const double sigmaR = std::abs(sigmaR_);
   const double alphaL = alphaL_ ? std::abs(*alphaL_) : std::numeric_limits<double>::infinity();
   const double alphaR = alphaR_ ? std::abs(*alphaR_) : std::numeric_limits<double>::infinity();
   const double nL = nL_ ? *nL_ : 0.0;
   const double nR = nR_ ? *nR_ : 0.0;

   const double t = (x - x0) / (x < x0 ? sigmaL : sigmaR);

   if (t < -alphaL) {
      return evaluateCrystalBallTail(t, alphaL, nL);
   } else if (t <= alphaR) {
      return std::exp(-0.5 * t * t);
   } else {
      return evaluateCrystalBallTail(-t, alphaR, nR);
   }
}

////////////////////////////////////////////////////////////////////////////////

Int_t RooCrystalBall::getAnalyticalIntegral(RooArgSet &allVars, RooArgSet &analVars, const char * /*rangeName*/) const
{
   return matchArgs(allVars, analVars, x_) ? 1 : 0;
}

////////////////////////////////////////////////////////////////////////////////

Double_t RooCrystalBall::analyticalIntegral(Int_t code, const char *rangeName) const
{
   R__ASSERT(code == 1);

   const double x0 = x0_;
   const double sigmaL = std::abs(sigmaL_);
   const double sigmaR = std::abs(sigmaR_);
   const double alphaL = alphaL_ ? std::abs(*alphaL_) : std::numeric_limits<double>::infinity();
   const double alphaR = alphaR_ ? std::abs(*alphaR_) : std::numeric_limits<double>::infinity();
   const double nL = nL_ ? *nL_ : 0.0;
   const double nR = nR_ ? *nR_ : 0.0;

   constexpr double switchToLogThreshold = 1.0e-05;

   const double xmin = x_.min(rangeName);
   const double xmax = x_.max(rangeName);
   const double tmin = (xmin - x0) / (xmin < x0 ? sigmaL : sigmaR);
   const double tmax = (xmax - x0) / (xmax < x0 ? sigmaL : sigmaR);

   double result = 0.0;

   if (tmin < -alphaL) {
      auto integrateTailL = std::abs(nL - 1.0) < switchToLogThreshold ? integrateTailLogVersion : integrateTailRegular;
      result += integrateTailL(sigmaL, alphaL, nL, tmin, std::min(tmax, -alphaL));
   }
   if (tmax > alphaR) {
      auto integrateTailR = std::abs(nR - 1.0) < switchToLogThreshold ? integrateTailLogVersion : integrateTailRegular;
      result += integrateTailR(sigmaR, alphaR, nR, -tmax, std::min(-tmin, -alphaR));
   }
   if (tmin < alphaR && tmax > -alphaL) {
      result += integrateGaussian(sigmaL, sigmaR, std::max(tmin, -alphaL), std::min(tmax, alphaR));
   }

   return result;
}

////////////////////////////////////////////////////////////////////////////////
/// Advertise that we know the maximum of self for given (m0,alpha,n,sigma)

Int_t RooCrystalBall::getMaxVal(const RooArgSet &vars) const
{
   RooArgSet dummy;
   return matchArgs(vars, dummy, x_) ? 1 : 0;
}

////////////////////////////////////////////////////////////////////////////////

Double_t RooCrystalBall::maxVal(Int_t code) const
{
   R__ASSERT(code == 1);

   // The maximum value for given (m0,alpha,n,sigma) is 1./ Integral in the variable range
   // For the crystal ball, the maximum is 1.0 in the current implementation,
   // but it's maybe better to keep this general in case the implementation changes.
   return 1.0 / analyticalIntegral(code);
}
