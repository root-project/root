/*****************************************************************************
 * Project: RooFit                                                           *
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2019, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

// Authors of this class:
//    T. Skwarnicki:
//      - modify RooCBShape to Asymmetrical Double-Sided CB
//    Michael Wilkinson
//      - add to RooFit source
//    Jonas Rembser, CERN  02/2021:
//      - merging RooDSCBShape with RooSDSCBShape to RooCrystalBall
//      - implement possibility to have asymmetrical Gaussian core
//      - complete rewrite of evaluation and integral code to reduce code
//        duplication

/** \class RooCrystalBall
    \ingroup Roofit

PDF implementing the generalized Asymmetrical Double-Sided Crystall Ball line shape.
\f[
  f(m;m_0,\sigma,\alpha_L,n_L,\alpha_R,n_R) =
  \begin{cases}
    A_L \cdot (B_L - \frac{m - m_0}{\sigma_L})^{-n_L}, & \mbox{for }\frac{m - m_0}{\sigma_L} < -\alpha_L \\
    \exp \left( - \frac{1}{2} \cdot \left[ \frac{m - m_0}{\sigma_L} \right]^2 \right), & \mbox{for }\frac{m - m_0}{\sigma_L} \leq 0 \\
    \exp \left( - \frac{1}{2} \cdot \left[ \frac{m - m_0}{\sigma_R} \right]^2 \right), & \mbox{for }\frac{m - m_0}{\sigma_R} \leq \alpha_R \\
    A_R \cdot (B_R + \frac{m - m_0}{\sigma_R})^{-n_R}, & \mbox{otherwise}, \\
  \end{cases}
\f]
times some normalization factor,
where
\f[
  \begin{align}
    A_i &= \left(\frac{n_i}{\left| \alpha_i \right|}\right)^{n_i} \cdot \exp\left(- \frac {\left| \alpha_i \right|^2}{2}\right) \\
    B_i &= \frac{n_i}{\left| \alpha_i \right|}  - \left| \alpha_i \right| \\
  \end{align}
\f]
**/

#include "RooCrystalBall.h"
#include "RooHelpers.h"
#include "TError.h"

#include <cmath>
#include <limits>
#include <memory>
#include <utility>

ClassImp(RooCrystalBall);

////////////////////////////////////////////////////////////////////////////////
/// Creates the fully parametrized crystal ball shape with asymmetric Gaussian core and asymmetric tails.
///
/// \param name Name that identifies the PDF in computations.
/// \param title Title for plotting.
/// \param x The variable of the PDF.
/// \param x0 Location parameter of the Gaussian component.
/// \param sigmaL Width parameter of the left side of the Gaussian component.
/// \param sigmaR Width parameter of the right side of the Gaussian component.
/// \param alphaL Location of transition to a power law on the left, in standard deviations away from the mean.
/// \param nL Exponent of power-law tail on the left.
/// \param alphaR Location of transition to a power law on the right, in standard deviations away from the mean.
/// \param nR Exponent of power-law tail on the right.
RooCrystalBall::RooCrystalBall(const char *name, const char *title, RooAbsReal &x, RooAbsReal &x0, RooAbsReal &sigmaL,
                               RooAbsReal &sigmaR, RooAbsReal &alphaL, RooAbsReal &nL, RooAbsReal &alphaR,
                               RooAbsReal &nR)
   : RooAbsPdf(name, title), x_("x", "Dependent", this, x), x0_("x0", "X0", this, x0),
     sigmaL_("sigmaL", "Left Sigma", this, sigmaL), sigmaR_("sigmaR", "Right Sigma", this, sigmaR),
     alphaL_{"alphaL", "Left Alpha", this, alphaL}, nL_{"nL", "Left Order", this, nL},
     alphaR_{std::make_unique<RooRealProxy>("alphaR", "Right Alpha", this, alphaR)},
     nR_{std::make_unique<RooRealProxy>("nR", "Right Order", this, nR)}
{
   RooHelpers::checkRangeOfParameters(this, {&sigmaL}, 0.0);
   RooHelpers::checkRangeOfParameters(this, {&sigmaR}, 0.0);
   RooHelpers::checkRangeOfParameters(this, {&alphaL}, 0.0);
   RooHelpers::checkRangeOfParameters(this, {&alphaR}, 0.0);
   RooHelpers::checkRangeOfParameters(this, {&nL}, 0.0);
   RooHelpers::checkRangeOfParameters(this, {&nR}, 0.0);
}

////////////////////////////////////////////////////////////////////////////////
/// Create a crystal ball shape with symmetric Gaussian core and asymmetric tails (just like `RooDSCBShape`).
///
/// \param name Name that identifies the PDF in computations.
/// \param title Title for plotting.
/// \param x The variable of the PDF.
/// \param x0 Location parameter of the Gaussian component.
/// \param sigmaLR Width parameter of the Gaussian component.
/// \param alphaL Location of transition to a power law on the left, in standard deviations away from the mean.
/// \param nL Exponent of power-law tail on the left.
/// \param alphaR Location of transition to a power law on the right, in standard deviations away from the mean.
/// \param nR Exponent of power-law tail on the right.
RooCrystalBall::RooCrystalBall(const char *name, const char *title, RooAbsReal &x, RooAbsReal &x0, RooAbsReal &sigmaLR,
                               RooAbsReal &alphaL, RooAbsReal &nL, RooAbsReal &alphaR, RooAbsReal &nR)
   : RooAbsPdf(name, title), x_("x", "Dependent", this, x), x0_("x0", "X0", this, x0),
     sigmaL_("sigmaL", "Left Sigma", this, sigmaLR), sigmaR_("sigmaR", "Right Sigma", this, sigmaLR),
     alphaL_{"alphaL", "Left Alpha", this, alphaL}, nL_{"nL", "Left Order", this, nL},
     alphaR_{std::make_unique<RooRealProxy>("alphaR", "Right Alpha", this, alphaR)},
     nR_{std::make_unique<RooRealProxy>("nR", "Right Order", this, nR)}
{
   RooHelpers::checkRangeOfParameters(this, {&sigmaLR}, 0.0);
   RooHelpers::checkRangeOfParameters(this, {&alphaL}, 0.0);
   RooHelpers::checkRangeOfParameters(this, {&alphaR}, 0.0);
   RooHelpers::checkRangeOfParameters(this, {&nL}, 0.0);
   RooHelpers::checkRangeOfParameters(this, {&nR}, 0.0);
}

////////////////////////////////////////////////////////////////////////////////
/// Create a crystal ball shape with symmetric Gaussian core and only a tail on
/// one side (just like `RooCBShape`) or two symmetric tails (like `RooSDSCBShape`).
///
/// \param name Name that identifies the PDF in computations.
/// \param title Title for plotting.
/// \param x The variable of the PDF.
/// \param x0 Location parameter of the Gaussian component.
/// \param sigmaLR Width parameter of the Gaussian component.
/// \param alpha Location of transition to a power law, in standard deviations away from the mean.
/// \param n Exponent of power-law tail.
/// \param doubleSided Whether the tail is only on one side or on both sides
RooCrystalBall::RooCrystalBall(const char *name, const char *title, RooAbsReal &x, RooAbsReal &x0, RooAbsReal &sigmaLR,
                               RooAbsReal &alpha, RooAbsReal &n, bool doubleSided)
   : RooAbsPdf(name, title), x_("x", "Dependent", this, x), x0_("x0", "X0", this, x0),
     sigmaL_{"sigmaL", "Left Sigma", this, sigmaLR}, sigmaR_{"sigmaR", "Right Sigma", this, sigmaLR},
     alphaL_{"alphaL", "Left Alpha", this, alpha},
     nL_{"nL", "Left Order", this, n}
{
   if (doubleSided) {
      alphaR_ = std::make_unique<RooRealProxy>("alphaR", "Right Alpha", this, alpha);
      nR_ = std::make_unique<RooRealProxy>("nR", "Right Order", this, n);
   }

   RooHelpers::checkRangeOfParameters(this, {&sigmaLR}, 0.0);
   RooHelpers::checkRangeOfParameters(this, {&n}, 0.0);
   if (doubleSided) {
      RooHelpers::checkRangeOfParameters(this, {&alpha}, 0.0);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Copy a RooCrystalBall.
RooCrystalBall::RooCrystalBall(const RooCrystalBall &other, const char *name)
   : RooAbsPdf(other, name), x_("x", this, other.x_), x0_("x0", this, other.x0_),
     sigmaL_("sigmaL", this, other.sigmaL_),
     sigmaR_("sigmaR", this, other.sigmaR_), alphaL_{"alphaL", this, other.alphaL_},
     nL_{"nL", this, other.nL_},
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

double RooCrystalBall::evaluate() const
{
   const double x = x_;
   const double x0 = x0_;
   const double sigmaL = std::abs(sigmaL_);
   const double sigmaR = std::abs(sigmaR_);
   double alphaL = std::abs(alphaL_);
   double nL = nL_;
   double alphaR = alphaR_ ? std::abs(*alphaR_) : std::numeric_limits<double>::infinity();
   double nR = nR_ ? *nR_ : 0.0;

   // If alphaL is negative, then the tail will be on the right side.
   // Like this, we follow the convention established by RooCBShape.
   if(!alphaR_ && alphaL_ < 0.0) {
      std::swap(alphaL, alphaR);
      std::swap(nL, nR);
   }

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

double RooCrystalBall::analyticalIntegral(Int_t code, const char *rangeName) const
{
   R__ASSERT(code == 1);

   const double x0 = x0_;
   const double sigmaL = std::abs(sigmaL_);
   const double sigmaR = std::abs(sigmaR_);
   double alphaL = std::abs(alphaL_);
   double nL = nL_;
   double alphaR = alphaR_ ? std::abs(*alphaR_) : std::numeric_limits<double>::infinity();
   double nR = nR_ ? *nR_ : 0.0;

   // If alphaL is negative, then the tail will be on the right side.
   // Like this, we follow the convention established by RooCBShape.
   if(!alphaR_ && alphaL_ < 0.0) {
      std::swap(alphaL, alphaR);
      std::swap(nL, nR);
   }

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
/// Advertise that we know the maximum of self for given (m0,alpha,n,sigma).

Int_t RooCrystalBall::getMaxVal(const RooArgSet &vars) const
{
   RooArgSet dummy;
   return matchArgs(vars, dummy, x_) ? 1 : 0;
}

////////////////////////////////////////////////////////////////////////////////

double RooCrystalBall::maxVal(Int_t code) const
{
   R__ASSERT(code == 1);

   // The maximum value for given (m0,alpha,n,sigma) is 1./ Integral in the variable range
   // For the crystal ball, the maximum is 1.0 in the current implementation,
   // but it's maybe better to keep this general in case the implementation changes.
   return 1.0 / analyticalIntegral(code);
}
