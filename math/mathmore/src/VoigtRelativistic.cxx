// @(#)root/mathmore:$Id$
// Author: J. Lindon Wed Jun 15 02:35:26 2022

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2022  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class VoigtRelativistic

#include "Math/VoigtRelativistic.h"
#include "Math/Integrator.h"
#include "Math/Functor.h"
#include "TMath.h"

namespace ROOT {
namespace Math {

////////////////////////////////////////////////////////////////////////////////
// Computation of a relativistic voigt function (normalised), the convolution of
// a guassian and relativistic breit wigner function.
//
// \f$ V(E;\mu,\sigma,\Gamma) = \left(\frac{\sqrt{2}}{2\pi^{\frac{3}{2}}}\right)\left(\frac{\Gamma\mu^{2}
// \sqrt{\mu^{2}+\Gamma^{2} }}{\sigma^{4}\sqrt{\mu^{2}+\mu\sqrt{\mu^{2}+\Gamma^{2} }}}\right) \int_{-\infty}^{+\infty}
// \frac{e^{-t^{2}}}{\left(\frac{1}{\sqrt{2}}\left(E-\mu\right)\sigma-t\right)^{2}\left(\frac{1}{\sqrt{2}}\left(E+\mu\right)\sigma-t\right)^{2}+\frac{\Gamma^{2}\mu^{2}}{4\sigma^{4}}}
// \f$ Where all varialbles are real.
//
// E is the independent variable (typically the energy)
// \f$\mu\f$ is the median (typically the pole mass of the resonance)
// \f$\sigma\f$ is the width of the gaussian
// \f$\Gamma\f$ is the width of the relativistic breit wigner
// for this function to be exact the integrationRange must be taken to infinity
// however, it converges rapidly and the integrationRange by default is taken
// to the maximum precision allowed by this method for a double.
////////////////////////////////////////////////////////////////////////////////

double VoigtRelativistic::evaluate(double x, double median, double sigma, double lg, double integrationRange)
{

   double inverse_sSqrt2 = 1 / (1.41421356237309504 * sigma); // 1.41421356237309504=sqrt(2)
   double ss = sigma * sigma;
   double mm = median * median;

   double u1 = (x - median) * inverse_sSqrt2;
   double u2 = (x + median) * inverse_sSqrt2;
   double a = (lg * median) / (2 * ss);
   double aa = a * a;

   double y = median * sqrt(mm + lg * lg);
   double k = (0.25397454373696387 * a * y) / (ss * sqrt(mm + y)); // 0.25397454373696387=sqrt(2)/(pi^(3/2))

   auto integrand = [&](double t) {
      double u1Minust = (u1 - t);
      double u2Minust = (u2 - t);
      return ((exp(-t * t)) / (u1Minust * u1Minust * u2Minust * u2Minust + (aa)));
   };

   ROOT::Math::Functor1D f(integrand);
   ROOT::Math::Integrator integrator(ROOT::Math::IntegrationOneDim::kDEFAULT);
   integrator.SetFunction(f);
   double voigt = k * integrator.Integral(-integrationRange, integrationRange);

   return voigt;
}

////////////////////////////////////////////////////////////////////////////////
// Computation of a relativistic voigt function's dumping function, the ratio of
// the peak of the voigt to the peak of the breit wigner that makes it up
//
// \f$ D(\sigma;\Gamma,\mu) = \frac{V(\mu;\mu,\sigma,\Gamma)}{BW(\mu;\mu,\Gamma)} \f$
// Where V(\mu;\mu,\sigma,\Gamma) is the value of the relativistic voigt at the median
// and   BW(\mu;\mu,\Gamma) is the value of the relativistic breit wigner at the median
// where all varialbles are real.
// \f$\mu\f$ is the median (typically the pole mass of the resonance).
// \f$\sigma\f$ is the width of the gaussian
// \f$\Gamma\f$ is the width of the relativistic breit wigner
// for this function to be exact the integrationRange must be taken to infinity
// however, it converges rapidly and the integrationRange by default is taken
// to the maximum precision allowed by this method for a double.
////////////////////////////////////////////////////////////////////////////////
double VoigtRelativistic::dumpingFunction(double median, double sigma, double lg, double integrationRange)
{
   return VoigtRelativistic::evaluate(median, median, sigma, lg, integrationRange) /
          TMath::BreitWignerRelativistic(median, median, lg);
}

} // namespace Math
} // namespace ROOT
