/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN 2023
 *   Garima Singh, CERN 2023
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFit_Detail_AnalyticalIntegrals_h
#define RooFit_Detail_AnalyticalIntegrals_h

#include <RooFit/Detail/EvaluateFuncs.h>

#include <TMath.h>
#include <Math/ProbFuncMathCore.h>

#include <cmath>

namespace RooFit {

namespace Detail {

// For RooCBShape
inline double approxErf(double arg)
{
   if (arg > 5.0)
      return 1.0;
   if (arg < -5.0)
      return -1.0;

   return TMath::Erf(arg);
}

namespace AnalyticalIntegrals {

/// @brief Function to calculate the integral of an un-normalized RooGaussian over x. To calculate the integral over
/// mean, just interchange the respective values of x and mean.
/// @param xMin Minimum value of variable to integrate wrt.
/// @param xMax Maximum value of of variable to integrate wrt.
/// @param mean Mean.
/// @param sigma Sigma.
/// @return The integral of an un-normalized RooGaussian over the value in x.
inline double gaussianIntegral(double xMin, double xMax, double mean, double sigma)
{
   // The normalisation constant 1./sqrt(2*pi*sigma^2) is left out in evaluate().
   // Therefore, the integral is scaled up by that amount to make RooFit normalise
   // correctly.
   double resultScale = 0.5 * std::sqrt(TMath::TwoPi()) * sigma;

   // Here everything is scaled and shifted into a standard normal distribution:
   double xscale = TMath::Sqrt2() * sigma;
   double scaledMin = 0.;
   double scaledMax = 0.;
   scaledMin = (xMin - mean) / xscale;
   scaledMax = (xMax - mean) / xscale;

   // Here we go for maximum precision: We compute all integrals in the UPPER
   // tail of the Gaussian, because erfc has the highest precision there.
   // Therefore, the different cases for range limits in the negative hemisphere are mapped onto
   // the equivalent points in the upper hemisphere using erfc(-x) = 2. - erfc(x)
   double ecmin = TMath::Erfc(std::abs(scaledMin));
   double ecmax = TMath::Erfc(std::abs(scaledMax));

   double cond = 0.0;
   // Don't put this "prd" inside the "if" because clad will not be able to differentiate the code correctly (as of
   // v1.1)!
   double prd = scaledMin * scaledMax;
   if (prd < 0.0) {
      cond = 2.0 - (ecmin + ecmax);
   } else if (scaledMax <= 0.0) {
      cond = ecmax - ecmin;
   } else {
      cond = ecmin - ecmax;
   }
   return resultScale * cond;
}

inline double bifurGaussIntegral(double xMin, double xMax, double mean, double sigmaL, double sigmaR)
{
   const double xscaleL = TMath::Sqrt2() * sigmaL;
   const double xscaleR = TMath::Sqrt2() * sigmaR;

   const double resultScale = 0.5 * std::sqrt(TMath::TwoPi());

   if (xMax < mean) {
      return resultScale * (sigmaL * (TMath::Erf((xMax - mean) / xscaleL) - TMath::Erf((xMin - mean) / xscaleL)));
   } else if (xMin > mean) {
      return resultScale * (sigmaR * (TMath::Erf((xMax - mean) / xscaleR) - TMath::Erf((xMin - mean) / xscaleR)));
   } else {
      return resultScale *
             (sigmaR * TMath::Erf((xMax - mean) / xscaleR) - sigmaL * TMath::Erf((xMin - mean) / xscaleL));
   }
}

inline double exponentialIntegral(double xMin, double xMax, double constant)
{
   if (constant == 0.0) {
      return xMax - xMin;
   }

   return (std::exp(constant * xMax) - std::exp(constant * xMin)) / constant;
}

/// In pdfMode, a coefficient for the constant term of 1.0 is implied if lowestOrder > 0.
template <bool pdfMode = false>
inline double polynomialIntegral(double const *coeffs, int nCoeffs, int lowestOrder, double xMin, double xMax)
{
   int denom = lowestOrder + nCoeffs;
   double min = coeffs[nCoeffs - 1] / double(denom);
   double max = coeffs[nCoeffs - 1] / double(denom);

   for (int i = nCoeffs - 2; i >= 0; i--) {
      denom--;
      min = (coeffs[i] / double(denom)) + xMin * min;
      max = (coeffs[i] / double(denom)) + xMax * max;
   }

   max = max * std::pow(xMax, 1 + lowestOrder);
   min = min * std::pow(xMin, 1 + lowestOrder);

   return max - min + (pdfMode && lowestOrder > 0.0 ? xMax - xMin : 0.0);
}

/// use fast FMA if available, fall back to normal arithmetic if not
inline double fast_fma(double x, double y, double z) noexcept
{
#if defined(FP_FAST_FMA) // check if std::fma has fast hardware implementation
   return std::fma(x, y, z);
#else // defined(FP_FAST_FMA)
   // std::fma might be slow, so use a more pedestrian implementation
#if defined(__clang__)
#pragma STDC FP_CONTRACT ON // hint clang that using an FMA is okay here
#endif                      // defined(__clang__)
   return (x * y) + z;
#endif                      // defined(FP_FAST_FMA)
}

inline double chebychevIntegral(double const *coeffs, unsigned int nCoeffs, double xMin, double xMax, double xMinFull,
                                double xMaxFull)
{
   const double halfrange = .5 * (xMax - xMin);
   const double mid = .5 * (xMax + xMin);

   // the full range of the function is mapped to the normalised [-1, 1] range
   const double b = (xMaxFull - mid) / halfrange;
   const double a = (xMinFull - mid) / halfrange;

   // coefficient for integral(T_0(x)) is 1 (implicit), integrate by hand
   // T_0(x) and T_1(x), and use for n > 1: integral(T_n(x) dx) =
   // (T_n+1(x) / (n + 1) - T_n-1(x) / (n - 1)) / 2
   double sum = b - a; // integrate T_0(x) by hand

   const unsigned int iend = nCoeffs;
   if (iend > 0) {
      {
         // integrate T_1(x) by hand...
         const double c = coeffs[0];
         sum = fast_fma(0.5 * (b + a) * (b - a), c, sum);
      }
      if (1 < iend) {
         double bcurr = b;
         double btwox = 2 * b;
         double blast = 1;

         double acurr = a;
         double atwox = 2 * a;
         double alast = 1;

         double newval = atwox * acurr - alast;
         alast = acurr;
         acurr = newval;

         newval = btwox * bcurr - blast;
         blast = bcurr;
         bcurr = newval;
         double nminus1 = 1.;
         for (unsigned int i = 1; iend != i; ++i) {
            // integrate using recursion relation
            const double c = coeffs[i];
            const double term2 = (blast - alast) / nminus1;

            newval = atwox * acurr - alast;
            alast = acurr;
            acurr = newval;

            newval = btwox * bcurr - blast;
            blast = bcurr;
            bcurr = newval;

            ++nminus1;
            const double term1 = (bcurr - acurr) / (nminus1 + 1.);
            const double intTn = 0.5 * (term1 - term2);
            sum = fast_fma(intTn, c, sum);
         }
      }
   }

   // take care to multiply with the right factor to account for the mapping to
   // normalised range [-1, 1]
   return halfrange * sum;
}

// Clad does not like std::max and std::min so redefined here for simplicity.
inline double max(double x, double y)
{
   return x >= y ? x : y;
}

inline double min(double x, double y)
{
   return x <= y ? x : y;
}

// The last param should be of type bool but it is not as that causes some issues with Cling for some reason...
inline double
poissonIntegral(int code, double mu, double x, double integrandMin, double integrandMax, unsigned int protectNegative)
{
   if (protectNegative && mu < 0.0) {
      return std::exp(-2.0 * mu); // make it fall quickly
   }

   if (code == 1) {
      // Implement integral over x as summation. Add special handling in case
      // range boundaries are not on integer values of x
      integrandMin = max(0, integrandMin);

      if (integrandMax < 0. || integrandMax < integrandMin) {
         return 0;
      }
      const double delta = 100.0 * std::sqrt(mu);
      // If the limits are more than many standard deviations away from the mean,
      // we might as well return the integral of the full Poisson distribution to
      // save computing time.
      if (integrandMin < max(mu - delta, 0.0) && integrandMax > mu + delta) {
         return 1.;
      }

      // The range as integers. ixMin is included, ixMax outside.
      const unsigned int ixMin = integrandMin;
      const unsigned int ixMax = min(integrandMax + 1, (double)std::numeric_limits<unsigned int>::max());

      // Sum from 0 to just before the bin outside of the range.
      if (ixMin == 0) {
         return ROOT::Math::gamma_cdf_c(mu, ixMax, 1);
      } else {
         // If necessary, subtract from 0 to the beginning of the range
         if (ixMin <= mu) {
            return ROOT::Math::gamma_cdf_c(mu, ixMax, 1) - ROOT::Math::gamma_cdf_c(mu, ixMin, 1);
         } else {
            // Avoid catastrophic cancellation in the high tails:
            return ROOT::Math::gamma_cdf(mu, ixMin, 1) - ROOT::Math::gamma_cdf(mu, ixMax, 1);
         }
      }
   }

   // the integral with respect to the mean is the integral of a gamma distribution
   // negative ix does not need protection (gamma returns 0.0)
   const double ix = 1 + x;

   return ROOT::Math::gamma_cdf(integrandMax, ix, 1.0) - ROOT::Math::gamma_cdf(integrandMin, ix, 1.0);
}

inline double logNormalIntegral(double xMin, double xMax, double m0, double k)
{
   const double root2 = std::sqrt(2.);

   double ln_k = std::abs(std::log(k));
   double ret =
      0.5 * (TMath::Erf(std::log(xMax / m0) / (root2 * ln_k)) - TMath::Erf(std::log(xMin / m0) / (root2 * ln_k)));

   return ret;
}

inline double logNormalIntegralStandard(double xMin, double xMax, double mu, double sigma)
{
   const double root2 = std::sqrt(2.);

   double ln_k = std::abs(sigma);
   double ret =
      0.5 * (TMath::Erf((std::log(xMax) - mu) / (root2 * ln_k)) - TMath::Erf((std::log(xMin) - mu) / (root2 * ln_k)));

   return ret;
}

inline double cbShapeIntegral(double mMin, double mMax, double m0, double sigma, double alpha, double n)
{
   const double sqrtPiOver2 = 1.2533141373;
   const double sqrt2 = 1.4142135624;

   double result = 0.0;
   bool useLog = false;

   if (std::abs(n - 1.0) < 1.0e-05)
      useLog = true;

   double sig = std::abs((double)sigma);

   double tmin = (mMin - m0) / sig;
   double tmax = (mMax - m0) / sig;

   if (alpha < 0) {
      double tmp = tmin;
      tmin = -tmax;
      tmax = -tmp;
   }

   double absAlpha = std::abs((double)alpha);

   if (tmin >= -absAlpha) {
      result += sig * sqrtPiOver2 * (RooFit::Detail::approxErf(tmax / sqrt2) - RooFit::Detail::approxErf(tmin / sqrt2));
   } else if (tmax <= -absAlpha) {
      double a = std::pow(n / absAlpha, n) * std::exp(-0.5 * absAlpha * absAlpha);
      double b = n / absAlpha - absAlpha;

      if (useLog) {
         result += a * sig * (std::log(b - tmin) - std::log(b - tmax));
      } else {
         result += a * sig / (1.0 - n) * (1.0 / (std::pow(b - tmin, n - 1.0)) - 1.0 / (std::pow(b - tmax, n - 1.0)));
      }
   } else {
      double a = std::pow(n / absAlpha, n) * std::exp(-0.5 * absAlpha * absAlpha);
      double b = n / absAlpha - absAlpha;

      double term1 = 0.0;
      if (useLog) {
         term1 = a * sig * (std::log(b - tmin) - std::log(n / absAlpha));
      } else {
         term1 = a * sig / (1.0 - n) * (1.0 / (std::pow(b - tmin, n - 1.0)) - 1.0 / (std::pow(n / absAlpha, n - 1.0)));
      }

      double term2 =
         sig * sqrtPiOver2 * (RooFit::Detail::approxErf(tmax / sqrt2) - RooFit::Detail::approxErf(-absAlpha / sqrt2));

      result += term1 + term2;
   }

   if (result == 0)
      return 1.E-300;
   return result;
}

inline double bernsteinIntegral(double xlo, double xhi, double xmin, double xmax, double *coefs, int nCoefs)
{
   double xloScaled = (xlo - xmin) / (xmax - xmin);
   double xhiScaled = (xhi - xmin) / (xmax - xmin);

   int degree = nCoefs - 1; // n+1 polys of degree n
   double norm = 0.;

   for (int i = 0; i <= degree; ++i) {
      // for each of the i Bernstein basis polynomials
      // represent it in the 'power basis' (the naive polynomial basis)
      // where the integral is straight forward.
      double temp = 0.;
      for (int j = i; j <= degree; ++j) { // power basis≈ß
         double binCoefs = binomial(degree, j) * binomial(j, i);
         double oneOverJPlusOne = 1. / (j + 1.);
         double powDiff = std::pow(xhiScaled, j + 1.) - std::pow(xloScaled, j + 1.);
         temp += std::pow(-1., j - i) * binCoefs * powDiff * oneOverJPlusOne;
      }
      temp *= coefs[i]; // include coeff
      norm += temp;     // add this basis's contribution to total
   }

   return norm * (xmax - xmin);
}

} // namespace AnalyticalIntegrals

} // namespace Detail

} // namespace RooFit

#endif
