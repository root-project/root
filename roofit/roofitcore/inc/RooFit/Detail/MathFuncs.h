/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN 2024
 *   Garima Singh, CERN 2023
 *
 * Copyright (c) 2024, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFit_Detail_MathFuncs_h
#define RooFit_Detail_MathFuncs_h

#include <TMath.h>
#include <Math/PdfFuncMathCore.h>
#include <Math/ProbFuncMathCore.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace RooFit {
namespace Detail {
namespace MathFuncs {

/// Calculates the binomial coefficient n over k.
/// Equivalent to TMath::Binomial, but inlined.
inline double binomial(int n, int k)
{
   if (n < 0 || k < 0 || n < k)
      return TMath::SignalingNaN();
   if (k == 0 || n == k)
      return 1;

   int k1 = std::min(k, n - k);
   int k2 = n - k1;
   double fact = k2 + 1;
   for (double i = k1; i > 1.; --i) {
      fact *= (k2 + i) / i;
   }
   return fact;
}

/// The caller needs to make sure that there is at least one coefficient.
inline double bernstein(double x, double xmin, double xmax, double *coefs, int nCoefs)
{
   double xScaled = (x - xmin) / (xmax - xmin); // rescale to [0,1]
   int degree = nCoefs - 1;                     // n+1 polys of degree n

   // in case list of arguments passed is empty
   if (degree < 0) {
      return TMath::SignalingNaN();
   } else if (degree == 0) {
      return coefs[0];
   } else if (degree == 1) {

      double a0 = coefs[0];      // c0
      double a1 = coefs[1] - a0; // c1 - c0
      return a1 * xScaled + a0;

   } else if (degree == 2) {

      double a0 = coefs[0];            // c0
      double a1 = 2 * (coefs[1] - a0); // 2 * (c1 - c0)
      double a2 = coefs[2] - a1 - a0;  // c0 - 2 * c1 + c2
      return (a2 * xScaled + a1) * xScaled + a0;
   }

   double t = xScaled;
   double s = 1. - xScaled;

   double result = coefs[0] * s;
   for (int i = 1; i < degree; i++) {
      result = (result + t * binomial(degree, i) * coefs[i]) * s;
      t *= xScaled;
   }
   result += t * coefs[degree];

   return result;
}

/// @brief Function to evaluate an un-normalized RooGaussian.
inline double gaussian(double x, double mean, double sigma)
{
   const double arg = x - mean;
   const double sig = sigma;
   return std::exp(-0.5 * arg * arg / (sig * sig));
}

inline double product(double const *factors, std::size_t nFactors)
{
   double out = 1.0;
   for (std::size_t i = 0; i < nFactors; ++i) {
      out *= factors[i];
   }
   return out;
}

// RooRatio evaluate function.
inline double ratio(double numerator, double denominator)
{
   return numerator / denominator;
}

inline double bifurGauss(double x, double mean, double sigmaL, double sigmaR)
{
   // Note: this simplification does not work with Clad as of v1.1!
   // return gaussian(x, mean, x < mean ? sigmaL : sigmaR);
   if (x < mean)
      return gaussian(x, mean, sigmaL);
   return gaussian(x, mean, sigmaR);
}

inline double efficiency(double effFuncVal, int catIndex, int sigCatIndex)
{
   // Truncate efficiency function in range 0.0-1.0
   effFuncVal = std::clamp(effFuncVal, 0.0, 1.0);

   if (catIndex == sigCatIndex)
      return effFuncVal; // Accept case
   else
      return 1 - effFuncVal; // Reject case
}

/// In pdfMode, a coefficient for the constant term of 1.0 is implied if lowestOrder > 0.
template <bool pdfMode = false>
inline double polynomial(double const *coeffs, int nCoeffs, int lowestOrder, double x)
{
   double retVal = coeffs[nCoeffs - 1];
   for (int i = nCoeffs - 2; i >= 0; i--) {
      retVal = coeffs[i] + x * retVal;
   }
   retVal = retVal * std::pow(x, lowestOrder);
   return retVal + (pdfMode && lowestOrder > 0 ? 1.0 : 0.0);
}

inline double chebychev(double *coeffs, unsigned int nCoeffs, double x_in, double xMin, double xMax)
{
   // transform to range [-1, +1]
   const double xPrime = (x_in - 0.5 * (xMax + xMin)) / (0.5 * (xMax - xMin));

   // extract current values of coefficients
   double sum = 1.;
   if (nCoeffs > 0) {
      double curr = xPrime;
      double twox = 2 * xPrime;
      double last = 1;
      double newval = twox * curr - last;
      last = curr;
      curr = newval;
      for (unsigned int i = 0; nCoeffs != i; ++i) {
         sum += last * coeffs[i];
         newval = twox * curr - last;
         last = curr;
         curr = newval;
      }
   }
   return sum;
}

inline double multipdf(int idx, double const *pdfs)
{
   /* if (idx < 0 || idx >= static_cast<int>(pdfs.size())){
        throw std::out_of_range("Invalid PDF index");

   }
   */
   return pdfs[idx];
}
inline double constraintSum(double const *comp, unsigned int compSize)
{
   double sum = 0;
   for (unsigned int i = 0; i < compSize; i++) {
      sum -= std::log(comp[i]);
   }
   return sum;
}

inline unsigned int uniformBinNumber(double low, double high, double val, unsigned int numBins, double coef)
{
   double binWidth = (high - low) / numBins;
   return coef * (val >= high ? numBins - 1 : std::abs((val - low) / binWidth));
}

inline unsigned int rawBinNumber(double x, double const *boundaries, std::size_t nBoundaries)
{
   double const *end = boundaries + nBoundaries;
   double const *it = std::lower_bound(boundaries, end, x);
   // always return valid bin number
   while (boundaries != it && (end == it || end == it + 1 || x < *it)) {
      --it;
   }
   return it - boundaries;
}

inline unsigned int
binNumber(double x, double coef, double const *boundaries, unsigned int nBoundaries, int nbins, int blo)
{
   const int rawBin = rawBinNumber(x, boundaries, nBoundaries);
   int tmp = std::min(nbins, rawBin - blo);
   return coef * std::max(0, tmp);
}

inline double interpolate1d(double low, double high, double val, unsigned int numBins, double const *vals)
{
   double binWidth = (high - low) / numBins;
   int idx = val >= high ? numBins - 1 : std::abs((val - low) / binWidth);

   // interpolation
   double central = low + (idx + 0.5) * binWidth;
   if (val > low + 0.5 * binWidth && val < high - 0.5 * binWidth) {
      double slope;
      if (val < central) {
         slope = vals[idx] - vals[idx - 1];
      } else {
         slope = vals[idx + 1] - vals[idx];
      }
      return vals[idx] + slope * (val - central) / binWidth;
   }

   return vals[idx];
}

inline double poisson(double x, double par)
{
   if (par < 0)
      return TMath::QuietNaN();

   if (x < 0) {
      return 0;
   } else if (x == 0.0) {
      return std::exp(-par);
   } else {
      double out = x * std::log(par) - TMath::LnGamma(x + 1.) - par;
      return std::exp(out);
   }
}

inline double flexibleInterpSingle(unsigned int code, double low, double high, double boundary, double nominal,
                                   double paramVal, double res)
{
   if (code == 0) {
      // piece-wise linear
      if (paramVal > 0) {
         return paramVal * (high - nominal);
      } else {
         return paramVal * (nominal - low);
      }
   } else if (code == 1) {
      // piece-wise log
      if (paramVal >= 0) {
         return res * (std::pow(high / nominal, +paramVal) - 1);
      } else {
         return res * (std::pow(low / nominal, -paramVal) - 1);
      }
   } else if (code == 2) {
      // parabolic with linear
      double a = 0.5 * (high + low) - nominal;
      double b = 0.5 * (high - low);
      double c = 0;
      if (paramVal > 1) {
         return (2 * a + b) * (paramVal - 1) + high - nominal;
      } else if (paramVal < -1) {
         return -1 * (2 * a - b) * (paramVal + 1) + low - nominal;
      } else {
         return a * paramVal * paramVal + b * paramVal + c;
      }
      // According to an old comment in the source code, code 3 was apparently
      // meant to be a "parabolic version of log-normal", but it never got
      // implemented. If someone would need it, it could be implemented as doing
      // code 2 in log space.
   } else if (code == 4 || code == 6) {
      double x = paramVal;
      double mod = 1.0;
      if (code == 6) {
         high /= nominal;
         low /= nominal;
         nominal = 1;
      }
      if (x >= boundary) {
         mod = x * (high - nominal);
      } else if (x <= -boundary) {
         mod = x * (nominal - low);
      } else {
         // interpolate 6th degree
         double t = x / boundary;
         double eps_plus = high - nominal;
         double eps_minus = nominal - low;
         double S = 0.5 * (eps_plus + eps_minus);
         double A = 0.0625 * (eps_plus - eps_minus);

         mod = x * (S + t * A * (15 + t * t * (-10 + t * t * 3)));
      }

      // code 6 is multiplicative version of code 4
      if (code == 6) {
         mod *= res;
      }
      return mod;

   } else if (code == 5) {
      double x = paramVal;
      double mod = 1.0;
      if (x >= boundary) {
         mod = std::pow(high / nominal, +paramVal);
      } else if (x <= -boundary) {
         mod = std::pow(low / nominal, -paramVal);
      } else {
         // interpolate 6th degree exp
         double x0 = boundary;

         high /= nominal;
         low /= nominal;

         // GHL: Swagato's suggestions
         double logHi = std::log(high);
         double logLo = std::log(low);
         double powUp = std::exp(x0 * logHi);
         double powDown = std::exp(x0 * logLo);
         double powUpLog = high <= 0.0 ? 0.0 : powUp * logHi;
         double powDownLog = low <= 0.0 ? 0.0 : -powDown * logLo;
         double powUpLog2 = high <= 0.0 ? 0.0 : powUpLog * logHi;
         double powDownLog2 = low <= 0.0 ? 0.0 : -powDownLog * logLo;

         double S0 = 0.5 * (powUp + powDown);
         double A0 = 0.5 * (powUp - powDown);
         double S1 = 0.5 * (powUpLog + powDownLog);
         double A1 = 0.5 * (powUpLog - powDownLog);
         double S2 = 0.5 * (powUpLog2 + powDownLog2);
         double A2 = 0.5 * (powUpLog2 - powDownLog2);

         // fcns+der+2nd_der are eq at bd

         double x0Sq = x0 * x0;

         double a = 1. / (8 * x0) * (15 * A0 - 7 * x0 * S1 + x0 * x0 * A2);
         double b = 1. / (8 * x0Sq) * (-24 + 24 * S0 - 9 * x0 * A1 + x0 * x0 * S2);
         double c = 1. / (4 * x0Sq * x0) * (-5 * A0 + 5 * x0 * S1 - x0 * x0 * A2);
         double d = 1. / (4 * x0Sq * x0Sq) * (12 - 12 * S0 + 7 * x0 * A1 - x0 * x0 * S2);
         double e = 1. / (8 * x0Sq * x0Sq * x0) * (+3 * A0 - 3 * x0 * S1 + x0 * x0 * A2);
         double f = 1. / (8 * x0Sq * x0Sq * x0Sq) * (-8 + 8 * S0 - 5 * x0 * A1 + x0 * x0 * S2);

         // evaluate the 6-th degree polynomial using Horner's method
         double value = 1. + x * (a + x * (b + x * (c + x * (d + x * (e + x * f)))));
         mod = value;
      }
      return res * (mod - 1.0);
   }

   return 0.0;
}

inline double flexibleInterp(unsigned int code, double const *params, unsigned int n, double const *low,
                             double const *high, double boundary, double nominal, int doCutoff)
{
   double total = nominal;
   for (std::size_t i = 0; i < n; ++i) {
      total += flexibleInterpSingle(code, low[i], high[i], boundary, nominal, params[i], total);
   }

   return doCutoff && total <= 0 ? TMath::Limits<double>::Min() : total;
}

inline double landau(double x, double mu, double sigma)
{
   if (sigma <= 0.)
      return 0.;
   return ROOT::Math::landau_pdf((x - mu) / sigma);
}

inline double logNormal(double x, double k, double m0)
{
   return ROOT::Math::lognormal_pdf(x, std::log(m0), std::abs(std::log(k)));
}

inline double logNormalStandard(double x, double sigma, double mu)
{
   return ROOT::Math::lognormal_pdf(x, mu, std::abs(sigma));
}

inline double effProd(double eff, double pdf)
{
   return eff * pdf;
}

inline double nll(double pdf, double weight, int binnedL, int doBinOffset)
{
   if (binnedL) {
      // Special handling of this case since std::log(Poisson(0,0)=0 but can't be
      // calculated with usual log-formula since std::log(mu)=0. No update of result
      // is required since term=0.
      if (std::abs(pdf) < 1e-10 && std::abs(weight) < 1e-10) {
         return 0.0;
      }
      if (doBinOffset) {
         return pdf - weight - weight * (std::log(pdf) - std::log(weight));
      }
      return pdf - weight * std::log(pdf) + TMath::LnGamma(weight + 1);
   } else {
      return -weight * std::log(pdf);
   }
}

inline double recursiveFraction(double *a, unsigned int n)
{
   double prod = a[0];

   for (unsigned int i = 1; i < n; ++i) {
      prod *= 1.0 - a[i];
   }

   return prod;
}

inline double cbShape(double m, double m0, double sigma, double alpha, double n)
{
   double t = (m - m0) / sigma;
   if (alpha < 0)
      t = -t;

   double absAlpha = std::abs(alpha);

   if (t >= -absAlpha) {
      return std::exp(-0.5 * t * t);
   } else {
      double r = n / absAlpha;
      double a = std::exp(-0.5 * absAlpha * absAlpha);
      double b = r - absAlpha;

      return a * std::pow(r / (b - t), n);
   }
}

// For RooCBShape
inline double approxErf(double arg)
{
   if (arg > 5.0)
      return 1.0;
   if (arg < -5.0)
      return -1.0;

   return std::erf(arg);
}

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
   double ecmin = std::erfc(std::abs(scaledMin));
   double ecmax = std::erfc(std::abs(scaledMax));

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
      return resultScale * (sigmaL * (std::erf((xMax - mean) / xscaleL) - std::erf((xMin - mean) / xscaleL)));
   } else if (xMin > mean) {
      return resultScale * (sigmaR * (std::erf((xMax - mean) / xscaleR) - std::erf((xMin - mean) / xscaleR)));
   } else {
      return resultScale * (sigmaR * std::erf((xMax - mean) / xscaleR) - sigmaL * std::erf((xMin - mean) / xscaleL));
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
      integrandMin = std::max(0., integrandMin);

      if (integrandMax < 0. || integrandMax < integrandMin) {
         return 0;
      }
      const double delta = 100.0 * std::sqrt(mu);
      // If the limits are more than many standard deviations away from the mean,
      // we might as well return the integral of the full Poisson distribution to
      // save computing time.
      if (integrandMin < std::max(mu - delta, 0.0) && integrandMax > mu + delta) {
         return 1.;
      }

      // The range as integers. ixMin is included, ixMax outside.
      const unsigned int ixMin = integrandMin;
      const unsigned int ixMax = std::min(integrandMax + 1, (double)std::numeric_limits<unsigned int>::max());

      // Sum from 0 to just before the bin outside of the range.
      if (ixMin == 0) {
         return ROOT::Math::inc_gamma_c(ixMax, mu);
      } else {
         // If necessary, subtract from 0 to the beginning of the range
         if (ixMin <= mu) {
            return ROOT::Math::inc_gamma_c(ixMax, mu) - ROOT::Math::inc_gamma_c(ixMin, mu);
         } else {
            // Avoid catastrophic cancellation in the high tails:
            return ROOT::Math::inc_gamma(ixMin, mu) - ROOT::Math::inc_gamma(ixMax, mu);
         }
      }
   }

   // the integral with respect to the mean is the integral of a gamma distribution
   // negative ix does not need protection (gamma returns 0.0)
   const double ix = 1 + x;

   return ROOT::Math::inc_gamma(ix, integrandMax) - ROOT::Math::inc_gamma(ix, integrandMin);
}

inline double logNormalIntegral(double xMin, double xMax, double m0, double k)
{
   const double root2 = std::sqrt(2.);

   double ln_k = std::abs(std::log(k));
   double ret = 0.5 * (std::erf(std::log(xMax / m0) / (root2 * ln_k)) - std::erf(std::log(xMin / m0) / (root2 * ln_k)));

   return ret;
}

inline double logNormalIntegralStandard(double xMin, double xMax, double mu, double sigma)
{
   const double root2 = std::sqrt(2.);

   double ln_k = std::abs(sigma);
   double ret =
      0.5 * (std::erf((std::log(xMax) - mu) / (root2 * ln_k)) - std::erf((std::log(xMin) - mu) / (root2 * ln_k)));

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

   double sig = std::abs(sigma);

   double tmin = (mMin - m0) / sig;
   double tmax = (mMax - m0) / sig;

   if (alpha < 0) {
      double tmp = tmin;
      tmin = -tmax;
      tmax = -tmp;
   }

   double absAlpha = std::abs(alpha);

   if (tmin >= -absAlpha) {
      result += sig * sqrtPiOver2 * (approxErf(tmax / sqrt2) - approxErf(tmin / sqrt2));
   } else if (tmax <= -absAlpha) {
      double r = n / absAlpha;
      double a = r * std::exp(-0.5 * absAlpha * absAlpha);
      double b = r - absAlpha;

      if (useLog) {
         result += a * std::pow(r, n - 1) * sig * (std::log(b - tmin) - std::log(b - tmax));
      } else {
         result += a * sig / (1.0 - n) * (std::pow(r / (b - tmin), n - 1.0) - std::pow(r / (b - tmax), n - 1.0));
      }
   } else {
      double r = n / absAlpha;
      double a = r * std::exp(-0.5 * absAlpha * absAlpha);
      double b = r - absAlpha;

      double term1 = 0.0;
      if (useLog) {
         term1 = a * std::pow(r, n - 1) * sig * (std::log(b - tmin) - std::log(r));
      } else {
         term1 = a * sig / (1.0 - n) * (std::pow(r / (b - tmin), n - 1.0) - 1.0);
      }

      double term2 = sig * sqrtPiOver2 * (approxErf(tmax / sqrt2) - approxErf(-absAlpha / sqrt2));

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

inline double multiVarGaussian(int n, const double *x, const double *mu, const double *covI)
{
   double result = 0.0;

   // Compute the bilinear form (x-mu)^T * covI * (x-mu)
   for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
         result += (x[i] - mu[i]) * covI[i * n + j] * (x[j] - mu[j]);
      }
   }
   return std::exp(-0.5 * result);
}

// Integral of a step function defined by `nBins` intervals, where the
// intervals have values `coefs` and the boundary on the interval `iBin` is
// given by `[boundaries[i], boundaries[i+1])`.
inline double
stepFunctionIntegral(double xmin, double xmax, std::size_t nBins, double const *boundaries, double const *coefs)
{
   double out = 0.0;
   for (std::size_t i = 0; i < nBins; ++i) {
      double a = boundaries[i];
      double b = boundaries[i + 1];
      out += coefs[i] * std::max(0.0, std::min(b, xmax) - std::max(a, xmin));
   }
   return out;
}

} // namespace MathFuncs
} // namespace Detail
} // namespace RooFit

namespace clad {
namespace custom_derivatives {
namespace RooFit {
namespace Detail {
namespace MathFuncs {

// Clad can't generate the pullback for binNumber because of the
// std::lower_bound usage. But since binNumber returns an integer, and such
// functions have mathematically no derivatives anyway, we just declare a
// custom dummy pullback that does nothing.

template <class... Types>
void binNumber_pullback(Types...)
{
}

} // namespace MathFuncs
} // namespace Detail
} // namespace RooFit
} // namespace custom_derivatives
} // namespace clad

#endif
