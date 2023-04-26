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

#include <TMath.h>

#include <cmath>

namespace RooFit {

namespace Detail {

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
   double resultScale = std::sqrt(TMath::TwoPi()) * sigma;

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
   if (prd < 0.0)
      cond = 2.0 - (ecmin + ecmax);
   else if (scaledMax <= 0.0)
      cond = ecmax - ecmin;
   else
      cond = ecmin - ecmax;
   return resultScale * 0.5 * cond;
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

} // namespace AnalyticalIntegrals

} // namespace Detail

} // namespace RooFit

#endif
