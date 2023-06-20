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

#ifndef RooFit_Detail_EvaluateFuncs_h
#define RooFit_Detail_EvaluateFuncs_h

#include <TMath.h>

#include <cmath>

namespace RooFit {

namespace Detail {

namespace EvaluateFuncs {

/// @brief Function to evaluate an un-normalized RooGaussian.
inline double gaussianEvaluate(double x, double mean, double sigma)
{
   const double arg = x - mean;
   const double sig = sigma;
   return std::exp(-0.5 * arg * arg / (sig * sig));
}

/// In pdfMode, a coefficient for the constant term of 1.0 is implied if lowestOrder > 0.
template <bool pdfMode = false>
inline double polynomialEvaluate(double const *coeffs, int nCoeffs, int lowestOrder, double x)
{
   double retVal = coeffs[nCoeffs - 1];
   for (int i = nCoeffs - 2; i >= 0; i--)
      retVal = coeffs[i] + x * retVal;
   retVal = retVal * std::pow(x, lowestOrder);
   return retVal + (pdfMode && lowestOrder > 0 ? 1.0 : 0.0);
}

inline double chebychevEvaluate(double *coeffs, unsigned int nCoeffs, double x_in, double xMin, double xMax)
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

inline double constraintSumEvaluate(double const *comp, unsigned int compSize)
{
   double sum = 0;
   for (unsigned int i = 0; i < compSize; i++) {
      sum -= std::log(comp[i]);
   }
   return sum;
}

inline unsigned int getUniformBinning(double low, double high, double val, unsigned int numBins)
{
   double binWidth = (high - low) / numBins;
   return val >= high ? numBins - 1 : std::abs((val - low) / binWidth);
}

inline double poissonEvaluate(double x, double par)
{
   if (par < 0)
      return TMath::QuietNaN();

   if (x < 0)
      return 0;
   else if (x == 0.0)
      return std::exp(-par);
   else {
      double out = x * std::log(par) - TMath::LnGamma(x + 1.) - par;
      return std::exp(out);
   }
}

/// Evaluate the 6-th degree polynomial using Horner's method.
inline double interpolate6thDegreeHornerPolynomial(double const *p, double x)
{
   return 1. + x * (p[0] + x * (p[1] + x * (p[2] + x * (p[3] + x * (p[4] + x * p[5])))));
}

inline double flexibleInterp(unsigned int code, double *polCoeff, double low, double high, double boundary,
                             double nominal, double paramVal, double total)
{
   if (code == 0) {
      // piece-wise linear
      if (paramVal > 0)
         return total + paramVal * (high - nominal);
      else
         return total + paramVal * (nominal - low);
   } else if (code == 1) {
      // pice-wise log
      if (paramVal >= 0)
         return total * std::pow(high / nominal, +paramVal);
      else
         return total * std::pow(low / nominal, -paramVal);
   } else if (code == 2) {
      // parabolic with linear
      double a = 0.5 * (high + low) - nominal;
      double b = 0.5 * (high - low);
      double c = 0;
      if (paramVal > 1) {
         return total + (2 * a + b) * (paramVal - 1) + high - nominal;
      } else if (paramVal < -1) {
         return total + -1 * (2 * a - b) * (paramVal + 1) + low - nominal;
      } else {
         return total + a * std::pow(paramVal, 2) + b * paramVal + c;
      }
   } else if (code == 3) {
      // parabolic version of log-normal
      double a = 0.5 * (high + low) - nominal;
      double b = 0.5 * (high - low);
      double c = 0;
      if (paramVal > 1) {
         return total + (2 * a + b) * (paramVal - 1) + high - nominal;
      } else if (paramVal < -1) {
         return total + -1 * (2 * a - b) * (paramVal + 1) + low - nominal;
      } else {
         return total + a * std::pow(paramVal, 2) + b * paramVal + c;
      }
   } else if (code == 4) {
      double x = paramVal;
      if (x >= boundary) {
         return total * std::pow(high / nominal, +paramVal);
      } else if (x <= -boundary) {
         return total * std::pow(low / nominal, -paramVal);
      }

      return total * interpolate6thDegreeHornerPolynomial(polCoeff, x);
   }

   return total;
}

} // namespace EvaluateFuncs

} // namespace Detail

} // namespace RooFit

#endif
