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

inline double addPdfEvaluate(double const *coeffs, unsigned int nCoeffs, double const *pdfs, unsigned int pdfSize)
{
   bool noLastCoeff = pdfSize != nCoeffs;
   double extraCoeff = 1;
   double sum = 0;

   unsigned int i;
   for (i = 0; i < nCoeffs; i++) {
      sum += pdfs[i] * coeffs[i];
      if (noLastCoeff)
         extraCoeff -= coeffs[i];
   }

   if (noLastCoeff) {
      sum += pdfs[i] * extraCoeff;
   }

   return sum;
}

} // namespace EvaluateFuncs

} // namespace Detail

} // namespace RooFit

#endif
