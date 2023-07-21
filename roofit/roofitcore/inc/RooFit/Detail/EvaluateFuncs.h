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
#include <Math/PdfFuncMathCore.h>

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

inline double interpolate6thDegree(double x, double low, double high, double nominal, double boundary)
{
   double t = x / boundary;
   double eps_plus = high - nominal;
   double eps_minus = nominal - low;
   double S = 0.5 * (eps_plus + eps_minus);
   double A = 0.0625 * (eps_plus - eps_minus);

   return x * (S + t * A * (15 + t * t * (-10 + t * t * 3)));
}

inline double flexibleInterp(unsigned int code, double low, double high, double boundary, double nominal,
                             double paramVal, double total)
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

      return total * std::exp(interpolate6thDegree(x, low<=0 ? 0. : std::log(low), high <= 0 ? : 0. : std::log(high), std::log(nominal), boundary));
   }

   return total;
}

inline double
piecewiseInterpolation(unsigned int code, double low, double high, double nominal, double param, double sum)
{
   if (code == 4) {

      // WVE ****************************************************************
      // WVE *** THIS CODE IS CRITICAL TO HISTFACTORY FIT CPU PERFORMANCE ***
      // WVE *** Do not modify unless you know what you are doing...      ***
      // WVE ****************************************************************

      double x = param;
      if (x > 1.0) {
         return sum + x * (high - nominal);
      } else if (x < -1.0) {
         return sum + x * (nominal - low);
      } else {
         // fcns+der+2nd_der are eq at bd
         double val = nominal + interpolate6thDegree(x, low, high, nominal, 1.0);

         if (val < 0)
            val = 0;
         return sum + val - nominal;
      }
      // WVE ****************************************************************
   } else {

      double x0 = 1.0; // boundary;
      double x = param;

      if (x > x0 || x < -x0) {
         if (x > 0)
            return sum + x * (high - nominal);
         else
            return sum + x * (nominal - low);
      } else if (nominal != 0) {
         double eps_plus = high - nominal;
         double eps_minus = nominal - low;
         double S = (eps_plus + eps_minus) / 2;
         double A = (eps_plus - eps_minus) / 2;

         // fcns+der are eq at bd
         double a = S;
         double b = 3 * A / (2 * x0);
         // double c = 0;
         double d = -A / (2 * x0 * x0 * x0);

         double val = nominal + a * x + b * std::pow(x, 2) + 0 /*c*pow(x, 3)*/ + d * std::pow(x, 4);
         if (val < 0)
            val = 0;

         return sum + val - nominal;
      }
   }

   return sum;
}

inline double logNormalEvaluate(double x, double k, double m0)
{
   double ln_k = TMath::Abs(TMath::Log(k));
   double ln_m0 = TMath::Log(m0);

   double ret = ROOT::Math::lognormal_pdf(x, ln_m0, ln_k);
   return ret;
}

} // namespace EvaluateFuncs

} // namespace Detail

} // namespace RooFit

#endif
