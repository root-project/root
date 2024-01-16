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

// RooRatio evaluate function.
inline double ratioEvaluate(double numerator, double denominator) {
   return numerator / denominator;
}

inline double bifurGaussEvaluate(double x, double mean, double sigmaL, double sigmaR)
{
   // Note: this simplification does not work with Clad as of v1.1!
   // return gaussianEvaluate(x, mean, x < mean ? sigmaL : sigmaR);
   if(x < mean) return gaussianEvaluate(x, mean, sigmaL);
   return gaussianEvaluate(x, mean, sigmaR);
}

inline double efficiencyEvaluate(double effFuncVal, int catIndex, int sigCatIndex)
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

   if (x < 0) {
      return 0;
   } else if (x == 0.0) {
      return std::exp(-par);
   } else {
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

inline double interpolate6thDegreeExp(double x, double low, double high, double nominal, double boundary)
{
   double x0 = boundary;

   // GHL: Swagato's suggestions
   double powUp = std::pow(high / nominal, x0);
   double powDown = std::pow(low / nominal, x0);
   double logHi = std::log(high);
   double logLo = std::log(low);
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

   double a = 1. / (8 * x0) * (15 * A0 - 7 * x0 * S1 + x0 * x0 * A2);
   double b = 1. / (8 * x0 * x0) * (-24 + 24 * S0 - 9 * x0 * A1 + x0 * x0 * S2);
   double c = 1. / (4 * std::pow(x0, 3)) * (-5 * A0 + 5 * x0 * S1 - x0 * x0 * A2);
   double d = 1. / (4 * std::pow(x0, 4)) * (12 - 12 * S0 + 7 * x0 * A1 - x0 * x0 * S2);
   double e = 1. / (8 * std::pow(x0, 5)) * (+3 * A0 - 3 * x0 * S1 + x0 * x0 * A2);
   double f = 1. / (8 * std::pow(x0, 6)) * (-8 + 8 * S0 - 5 * x0 * A1 + x0 * x0 * S2);

   // evaluate the 6-th degree polynomial using Horner's method
   double value = 1. + x * (a + x * (b + x * (c + x * (d + x * (e + x * f)))));
   return value;
}

inline double
flexibleInterp(unsigned int code, double low, double high, double boundary, double nominal, double paramVal, double res)
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
         return a * std::pow(paramVal, 2) + b * paramVal + c;
      }
   } else if (code == 3) {
      // parabolic version of log-normal
      double a = 0.5 * (high + low) - nominal;
      double b = 0.5 * (high - low);
      double c = 0;
      if (paramVal > 1) {
         return (2 * a + b) * (paramVal - 1) + high - nominal;
      } else if (paramVal < -1) {
         return -1 * (2 * a - b) * (paramVal + 1) + low - nominal;
      } else {
         return a * std::pow(paramVal, 2) + b * paramVal + c;
      }
   } else if (code == 4) {
      double x = paramVal;
      if (x >= boundary) {
         return x * (high - nominal);
      } else if (x <= -boundary) {
         return x * (nominal - low);
      }

      return interpolate6thDegree(x, low, high, nominal, boundary);
   } else if (code == 5) {
      double x = paramVal;
      double mod = 1.0;
      if (x >= boundary) {
         mod = std::pow(high / nominal, +paramVal);
      } else if (x <= -boundary) {
         mod = std::pow(low / nominal, -paramVal);
      } else {
         mod = interpolate6thDegreeExp(x, low, high, nominal, boundary);
      }
      return res * (mod - 1.0);
   }

   return 0.0;
}

inline double logNormalEvaluate(double x, double k, double m0)
{
   return ROOT::Math::lognormal_pdf(x, std::log(m0), std::abs(std::log(k)));
}

inline double logNormalEvaluateStandard(double x, double sigma, double mu)
{
   return ROOT::Math::lognormal_pdf(x, mu, std::abs(sigma));
}

inline double effProdEvaluate(double eff, double pdf) {
   return eff * pdf;
}

} // namespace EvaluateFuncs

} // namespace Detail

} // namespace RooFit

#endif
