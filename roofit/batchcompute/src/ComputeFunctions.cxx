/*
 * Project: RooFit
 * Authors:
 *   Emmanouil Michalainas, CERN, Summer 2019
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

/**
\file ComputeFunctions.cxx
\ingroup roofit_dev_docs_batchcompute

This file contains vectorizable computation functions for PDFs and other Roofit objects.
The same source file can also be compiled with nvcc. All functions have a single `Batches`
object as an argument passed by value, which contains all the information necessary for the
computation. In case of cuda computations, the loops have a step (stride) the size of the grid
which allows for reusing the same code as the cpu implementations, easier debugging and in terms
of performance, maximum memory coalescing. For more details, see
https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
**/

#include "RooBatchCompute.h"
#include "RooNaNPacker.h"
#include "RooVDTHeaders.h"
#include "Batches.h"

#include <TMath.h>

#include <RooHeterogeneousMath.h>

#include <vector>

#ifdef __CUDACC__
#define BEGIN blockDim.x *blockIdx.x + threadIdx.x
#define STEP blockDim.x *gridDim.x
#else
#define BEGIN 0
#define STEP 1
#endif // #ifdef __CUDACC__

namespace RooBatchCompute {
namespace RF_ARCH {

__rooglobal__ void computeAddPdf(Batches &batches)
{
   const int nPdfs = batches.nExtra;
   for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      batches.output[i] = batches.extra[0] * batches.args[0][i];
   }
   for (int pdf = 1; pdf < nPdfs; pdf++) {
      for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
         batches.output[i] += batches.extra[pdf] * batches.args[pdf][i];
      }
   }
}

__rooglobal__ void computeArgusBG(Batches &batches)
{
   Batch m = batches.args[0];
   Batch m0 = batches.args[1];
   Batch c = batches.args[2];
   Batch p = batches.args[3];
   for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      const double t = m[i] / m0[i];
      const double u = 1 - t * t;
      batches.output[i] = c[i] * u + p[i] * fast_log(u);
   }
   for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      if (m[i] >= m0[i]) {
         batches.output[i] = 0.0;
      } else {
         batches.output[i] = m[i] * fast_exp(batches.output[i]);
      }
   }
}

__rooglobal__ void computeBMixDecay(Batches &batches)
{
   Batch coef0 = batches.args[0];
   Batch coef1 = batches.args[1];
   Batch tagFlav = batches.args[2];
   Batch delMistag = batches.args[3];
   Batch mixState = batches.args[4];
   Batch mistag = batches.args[5];

   for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      batches.output[i] =
         coef0[i] * (1.0 - tagFlav[i] * delMistag[0]) + coef1[i] * (mixState[i] * (1.0 - 2.0 * mistag[0]));
   }
}

__rooglobal__ void computeBernstein(Batches &batches)
{
   const int nCoef = batches.nExtra - 2;
   const int degree = nCoef - 1;
   const double xmin = batches.extra[nCoef];
   const double xmax = batches.extra[nCoef + 1];
   Batch xData = batches.args[0];

   // apply binomial coefficient in-place so we don't have to allocate new memory
   double binomial = 1.0;
   for (int k = 0; k < nCoef; k++) {
      batches.extra[k] = batches.extra[k] * binomial;
      binomial = (binomial * (degree - k)) / (k + 1);
   }

   if (STEP == 1) {
      double X[bufferSize];
      double _1_X[bufferSize];
      double powX[bufferSize];
      double pow_1_X[bufferSize];
      for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
         powX[i] = pow_1_X[i] = 1.0;
         X[i] = (xData[i] - xmin) / (xmax - xmin);
         _1_X[i] = 1 - X[i];
         batches.output[i] = 0.0;
      }

      // raising 1-x to the power of degree
      for (int k = 2; k <= degree; k += 2) {
         for (size_t i = BEGIN; i < batches.nEvents; i += STEP)
            pow_1_X[i] *= _1_X[i] * _1_X[i];
      }

      if (degree % 2 == 1) {
         for (size_t i = BEGIN; i < batches.nEvents; i += STEP)
            pow_1_X[i] *= _1_X[i];
      }

      // inverting 1-x ---> 1/(1-x)
      for (size_t i = BEGIN; i < batches.nEvents; i += STEP)
         _1_X[i] = 1 / _1_X[i];

      for (int k = 0; k < nCoef; k++) {
         for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
            batches.output[i] += batches.extra[k] * powX[i] * pow_1_X[i];

            // calculating next power for x and 1-x
            powX[i] *= X[i];
            pow_1_X[i] *= _1_X[i];
         }
      }
   } else {
      for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
         batches.output[i] = 0.0;
         const double X = (xData[i] - xmin) / (xmax - xmin);
         double powX = 1.0;
         double pow_1_X = 1.0;
         for (int k = 1; k <= degree; k++)
            pow_1_X *= 1 - X;
         const double _1_X = 1 / (1 - X);
         for (int k = 0; k < nCoef; k++) {
            batches.output[i] += batches.extra[k] * powX * pow_1_X;
            powX *= X;
            pow_1_X *= _1_X;
         }
      }
   }

   // reset extraArgs values so we don't mutate the Batches object
   binomial = 1.0;
   for (int k = 0; k < nCoef; k++) {
      batches.extra[k] = batches.extra[k] / binomial;
      binomial = (binomial * (degree - k)) / (k + 1);
   }
}

__rooglobal__ void computeBifurGauss(Batches &batches)
{
   Batch X = batches.args[0];
   Batch M = batches.args[1];
   Batch SL = batches.args[2];
   Batch SR = batches.args[3];
   for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      double arg = X[i] - M[i];
      if (arg < 0) {
         arg /= SL[i];
      } else {
         arg /= SR[i];
      }
      batches.output[i] = fast_exp(-0.5 * arg * arg);
   }
}

__rooglobal__ void computeBreitWigner(Batches &batches)
{
   Batch X = batches.args[0];
   Batch M = batches.args[1];
   Batch W = batches.args[2];
   for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      const double arg = X[i] - M[i];
      batches.output[i] = 1 / (arg * arg + 0.25 * W[i] * W[i]);
   }
}

__rooglobal__ void computeBukin(Batches &batches)
{
   Batch X = batches.args[0];
   Batch XP = batches.args[1];
   Batch SP = batches.args[2];
   Batch XI = batches.args[3];
   Batch R1 = batches.args[4];
   Batch R2 = batches.args[5];
   const double r3 = log(2.0);
   const double r6 = exp(-6.0);
   const double r7 = 2 * sqrt(2 * log(2.0));

   for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      const double r1 = XI[i] * fast_isqrt(XI[i] * XI[i] + 1);
      const double r4 = 1 / fast_isqrt(XI[i] * XI[i] + 1);
      const double hp = 1 / (SP[i] * r7);
      const double x1 = XP[i] + 0.5 * SP[i] * r7 * (r1 - 1);
      const double x2 = XP[i] + 0.5 * SP[i] * r7 * (r1 + 1);

      double r5 = 1.0;
      if (XI[i] > r6 || XI[i] < -r6)
         r5 = XI[i] / fast_log(r4 + XI[i]);

      double factor = 1;
      double y = X[i] - x1;
      double Yp = XP[i] - x1;
      double yi = r4 - XI[i];
      double rho = R1[i];
      if (X[i] >= x2) {
         factor = -1;
         y = X[i] - x2;
         Yp = XP[i] - x2;
         yi = r4 + XI[i];
         rho = R2[i];
      }

      batches.output[i] = rho * y * y / Yp / Yp - r3 + factor * 4 * r3 * y * hp * r5 * r4 / yi / yi;
      if (X[i] >= x1 && X[i] < x2) {
         batches.output[i] =
            fast_log(1 + 4 * XI[i] * r4 * (X[i] - XP[i]) * hp) / fast_log(1 + 2 * XI[i] * (XI[i] - r4));
         batches.output[i] *= -batches.output[i] * r3;
      }
      if (X[i] >= x1 && X[i] < x2 && XI[i] < r6 && XI[i] > -r6)
         batches.output[i] = -4 * r3 * (X[i] - XP[i]) * (X[i] - XP[i]) * hp * hp;
   }
   for (size_t i = BEGIN; i < batches.nEvents; i += STEP)
      batches.output[i] = fast_exp(batches.output[i]);
}

__rooglobal__ void computeCBShape(Batches &batches)
{
   Batch M = batches.args[0];
   Batch M0 = batches.args[1];
   Batch S = batches.args[2];
   Batch A = batches.args[3];
   Batch N = batches.args[4];
   for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      const double t = (M[i] - M0[i]) / S[i];
      if ((A[i] > 0 && t >= -A[i]) || (A[i] < 0 && -t >= A[i])) {
         batches.output[i] = -0.5 * t * t;
      } else {
         batches.output[i] = N[i] / (N[i] - A[i] * A[i] - A[i] * t);
         batches.output[i] = fast_log(batches.output[i]);
         batches.output[i] *= N[i];
         batches.output[i] -= 0.5 * A[i] * A[i];
      }
   }
   for (size_t i = BEGIN; i < batches.nEvents; i += STEP)
      batches.output[i] = fast_exp(batches.output[i]);
}

__rooglobal__ void computeChebychev(Batches &batches)
{
   Batch xData = batches.args[0];
   const int nCoef = batches.nExtra - 2;
   const double xmin = batches.extra[nCoef];
   const double xmax = batches.extra[nCoef + 1];

   if (STEP == 1) {
      double prev[bufferSize][2];
      double X[bufferSize];

      for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
         // set a0-->prev[i][0] and a1-->prev[i][1]
         // and x tranfsformed to range[-1..1]-->X[i]
         prev[i][0] = batches.output[i] = 1.0;
         prev[i][1] = X[i] = 2 * (xData[i] - 0.5 * (xmax + xmin)) / (xmax - xmin);
      }
      for (int k = 0; k < nCoef; k++) {
         for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
            batches.output[i] += prev[i][1] * batches.extra[k];

            // compute next order
            const double next = 2 * X[i] * prev[i][1] - prev[i][0];
            prev[i][0] = prev[i][1];
            prev[i][1] = next;
         }
      }
   } else {
      for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
         double prev0 = 1.0;
         double prev1 = 2 * (xData[i] - 0.5 * (xmax + xmin)) / (xmax - xmin);
         double X = prev1;
         batches.output[i] = 1.0;
         for (int k = 0; k < nCoef; k++) {
            batches.output[i] += prev1 * batches.extra[k];

            // compute next order
            const double next = 2 * X * prev1 - prev0;
            prev0 = prev1;
            prev1 = next;
         }
      }
   }
}

__rooglobal__ void computeChiSquare(Batches &batches)
{
   Batch X = batches.args[0];
   const double ndof = batches.extra[0];
   const double gamma = 1 / std::tgamma(ndof / 2.0);
   for (size_t i = BEGIN; i < batches.nEvents; i += STEP)
      batches.output[i] = gamma;

   constexpr double ln2 = 0.693147180559945309417232121458;
   for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      double arg = (ndof - 2) * fast_log(X[i]) - X[i] - ndof * ln2;
      batches.output[i] *= fast_exp(0.5 * arg);
   }
}

__rooglobal__ void computeDeltaFunction(Batches &batches)
{
   for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      batches.output[i] = 0.0 + (batches.args[0][i] == 1.0);
   }
}

__rooglobal__ void computeDstD0BG(Batches &batches)
{
   Batch DM = batches.args[0];
   Batch DM0 = batches.args[1];
   Batch C = batches.args[2];
   Batch A = batches.args[3];
   Batch B = batches.args[4];
   for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      const double ratio = DM[i] / DM0[i];
      const double arg1 = (DM0[i] - DM[i]) / C[i];
      const double arg2 = A[i] * fast_log(ratio);
      batches.output[i] = (1 - fast_exp(arg1)) * fast_exp(arg2) + B[i] * (ratio - 1);
   }

   for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      if (batches.output[i] < 0)
         batches.output[i] = 0;
   }
}

__rooglobal__ void computeExpPoly(Batches &batches)
{
   int lowestOrder = batches.extra[0];
   int nTerms = batches.extra[1];
   auto x = batches.args[0];

   for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      batches.output[i] = 0.0;
      double xTmp = std::pow(x[i], lowestOrder);
      for (int k = 0; k < nTerms; ++k) {
         batches.output[i] += batches.args[k + 1][i] * xTmp;
         xTmp *= x[i];
      }
      batches.output[i] = std::exp(batches.output[i]);
   }
}

__rooglobal__ void computeExponential(Batches &batches)
{
   Batch x = batches.args[0];
   Batch c = batches.args[1];
   for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      batches.output[i] = fast_exp(x[i] * c[i]);
   }
}

__rooglobal__ void computeExponentialNeg(Batches &batches)
{
   Batch x = batches.args[0];
   Batch c = batches.args[1];
   for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      batches.output[i] = fast_exp(-x[i] * c[i]);
   }
}

__rooglobal__ void computeGamma(Batches &batches)
{
   Batch X = batches.args[0];
   Batch G = batches.args[1];
   Batch B = batches.args[2];
   Batch M = batches.args[3];
   double gamma = -std::lgamma(G[0]);
   for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      if (X[i] == M[i]) {
         batches.output[i] = (G[i] == 1.0) / B[i];
      } else if (G._isVector) {
         batches.output[i] = -std::lgamma(G[i]);
      } else {
         batches.output[i] = gamma;
      }
   }

   for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      if (X[i] != M[i]) {
         const double invBeta = 1 / B[i];
         double arg = (X[i] - M[i]) * invBeta;
         batches.output[i] -= arg;
         arg = fast_log(arg);
         batches.output[i] += arg * (G[i] - 1);
         batches.output[i] = fast_exp(batches.output[i]);
         batches.output[i] *= invBeta;
      }
   }
}

__rooglobal__ void computeGaussModelExpBasis(Batches &batches)
{
   const double root2 = std::sqrt(2.);
   const double root2pi = std::sqrt(2. * std::atan2(0., -1.));

   const bool isMinus = batches.extra[0] < 0.0;
   const bool isPlus = batches.extra[0] > 0.0;

   for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {

      const double x = batches.args[0][i];
      const double mean = batches.args[1][i] * batches.args[2][i];
      const double sigma = batches.args[3][i] * batches.args[4][i];
      const double tau = batches.args[5][i];

      if (tau == 0.0) {
         // Straight Gaussian, used for unconvoluted PDF or expBasis with 0 lifetime
         double xprime = (x - mean) / sigma;
         double result = std::exp(-0.5 * xprime * xprime) / (sigma * root2pi);
         if (!isMinus && !isPlus)
            result *= 2;
         batches.output[i] = result;
      } else {
         // Convolution with exp(-t/tau)
         const double xprime = (x - mean) / tau;
         const double c = sigma / (root2 * tau);
         const double u = xprime / (2 * c);

         double result = 0.0;
         if (!isMinus)
            result += RooHeterogeneousMath::evalCerf(0, -u, c).real();
         if (!isPlus)
            result += RooHeterogeneousMath::evalCerf(0, u, c).real();
         batches.output[i] = result;
      }
   }
}

__rooglobal__ void computeGaussian(Batches &batches)
{
   auto x = batches.args[0];
   auto mean = batches.args[1];
   auto sigma = batches.args[2];
   for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      const double arg = x[i] - mean[i];
      const double halfBySigmaSq = -0.5 / (sigma[i] * sigma[i]);
      batches.output[i] = fast_exp(arg * arg * halfBySigmaSq);
   }
}

__rooglobal__ void computeIdentity(Batches &batches)
{
   for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      batches.output[i] = batches.args[0][i];
   }
}

__rooglobal__ void computeNegativeLogarithms(Batches &batches)
{
   for (size_t i = BEGIN; i < batches.nEvents; i += STEP)
      batches.output[i] = -fast_log(batches.args[0][i]);
   // Multiply by weights if they exist
   if (batches.extra[0]) {
      for (size_t i = BEGIN; i < batches.nEvents; i += STEP)
         batches.output[i] *= batches.args[1][i];
   }
}

__rooglobal__ void computeJohnson(Batches &batches)
{
   Batch mass = batches.args[0];
   Batch mu = batches.args[1];
   Batch lambda = batches.args[2];
   Batch gamma = batches.args[3];
   Batch delta = batches.args[4];
   const double sqrtTwoPi = std::sqrt(TMath::TwoPi());
   const double massThreshold = batches.extra[0];

   for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      const double arg = (mass[i] - mu[i]) / lambda[i];
#ifdef R__HAS_VDT
      const double asinh_arg = fast_log(arg + 1 / fast_isqrt(arg * arg + 1));
#else
      const double asinh_arg = asinh(arg);
#endif
      const double expo = gamma[i] + delta[i] * asinh_arg;
      const double result =
         delta[i] * fast_exp(-0.5 * expo * expo) * fast_isqrt(1. + arg * arg) / (sqrtTwoPi * lambda[i]);

      const double passThrough = mass[i] >= massThreshold;
      batches.output[i] = result * passThrough;
   }
}

/* Actual computation of Landau(x,mean,sigma) in a vectorization-friendly way
 * Code copied from function landau_pdf (math/mathcore/src/PdfFuncMathCore.cxx)
 * and rewritten to enable vectorization.
 */
__rooglobal__ void computeLandau(Batches &batches)
{
   auto case0 = [](double x) {
      const double a1[3] = {0.04166666667, -0.01996527778, 0.02709538966};
      const double u = fast_exp(x + 1.0);
      return 0.3989422803 * fast_exp(-1 / u - 0.5 * (x + 1)) * (1 + (a1[0] + (a1[1] + a1[2] * u) * u) * u);
   };
   auto case1 = [](double x) {
      constexpr double p1[5] = {0.4259894875, -0.1249762550, 0.03984243700, -0.006298287635, 0.001511162253};
      constexpr double q1[5] = {1.0, -0.3388260629, 0.09594393323, -0.01608042283, 0.003778942063};
      const double u = fast_exp(-x - 1);
      return fast_exp(-u - 0.5 * (x + 1)) * (p1[0] + (p1[1] + (p1[2] + (p1[3] + p1[4] * x) * x) * x) * x) /
             (q1[0] + (q1[1] + (q1[2] + (q1[3] + q1[4] * x) * x) * x) * x);
   };
   auto case2 = [](double x) {
      constexpr double p2[5] = {0.1788541609, 0.1173957403, 0.01488850518, -0.001394989411, 0.0001283617211};
      constexpr double q2[5] = {1.0, 0.7428795082, 0.3153932961, 0.06694219548, 0.008790609714};
      return (p2[0] + (p2[1] + (p2[2] + (p2[3] + p2[4] * x) * x) * x) * x) /
             (q2[0] + (q2[1] + (q2[2] + (q2[3] + q2[4] * x) * x) * x) * x);
   };
   auto case3 = [](double x) {
      constexpr double p3[5] = {0.1788544503, 0.09359161662, 0.006325387654, 0.00006611667319, -0.000002031049101};
      constexpr double q3[5] = {1.0, 0.6097809921, 0.2560616665, 0.04746722384, 0.006957301675};
      return (p3[0] + (p3[1] + (p3[2] + (p3[3] + p3[4] * x) * x) * x) * x) /
             (q3[0] + (q3[1] + (q3[2] + (q3[3] + q3[4] * x) * x) * x) * x);
   };
   auto case4 = [](double x) {
      constexpr double p4[5] = {0.9874054407, 118.6723273, 849.2794360, -743.7792444, 427.0262186};
      constexpr double q4[5] = {1.0, 106.8615961, 337.6496214, 2016.712389, 1597.063511};
      const double u = 1 / x;
      return u * u * (p4[0] + (p4[1] + (p4[2] + (p4[3] + p4[4] * u) * u) * u) * u) /
             (q4[0] + (q4[1] + (q4[2] + (q4[3] + q4[4] * u) * u) * u) * u);
   };
   auto case5 = [](double x) {
      constexpr double p5[5] = {1.003675074, 167.5702434, 4789.711289, 21217.86767, -22324.94910};
      constexpr double q5[5] = {1.0, 156.9424537, 3745.310488, 9834.698876, 66924.28357};
      const double u = 1 / x;
      return u * u * (p5[0] + (p5[1] + (p5[2] + (p5[3] + p5[4] * u) * u) * u) * u) /
             (q5[0] + (q5[1] + (q5[2] + (q5[3] + q5[4] * u) * u) * u) * u);
   };
   auto case6 = [](double x) {
      constexpr double p6[5] = {1.000827619, 664.9143136, 62972.92665, 475554.6998, -5743609.109};
      constexpr double q6[5] = {1.0, 651.4101098, 56974.73333, 165917.4725, -2815759.939};
      const double u = 1 / x;
      return u * u * (p6[0] + (p6[1] + (p6[2] + (p6[3] + p6[4] * u) * u) * u) * u) /
             (q6[0] + (q6[1] + (q6[2] + (q6[3] + q6[4] * u) * u) * u) * u);
   };
   auto case7 = [](double x) {
      const double a2[2] = {-1.845568670, -4.284640743};
      const double u = 1 / (x - x * fast_log(x) / (x + 1));
      return u * u * (1 + (a2[0] + a2[1] * u) * u);
   };

   Batch X = batches.args[0];
   Batch M = batches.args[1];
   Batch S = batches.args[2];

   for (size_t i = BEGIN; i < batches.nEvents; i += STEP)
      batches.output[i] = (X[i] - M[i]) / S[i];

   for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      if (S[i] <= 0.0) {
         batches.output[i] = 0;
      } else if (batches.output[i] < -5.5) {
         batches.output[i] = case0(batches.output[i]);
      } else if (batches.output[i] < -1.0) {
         batches.output[i] = case1(batches.output[i]);
      } else if (batches.output[i] < 1.0) {
         batches.output[i] = case2(batches.output[i]);
      } else if (batches.output[i] < 5.0) {
         batches.output[i] = case3(batches.output[i]);
      } else if (batches.output[i] < 12.0) {
         batches.output[i] = case4(batches.output[i]);
      } else if (batches.output[i] < 50.0) {
         batches.output[i] = case5(batches.output[i]);
      } else if (batches.output[i] < 300.) {
         batches.output[i] = case6(batches.output[i]);
      } else {
         batches.output[i] = case7(batches.output[i]);
      }
   }
}

__rooglobal__ void computeLognormal(Batches &batches)
{
   Batch X = batches.args[0];
   Batch M0 = batches.args[1];
   Batch K = batches.args[2];
   constexpr double rootOf2pi = 2.506628274631000502415765284811;
   for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      double lnxOverM0 = fast_log(X[i] / M0[i]);
      double lnk = fast_log(K[i]);
      if (lnk < 0)
         lnk = -lnk;
      double arg = lnxOverM0 / lnk;
      arg *= -0.5 * arg;
      batches.output[i] = fast_exp(arg) / (X[i] * lnk * rootOf2pi);
   }
}

__rooglobal__ void computeLognormalStandard(Batches &batches)
{
   Batch X = batches.args[0];
   Batch M0 = batches.args[1];
   Batch K = batches.args[2];
   constexpr double rootOf2pi = 2.506628274631000502415765284811;
   for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      double lnxOverM0 = fast_log(X[i]) - M0[i];
      double lnk = K[i];
      if (lnk < 0)
         lnk = -lnk;
      double arg = lnxOverM0 / lnk;
      arg *= -0.5 * arg;
      batches.output[i] = fast_exp(arg) / (X[i] * lnk * rootOf2pi);
   }
}

__rooglobal__ void computeNormalizedPdf(Batches &batches)
{
   auto rawVal = batches.args[0];
   auto normVal = batches.args[1];

   int nEvalErrorsType0 = 0;
   int nEvalErrorsType1 = 0;
   int nEvalErrorsType2 = 0;

   for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      double out = 0.0;
      // batches.output[i] = rawVal[i] / normVar[i];
      if (normVal[i] < 0. || (normVal[i] == 0. && rawVal[i] != 0)) {
         // Unreasonable normalisations. A zero integral can be tolerated if the function vanishes, though.
         out = RooNaNPacker::packFloatIntoNaN(-normVal[i] + (rawVal[i] < 0. ? -rawVal[i] : 0.));
         nEvalErrorsType0++;
      } else if (rawVal[i] < 0.) {
         // The pdf value is less than zero.
         out = RooNaNPacker::packFloatIntoNaN(-rawVal[i]);
         nEvalErrorsType1++;
      } else if (std::isnan(rawVal[i])) {
         // The pdf value is Not-a-Number.
         out = rawVal[i];
         nEvalErrorsType2++;
      } else {
         out = (rawVal[i] == 0. && normVal[i] == 0.) ? 0. : rawVal[i] / normVal[i];
      }
      batches.output[i] = out;
   }

   if (nEvalErrorsType0 > 0)
      batches.extra[0] = batches.extra[0] + nEvalErrorsType0;
   if (nEvalErrorsType1 > 1)
      batches.extra[1] = batches.extra[1] + nEvalErrorsType1;
   if (nEvalErrorsType2 > 2)
      batches.extra[2] = batches.extra[2] + nEvalErrorsType2;
}

/* TMath::ASinH(x) needs to be replaced with ln( x + sqrt(x^2+1))
 * argasinh -> the argument of TMath::ASinH()
 * argln -> the argument of the logarithm that replaces AsinH
 * asinh -> the value that the function evaluates to
 *
 * ln is the logarithm that was solely present in the initial
 * formula, that is before the asinh replacement
 */
__rooglobal__ void computeNovosibirsk(Batches &batches)
{
   Batch X = batches.args[0];
   Batch P = batches.args[1];
   Batch W = batches.args[2];
   Batch T = batches.args[3];
   constexpr double xi = 2.3548200450309494; // 2 Sqrt( Ln(4) )
   for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      double argasinh = 0.5 * xi * T[i];
      double argln = argasinh + 1 / fast_isqrt(argasinh * argasinh + 1);
      double asinh = fast_log(argln);

      double argln2 = 1 - (X[i] - P[i]) * T[i] / W[i];
      double ln = fast_log(argln2);
      batches.output[i] = ln / asinh;
      batches.output[i] *= -0.125 * xi * xi * batches.output[i];
      batches.output[i] -= 2.0 / xi / xi * asinh * asinh;
   }

   // faster if you exponentiate in a separate loop (dark magic!)
   for (size_t i = BEGIN; i < batches.nEvents; i += STEP)
      batches.output[i] = fast_exp(batches.output[i]);
}

__rooglobal__ void computePoisson(Batches &batches)
{
   Batch x = batches.args[0];
   Batch mean = batches.args[1];
   bool protectNegative = batches.extra[0];
   bool noRounding = batches.extra[1];
   for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      const double x_i = noRounding ? x[i] : floor(x[i]);
      batches.output[i] = std::lgamma(x_i + 1.);
   }

   for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      const double x_i = noRounding ? x[i] : floor(x[i]);
      const double logMean = fast_log(mean[i]);
      const double logPoisson = x_i * logMean - mean[i] - batches.output[i];
      batches.output[i] = fast_exp(logPoisson);

      // Cosmetics
      if (x_i < 0) {
         batches.output[i] = 0;
      } else if (x_i == 0) {
         batches.output[i] = 1 / fast_exp(mean[i]);
      }

      if (protectNegative && mean[i] < 0)
         batches.output[i] = 1.E-3;
   }
}

__rooglobal__ void computePolynomial(Batches &batches)
{
   const int nCoef = batches.extra[0];
   const std::size_t nEvents = batches.nEvents;
   Batch x = batches.args[nCoef];

   for (size_t i = BEGIN; i < nEvents; i += STEP) {
      batches.output[i] = batches.args[nCoef - 1][i];
   }

   // Indexes are in range 0..nCoef-1 but coefList[nCoef-1] has already been
   // processed.
   for (int k = nCoef - 2; k >= 0; k--) {
      for (size_t i = BEGIN; i < nEvents; i += STEP) {
         batches.output[i] = batches.args[k][i] + x[i] * batches.output[i];
      }
   }
}

__rooglobal__ void computePower(Batches &batches)
{
   const int nCoef = batches.extra[0];
   Batch x = batches.args[0];

   for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      batches.output[i] = 0.0;
      for (int k = 0; k < nCoef; ++k) {
         batches.output[i] += batches.args[2 * k + 1][i] * std::pow(x[i], batches.args[2 * k + 2][i]);
      }
   }
}

__rooglobal__ void computeProdPdf(Batches &batches)
{
   const int nPdfs = batches.extra[0];
   for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      batches.output[i] = 1.;
   }
   for (int pdf = 0; pdf < nPdfs; pdf++) {
      for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
         batches.output[i] *= batches.args[pdf][i];
      }
   }
}

__rooglobal__ void computeRatio(Batches &batches)
{
   for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      batches.output[i] = batches.args[0][i] / batches.args[1][i];
   }
}

__rooglobal__ void computeTruthModelExpBasis(Batches &batches)
{

   const bool isMinus = batches.extra[0] < 0.0;
   const bool isPlus = batches.extra[0] > 0.0;
   for (std::size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      double x = batches.args[0][i];
      // Enforce sign compatibility
      const bool isOutOfSign = (isMinus && x > 0.0) || (isPlus && x < 0.0);
      batches.output[i] = isOutOfSign ? 0.0 : fast_exp(-std::abs(x) / batches.args[1][i]);
   }
}

__rooglobal__ void computeTruthModelSinBasis(Batches &batches)
{
   const bool isMinus = batches.extra[0] < 0.0;
   const bool isPlus = batches.extra[0] > 0.0;
   for (std::size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      double x = batches.args[0][i];
      // Enforce sign compatibility
      const bool isOutOfSign = (isMinus && x > 0.0) || (isPlus && x < 0.0);
      batches.output[i] =
         isOutOfSign ? 0.0 : fast_exp(-std::abs(x) / batches.args[1][i]) * fast_sin(x * batches.args[2][i]);
   }
}

__rooglobal__ void computeTruthModelCosBasis(Batches &batches)
{
   const bool isMinus = batches.extra[0] < 0.0;
   const bool isPlus = batches.extra[0] > 0.0;
   for (std::size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      double x = batches.args[0][i];
      // Enforce sign compatibility
      const bool isOutOfSign = (isMinus && x > 0.0) || (isPlus && x < 0.0);
      batches.output[i] =
         isOutOfSign ? 0.0 : fast_exp(-std::abs(x) / batches.args[1][i]) * fast_cos(x * batches.args[2][i]);
   }
}

__rooglobal__ void computeTruthModelLinBasis(Batches &batches)
{
   const bool isMinus = batches.extra[0] < 0.0;
   const bool isPlus = batches.extra[0] > 0.0;
   for (std::size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      double x = batches.args[0][i];
      // Enforce sign compatibility
      const bool isOutOfSign = (isMinus && x > 0.0) || (isPlus && x < 0.0);
      if (isOutOfSign) {
         batches.output[i] = 0.0;
      } else {
         const double tscaled = std::abs(x) / batches.args[1][i];
         batches.output[i] = fast_exp(-tscaled) * tscaled;
      }
   }
}

__rooglobal__ void computeTruthModelQuadBasis(Batches &batches)
{
   const bool isMinus = batches.extra[0] < 0.0;
   const bool isPlus = batches.extra[0] > 0.0;
   for (std::size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      double x = batches.args[0][i];
      // Enforce sign compatibility
      const bool isOutOfSign = (isMinus && x > 0.0) || (isPlus && x < 0.0);
      if (isOutOfSign) {
         batches.output[i] = 0.0;
      } else {
         const double tscaled = std::abs(x) / batches.args[1][i];
         batches.output[i] = fast_exp(-tscaled) * tscaled * tscaled;
      }
   }
}

__rooglobal__ void computeTruthModelSinhBasis(Batches &batches)
{
   const bool isMinus = batches.extra[0] < 0.0;
   const bool isPlus = batches.extra[0] > 0.0;
   for (std::size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      double x = batches.args[0][i];
      // Enforce sign compatibility
      const bool isOutOfSign = (isMinus && x > 0.0) || (isPlus && x < 0.0);
      batches.output[i] =
         isOutOfSign ? 0.0 : fast_exp(-std::abs(x) / batches.args[1][i]) * sinh(x * batches.args[2][i] * 0.5);
   }
}

__rooglobal__ void computeTruthModelCoshBasis(Batches &batches)
{
   const bool isMinus = batches.extra[0] < 0.0;
   const bool isPlus = batches.extra[0] > 0.0;
   for (std::size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      double x = batches.args[0][i];
      // Enforce sign compatibility
      const bool isOutOfSign = (isMinus && x > 0.0) || (isPlus && x < 0.0);
      batches.output[i] =
         isOutOfSign ? 0.0 : fast_exp(-std::abs(x) / batches.args[1][i]) * cosh(x * batches.args[2][i] * .5);
   }
}

__rooglobal__ void computeVoigtian(Batches &batches)
{
   Batch X = batches.args[0];
   Batch M = batches.args[1];
   Batch W = batches.args[2];
   Batch S = batches.args[3];
   const double invSqrt2 = 0.707106781186547524400844362105;
   for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      const double arg = (X[i] - M[i]) * (X[i] - M[i]);
      if (S[i] == 0.0 && W[i] == 0.0) {
         batches.output[i] = 1.0;
      } else if (S[i] == 0.0) {
         batches.output[i] = 1 / (arg + 0.25 * W[i] * W[i]);
      } else if (W[i] == 0.0) {
         batches.output[i] = fast_exp(-0.5 * arg / (S[i] * S[i]));
      } else {
         batches.output[i] = invSqrt2 / S[i];
      }
   }

   for (size_t i = BEGIN; i < batches.nEvents; i += STEP) {
      if (S[i] != 0.0 && W[i] != 0.0) {
         if (batches.output[i] < 0)
            batches.output[i] = -batches.output[i];
         const double factor = W[i] > 0.0 ? 0.5 : -0.5;
         RooHeterogeneousMath::STD::complex<double> z(batches.output[i] * (X[i] - M[i]),
                                                      factor * batches.output[i] * W[i]);
         batches.output[i] *= RooHeterogeneousMath::faddeeva(z).real();
      }
   }
}

/// Returns a std::vector of pointers to the compute functions in this file.
std::vector<void (*)(Batches &)> getFunctions()
{
   return {computeAddPdf,
           computeArgusBG,
           computeBMixDecay,
           computeBernstein,
           computeBifurGauss,
           computeBreitWigner,
           computeBukin,
           computeCBShape,
           computeChebychev,
           computeChiSquare,
           computeDeltaFunction,
           computeDstD0BG,
           computeExpPoly,
           computeExponential,
           computeExponentialNeg,
           computeGamma,
           computeGaussModelExpBasis,
           computeGaussian,
           computeIdentity,
           computeJohnson,
           computeLandau,
           computeLognormal,
           computeLognormalStandard,
           computeNegativeLogarithms,
           computeNormalizedPdf,
           computeNovosibirsk,
           computePoisson,
           computePolynomial,
           computePower,
           computeProdPdf,
           computeRatio,
           computeTruthModelExpBasis,
           computeTruthModelSinBasis,
           computeTruthModelCosBasis,
           computeTruthModelLinBasis,
           computeTruthModelQuadBasis,
           computeTruthModelSinhBasis,
           computeTruthModelCoshBasis,
           computeVoigtian};
}
} // End namespace RF_ARCH
} // End namespace RooBatchCompute
