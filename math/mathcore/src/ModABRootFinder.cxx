// @(#)root/mathcore:$Id$
// Authors: Nedelcho Ganchovski    03/03/2026

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2026  CERN                                           *
 * All rights reserved.                                               *
 *                                                                    *
 * For the licensing terms see $ROOTSYS/LICENSE.                      *
 * For the list of contributors see $ROOTSYS/README/CREDITS.          *
 *                                                                    *
 **********************************************************************/

#include "Math/ModABRootFinder.h"
#include "Math/IFunctionfwd.h"
#include "Math/Error.h"
#include <cmath>

namespace ROOT::Math {

bool ModABRootFinder::SetFunction(const ROOT::Math::IGenFunction &f, double xmin, double xmax)
{
   // Set function to solve and the interval in where to look for the root.

   fFunction = &f;
   // invalid previous status
   fStatus = -1;
   if (xmin > xmax) {
      fXMin = xmax;
      fXMax = xmin;
   } else {
      fXMin = xmin;
      fXMax = xmax;
   }
   return true;
}

const char *ModABRootFinder::Name() const
{
   return "ModABRootFinder";
}

namespace {
int sign(double x)
{
   return (x > 0) - (x < 0);
}
} // namespace

bool ModABRootFinder::Solve(int maxIter, double absTol, double relTol)
{
   // Returns the X value corresponding to the function value fy for (xmin<x<xmax).

   if (!fFunction) {
      MATH_ERROR_MSG("ModABRootFinder::Solve", "Function has not been set");
      return false;
   }
   double x1 = fXMin, y1 = (*fFunction)(x1);
   if (y1 == 0.0) {
      fRoot = x1;
      fStatus = 0;
      return true;
   }
   double x2 = fXMax, y2 = (*fFunction)(x2);
   if (y2 == 0.0) {
      fRoot = x2;
      fStatus = 0;
      return true;
   }
   if (sign(y1) == sign(y2)) {
      MATH_ERROR_MSG("ModABRootFinder::Solve", "Function values at the interval endpoints have the same sign");
      return false;
   }
   int side = 0;
   bool bisection = true;
   double threshold = x2 - x1;
   const double C = 16;
   for (int i = 1; i <= maxIter; ++i) {
      double x3, y3;
      if (bisection) {
         x3 = (x1 + x2) / 2.0;
         y3 = (*fFunction)(x3);
         double ym = (y1 + y2) / 2.0;
         double r = 1 - std::fabs(ym / (y2 - y1));
         double k = r * r;
         if (std::fabs(ym - y3) < k * (std::fabs(y3) + std::fabs(ym))) {
            bisection = false;
            threshold = (x2 - x1) * C;
         }
      } else {
         x3 = (x1 * y2 - y1 * x2) / (y2 - y1);
         // Clamp x3 when floating point round-off errors shoots it out of the bracketing interval. Assign also y values
         // to avoid redundant re-evaluation
         if (x3 <= x1) {
            x3 = x1;
            y3 = y1;
         } else if (x3 >= x2) {
            x3 = x2;
            y3 = y2;
         } else
            y3 = (*fFunction)(x3);

         threshold /= 2.0;
      }
      fRoot = x3;
      fNIter = i;
      double eps = absTol + relTol * std::abs(x3);
      if (std::fabs(y3) == 0.0 || x2 - x1 <= eps) {
         fStatus = 0;
         return true;
      }
      if (y1 * y3 > 0.0) {
         if (side == 1) {
            double m = 1 - y3 / y1;
            if (m <= 0)
               y2 /= 2;
            else
               y2 *= m;
         } else if (!bisection)
            side = 1;

         x1 = x3;
         y1 = y3;
      } else {
         if (side == -1) {
            double m = 1 - y3 / y2;
            if (m <= 0)
               y1 /= 2;
            else
               y1 *= m;
         } else if (!bisection)
            side = -1;

         x2 = x3;
         y2 = y3;
      }
      if (x2 - x1 > threshold) {
         bisection = true;
         side = 0;
      }
   }
   fNIter = maxIter;
   MATH_ERROR_MSG("ModABRootFinder::Solve", "Search didn't converge");
   fStatus = -2;
   return false;
}
} // namespace ROOT::Math
