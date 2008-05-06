// @(#)root/mathcore:$Id$
// Authors: David Gonzalez Maline    01/2008 

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

#include "Math/BrentRootFinder.h"
#include "Math/BrentMethods.h"
#include <cmath>

#ifndef ROOT_Math_Error
#include "Math/Error.h"
#endif

namespace ROOT {
namespace Math {


BrentRootFinder::BrentRootFinder() : fFunction(0) {}

BrentRootFinder::~BrentRootFinder() {}

int BrentRootFinder::SetFunction(const ROOT::Math::IGenFunction& f, double xlow, double xup)
{
// Set function to solve and the interval in where to look for the root. 

   fFunction = &f;

   if (xlow >= xup) 
   {
      double tmp = xlow;
      xlow = xup; 
      xup = tmp;
   }
   fXMin = xlow;
   fXMax = xup;

   return 0;
}

const char* BrentRootFinder::Name() const
{   return "BrentRootFinder";  }

double BrentRootFinder::Root() const
{   return fRoot;  }

int BrentRootFinder::Solve(int, double /*absTol*/, double /*relTol*/)
{
// Returns the X value corresponding to the function value fy for (xmin<x<xmax).

   double fy = 0; // To find the root

   double xmin = fXMin;
   double xmax = fXMax;

   int niter=0;
   double x;
   x = MinimStep(fFunction, 4, xmin, xmax, fy);
   bool ok = true;
   x = MinimBrent(fFunction, 4, xmin, xmax, x, fy, ok);
   while (!ok){
      if (niter>10){
         MATH_ERROR_MSG("Root", "Search didn't converge");
         break;
      }
      x=MinimStep(fFunction, 4, xmin, xmax, fy);
      x = MinimBrent(fFunction, 4, xmin, xmax, x, fy, ok);
      niter++;
   }

   fRoot = x;
   return 1;
}

} // namespace Math
} // namespace ROOT
