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


static int gDefaultNpx = 100; // default nunmber of points used in the grid to bracked the root
static int gDefaultNSearch = 10;  // nnumber of time the iteration (bracketing -Brent ) is repeted

   BrentRootFinder::BrentRootFinder() : fFunction(0),
                                        fLogScan(false), fNIter(0), 
                                        fNpx(0), fStatus(-1), 
                                        fXMin(0), fXMax(0), fRoot(0) 
{
   // default constructor (number of points used to bracket value is set to 100)
   fNpx = gDefaultNpx; 
}

void BrentRootFinder::SetDefaultNpx(int n) { gDefaultNpx = n; }

void BrentRootFinder::SetDefaultNSearch(int n) { gDefaultNSearch = n; }


bool BrentRootFinder::SetFunction(const ROOT::Math::IGenFunction& f, double xlow, double xup)
{
// Set function to solve and the interval in where to look for the root. 

   fFunction = &f;
   // invalid previous status
   fStatus = -1; 

   if (xlow >= xup) 
   {
      double tmp = xlow;
      xlow = xup; 
      xup = tmp;
   }
   fXMin = xlow;
   fXMax = xup;

   return true;
}

const char* BrentRootFinder::Name() const
{   return "BrentRootFinder";  }


bool BrentRootFinder::Solve(int maxIter, double absTol, double relTol)
{
  // Returns the X value corresponding to the function value fy for (xmin<x<xmax).

   if (!fFunction) { 
       MATH_ERROR_MSG("BrentRootFinder::Solve", "Function has not been set");
       return false;
   }

   if (fLogScan && fXMin <= 0) { 
      MATH_ERROR_MSG("BrentRootFinder::Solve", "xmin is < 0 and log scan is set - disable it");
      fLogScan = false; 
   }


   const double fy = 0; // To find the root
   fNIter = 0; 
   fStatus = -1; 

   double xmin = fXMin;
   double xmax = fXMax;

   int maxIter1 = gDefaultNSearch;  // external loop (number of search )
   int maxIter2 = maxIter;          // internal loop inside the Brent algorithm 

   int niter1 = 0;
   int niter2 = 0;
   bool ok = false; 
   while (!ok){
      if (niter1 > maxIter1){
         MATH_ERROR_MSG("BrentRootFinder::Solve", "Search didn't converge");
         fStatus = -2; 
         return false;
      }
      double x = BrentMethods::MinimStep(fFunction, 4, xmin, xmax, fy, fNpx,fLogScan);
      x = BrentMethods::MinimBrent(fFunction, 4, xmin, xmax, x, fy, ok, niter2, absTol, relTol, maxIter2);
      fNIter += niter2;  // count the total number of iterations
      niter1++;
      fRoot = x; 
   }

   fStatus = 0;
   return true;
}

} // namespace Math
} // namespace ROOT
