// @(#)root/mathcore:$Id$
// Author: David Gonzalez Maline 2/2008
 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2004 Maline,  CERN/PH-SFT                            *
  *                                                                    *
  * This library is free software; you can redistribute it and/or      *
  * modify it under the terms of the GNU General Public License        *
  * as published by the Free Software Foundation; either version 2     *
  * of the License, or (at your option) any later version.             *
  *                                                                    *
  * This library is distributed in the hope that it will be useful,    *
  * but WITHOUT ANY WARRANTY; without even the implied warranty of     *
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU   *
  * General Public License for more details.                           *
  *                                                                    *
  * You should have received a copy of the GNU General Public License  *
  * along with this library (see file COPYING); if not, write          *
  * to the Free Software Foundation, Inc., 59 Temple Place, Suite      *
  * 330, Boston, MA 02111-1307 USA, or contact the author.             *
  *                                                                    *
  **********************************************************************/

// Header file for class BrentMinimizer1D
//
// Created by: Maline  at Mon Feb  4 09:32:36 2008
//
//

#include "Math/BrentMinimizer1D.h"
#include "Math/BrentMethods.h"
#include "Math/IFunction.h"

#ifndef ROOT_Math_Error
#include "Math/Error.h"
#endif

namespace ROOT {
namespace Math {

static int gDefaultNpx = 100; // default nunmber of points used in the grid to bracked the minimum
static int gDefaultNSearch = 10;  // nnumber of time the iteration (bracketing -Brent ) is repeted


   BrentMinimizer1D::BrentMinimizer1D(): IMinimizer1D(),
                                         fFunction(0),
                                         fLogScan(false), fNIter(0),
                                         fNpx(0), fStatus(-1),
                                         fXMin(0), fXMax(0), fXMinimum(0)
{
// Default Constructor.
   fNpx = gDefaultNpx;
}

void BrentMinimizer1D::SetDefaultNpx(int n) { gDefaultNpx = n; }

void BrentMinimizer1D::SetDefaultNSearch(int n) { gDefaultNSearch = n; }


void BrentMinimizer1D::SetFunction(const ROOT::Math::IGenFunction& f, double xlow, double xup)
{
// Sets function to be minimized.

   fFunction = &f;
   fStatus = -1;  // reset the status

   if (xlow >= xup)
   {
      double tmp = xlow;
      xlow = xup;
      xup = tmp;
   }
   fXMin = xlow;
   fXMax = xup;
}



double BrentMinimizer1D::FValMinimum() const
{   return (*fFunction)(fXMinimum); }

double BrentMinimizer1D::FValLower() const
{   return (*fFunction)(fXMin);  }

double BrentMinimizer1D::FValUpper() const
{   return (*fFunction)(fXMax);  }

bool BrentMinimizer1D::Minimize( int maxIter, double absTol , double relTol)
{
// Find minimum position iterating until convergence specified by the
// absolute and relative tolerance or the maximum number of iteration
// is reached.
// repet search (Bracketing + Brent) until max number of search is reached (default is 10)
// maxITer refers to the iterations inside the Brent algorithm

   if (!fFunction) {
       MATH_ERROR_MSG("BrentMinimizer1D::Minimize", "Function has not been set");
       return false;
   }

   if (fLogScan && fXMin <= 0) {
      MATH_ERROR_MSG("BrentMinimizer1D::Minimize", "xmin is < 0 and log scan is set - disable it");
      fLogScan = false;
   }

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
         MATH_ERROR_MSG("BrentMinimizer1D::Minimize", "Search didn't converge");
         fStatus = -2;
         return false;
      }
      double x = BrentMethods::MinimStep(fFunction, 0, xmin, xmax, 0, fNpx,fLogScan);
      x = BrentMethods::MinimBrent(fFunction, 0, xmin, xmax, x, 0,  ok, niter2, absTol, relTol, maxIter2 );
      fNIter += niter2;  // count the total number of iterations
      niter1++;
      fXMinimum = x;
   }

   fStatus = 0;
   return true;
}


const char * BrentMinimizer1D::Name() const
{   return "BrentMinimizer1D";  }

} // Namespace Math

} // Namespace ROOT
