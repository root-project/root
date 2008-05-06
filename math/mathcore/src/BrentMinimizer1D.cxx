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

#ifndef ROOT_Math_Error
#include "Math/Error.h"
#endif

namespace ROOT {
namespace Math {

BrentMinimizer1D::BrentMinimizer1D(): IMinimizer1D() 
{
// Default Constructor.

   fFunction = 0;
   fXMin = 0;
   fXMax = 0;
}
 
BrentMinimizer1D::~BrentMinimizer1D() {}

int BrentMinimizer1D::SetFunction(const ROOT::Math::IGenFunction& f, double xlow, double xup)
{
// Sets function to be minimized. 

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

double BrentMinimizer1D::XMinimum() const
{   return fXMinimum;  }

double BrentMinimizer1D::XLower() const
{   return fXMin;  }

double BrentMinimizer1D::XUpper() const
{   return fXMax;  }
 
double BrentMinimizer1D::FValMinimum() const
{   return (*fFunction)(fXMinimum); }

double BrentMinimizer1D::FValLower() const
{   return (*fFunction)(fXMin);  }

double BrentMinimizer1D::FValUpper() const
{   return (*fFunction)(fXMax);  }

int BrentMinimizer1D::Minimize( int maxIter, double , double )
{
// Find minimum position iterating until convergence specified by the
// absolute and relative tolerance or the maximum number of iteration
// is reached.

   int niter=0;
   double x;
   double xmin = fXMin;
   double xmax = fXMax;

   //ROOT::Math::WrappedFunction<const TF1&> wf1(*this);
   x = MinimStep(fFunction, 0, xmin, xmax, 0);
   bool ok = true;
   x = MinimBrent(fFunction, 0, xmin, xmax, x, 0, ok);
   while (!ok){
      if (niter>maxIter){
         MATH_ERROR_MSG("BrentMinimizer1D::Minimize", "Search didn't converge");
         return -1;
      }
      x=MinimStep(fFunction, 0, xmin, xmax, 0);
      x = MinimBrent(fFunction, 0, xmin, xmax, x, 0, ok);
      niter++;
   }

   fNIter = niter;
   fXMinimum = x;

   return 1;
} 

int BrentMinimizer1D::Iterations() const
{   return fNIter;  }


const char * BrentMinimizer1D::Name() const
{   return "BrentMinimizer1D";  }

} // Namespace Math

} // Namespace ROOT
