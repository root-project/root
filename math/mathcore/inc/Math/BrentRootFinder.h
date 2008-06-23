// @(#)root/mathcore:$Id$
// Authors: David Gonzalez Maline    01/2008 

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006 , LCG ROOT MathLib Team                         *
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

// Header for the RootFinder
// 
// Created by: David Gonzalez Maline  : Wed Jan 21 2008
// 

#ifndef ROOT_Math_BrentRootFinder
#define ROOT_Math_BrentRootFinder

#include <Math/IFunction.h>
#include <Math/IRootFinderMethod.h>

namespace ROOT {
namespace Math {

//___________________________________________________________________________________________
/**
   User class for finding function roots.

   It will use the Brent Method for finding function roots in a given interval. 
   This class is implemented from TF1::GetX() method.

   @ingroup RootFinders
  
 */

   class BrentRootFinder: public IRootFinderMethod {
   public:

      /** Default Destructor. */
      virtual ~BrentRootFinder();

      /** Default Constructor. */
      BrentRootFinder();
      
      /** Set function to solve and the interval in where to look for the root. 

          \@param f Function to be minimized.
          \@param xlow Lower bound of the search interval.
          \@param xup Upper bound of the search interval.
      */
      using IRootFinderMethod::SetFunction;
      int SetFunction(const ROOT::Math::IGenFunction& f, double xlow, double xup);
      

      /** Returns the X value corresponding to the function value fy for (xmin<x<xmax).
          Method:
          First, the grid search is used to bracket the maximum
          with the step size = (xmax-xmin)/fNpx. This way, the step size
          can be controlled via the SetNpx() function. If the function is
          unimodal or if its extrema are far apart, setting the fNpx to
          a small value speeds the algorithm up many times.
          Then, Brent's method is applied on the bracketed interval.

          \@param maxIter maximum number of iterations.
          \@param absTol desired absolute error in the minimum position.
          \@param absTol desired relative error in the minimum position.
      */
      int Solve(int maxIter = 100, double absTol = 1E-3, double relTol = 1E-6);

      /** Returns root value. Need to call first Solve(). */
      double Root() const;
      
      /** Return name of root finder algorithm ("BrentRootFinder"). */
      const char* Name() const;
      
   protected:
      const IGenFunction* fFunction; // Pointer to the function.
      double fXMin;                  // Lower bound of the search interval.
      double fXMax;                  // Upper bound of the search interval
      double fRoot;                  // Current stimation of the function root.
   };
   
} // namespace Math
} // namespace ROOT

#endif /* ROOT_Math_BrentRootFinder */
