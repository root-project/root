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

#include "Math/IFunctionfwd.h"

#include "Math/IRootFinderMethod.h"

namespace ROOT {
namespace Math {

//___________________________________________________________________________________________
/**
   Class for finding the root of a one dimensional function using the Brent algorithm.
   It will use the Brent Method for finding function roots in a given interval.
   First, a grid search is used to bracket the root value
   with the a step size = (xmax-xmin)/npx. The step size
   can be controlled via the SetNpx() function. A default value of npx = 100 is used.
   The default value con be changed using the static method SetDefaultNpx.
   If the function is unimodal or if its extrema are far apart, setting the fNpx to
   a small value speeds the algorithm up many times.
   Then, Brent's method is applied on the bracketed interval.
   It will use the Brent Method for finding function roots in a given interval.
   If the Brent method fails to converge the bracketing is repeted on the latest best estimate of the
   interval. The procedure is repeted with a maximum value (default =10) which can be set for all
   BrentRootFinder classes with the method SetDefaultNSearch

   This class is implemented from TF1::GetX() method.

   @ingroup RootFinders

 */

   class BrentRootFinder: public IRootFinderMethod {
   public:


      /** Default Constructor. */
      BrentRootFinder();


      /** Default Destructor. */
      virtual ~BrentRootFinder() {}


      /** Set function to solve and the interval in where to look for the root.

          \@param f Function to be minimized.
          \@param xlow Lower bound of the search interval.
          \@param xup Upper bound of the search interval.
      */
      using IRootFinderMethod::SetFunction;
      bool SetFunction(const ROOT::Math::IGenFunction& f, double xlow, double xup);


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
      bool Solve(int maxIter = 100, double absTol = 1E-8, double relTol = 1E-10);

      /** Set the number of point used to bracket root using a grid */
      void SetNpx(int npx) { fNpx = npx; }

      /**
          Set a log grid scan (default is equidistant bins)
          will work only if xlow > 0
      */
      void SetLogScan(bool on) { fLogScan = on; }

      /** Returns root value. Need to call first Solve(). */
      double Root() const { return fRoot; }

      /** Returns status of last estimate. If = 0 is OK */
      int Status() const { return fStatus; }

      /** Return number of iteration used to find minimum */
      int Iterations() const { return fNIter; }

      /** Return name of root finder algorithm ("BrentRootFinder"). */
      const char* Name() const;

      // static function used to modify the default parameters

      /** set number of default Npx used at construction time (when SetNpx is not called)
          Default value is 100
       */
      static void SetDefaultNpx(int npx);

      /** set number of  times the bracketing search in combination with is done to find a good interval
          Default value is 10
       */
      static void SetDefaultNSearch(int n);


   private:

      const IGenFunction* fFunction; // Pointer to the function.
      bool fLogScan;                 // flag to control usage of a log scan
      int fNIter;                    // Number of iterations needed for the last estimation.
      int fNpx;                      // Number of points to bracket root with initial grid (def is 100)
      int fStatus;                   // Status of code of the last estimate
      double fXMin;                  // Lower bound of the search interval.
      double fXMax;                  // Upper bound of the search interval
      double fRoot;                  // Current stimation of the function root.
   };

} // namespace Math
} // namespace ROOT

#endif /* ROOT_Math_BrentRootFinder */
