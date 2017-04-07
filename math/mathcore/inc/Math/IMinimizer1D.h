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

// Header file for class Minimizer1D
//
// Created by: Maline  at Fri Feb  1 11:30:26 2008
//

#ifndef ROOT_Math_IMinimizer1D
#define ROOT_Math_IMinimizer1D

/**

   @defgroup Min1D One-dimensional Minimization
   Classes for one-dimensional minimization
   @ingroup NumAlgo
 */

namespace ROOT {
namespace Math {

//___________________________________________________________________________________________
/**
   Interface class for numerical methods for one-dimensional minimization

   @ingroup Min1D

 */

   class IMinimizer1D {

   public:

      IMinimizer1D() {}
      virtual ~IMinimizer1D() {}

   public:

      /**
       * Return current estimate of the position of the minimum
       */
      virtual double XMinimum() const = 0;

      /**
       * Return current lower bound of the minimization interval
       */
      virtual double XLower() const = 0;

      /**
       * Return current upper bound of the minimization interval
       */
      virtual double XUpper() const = 0;

      /**
       * Return function value at current estimate of the minimum
       */
      virtual double FValMinimum() const = 0;

      /**
       * Return function value at current lower bound of the minimization interval
       */
      virtual double FValLower() const = 0;

      /**
       * Return function value at current upper bound of the minimization interval
       */
      virtual double FValUpper() const = 0;

      /**
       * Find minimum position iterating until convergence specified by the absolute and relative tolerance or
       * the maximum number of iteration is reached
       * Return true if iterations converged successfully
       * \@param maxIter maximum number of iteration
       * \@param absTol desired absolute error in the minimum position
       * \@param absTol desired relative error in the minimum position
       */
      virtual bool Minimize( int maxIter, double absTol, double relTol) = 0;

      /**
       * Return number of iteration used to find minimum
       */
      virtual int Iterations() const = 0;

      /**
       * Return name of minimization algorithm
       */
      virtual const char * Name() const = 0;

      /** Returns the status of the previous estimate */
      virtual int Status() const = 0;


   };  // end class IMinimizer1D

} // end namespace Math

} // end namespace ROOT

#endif /* ROOT_Math_IMinimizer1D */
