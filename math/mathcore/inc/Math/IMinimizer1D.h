// @(#)root/mathcore:$Id$
// Author: David Gonzalez Maline 2/2008
 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2006  CERN                                           *
  * All rights reserved.                                               *
  *                                                                    *
  * For the licensing terms see $ROOTSYS/LICENSE.                      *
  * For the list of contributors see $ROOTSYS/README/CREDITS.          *
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
