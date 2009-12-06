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

#include "Math/IMinimizer1D.h"
#include "Math/IFunction.h"

#ifndef ROOT_Math_BrentMinimizer1D
#define ROOT_Math_BrentMinimizer1D

namespace ROOT { 
namespace Math { 

//___________________________________________________________________________________________
/**
   User class for performing function minimization 

   It will use the Brent Method for function minimization in a given interval. 
   This class is implemented from TF1::GetMinimum.

   To use the class, three steps have to be taken:
       1. Create the class.
       2. Set a function within an interval to look for the minimum.
       3. Call the Minimize function with the error parameters.

   If another minimization is to be performed, repeat the last two steps.

   @ingroup Min1D
  
 */

   class BrentMinimizer1D: IMinimizer1D {
      
   public: 

      /** Default Constructor. */
      BrentMinimizer1D(); 

      /** Default Destructor. */
      virtual ~BrentMinimizer1D();
      
   public: 
      
      /** Return current estimate of the position of the minimum. */
      virtual double XMinimum() const; 

      /** Return current lower bound of the minimization interval. */
      virtual double XLower() const; 

      /** Return current upper bound of the minimization interval. */
      virtual double XUpper() const; 

      /** Return function value at current estimate of the minimum. */
      virtual double FValMinimum() const; 

      /** Return function value at current lower bound of the minimization interval. */
      virtual double FValLower() const; 

      /** Return function value at current upper bound of the minimization interval. */
      virtual double FValUpper() const; 

      /** Find minimum position iterating until convergence specified by the absolute and relative tolerance or
          the maximum number of iteration is reached.

          \@param maxIter maximum number of iterations.
          \@param absTol desired absolute error in the minimum position.
          \@param absTol desired relative error in the minimum position.
      */
      virtual int Minimize( int maxIter, double absTol, double relTol); 

      /** Return number of iteration used to find minimum */
      virtual int Iterations() const;


      /** Return name of minimization algorithm ("BrentMinimizer1D") */
      virtual const char * Name() const;  

      /** Sets function to be minimized. 

          \@param f Function to be minimized.
          \@param xlow Lower bound of the search interval.
          \@param xup Upper bound of the search interval.
      */
      int SetFunction(const ROOT::Math::IGenFunction& f, double xlow, double xup);

   protected:
      const IGenFunction* fFunction; // Pointer to the function.
      double fXMin;                  // Lower bound of the search interval.
      double fXMax;                  // Upper bound of the search interval
      double fXMinimum;              // Position of the stimated minimum.
      int fNIter;                    // Number of iterations needed for the last stimation.

   };  // end class BrentMinimizer1D
   
} // end namespace Math
   
} // end namespace ROOT

#endif /* ROOT_Math_BrentMinimizer1D */
