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


#ifndef ROOT_Math_BrentMinimizer1D
#define ROOT_Math_BrentMinimizer1D

#ifndef ROOT_Math_IMinimizer1D
#include "Math/IMinimizer1D.h"
#endif

#ifndef ROOT_Math_IFunctionfwd
#include "Math/IFunctionfwd.h"
#endif


namespace ROOT { 
namespace Math { 

//___________________________________________________________________________________________
/**
   User class for performing function minimization 

   It will use the Brent Method for function minimization in a given interval.
   First, a grid search is used to bracket the minimum value
   with the a step size = (xmax-xmin)/npx. The step size
   can be controlled via the SetNpx() function. A default value of npx = 100 is used. 
   The default value con be changed using the static method SetDefaultNpx.
   If the function is unimodal or if its extrema are far apart, setting the fNpx to 
   a small value speeds the algorithm up many times.  
   Then, Brent's method is applied on the bracketed interval. 
   If the Brent method fails to converge the bracketing is repeted on the latest best estimate of the 
   interval. The procedure is repeted with a maximum value (default =10) which can be set for all
   BrentRootFinder classes with the method SetDefaultNSearch


 
   This class is implemented from TF1::GetMinimum.

   To use the class, three steps have to be taken:
       1. Create the class.
       2. Set a function within an interval to look for the minimum.
       3. Call the Minimize function with the error parameters.

   If another minimization is to be performed, repeat the last two steps.

   @ingroup Min1D
  
 */

   class BrentMinimizer1D: ROOT::Math::IMinimizer1D {
      
   public: 

      /** Default Constructor. */
      BrentMinimizer1D(); 

      /** Default Destructor. */
      virtual ~BrentMinimizer1D() {}
      
   public: 
      
      /** Return current estimate of the position of the minimum. */
      virtual double XMinimum() const {   return fXMinimum;  }

      /** Return current lower bound of the minimization interval. */
      virtual double XLower() const {   return fXMin;  }

      /** Return current upper bound of the minimization interval. */
      virtual double XUpper() const {   return fXMax;  }

      /** Return function value at current estimate of the minimum. */
      virtual double FValMinimum() const; 

      /** Return function value at current lower bound of the minimization interval. */
      virtual double FValLower() const; 

      /** Return function value at current upper bound of the minimization interval. */
      virtual double FValUpper() const; 

      /** Find minimum position iterating until convergence specified by the absolute and relative tolerance or
          the maximum number of iteration is reached.
          Return true if iterations converged successfully
          \@param maxIter maximum number of iterations.
          \@param absTol desired absolute error in the minimum position (default 1.E-8)
          \@param absTol desired relative error in the minimum position (default = 1.E-10)
      */
      virtual bool Minimize( int maxIter, double absTol = 1.E-8, double relTol = 1.E-10); 

      /** Return number of iteration used to find minimum */
      virtual int Iterations() const { return fNIter; }

      /** Return name of minimization algorithm ("BrentMinimizer1D") */
      virtual const char * Name() const;  

      /** Sets function to be minimized. 

          \@param f Function to be minimized.
          \@param xlow Lower bound of the search interval.
          \@param xup Upper bound of the search interval.
      */
      void SetFunction(const ROOT::Math::IGenFunction& f, double xlow, double xup);

      /** Set the number of point used to bracket root using a grid */
      void SetNpx(int npx) { fNpx = npx; }

      /** 
          Set a log grid scan (default is equidistant bins) 
          will work only if xlow > 0
      */ 
      void SetLogScan(bool on) { fLogScan = on; }
          

      /** Returns status of last estimate. If = 0 is OK */
      int Status() const { return fStatus; }

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
      int fNpx;                      // Number of points to bracket minimum with grid (def is 100)
      int fStatus;                   // Status of code of the last estimate
      double fXMin;                  // Lower bound of the search interval.
      double fXMax;                  // Upper bound of the search interval
      double fXMinimum;              // Position of the stimated minimum.
 
   };  // end class BrentMinimizer1D
   
} // end namespace Math
   
} // end namespace ROOT

#endif /* ROOT_Math_BrentMinimizer1D */
