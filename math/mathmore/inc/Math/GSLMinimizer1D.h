// @(#)root/mathmore:$Id$
// Author: L. Moneta, A. Zsenei   08/2005
 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2004 moneta,  CERN/PH-SFT                            *
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

// Header file for class GSLMinimizer1D
// 
// Created by: moneta  at Wed Dec  1 15:04:51 2004
// 
// Last update: Wed Dec  1 15:04:51 2004
// 

#ifndef ROOT_Math_GSLMinimizer1D
#define ROOT_Math_GSLMinimizer1D

#include "Math/IMinimizer1D.h"
#include "Math/GSLFunctionAdapter.h"


namespace ROOT { 
namespace Math { 

   namespace Minim1D {
      
      /** 
          Enumeration with One Dimensional Minimizer Algorithms. 
          The algorithms are implemented using GSL, see the 
          <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_33.html#SEC447">GSL manual</A>.
          
          The algorithms available are: 
          <ul>
          <li><em>Golden Section Algorithm</em>, simplest method of bracketing the minimum of a function 
          <li><em>Brent Algorithm</em>, which combines a parabolic interpolation with the golden section algorithm
          </ul>
          @ingroup Min1D
      */
      
      enum Type {kGOLDENSECTION, 
                 kBRENT
      };
   }
   
   class GSL1DMinimizerWrapper; 
   class GSLFunctionWrapper;

//______________________________________________________________________________________
/** 

Minimizer for arbitrary one dimensional functions. 

Implemented using GSL, for detailed description see: 
<A HREF="http://www.gnu.org/software/gsl/manual/html_node/One-dimensional-Minimization.html">GSL online doc</A>

The algorithms uspported are only bracketing algorithm which do not use derivatives information. 
The algorithms which can be choosen at construction time are  GOLDENSECTION, whic is the simplest method 
but the slowest and BRENT (the default one) which combines the golden section with a parabolic interpolation. 


This class does not support copying
@ingroup Min1D
*/

   class GSLMinimizer1D: public IMinimizer1D {

   public: 

      /**
         Construct the minimizer passing the minimizer type using the Minim1D::Algorithm enumeration
      */
      
      explicit GSLMinimizer1D(Minim1D::Type type=Minim1D::kBRENT);
 
      /**
         Destructor: free allocated resources
      */
      virtual ~GSLMinimizer1D(); 

   private:
      // usually copying is non trivial, so we make this unaccessible
      GSLMinimizer1D(const GSLMinimizer1D &); 
      GSLMinimizer1D & operator = (const GSLMinimizer1D &); 
    
   public: 
      
     
      /** 
          Set, or reset, minimizer to use the function f and the initial search interval [xlow, xup], with a guess for the location of the minimum xmin.
          The condition : \f$ f(xlow) > f(xmin) < f(xup)\f$  must be satisfied
      */
      template <class UserFunc> 
      void SetFunction( const UserFunc & f, double xmin, double xlow, double xup) { 
         const void * p = &f; 
         SetFunction(  &GSLFunctionAdapter<UserFunc>::F, const_cast<void *>(p), xmin, xlow, xup ); 
      }
    
      /** 
          Set, or reset, minimizer to use the function f and the initial search interval [xlow, xup], with a guess for the location of the minimum xmin.
          The condition : \f$ f(xlow) > f(xmin) < f(xup) \f$ must be satisfied
        
          Method specialized on the GSL function type 
      */
      void SetFunction( GSLFuncPointer  f, void * params, double xmin, double xlow, double xup); 
    
      /** 
          Perform a minimizer iteration and  
          if an unexepcted problem occurr then an error code will be returned
      */
      int Iterate(); 


      /** 
          Return current estimate of the position of the minimum
      */
      double XMinimum() const; 

      /**
         Return current lower bound of the minimization interval
      */
      double XLower() const; 
    
      /**
         Return current upper bound of the minimization interval
      */
      double XUpper() const; 

      /** 
          Return function value at current estimate of the minimum
      */
      double FValMinimum() const; 

      /**
         Return function value at current lower bound of the minimization interval
      */
      double FValLower() const; 
    
      /**
         Return function value at current upper bound of the minimization interval
      */
      double FValUpper() const; 
        
    
      /**
         Find minimum position iterating until convergence specified by the absolute and relative tolerance or 
         the maximum number of iteration is reached 
         Return true is result is successfull
         \@param maxIter maximum number of iteration
         \@param absTol desired absolute error in the minimum position
         \@param absTol desired relative error in the minimum position
      */
      bool Minimize( int maxIter, double absTol, double relTol); 


      /**
         Return number of iteration used to find minimum
      */
      int Iterations() const {
         return fIter; 
      }

      /**
         Return status of last minimization
       */
      int Status() const { return fStatus; }

      /**
         Return name of minimization algorithm
      */
      const char * Name() const;  

      /**
         Test convergence of the interval. 
         The test returns success if 
         \f[
         |x_{min}-x_{truemin}| < epsAbs + epsRel *x_{truemin}
         \f]
      */
      static int TestInterval( double xlow, double xup, double epsAbs, double epsRel); 


   protected: 


   private: 

      double fXmin; 
      double fXlow;
      double fXup; 
      double fMin; 
      double fLow;
      double fUp; 
      int fIter; 
      int fStatus;    // status of last minimization (==0 ok =1 failed)
      bool fIsSet; 


      GSL1DMinimizerWrapper * fMinimizer; 
      GSLFunctionWrapper * fFunction;  

   }; 

} // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_GSLMinimizer1D */
