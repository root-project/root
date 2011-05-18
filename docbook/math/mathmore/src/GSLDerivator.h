// @(#)root/mathmore:$Id$
// Authors: L. Moneta, A. Zsenei   08/2005 

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2004 ROOT Foundation,  CERN/PH-SFT                   *
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

// Header file for class Derivator
// 
// class for calculating Derivative of functions
// 
// Created by: moneta  at Sat Nov 13 14:46:00 2004
// 
// Last update: Sat Nov 13 14:46:00 2004
// 
#ifndef ROOT_Math_GSLDerivator
#define ROOT_Math_GSLDerivator

/** 
@defgroup Deriv Numerical Differentiation
*/
 
#include "Math/GSLFunctionAdapter.h"
#include "GSLFunctionWrapper.h"


#include "Math/IFunctionfwd.h"
#include "Math/IFunction.h"

namespace ROOT {
namespace Math {


class GSLFunctionWrapper; 


/** 
    Class for computing numerical derivative of a function based on the GSL numerical algorithm 
    This class is implemented using the numerical derivatives algorithms provided by GSL
    (see <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Numerical-Differentiation.html">GSL Online Manual</A> ).
    
    @ingroup Deriv
*/

class GSLDerivator {

public:
   /**
      Default Constructor of  a GSLDerivator class based on GSL numerical differentiation algorithms
   */
   GSLDerivator() : fStatus(0), fResult(0), fError(0)   {}

   /// destructor (no operations)
   virtual ~GSLDerivator() {}

//    // disable copying
// private: 

//    GSLDerivator(const GSLDerivator &);
//    GSLDerivator & operator = (const GSLDerivator &); 

// public: 



   /**
      Set the function for calculating the derivatives. 
      The function must implement the ROOT::Math::IGenFunction signature
    */
   void SetFunction(const IGenFunction &f);

   /** 
       Set the function f for evaluating the derivative using a GSL function pointer type 
       @param f :  free function pointer of the GSL required type
       @param p :  pointer to the object carrying the function state 
                    (for example the function object itself)
   */
   void SetFunction( GSLFuncPointer f, void * p = 0);
	
   /** 
       Computes the numerical derivative at a point x using an adaptive central 
       difference algorithm with a step size h. 
   */
   double EvalCentral( double x, double h); 

   /** 
       Computes the numerical derivative at a point x using an adaptive forward 
       difference algorithm with a step size h.
       The function is evaluated only at points greater than x and at x itself.
   */
   double EvalForward( double x, double h); 

   /** 
       Computes the numerical derivative at a point x using an adaptive backward 
       difference algorithm with a step size h.
       The function is evaluated only at points less than x and at x itself.
   */
   double EvalBackward( double x, double h); 

   /** @name --- Static methods --- **/

   /** 
       Computes the numerical derivative of a function f at a point x using an adaptive central 
       difference algorithm with a step size h
   */
   static double EvalCentral(const IGenFunction & f, double x, double h); 


   /** 
       Computes the numerical derivative of a function f at a point x using an adaptive forward 
       difference algorithm with a step size h.
       The function is evaluated only at points greater than x and at x itself
   */    
   static double EvalForward(const IGenFunction & f, double x, double h);

   /** 
       Computes the numerical derivative of a function f at a point x using an adaptive backward 
       difference algorithm with a step size h.
       The function is evaluated only at points less than x and at x itself
   */
    
   static double EvalBackward(const IGenFunction & f, double x, double h); 

    

   /**
      return the error status of the last integral calculation
   */     
   int Status() const; 

   /**
      return  the result of the last derivative calculation
   */
   double Result() const; 

   /**
      return the estimate of the absolute error of the last derivative calculation
   */
   double Error() const; 


private: 

   int fStatus;
   double fResult; 
   double fError; 

   GSLFunctionWrapper fFunction;  

}; 




} // namespace Math
} // namespace ROOT


#endif /* ROOT_Math_GSLDerivator */
