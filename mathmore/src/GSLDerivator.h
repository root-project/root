// @(#)root/mathmore:$Name:  $:$Id: GSLDerivator.h,v 1.2 2005/09/18 20:41:25 brun Exp $
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

#include "Math/IFunctionfwd.h"
#include "Math/IFunction.h"

namespace ROOT {
namespace Math {


   class GSLFunctionWrapper; 


  /** 
      Class for computing numerical derivative of a function. 
      This class is implemented using the numerical derivatives algorithms provided by GSL
      (see <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_27.html#SEC400">GSL Online Manual</A> ).

      This class does not support copying
      @ingroup Deriv
  */

   class GSLDerivator {

   public:
     /**
	Construct a GSLDerivator class based on GSL numerical differentiation algorithms
      */
 
     GSLDerivator(const IGenFunction &f);
     GSLDerivator(const GSLFuncPointer &f);
     virtual ~GSLDerivator(); 

     // disable copying
   private: 

     GSLDerivator(const GSLDerivator &);
     GSLDerivator & operator = (const GSLDerivator &); 

   public: 


    // template methods for generic functors
     /** 
	 Set the function f for evaluating the derivative.
	 The function type must implement the assigment operator, <em>  double  operator() (  double  x ) </em>
     
     template <class UserFunc>
     inline void SetFunction(const UserFunc &f) { 
       const void * p = &f; 
       FillGSLFunction(  &GSLFunctionAdapter<UserFunc>::F, const_cast<void *>(p) ); 
    }
     */
     inline void SetFunction(const IGenFunction &f) { 
       //const void * p = &f; 
       //FillGSLFunction(  &GSLFunctionAdapter<IGenFunction>::F, const_cast<void *>(p) ); 
       FillGSLFunction(f);

     }


    /**
       signature for function pointers used by GSL
    */
     //typedef double ( * GSLFuncPointer ) ( double, void * ); 


     /** 
	 Set the function f for evaluating the derivative (use of a free function pointer)
     */
     inline void SetFunction( const GSLFuncPointer &f) { 
	 FillGSLFunction( f, 0); 
       }
	

     /** 
	 Computes the numerical derivative of a function f at a point x. 
	 It uses GSLDerivator::EvalCentral to compute the derivative using an 
	 adaptive central difference algorithm with a step size h
     */

    inline double Eval(const IGenFunction & f, double x, double h ) { 
      return EvalCentral( f, x, h); 
    }
     
   


     /** 
	 Computes the numerical derivative of a function f at a point x using an adaptive central 
	 difference algorithm with a step size h
     
    template <class UserFunc>
    inline double EvalCentral(const UserFunc & f, double x, double h) { 
      SetFunction(f); 
      return EvalCentral(x,h); 
    }
     */
    inline double EvalCentral(const IGenFunction & f, double x, double h) { 
      SetFunction(f); 
      return EvalCentral(x,h); 
    }



     /** 
	 Computes the numerical derivative of a function f at a point x using an adaptive forward 
	 difference algorithm with a step size h.
	 The function is evaluated only at points greater than x and at x itself
     */
    
    inline double EvalForward(const IGenFunction & f, double x, double h) { 
      SetFunction(f); 
      return EvalForward(x,h); 
    }

     /** 
	 Computes the numerical derivative of a function f at a point x using an adaptive backward 
	 difference algorithm with a step size h.
	 The function is evaluated only at points less than x and at x itself
     */
    
    inline double EvalBackward(const IGenFunction & f, double x, double h) { 
      SetFunction(f); 
      return EvalBackward(x,h); 
    }

    
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

protected: 

    // internal method to create GSL function adapter
    void FillGSLFunction( GSLFuncPointer fp, void *); 
    void FillGSLFunction(const IGenFunction & f); 


private: 

    int fStatus;
    double fResult; 
    double fError; 

    GSLFunctionWrapper * fFunction;  

}; 




} // namespace Math
} // namespace ROOT


#endif /* ROOT_Math_GSLDerivator */
