// @(#)root/mathmore:$Name:  $:$Id: IGenFunction.h,v 1.1 2005/09/18 17:33:47 brun Exp $
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

// Header file for class IGenFunction
// 
// Generic interface for one-dimensional functions
//
// Created by: Lorenzo Moneta  at Wed Nov 10 16:15:35 2004
// 
// Last update: Wed Nov 10 16:15:35 2004
// 
#ifndef ROOT_Math_IGenFunction
#define ROOT_Math_IGenFunction

/** 
@defgroup CppFunctions Function Classes and Interfaces
*/

namespace ROOT {
namespace Math {

  /** 
      Interface for generic 1 Dimensional Functions.
      @ingroup  CppFunctions
  */

  class IGenFunction {

  public: 

    virtual ~IGenFunction() {}

    /** 
	Evaluate the function at a point x. 
	It is a pure virtual method and must be implemented by sub-classes
    */
    virtual double operator() (double x) = 0;
    //virtual double operator() (double x) {return 2.0;}


    /** 
	Evaluate the function derivative at a point x.
	Functions which do not implement it, can use the dummy implementation of this class
	which returns zero. 
	The method IgenFunction::ProvidesGradient() is used to query this information
    */

    virtual double Gradient(double /* x */ ) { return 0; }
  
    /** 
	Optimized method to evaluate at the same time the function value and derivative at a point x.
	Often both value and derivatives are needed and it is often more efficient to compute them at the same time.
	Derived class should implement this method if performances play an important role and if it is faster to 
	evaluate value and derivative at the same time

    */

    virtual void Fdf(double x, double & f, double & df) { 
      f = operator()(x); 
      df = Gradient(x); 
    }

    // should return a reference ? The returned function has no sense if this disappears
    //virtual GenFunction * gradientFunc();


    /** 
	Clone a function. 
	Pure virtual method needed to perform deep copy of the derived classes. 
    */
    virtual IGenFunction * Clone() const = 0;

    /**
       Return \a true if the calculation of derivatives is implemented
     */
    // overwrite by derived classes
    virtual bool ProvidesGradient() const { return false; } 


  }; 
  

} // namespace Math
} // namespace ROOT

#endif /* ROOT_Math_IGenFunction */
