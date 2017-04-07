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
#ifndef ROOT_Math_Derivator
#define ROOT_Math_Derivator


#include "Math/IFunctionfwd.h"

#include "Math/IParamFunctionfwd.h"


namespace ROOT {
namespace Math {



class GSLDerivator;

//_______________________________________________________________________
/**
    Class for computing numerical derivative of a function.
    Presently this class is implemented only using the numerical derivatives
    algorithms provided by GSL
    using the implementation class ROOT::Math::GSLDerivator

    This class does not support copying

    @ingroup Deriv
*/

class Derivator {

public:

   /**
      signature for function pointers used by GSL
   */
   typedef double ( * GSLFuncPointer ) ( double, void * );

   /**
      Empty Construct for a Derivator class
      Need to set the function afterwards with Derivator::SetFunction
   */
   Derivator();
   /**
      Construct using a ROOT::Math::IGenFunction interface
    */
   explicit Derivator(const IGenFunction &f);
   /**
      Construct using a GSL function pointer type
       @param f :  free function pointer of the GSL required type
       @param p :  pointer to the object carrying the function state
                    (for example the function object itself)
    */
   explicit Derivator(const GSLFuncPointer &f, void * p = 0);

   /// destructor
   virtual ~Derivator();

   // disable copying
private:

   Derivator(const Derivator &);
   Derivator & operator = (const Derivator &);

public:


#ifdef LATER
   /**
       Template methods for generic functions
       Set the function f for evaluating the derivative.
       The function type must implement the assigment operator,
       <em>  double  operator() (  double  x ) </em>
   */
   template <class UserFunc>
   inline void SetFunction(const UserFunc &f) {
      const void * p = &f;
      SetFunction(  &GSLFunctionAdapter<UserFunc>::F, const_cast<void *>(p) );
   }
#endif

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
   void SetFunction( const GSLFuncPointer &f, void * p = 0);



   /**
       Computes the numerical derivative of a function f at a point x.
       It uses Derivator::EvalCentral to compute the derivative using an
       adaptive central difference algorithm with a step size h
   */

   double Eval(double x, double h = 1E-8) const;



   /**
       Computes the numerical derivative at a point x using an adaptive central
       difference algorithm with a step size h.
   */
   double EvalCentral( double x, double h = 1E-8) const;

   /**
       Computes the numerical derivative at a point x using an adaptive forward
       difference algorithm with a step size h.
       The function is evaluated only at points greater than x and at x itself.
   */
   double EvalForward( double x, double h = 1E-8) const;

   /**
       Computes the numerical derivative at a point x using an adaptive backward
       difference algorithm with a step size h.
       The function is evaluated only at points less than x and at x itself.
   */
   double EvalBackward( double x, double h = 1E-8) const;

   /** @name --- Static methods ---
       This methods don't require to use a Derivator object, and are designed to be used in
       fast calculation. Error and status code cannot be retrieved in this case
    */

   /**
       Computes the numerical derivative of a function f at a point x.
       It uses Derivator::EvalCentral to compute the derivative using an
       adaptive central difference algorithm with a step size h
   */
   static double Eval(const IGenFunction & f, double x, double h = 1E-8);

   /**
       Computes the numerical derivative of a function f at a point x using an adaptive central
       difference algorithm with a step size h
   */
   static double EvalCentral(const IGenFunction & f, double x, double h = 1E-8);


   /**
       Computes the numerical derivative of a function f at a point x using an adaptive forward
       difference algorithm with a step size h.
       The function is evaluated only at points greater than x and at x itself
   */
   static double EvalForward(const IGenFunction & f, double x, double h = 1E-8);

   /**
       Computes the numerical derivative of a function f at a point x using an adaptive backward
       difference algorithm with a step size h.
       The function is evaluated only at points less than x and at x itself
   */
   static double EvalBackward(const IGenFunction & f, double x, double h = 1E-8);

   // Derivatives for multi-dimension functions
   /**
      Evaluate the partial derivative of a multi-dim function
      with respect coordinate x_icoord at the point x[]
    */
   static double Eval(const IMultiGenFunction & f, const double * x, unsigned int icoord = 0, double h = 1E-8);

   /**
      Evaluate the derivative with respect a parameter for one-dim parameteric function
      at the point ( x,p[]) with respect the parameter p_ipar
    */
   static double Eval(IParamFunction & f, double x, const double * p, unsigned int ipar = 0, double h = 1E-8);

   /**
      Evaluate the derivative with respect a parameter for a multi-dim parameteric function
      at the point ( x[],p[]) with respect the parameter p_ipar
    */
   static double Eval(IParamMultiFunction & f, const double * x, const double * p, unsigned int ipar = 0, double h = 1E-8);


   /**
      return the error status of the last derivative calculation
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


   mutable GSLDerivator * fDerivator;

};




} // namespace Math
} // namespace ROOT


#endif /* ROOT_Math_Derivator */
