// @(#)root/mathcore:$Id$
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

#ifndef ROOT_Math_WrappedFunction
#define ROOT_Math_WrappedFunction

#ifndef ROOT_Math_IFunction
#include "IFunction.h"
#endif


namespace ROOT {
namespace Math {




struct NullTypeFunc1D {};

typedef double(*FreeFunctionPtr)(double);

typedef double(*FreeMultiFunctionPtr)(const double*);

/**
   Template class to wrap any C++ callable object which takes one argument
   i.e. implementing operator() (double x) in a One-dimensional function interface.
   It provides a ROOT::Math::IGenFunction-like signature

   Note: If you want to wrap just the reference (to avoid copying) you need to use
   Func& or const Func & as template parameter.  The former should be used when the
   operator() is not a const method of Func

   @ingroup  GenFunc

 */
template< typename Func =  FreeFunctionPtr   >
class WrappedFunction : public IGenFunction {


 public:

   /**
      construct from the pointer to the object and the member function
    */
   WrappedFunction( Func f ) :
      fFunc( f )
   { /* no op */ }

   // use default  copy contructor and assignment operator

   /// clone (required by the interface)
   WrappedFunction * Clone() const {
      return new WrappedFunction(fFunc);
   }

   //  virtual ~WrappedFunction() { /**/ }

private:

   virtual double DoEval (double x) const {
      return fFunc( x );
   }


   Func fFunc;


}; // WrappedFunction


/**
   Template class to wrap any member function of a class
   taking a double and returning a double in a 1D function interface
   For example, if you have a class like:
   struct X {
       double Eval(double x);
   };
   you can wrapped in the following way:
   WrappedMemFunction<X, double ( X::* ) (double) > f;


   @ingroup  GenFunc

 */

template<typename FuncObj, typename MemFuncPtr >
class WrappedMemFunction : public IGenFunction {


 public:

   /**
      construct from the pointer to the object and the member function
    */
   WrappedMemFunction( FuncObj & obj, MemFuncPtr memFn ) :
      fObj(&obj),
      fMemFunc( memFn )
   { /* no op */ }

   // use default  copy contructor and assignment operator

   /// clone (required by the interface)
   WrappedMemFunction * Clone() const {
      return new WrappedMemFunction(*fObj,fMemFunc);
   }


private:

   virtual double DoEval (double x) const {
      return ((*fObj).*fMemFunc)( x );
   }


   FuncObj * fObj;
   MemFuncPtr fMemFunc;


}; // WrappedMemFunction


/**
   Template class to wrap any C++ callable object
   implementing operator() (const double * x) in a multi-dimensional function interface.
   It provides a ROOT::Math::IGenMultiFunction-like signature

   Note: If you want to wrap just the reference (to avoid copying) you need to use
   Func& or const Func & as template parameter. The former should be used when the
   operator() is not a const method of Func

   @ingroup  GenFunc

 */
template< typename Func =  FreeMultiFunctionPtr   >
class WrappedMultiFunction : public IMultiGenFunction {


 public:

   /**
      construct from the pointer to the object and the member function
    */
   WrappedMultiFunction( Func f , unsigned int dim = 1) :
      fFunc( f ),
      fDim( dim)
   { /* no op */ }

   // use default  copy contructor and assignment operator

   /// clone (required by the interface)
   WrappedMultiFunction * Clone() const {
      return new WrappedMultiFunction(fFunc,fDim);
   }

   unsigned int NDim() const { return fDim; }

   //  virtual ~WrappedFunction() { /**/ }

private:

   virtual double DoEval (const double * x) const {
      return fFunc( x );
   }


   Func fFunc;
   unsigned int fDim;


}; // WrappedMultiFunction


template<typename FuncObj, typename MemFuncPtr >
class WrappedMemMultiFunction : public IMultiGenFunction {


 public:

   /**
      construct from the pointer to the object and the member function
    */
   WrappedMemMultiFunction( FuncObj & obj, MemFuncPtr memFn, unsigned int dim = 1 ) :
      fObj(&obj),
      fMemFunc( memFn ),
      fDim(dim)
   { /* no op */ }

   // use default  copy contructor and assignment operator

   /// clone (required by the interface)
   WrappedMemMultiFunction * Clone() const {
      return new WrappedMemMultiFunction(*fObj,fMemFunc,fDim);
   }


   unsigned int NDim() const { return fDim; }

private:

   virtual double DoEval (const double * x) const {
      return ((*fObj).*fMemFunc)( x );
   }


   FuncObj * fObj;
   MemFuncPtr fMemFunc;
   unsigned int fDim;


}; // WrappedMemMultiFunction


} // namespace Math
} // namespace ROOT



#endif // ROOT_Math_WrappedFunction
