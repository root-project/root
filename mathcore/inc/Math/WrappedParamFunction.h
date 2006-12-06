// @(#)root/mathcore:$Name:  $:$Id: inc/Math/WrappedParamFunction.h,v 1.0 2006/01/01 12:00:00 moneta Exp $
// Author: L. Moneta Thu Nov 23 10:38:32 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class WrappedParamFunction

#ifndef ROOT_Math_WrappedParamFunction
#define ROOT_Math_WrappedParamFunction

#ifndef ROOT_Math_IParamFunction
#include "Math/IParamFunction.h"
#endif

#include <iostream>

namespace ROOT { 

   namespace Math { 


typedef double( * FreeParamMultiFunctionPtr ) (const double *, const double * ); 

/** 
   WrappedParamFunction class to wrap any multi-dimensional parameteric function 
   in an interface-like IParamFunciton
*/ 
template< typename FuncPtr =  FreeParamMultiFunctionPtr   >
class WrappedParamFunction : public IParamMultiFunction {

public: 

   /** 
      Constructor a wrapped function from a pointer to a callable object and an iterator specifying begin and end 
      of parameters
   */ 
   template<class Iterator> 
   WrappedParamFunction (const FuncPtr & func, unsigned int dim, Iterator begin, Iterator end) : 
      fFunc(func),
      fDim(dim),
      fParams(std::vector<double>(begin,end) )
   {}

   /** 
      Constructor a wrapped function from a non - const pointer to a callable object and an iterator specifying begin and end of parameters. This constructor is needed in the case FuncPtr is a std::auto_ptr which has a copy ctor taking non const objects
   */ 
   template<class Iterator> 
   WrappedParamFunction (FuncPtr & func, unsigned int dim, Iterator begin, Iterator end) : 
      fFunc(func),
      fDim(dim),
      fParams(std::vector<double>(begin,end) )
   {}

   /// clone the function
   IMultiGenFunction * Clone() const { 
      return new WrappedParamFunction(fFunc, fDim, fParams.begin(), fParams.end()); 
   }

   const double * Parameters() const { 
      return  &(fParams.front()); 
   }

   void SetParameters(const double * p)  { 
      std::copy(p, p+NPar(), fParams.begin() );
   }

   unsigned int NPar() const { return fParams.size(); }

   unsigned int NDim() const { return fDim; }

   // re-implement this since is more efficient
   double operator() (const double * x, const double * p) { 
      return (*fFunc)( x, p );
   }

private: 
   
   /// evaluate the function
   double DoEval(const double * x) const { 
//      std::cout << x << "  " << *x << "   " << fParams.size() << "  " << &fParams[0] << "  " << fParams[0] << std::endl; 
      return (*fFunc)( x, &(fParams.front()) );
   }


   mutable FuncPtr fFunc; 
   unsigned int fDim; 
   std::vector<double> fParams; 
      


}; 

   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_WrappedParamFunction */
