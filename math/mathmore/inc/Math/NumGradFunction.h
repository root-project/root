// @(#)root/mathmore:$Id$
// Author: L. Moneta Wed Dec 20 14:36:31 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
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

// Header file for class NumGradFunction

#ifndef ROOT_Math_NumGradFunction
#define ROOT_Math_NumGradFunction


#ifndef ROOT_Math_IFunction
#include "Math/IFunction.h"
#endif

#ifndef ROOT_Math_WrappedFunction
#include "Math/WrappedFunction.h"
#endif

#ifndef ROOT_Math_Derivator
#include "Math/Derivator.h"
#endif

namespace ROOT { 

   namespace Math { 


/** 
   NumGradMultiFunction class to wrap a normal function in  a
   gradient function using numerical gradient calculation


   @ingroup MultiMin
*/ 
class MultiNumGradFunction : public IMultiGradFunction { 

public: 


   /** 
     Constructor from a IMultiGenFunction interface
   */ 
   MultiNumGradFunction (const IMultiGenFunction & f) : 
      fFunc(&f), 
      fDim(f.NDim() ),
      fOwner(false)
   {}

   /** 
     Constructor from a generic function (pointer or reference) and number of dimension
     implementiong operator () (double * x)
   */ 

   template<class FuncType> 
   MultiNumGradFunction (FuncType f, int n) : 
      fDim( n ), 
      fOwner(true)
   {
      // create a wrapped function
      fFunc = new ROOT::Math::WrappedMultiFunction<FuncType> (f, n);
   }

   /** 
      Destructor (no operations)
   */ 
   ~MultiNumGradFunction ()  {
      if (fOwner) delete fFunc; 
   }  


   // method inheritaed from IFunction interface 

   unsigned int NDim() const { return fDim; } 

   IMultiGenFunction * Clone() const { 
      if (!fOwner) 
         return new MultiNumGradFunction(*fFunc); 
      else { 
         // we need to copy the pointer to the wrapped function
         MultiNumGradFunction * f =  new MultiNumGradFunction(*(fFunc->Clone()) );
         f->fOwner = true; 
         return f;
      }
   }


private: 


   double DoEval(const double * x) const { 
      return (*fFunc)(x); 
   }

   // calculate derivative using mathcore derivator 
   double DoDerivative (const double * x, unsigned int icoord  ) const { 
      static double kEps = 1.E-6;
      static double kPrecision = 1.E-8; // sqrt(epsilon)
      double x0 = x[icoord];
      double step = std::max( kEps* std::abs(x0), 8.0*kPrecision*(std::abs(x0) + kPrecision) );
      return ROOT::Math::Derivator::Eval(*fFunc, x, icoord, step); 
   }  

   // adapat internal function type to IMultiGenFunction needed by derivative calculation
   const IMultiGenFunction * fFunc;  
   unsigned int fDim; 
   bool fOwner; 
}; 

   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_NumGradFunction */
