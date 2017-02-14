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

// Header file for class MultiNumGradFunction

#ifndef ROOT_Math_MultiNumGradFunction
#define ROOT_Math_MultiNumGradFunction


#include "Math/IFunction.h"

#include "Math/WrappedFunction.h"


namespace ROOT {

   namespace Math {


/**
   MultiNumGradFunction class to wrap a normal function in  a
   gradient function using numerical gradient calculation
   provided by the class Derivator (based on GSL numerical derivation)


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
      fNCalls(0),
      fOwner(false)
   {}

   /**
     Constructor from a generic function (pointer or reference) and number of dimension
     implementiong operator () (double * x)
   */

   template<class FuncType>
   MultiNumGradFunction (FuncType f, int n) :
      fDim( n ),
      fNCalls(0),
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

   unsigned int NCalls() const { return fNCalls; }

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

   // set ownership
   void SetOwnership(bool on = true) { fOwner = on;  }

   /// precision value used for calculating the derivative step-size
   /// h = eps * |x|. The default is 0.001, give a smaller in case function chanes rapidly
   static void SetDerivPrecision(double eps);

   /// get precision value used for calculating the derivative step-size
   static double GetDerivPrecision();


private:


   double DoEval(const double * x) const {
      fNCalls++;
      return (*fFunc)(x);
   }

   // calculate derivative using mathcore derivator
   double DoDerivative (const double * x, unsigned int icoord  ) const;

   // adapat internal function type to IMultiGenFunction needed by derivative calculation
   const IMultiGenFunction * fFunc;
   unsigned int fDim;
   mutable unsigned int fNCalls;
   bool fOwner;

   static double fgEps;          // epsilon used in derivative calculation h ~ eps |x|

};

   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_NumGradFunction */
