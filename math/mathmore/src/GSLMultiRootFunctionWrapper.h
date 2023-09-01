// @(#)root/mathmore:$Id$
// Authors: L. Moneta Dec 2006

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

// Header file for class GSLMultiRootFunctionWrapper
//
// Created by: moneta  at Sat Nov 13 14:54:41 2004
//
// Last update: Sat Nov 13 14:54:41 2004
//
#ifndef ROOT_Math_GSLMultiRootFunctionWrapper
#define ROOT_Math_GSLMultiRootFunctionWrapper

#include "gsl/gsl_multiroots.h"

#include "GSLMultiRootFunctionAdapter.h"


#include <cassert>

namespace ROOT {
namespace Math {



// can re-use same type for multi-fit

   typedef double ( * GSLMultiRootFPointer ) ( const gsl_vector *, void *, gsl_vector *);
   typedef void   ( * GSLMultiRootDfPointer )   ( const gsl_vector *, void *, gsl_matrix *);
   typedef void   ( * GSLMultiRootFdfPointer ) ( const gsl_vector *, void *, gsl_vector *, gsl_matrix *);


/**
   wrapper to a multi-dim function without  derivatives for multi roots
   algorithm

   @ingroup MultiRoots
*/
class GSLMultiRootFunctionWrapper {

public:

   GSLMultiRootFunctionWrapper()
   {
      fFunc.f = 0;
      fFunc.n = 0;
      fFunc.params = 0;
   }


   /// Fill gsl function structure from a C++ function iterator and size and number of residuals
   template<class FuncVector>
   void SetFunctions(const FuncVector & f, unsigned int n ) {
      const void * p = &f;
      assert (p != 0);
      fFunc.f   = &GSLMultiRootFunctionAdapter<FuncVector >::F;
      fFunc.n = n;
      fFunc.params =  const_cast<void *>(p);
   }

   gsl_multiroot_function * GetFunctions() { return &fFunc; }


  private:

   gsl_multiroot_function fFunc;

};


/**
   wrapper to a multi-dim function with  derivatives for multi roots
   algorithm

   @ingroup MultiRoot
*/

class GSLMultiRootDerivFunctionWrapper {

public:

   GSLMultiRootDerivFunctionWrapper()
   {
      fFunc.f = 0;
      fFunc.df = 0;
      fFunc.fdf = 0;
      fFunc.n = 0;
      fFunc.params = 0;
   }


   /// Fill gsl function structure from a C++ function iterator and size and number of residuals
   template<class FuncVector>
   void SetFunctions(const FuncVector & f, unsigned int n  ) {
      const void * p = &f;
      assert (p != 0);
      fFunc.f   = &GSLMultiRootFunctionAdapter<FuncVector >::F;
      fFunc.df  = &GSLMultiRootFunctionAdapter<FuncVector >::Df;
      fFunc.fdf = &GSLMultiRootFunctionAdapter<FuncVector >::FDf;
      fFunc.n = n;
      fFunc.params =  const_cast<void *>(p);
   }

   gsl_multiroot_function_fdf * GetFunctions() { return &fFunc; }


  private:

   gsl_multiroot_function_fdf fFunc;

};



} // namespace Math
} // namespace ROOT

#endif /* ROOT_Math_GSLMultiMinFunctionWrapper */
