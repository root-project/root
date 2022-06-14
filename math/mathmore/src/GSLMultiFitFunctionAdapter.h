// @(#)root/mathmore:$Id$
// Authors: L. Moneta, Dec 2006

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

// Header file for class GSLMultiMinFunctionAdapter
//
// Generic adapter for gsl_multimin_function signature
// usable for any c++ class which defines operator( )
//
// Created by: Lorenzo Moneta  at Fri Nov 12 16:58:51 2004
//
// Last update: Fri Nov 12 16:58:51 2004
//
#ifndef ROOT_Math_GSLMultiFitFunctionAdapter
#define ROOT_Math_GSLMultiFitFunctionAdapter

#include "gsl/gsl_vector.h"
#include "gsl/gsl_matrix.h"

#include <cassert>

namespace ROOT {
namespace Math {



  /**
     Class for adapting a C++ functor class to C function pointers used by GSL MultiFit
     Algorithm
     The templated C++ function class must implement:

    <em> double operator( const double *  x)</em>
    and if the derivatives are required:
    <em> void Gradient( const double *   x, double * g)</em>
    and
    <em> void FdF( const double *   x, double &f, double * g)</em>

    This class defines static methods with will be used to fill the
    \a gsl_multimin_function and
    \a gsl_multimin_function_fdf structs used by GSL.
    See for examples the
    <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Providing-a-function-to-minimize.html#Providing-a-function-to-minimize">GSL online manual</A>

   @ingroup MultiMin
  */


template<class FuncVector>
class GSLMultiFitFunctionAdapter {

public:

   static int F( const gsl_vector * x, void * p, gsl_vector * f ) {
      // p is a pointer to an iterator of functions
      unsigned int n = f->size;
      // need to copy iterator otherwise next time the function is called it wont work
      FuncVector  & funcVec = *( reinterpret_cast< FuncVector *> (p) );
      if (n == 0) return -1;
      for (unsigned int i = 0; i < n ; ++i) {
         gsl_vector_set(f, i, (funcVec[i])(x->data) );
      }
      return 0;
   }


   static int Df(  const gsl_vector * x, void * p, gsl_matrix * h) {

      // p is a pointer to an iterator of functions
      unsigned int n = h->size1;
      unsigned int npar = h->size2;
      if (n == 0) return -1;
      if (npar == 0) return -2;
      FuncVector  & funcVec = *( reinterpret_cast< FuncVector *> (p) );
      for (unsigned int i = 0; i < n ; ++i) {
         double * g = (h->data)+i*npar;   //pointer to start  of i-th row
         assert ( npar == (funcVec[i]).NDim() );
         (funcVec[i]).Gradient(x->data, g);
      }
      return 0;
   }

   /// evaluate derivative and function at the same time
   static int FDf(  const gsl_vector * x, void * p,  gsl_vector * f, gsl_matrix * h) {
      // should be implemented in the function
      // p is a pointer to an iterator of functions
      unsigned int n = h->size1;
      unsigned int npar = h->size2;
      if (n == 0) return -1;
      if (npar == 0) return -2;
      FuncVector  & funcVec = *( reinterpret_cast< FuncVector *> (p) );
      assert ( f->size == n);
      for (unsigned int i = 0; i < n ; ++i) {
         assert ( npar == (funcVec[i]).NDim() );
         double fval = 0;
         double * g = (h->data)+i*npar;   //pointer to start  of i-th row
         (funcVec[i]).FdF(x->data, fval, g);
         gsl_vector_set(f, i, fval  );
      }
      return 0;
   }

};


} // namespace Math
} // namespace ROOT


#endif /* ROOT_Math_GSLMultiMinFunctionAdapter */
