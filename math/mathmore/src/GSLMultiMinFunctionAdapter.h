// @(#)root/mathmore:$Id$
// Authors: L. Moneta, 12/2006

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
#ifndef ROOT_Math_GSLMultiMinFunctionAdapter
#define ROOT_Math_GSLMultiMinFunctionAdapter

#include "gsl/gsl_vector.h"

namespace ROOT {
namespace Math {




  /**
     Class for adapting any multi-dimension C++ functor class to C function pointers used by
     GSL MultiMin algorithms.
     The templated C++ function class must implement:

    <em> double operator( const double *  x)</em>
    and if the derivatives are required:
    <em> void Gradient( const double *   x, double * g)</em>

    This class defines static methods with will be used to fill the
    \a gsl_multimin_function and
    \a gsl_multimin_function_fdf structs used by GSL.
    See for examples the
    <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Providing-a-function-to-minimize.html#Providing-a-function-to-minimize">GSL online manual</A>

   @ingroup MultiMin

  */


  template<class UserFunc>
  struct  GSLMultiMinFunctionAdapter {

    static double F( const gsl_vector * x, void * p) {

      UserFunc * function = reinterpret_cast< UserFunc *> (p);
      // get pointer to data from gsl_vector
      return (*function)( x->data );
    }


    static void Df(  const gsl_vector * x, void * p,  gsl_vector * g) {

      UserFunc * function = reinterpret_cast< UserFunc *> (p);
      (*function).Gradient( x->data, g->data );

    }

    static void Fdf( const gsl_vector * x, void * p, double *f, gsl_vector * g ) {

      UserFunc * function = reinterpret_cast< UserFunc *> (p);
//       *f  = (*function) ( x );
//       *df = (*function).Gradient( x );

      (*function).FdF( x->data, *f, g->data);
    }

  };


} // namespace Math
} // namespace ROOT


#endif /* ROOT_Math_GSLMultiMinFunctionAdapter */
