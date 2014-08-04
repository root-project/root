// @(#)root/mathmore:$Id$
// Authors: L. Moneta, 08/2007

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
#ifndef ROOT_Math_GSLMonteFunctionAdapter
#define ROOT_Math_GSLMonteFunctionAdapter


namespace ROOT {
namespace Math {


  /**
     Class for adapting any multi-dimension C++ functor class to C function pointers used by
     GSL MonteCarlo integration algorithms.
     The templated C++ function class must implement:

    <em> double operator( const double *  x)</em>

    This class defines static methods with will be used to fill the
    \a gsl_monte_function  used by GSL.
    See for examples the
    <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Monte-Carlo-Interface.html">GSL online manual</A>

    @ingroup MCIntegration
  */
 typedef double ( * GSLMonteFuncPointer ) ( double *, size_t, void *);

  template<class UserFunc>
  struct  GSLMonteFunctionAdapter {

    static double F( double * x, size_t, void * p) {

      UserFunc * function = reinterpret_cast< UserFunc *> (p);
      return (*function)( x );
    }

  };



} // namespace Math
} // namespace ROOT


#endif /* ROOT_Math_GSLMonteFunctionAdapter */
