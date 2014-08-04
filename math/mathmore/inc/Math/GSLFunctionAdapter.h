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

// Header file for class GSLFunctionAdapter
//
// Generic adapter for gsl_function signature
// usable for any c++ class which defines operator( )
//
// Created by: Lorenzo Moneta  at Fri Nov 12 16:58:51 2004
//
// Last update: Fri Nov 12 16:58:51 2004
//
#ifndef ROOT_Math_GSLFunctionAdapter
#define ROOT_Math_GSLFunctionAdapter


namespace ROOT {
namespace Math {

  /**
     Function pointer corresponding to gsl_function signature
   */

  typedef double ( * GSLFuncPointer ) ( double, void *);


  /**
     Class for adapting any C++ functor class to C function pointers used by GSL.
     The templated C++ function class must implement:

    <em> double operator( double x)</em>
    and if the derivatives are required:
    <em> double Gradient( double x)</em>

    This class defines static methods with will be used to fill the
    \a gsl_function and \a gsl_function_fdf structs used by GSL.
    See for examples the <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_32.html#SEC432">GSL online manual</A>
  */


  template<class UserFunc>
  class GSLFunctionAdapter {

  public:

    GSLFunctionAdapter() {}
    virtual ~GSLFunctionAdapter() {}

    static double F( double x, void * p) {

      UserFunc * function = reinterpret_cast< UserFunc *> (p);
      return (*function)( x );
    }


    static double Df( double x, void * p) {

      UserFunc * function = reinterpret_cast< UserFunc *> (p);
      return (*function).Derivative( x );
    }

    static void Fdf( double x, void * p, double *f, double *df ) {

      UserFunc * function = reinterpret_cast< UserFunc *> (p);
      *f  = (*function) ( x );
      *df = (*function).Derivative( x );
    }

  };


} // namespace Math
} // namespace ROOT


#endif /* ROOT_Math_GSLFunctionAdapter */
