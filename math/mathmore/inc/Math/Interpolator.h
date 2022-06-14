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

// Header file for class Interpolator
//
// Created by: moneta  at Fri Nov 26 15:00:25 2004
//
// Last update: Fri Nov 26 15:00:25 2004
//
#ifndef ROOT_Math_Interpolator
#define ROOT_Math_Interpolator

#include "Math/InterpolationTypes.h"

#include <vector>
#include <string>

/**
@defgroup Interpolation Interpolation Classes

Classes for interpolation of points

@ingroup NumAlgo
*/


namespace ROOT {
namespace Math {


   class GSLInterpolator;

//_____________________________________________________________________________________
   /**
      Class for performing function interpolation of points.
      The class is instantiated with an interpolation methods, passed as an enumeration in the constructor.
      See Interpolation::Type for the available interpolation algorithms, which are implemented using GSL.
      See also the <A HREF=http://www.gnu.org/software/gsl/manual/html_node/Interpolation.html">GSL manual</A> for more information.
      The class provides additional methods for computing derivatives and integrals of interpolating functions.

      This class does not support copying.
      @ingroup Interpolation
   */

class Interpolator {

public:

   /**
      Constructs an interpolator class from  number of data points and with Interpolation::Type type.
      The data can be set later on with the SetData method.
      In case the data size is not known, better using the default of zero or the next constructor later on.
      The default interpolation type is Cubic spline
   */
   Interpolator(unsigned int ndata = 0, Interpolation::Type type = Interpolation::kCSPLINE);

   /**
      Constructs an interpolator class from vector of data points \f$ (x_i, y_i )\f$ and with Interpolation::Type type.
      The method will compute a continuous interpolating function \f$ y(x) \f$ such that \f$ y_i = y ( x_i )\f$.
      The default interpolation type is Cubic spline
   */
   Interpolator(const std::vector<double> & x, const std::vector<double> & y, Interpolation::Type type = Interpolation::kCSPLINE);

   virtual ~Interpolator();

private:
   // usually copying is non trivial, so we make this unaccessible
   Interpolator(const Interpolator &);
   Interpolator & operator = (const Interpolator &);

public:

   /**
      Set the data vector ( x[] and y[] )
      To be efficient, the size of the data must be the same of the value used in the constructor (ndata)
      If this is not  the case a new re-initialization is performed with the new data size
    */
   bool SetData(const std::vector<double> & x, const std::vector<double> & y);

   /**
      Set the data vector ( x[] and y[] )
      To be efficient, the size of the data must be the same of the value used when  constructing the class (ndata)
      If this is not  the case a new re-initialization is performed with the new data size.
    */
   bool SetData(unsigned int ndata, const double * x, const double *  y);

   /**
      Return the interpolated value at point x
   */
   double Eval( double x ) const;

   /**
      Return the derivative of the interpolated function at point x
   */
   double Deriv( double x ) const;

   /**
      Return the second derivative of the interpolated function at point x
   */
   double Deriv2( double x ) const;

   /**
      Return the Integral of the interpolated function over the range [a,b]
   */
   double Integ( double a, double b) const;

   /**
      Return the type of interpolation method
   */
   std::string Type() const;
   std::string TypeGet() const;

protected:


private:

   GSLInterpolator * fInterp;   // pointer to GSL interpolator class

};

} // namespace Math
} // namespace ROOT


#endif /* ROOT_Math_Interpolator */
