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

// Header file for class InterpolationTypes
// 
// Created by: moneta  at Fri Nov 26 15:40:58 2004
// 
// Last update: Fri Nov 26 15:40:58 2004
// 
#ifndef ROOT_Math_InterpolationTypes
#define ROOT_Math_InterpolationTypes


namespace ROOT {
namespace Math {


  namespace Interpolation { 

    /**
       Enumeration defining the types of interpolation methods availables. 
       Passed as argument to instantiate mathlib::Interpolator objects. 
       The types available are (more information is available in the 
       <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Interpolation-Types.html">GSL manual</A>):
       <ul>
       <li>LINEAR interpolation;
       <li>POLYNOMIAL interpolation, to be used for small number of points since introduces large oscillations;
       <li>CSPLINE cubic spline with natural boundary conditions;
       <li>CSPLINE_PERIODIC cubic spline with periodic boundary conditions;
       <li>AKIMA, Akima spline with natural boundary conditions ( requires a minimum of 5 points);
       <li>AKIMA_PERIODIC, Akima spline with periodic boundaries ( requires a minimum of 5 points); 
       </ul>


       @ingroup Interpolation
     */

    // enumerations for the type of interpolations
    enum Type {  kLINEAR, 
		 kPOLYNOMIAL, 
 		 kCSPLINE, 
 		 kCSPLINE_PERIODIC,  
		 kAKIMA, 
		 kAKIMA_PERIODIC
    };
  }


} // namespace Math
} // namespace ROOT

#endif /* ROOT_Math_InterpolationTypes */
