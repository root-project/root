// @(#)root/mathmore:$Id$
// Author: L. Moneta, A. Zsenei   08/2005

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

// Header file for GSL ROOT Finder Algorithms
//
// Created by: moneta  at Sun Nov 14 14:07:50 2004
//
// Last update: Sun Nov 14 14:07:50 2004
//
#ifndef ROOT_Math_GSLRootFinderAlgorithms
#define ROOT_Math_GSLRootFinderAlgorithms


#include "Math/GSLRootFinder.h"

#include "Math/GSLRootFinderDeriv.h"

namespace ROOT {
namespace Math {

  /**
     Root-Finding Algorithms

  */

namespace Roots {

//________________________________________________________________________________________________________
     /**
        Roots::Bisection
      Bisection algorithm, simplest algorithm for bracketing the roots of a function, but slowest one.
      See the <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Root-Bracketing-Algorithms.html">GSL manual</A> for more information
      @ingroup RootFinders
     */

   class Bisection : public GSLRootFinder {

   public:

      Bisection();
      ~Bisection() override;
   };

//________________________________________________________________________________________________________
   /**
      False Position algorithm based on linear interpolation.
      See the <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Root-Bracketing-Algorithms.html">GSL manual</A> for more information
      @ingroup RootFinders
   */

   class FalsePos : public GSLRootFinder {

   public:

      FalsePos();
      ~FalsePos() override;
   };



//________________________________________________________________________________________________________
   /**
      Brent-Dekker algorithm which combines an interpolation strategy with the bisection algorithm
      See the <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Root-Bracketing-Algorithms.html">
      GSL manual</A> for more information

      @ingroup RootFinders
   */

   class Brent : public GSLRootFinder {

   public:

      Brent();
      ~Brent() override;
   };


   //----------------------------------------------------------------------
   // algorithm with derivatives
   //----------------------------------------------------------------------

//________________________________________________________________________________________________________
   /**
      a Newton algorithm, which computes the derivative at each iteration
      See the <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Root-Finding-Algorithms-using-Derivatives.html">
      GSL manual</A> for more information

      @ingroup RootFinders
   */

   class Newton : public GSLRootFinderDeriv {

   public:

      Newton();
      ~Newton() override;
   };


//________________________________________________________________________________________________________
   /**
      \a Secant algorithm, simplified version of Newton method, which does not require the derivative at every step.
      See the <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Root-Finding-Algorithms-using-Derivatives.html">
      GSL manual</A> for more information
      @ingroup RootFinders
   */

   class Secant : public GSLRootFinderDeriv {

   public:

      Secant();
      ~Secant() override;
   };

//________________________________________________________________________________________________________
   /**
      \a Steffenson method, providing the fastes convergence.
      See the <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Root-Finding-Algorithms-using-Derivatives.html">
      GSL manual</A> for more information

      @ingroup RootFinders
   */

   class Steffenson : public GSLRootFinderDeriv {

   public:

      Steffenson();
      ~Steffenson() override;
   };


}

} // namespace Math
} // namespace ROOT


#endif /* ROOT_Math_GSLRootFinderAlgorithms */
