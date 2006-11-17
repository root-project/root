// @(#)root/mathmore:$Name:  $:$Id: RootFinderAlgorithms.h,v 1.1 2005/09/18 17:33:47 brun Exp $
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

// Header file for GSL ROOT Finder Algorithms
// 
// Created by: moneta  at Sun Nov 14 14:07:50 2004
// 
// Last update: Sun Nov 14 14:07:50 2004
// 
#ifndef ROOT_Math_GSLRootFinderAlgorithms
#define ROOT_Math_GSLRootFinderAlgorithms


#ifndef ROOT_Math_GSLRootFinder
#include "Math/GSLRootFinder.h"
#endif

#ifndef ROOT_Math_GSLRootFinderDeriv
#include "Math/GSLRootFinderDeriv.h"
#endif

namespace ROOT {
namespace Math {

  /**
     Root-Finding Algorithms

  */

   namespace Roots { 

     /**
	Bisection algorithms, simplest algorithms for bracketing the roots of a function, but slowest one
	See  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_32.html#SEC436">GSL manual</A> for more information
	@ingroup RootFinders
     */

     class Bisection : public GSLRootFinder {
       
     public: 

       Bisection(); 
       virtual ~Bisection();
       
     private:
       // usually copying is non trivial, so we make this unaccessible

       Bisection(const Bisection &); 
       Bisection & operator = (const Bisection &); 

     }; 

     /**
	False Position algorithm based on linear interpolation. 
	See  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_32.html#SEC436">GSL manual</A> for more information
	@ingroup RootFinders
     */

     class FalsePos : public GSLRootFinder {
       
     public: 

       FalsePos(); 
       virtual ~FalsePos();
       
     private:
       // usually copying is non trivial, so we make this unaccessible
       FalsePos(const FalsePos &); 
       FalsePos & operator = (const FalsePos &); 

     }; 



     /**
	Brent-Dekker algorithm which combines an interpolation strategy with the bisection algorithm 
	See  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_32.html#SEC436">GSL manual</A> for more information
	@ingroup RootFinders
     */

     class Brent : public GSLRootFinder {
       
     public: 

       Brent(); 
       virtual ~Brent();
       
     private:
       // usually copying is non trivial, so we make this unaccessible
       Brent(const Brent &); 
       Brent & operator = (const Brent &); 

     }; 

     
     //----------------------------------------------------------------------
     // algorithm with derivatives 
     //----------------------------------------------------------------------

     /**
	\a Newton algorithm, which computes the derivative at each iteration
	See  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_32.html#SEC437">GSL manual</A> for more information
	@ingroup RootFinders
     */

     class Newton : public GSLRootFinderDeriv {
       
     public: 

       Newton(); 
       virtual ~Newton();
       
     private:
       // usually copying is non trivial, so we make this unaccessible
       Newton(const Newton &); 
       Newton & operator = (const Newton &); 

     }; 

    
     /**
	\a Secant algorithm, simplified version of Newton method, which does not require the derivative at every step. 
	See  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_32.html#SEC437">GSL manual</A> for more information
	@ingroup RootFinders
     */

     class Secant : public GSLRootFinderDeriv {
       
     public: 

       Secant(); 
       virtual ~Secant();
       
     private:
       // usually copying is non trivial, so we make this unaccessible
       Secant(const Secant &); 
       Secant & operator = (const Secant &); 

     }; 

     /**
	\a Steffenson method, providing the fastes convergence.
	See  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_32.html#SEC437">GSL manual</A> for more information
	@ingroup RootFinders
     */

     class Steffenson : public GSLRootFinderDeriv {
       
     public: 

       Steffenson(); 
       virtual ~Steffenson();
       
     private:
       // usually copying is non trivial, so we make this unaccessible
       Steffenson(const Steffenson &); 
       Steffenson & operator = (const Steffenson &); 

     }; 


   }

} // namespace Math
} // namespace ROOT


#endif /* ROOT_Math_GSLRootFinderAlgorithms */
