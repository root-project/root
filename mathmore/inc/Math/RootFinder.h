// @(#)root/mathmore:$Name:  $:$Id: RootFinder.h,v 1.1 2005/09/18 17:33:47 brun Exp $
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

// Header file for class RootFinder
// 
// Created by: moneta  at Sun Nov 14 16:59:55 2004
// 
// Last update: Sun Nov 14 16:59:55 2004
// 
#ifndef ROOT_Math_RootFinder
#define ROOT_Math_RootFinder

#ifndef ROOT_Math_GSLRootHelper
#include "Math/GSLRootHelper.h"
#endif

#ifndef ROOT_Math_IFunctionfwd
#include "Math/IFunctionfwd.h"
#endif


/**
   @defgroup RootFinders Root-Finding algorithms
 */


namespace ROOT {
namespace Math {


  /**
     Class to find the Root of one dimensional functions. 
     The class is templated on the type of Root solver algorithms.
     The possible types of Root-finding algorithms are: 
     <ul>
     <li>Root Bracketing Algorithms which they do not require function derivatives
     <ol>
     <li>Roots::Bisection
     <li>Roots::FalsePos
     <li>Roots::Brent
     </ol>
     <li>Root Finding Algorithms using Derivatives
     <ol>
     <li>Roots::Newton
     <li>Roots::Secant
     <li>Roots::Steffenson
     </ol>
     </ul>

     This class does not cupport copying

     @ingroup RootFinder
  */

template <class SolverClass>
class RootFinder {

public: 

  /**
     Construct a Root-Finder algorithm
   */
  RootFinder() {}
  virtual ~RootFinder() {}

private:
// usually copying is non trivial, so we make this unaccessible
  RootFinder(const RootFinder & ) {}
  RootFinder & operator = (const RootFinder & rhs) 
    {   
      if (this == &rhs) return *this;  // time saving self-test
      return *this;
    } 

public: 

    /**
       Provide to the solver the function and the initial search interval [xlow, xup] 
       for algorithms not using derivatives (bracketing algorithms) 
       The templated function f must be of a type implementing the \a operator() method, 
       <em>  double  operator() (  double  x ) </em>
    */
 
    void SetFunction( const IGenFunction & f, double xlow, double xup) { 
    fSolver.SetFunction( f, xlow, xup); 
  }   


    /**
       Provide to the solver the function and an initial estimate of the Root, 
       for algorithms using derivatives. 
       The templated function f must be of a type implementing the \a operator()  
       and the \a Gradient() methods. 
       <em>  double  operator() (  double  x ) </em>
    */

    void SetFunction( const IGradFunction & f, double Root) { 
    fSolver.SetFunction( f, Root); 
  }   

  /** 
      Compute the roots iterating until the estimate of the Root is within the required tolerance returning 
      the iteration Status
  */
  int Solve( int maxIter = 100, double absTol = 1E-3, double relTol = 1E-6) { 
    return fSolver.Solve( maxIter, absTol, relTol ); 
  }

  /** 
      Return the number of iteration performed to find the Root. 
  */ 
  int Iterations() const {
    return fSolver.Iterations(); 
  }

  /**
     Perform a single iteration and return the Status
   */
  int Iterate() { 
    return fSolver.Iterate(); 
  }

  /**
     Return the current and latest estimate of the Root
   */ 
  double Root() const { 
    return fSolver.Root(); 
  }


  /**
     Return the current and latest estimate of the lower value of the Root-finding interval (for bracketing algorithms)
  */
/*   double XLower() const {  */
/*     return fSolver.XLower();  */
/*   } */

  /**
     Return the current and latest estimate of the upper value of the Root-finding interval (for bracketing algorithms)
  */
/*   double XUpper() const {  */
/*     return  fSolver.XUpper();  */
/*   } */

  /**
     Get Name of the Root-finding solver algorithm
   */
  const char * Name() const { 
    return fSolver.Name(); 
  }

  /**
     Test convertgence Status of current iteration using interval values (for bracketing algorithms)
  */
  static int TestInterval( double xlow, double xup, double epsAbs, double epsRel) { 
    return GSLRootHelper::TestInterval(xlow, xup, epsAbs, epsRel); 
  }

  /**
     Test convergence Status of current iteration using last Root estimates (for algorithms using function derivatives)
  */
  static int TestDelta( double r1, double r0, double epsAbs, double epsRel) { 
    return GSLRootHelper::TestDelta(r1, r0, epsAbs, epsRel); 
  }

  /**
     Test function residual
   */
  static int TestResidual(double f,  double epsAbs) { 
    return GSLRootHelper::TestResidual(f, epsAbs); 
  }
  


protected: 


private: 

  SolverClass fSolver;   // type of algorithm to be used 


}; 

} // namespace Math
} // namespace ROOT



#endif /* ROOT_Math_RootFinder */
