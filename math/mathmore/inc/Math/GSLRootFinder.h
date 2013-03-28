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

// Header file for class GSLRootFinder
// 
// Created by: moneta  at Sun Nov 14 11:27:11 2004
// 
// Last update: Sun Nov 14 11:27:11 2004
// 
#ifndef ROOT_Math_GSLRootFinder
#define ROOT_Math_GSLRootFinder


#ifndef ROOT_Math_GSLFunctionAdapter
#include "Math/GSLFunctionAdapter.h"
#endif

#ifndef ROOT_Math_IFunctionfwd
#include "Math/IFunctionfwd.h"
#endif

#ifndef ROOT_Math_IRootFinderMethod
#include "Math/IRootFinderMethod.h"
#endif

#include <iostream>

namespace ROOT {
namespace Math {


   class GSLRootFSolver; 
   class GSLFunctionWrapper; 


//________________________________________________________________________________________________________
  /**
     Base class for GSL Root-Finding algorithms for one dimensional functions which do not use function derivatives. 
     For finding the roots users should not use this class directly but instantiate the derived classes, 
     for example  ROOT::Math::Roots::Brent for using the Brent algorithm. 
     All the classes defining the alhorithms are defined in the header Math/RootFinderAlgorithm.h
     They possible types implementing root bracketing algorithms which they do not require function 
     derivatives are: 
     <ul>
         <li>ROOT::Math::Roots::Bisection
         <li>ROOT::Math::Roots::FalsePos
         <li>ROOT::Math::Roots::Brent
     </ul>

     See also the specific  classes for the documentation. 
     See the GSL <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Root-Bracketing-Algorithms.html"> online manual</A> for 
     information on the GSL Root-Finding algorithms

     @ingroup RootFinders
  */


 class GSLRootFinder: public IRootFinderMethod {
     
 public: 
    GSLRootFinder(); 
    virtual ~GSLRootFinder(); 
     
 private:
    // usually copying is non trivial, so we make this unaccessible
    GSLRootFinder(const GSLRootFinder &); 
    GSLRootFinder & operator = (const GSLRootFinder &); 
     
 public: 
     

#if defined(__MAKECINT__) || defined(G__DICTIONARY)  
    bool SetFunction( const IGradFunction & , double ) { 
       std::cerr <<"GSLRootFinder - Error : this method must be used with a Root Finder algorithm using derivatives" << std::endl;  
       return false;
    }
#endif
   
    bool SetFunction( const IGenFunction & f, double xlow, double xup);

    typedef double ( * GSLFuncPointer ) ( double, void *);
    bool SetFunction( GSLFuncPointer  f, void * params, double xlow, double xup); 

    using IRootFinderMethod::SetFunction;

    // iterate to find ROOTS return GSL_CONTINUE if iteration was successful or another error
    int Iterate(); 

    double Root() const; 

    //double XLower() const; 

    //double XUpper() const; 

    /// Find the root
    bool Solve( int maxIter = 100, double absTol = 1E-8, double relTol = 1E-10); 

    /// Return number of iterations
    int Iterations() const {
       return fIter; 
    }

    /// Return the status of last root finding
    int Status() const { return fStatus; }

    const char * Name() const;  

     
 protected:
     

    void SetSolver (  GSLRootFSolver * s ); 
      
    void FreeSolver(); 
     
 private: 
     
    GSLFunctionWrapper * fFunction;     
    GSLRootFSolver * fS; 

    double fRoot; 
    double fXlow;
    double fXup; 
    int fIter;
    int fStatus;
    bool fValidInterval;

 }; 

} // namespace Math
} // namespace ROOT


#endif /* ROOT_Math_GSLRootFinder */
