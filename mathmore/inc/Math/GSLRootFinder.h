// @(#)root/mathmore:$Name:  $:$Id: GSLRootFinder.h,v 1.1 2005/09/18 17:33:47 brun Exp $
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

#include <iostream>

namespace ROOT {
namespace Math {


   class GSLRootFSolver; 
   class GSLFunctionWrapper; 


  /**
     Base class for GSL Root-Finding algorithms for one dimensional functions which do not use function derivatives. 
     For finding the roots users should instantiate the RootFinder class with the corresponding algorithms 
     See mathlib::RootFinder class for documentation.
     See the GSL <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_32.html#SEC428"> online manual</A> for 
     information on the GSL Root-Finding algorithms

     @ingroup RootFinders
  */


   class GSLRootFinder {
     
   public: 
     GSLRootFinder(); 
     virtual ~GSLRootFinder(); 
     
   private:
     // usually copying is non trivial, so we make this unaccessible
     GSLRootFinder(const GSLRootFinder &); 
     GSLRootFinder & operator = (const GSLRootFinder &); 
     
   public: 
     

     void SetFunction( const IGradFunction & , double ) { 
        std::cerr <<"GSLRootFinder - Error : use other Setfunction" << std::endl;  
      //SetFunction( f, Root -1, Root + 1); // temporary 
     }
   
     /**  
     void SetFunction( const IGenFunction & f, double xlow, double xup) { 
       const void * p = &f; 
       SetFunction(  &GSLFunctionAdapter<IGenFunction>::F, const_cast<void *>(p), xlow, xup ); 
     }
     */
     void SetFunction( const IGenFunction & f, double xlow, double xup);

     typedef double ( * GSLFuncPointer ) ( double, void *);
     void SetFunction( GSLFuncPointer  f, void * params, double xlow, double xup); 

     int Iterate(); 

     double Root() const; 

     //double XLower() const; 

     //double XUpper() const; 

     // Solve for roots
     int Solve( int maxIter = 100, double absTol = 1E-3, double relTol = 1E-6); 

     int Iterations() const {
       return fIter; 
     }

     const char * Name() const;  

     
   protected:
     

     void SetSolver (  GSLRootFSolver * s ); 

     void FreeSolver(); 
     
   private: 
     
     GSLFunctionWrapper * fFunction;     
     GSLRootFSolver * fS; 
  
   protected: 



   private: 

     double fRoot; 
     double fXlow;
     double fXup; 
     int fIter; 

   }; 

} // namespace Math
} // namespace ROOT


#endif /* ROOT_Math_GSLRootFinder */
