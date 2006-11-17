// @(#)root/mathmore:$Name:  $:$Id: GSLRootFinderDeriv.h,v 1.1 2005/09/18 17:33:47 brun Exp $
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

// Header file for class GSLRootFinderDeriv
// 
// Created by: moneta  at Sun Nov 21 16:26:03 2004
// 
// Last update: Sun Nov 21 16:26:03 2004
// 
#ifndef ROOT_Math_GSL_RootFinderDeriv
#define ROOT_Math_GSL_RootFinderDeriv


#ifndef ROOT_Math_GSLFunctionAdapter
#include "Math/GSLFunctionAdapter.h"
#endif

#ifndef ROOT_Math_IFunctionfwd
#include "Math/IFunctionfwd.h"
#endif
#ifndef ROOT_Math_IFunction
#include "Math/IFunction.h"
#endif

#include <iostream>

namespace ROOT {
namespace Math {


   class GSLRootFdFSolver; 
   class GSLFunctionDerivWrapper; 


  /**
     Base class for GSL Root-Finding algorithms for one dimensional functions which use function derivatives. 
     For finding the roots users should instantiate the RootFinder class with the corresponding algorithms 
     See mathlib::RootFinder class for documentation.
     See the GSL <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_32.html#SEC428"> online manual</A> for 
     information on the GSL Root-Finding algorithms

     @ingroup RootFinders
  */


   class GSLRootFinderDeriv {

   public: 
     GSLRootFinderDeriv(); 
     virtual ~GSLRootFinderDeriv(); 

   private:
     // usually copying is non trivial, so we make this unaccessible
     GSLRootFinderDeriv(const GSLRootFinderDeriv &); 
     GSLRootFinderDeriv & operator = (const GSLRootFinderDeriv &); 

   public: 


     void SetFunction( const IGenFunction & , double , double ) { 
        std::cerr <<"GSLRootFinderDeriv - Error : Algorithm requirs derivatives" << std::endl;  
     }
    
     
     void SetFunction( const IGradFunction & f, double Root) { 
       const void * p = &f; 
       SetFunction(  &GSLFunctionAdapter<IGradFunction>::F, &GSLFunctionAdapter<IGradFunction>::Df, &GSLFunctionAdapter<IGradFunction>::Fdf, const_cast<void *>(p), Root ); 
       }

     
     typedef double ( * GSLFuncPointer ) ( double, void *);
     typedef void ( * GSLFdFPointer ) ( double, void *, double *, double *);
     void SetFunction( GSLFuncPointer f, GSLFuncPointer df, GSLFdFPointer fdf, void * p, double Root );   

     int Iterate(); 

     double Root() const; 

     // Solve for roots
     int Solve( int maxIter = 100, double absTol = 1E-3, double relTol = 1E-6);

     int Iterations() const {
       return fIter; 
     }

     const char * Name() const;  

   protected:
     
     void SetSolver (  GSLRootFdFSolver * s ); 

     void FreeSolver(); 
     
   private: 
     
     GSLFunctionDerivWrapper * fFunction;     
     GSLRootFdFSolver * fS; 
 

     mutable double fRoot; 
     mutable double fPrevRoot; 
     int fIter; 
     
   }; 
   
} // namespace Math
} // namespace ROOT


#endif /* ROOT_Math_GSL_RootFinderDeriv */
