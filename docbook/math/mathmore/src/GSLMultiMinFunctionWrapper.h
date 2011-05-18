// @(#)root/mathmore:$Id$
// Authors: L. Moneta, 12/2006 

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

// Header file for class GSLMultiMinFunctionWrapper
// 
// Created by: moneta  at Sat Nov 13 14:54:41 2004
// 
// Last update: Sat Nov 13 14:54:41 2004
// 
#ifndef ROOT_Math_GSLMultiMinFunctionWrapper
#define ROOT_Math_GSLMultiMinFunctionWrapper

#include "gsl/gsl_multimin.h"

#include "GSLMultiMinFunctionAdapter.h"


#include <cassert>

namespace ROOT {
namespace Math {



   typedef double ( * GSLMultiMinFuncPointer ) ( const gsl_vector *, void *);
   typedef void   ( * GSLMultiMinDfPointer )   ( const gsl_vector *, void *, gsl_vector *);
   typedef void   ( * GSLMultiMinFdfPointer ) ( const gsl_vector *, void *, double *, gsl_vector *);


/**
   wrapper to a multi-dim function withtout  derivatives for multi-dimensional 
   minimization algorithm

   @ingroup MultiMin
*/

class GSLMultiMinFunctionWrapper { 

public: 

   GSLMultiMinFunctionWrapper()  
   {
      fFunc.f = 0; 
      fFunc.n = 0; 
      fFunc.params = 0;
   }

   void SetFuncPointer( GSLMultiMinFuncPointer f) { fFunc.f = f; } 
   void SetDim  ( unsigned int n ) { fFunc.n = n; }
   void SetParams ( void * p) { fFunc.params = p; }

   /// Fill gsl function structure from a C++ Function class 
   template<class FuncType> 
   void SetFunction(const FuncType &f) { 
      const void * p = &f;
      assert (p != 0); 
      SetFuncPointer(&GSLMultiMinFunctionAdapter<FuncType >::F);
      SetDim( f.NDim() ); 
      SetParams(const_cast<void *>(p));
   }
   
   gsl_multimin_function * GetFunc() { return &fFunc; } 

    bool IsValid() { 
       return (fFunc.f != 0) ? true : false;  
    }


  private: 

    gsl_multimin_function fFunc; 

  };


/**
   Wrapper for a multi-dimensional function with derivatives used in GSL multidim
   minimization algorithm

   @ingroup MultiMin

 */
 class GSLMultiMinDerivFunctionWrapper { 

 public: 

    GSLMultiMinDerivFunctionWrapper()  
    {
       fFunc.f = 0; 
       fFunc.df = 0; 
       fFunc.fdf = 0; 
       fFunc.n = 0; 
       fFunc.params = 0;
    }


    void SetFuncPointer( GSLMultiMinFuncPointer f) { fFunc.f = f; } 
    void SetDerivPointer( GSLMultiMinDfPointer f) { fFunc.df = f; } 
    void SetFdfPointer( GSLMultiMinFdfPointer f) { fFunc.fdf = f; } 
    void SetDim  ( unsigned int n ) { fFunc.n = n; }
    void SetParams ( void * p) { fFunc.params = p; }

    /// Fill gsl function structure from a C++ Function class 
    template<class FuncType> 
    void SetFunction(const FuncType &f) { 
       const void * p = &f;
       assert (p != 0); 
       SetFuncPointer(&GSLMultiMinFunctionAdapter<FuncType >::F);
       SetDerivPointer(&GSLMultiMinFunctionAdapter<FuncType >::Df);
       SetFdfPointer(&GSLMultiMinFunctionAdapter<FuncType >::Fdf);
       SetDim( f.NDim() ); 
       SetParams(const_cast<void *>(p));
     }
    
    gsl_multimin_function_fdf * GetFunc() { return &fFunc; } 

#ifdef NEEDED_LATER
    // evaluate the function 
    double operator() (const double * x) {  
       // vx must be a gsl_vector
       return GSL_MULTIMIN_FN_EVAL(&fFunc, vx); 
    }
#endif

   /// check if function is valid (has been set) 
    bool IsValid() { 
       return (fFunc.f != 0) ? true : false;  
    }

 private: 

    gsl_multimin_function_fdf fFunc; 
    
  };




} // namespace Math
} // namespace ROOT

#endif /* ROOT_Math_GSLMultiMinFunctionWrapper */
