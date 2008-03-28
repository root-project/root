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

// Header file for class GSLFunctionWrapper
// 
// Created by: moneta  at Sat Nov 13 14:54:41 2004
// 
// Last update: Sat Nov 13 14:54:41 2004
// 
#ifndef ROOT_Math_GSLFunctionWrapper
#define ROOT_Math_GSLFunctionWrapper

#include "gsl/gsl_math.h"

#include "Math/GSLFunctionAdapter.h"

#include <cassert>

namespace ROOT {
namespace Math {



typedef double ( * GSLFuncPointer ) ( double, void *);
typedef void ( * GSLFdfPointer ) ( double, void *, double *, double *);


/**
   Wrapper class to the gsl_function C structure. 
   This class to fill the GSL C structure  gsl_function with 
   the C++ function objcet. 
   Use the class ROOT::Math::GSLFunctionAdapter to adapt the 
   C++ function object to the right signature (function pointer type) 
   requested by GSL 
*/
class GSLFunctionWrapper { 

public: 

   GSLFunctionWrapper() 
   {
      fFunc.function = 0; 
      fFunc.params = 0;
   }

   /// set in the GSL C struct the pointer to the function evaluation 
   void SetFuncPointer( GSLFuncPointer f) { fFunc.function = f; } 

   /// set in the GSL C struct the extra-object pointer
   void SetParams ( void * p) { fFunc.params = p; }

   /// fill the GSL C struct from a generic C++ callable object 
   /// implementing operator() 
   template<class FuncType> 
   void SetFunction(const FuncType &f) { 
      const void * p = &f;
      assert (p != 0); 
      SetFuncPointer(&GSLFunctionAdapter<FuncType >::F);
      SetParams(const_cast<void *>(p));
   }
    
   gsl_function * GetFunc() { return &fFunc; } 

   GSLFuncPointer FunctionPtr() { return fFunc.function; }

   // evaluate the function 
   double operator() (double x) {  return GSL_FN_EVAL(&fFunc, x); }

   /// check if function is valid (has been set) 
   bool IsValid() { 
      return (fFunc.function != 0) ? true : false;  
   }

private: 
   gsl_function fFunc; 


};


   /**
     class to wrap a gsl_function_fdf (with derivatives)   
   */
  class GSLFunctionDerivWrapper { 

  public: 

     GSLFunctionDerivWrapper() 
     {
        fFunc.f = 0; 
        fFunc.df = 0; 
        fFunc.fdf = 0; 
        fFunc.params = 0;
     }


    void SetFuncPointer( GSLFuncPointer f) { fFunc.f = f; } 
    void SetDerivPointer( GSLFuncPointer f) { fFunc.df = f; } 
    void SetFdfPointer( GSLFdfPointer f) { fFunc.fdf = f; } 
    void SetParams ( void * p) { fFunc.params = p; }

    
    gsl_function_fdf * GetFunc() { return &fFunc; } 

    // evaluate the function and derivatives
    double operator() (double x) {  return GSL_FN_FDF_EVAL_F(&fFunc, x); }

    double Derivative (double x) { return GSL_FN_FDF_EVAL_DF(&fFunc, x); } 

    void Fdf(double x, double & f, double & df) { 
      return GSL_FN_FDF_EVAL_F_DF(&fFunc, x, &f, &df);
    }

   /// check if function is valid (has been set) 
   bool IsValid() { 
      return (fFunc.f != 0 ) ? true : false;  
   }

  private: 
    gsl_function_fdf fFunc; 

  };



} // namespace Math
} // namespace ROOT

#endif /* ROOT_Math_GSLFunctionWrapper */
