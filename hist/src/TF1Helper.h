// @(#)root/hist:$Name:  $:$Id: TF1Helper.h,v 1.1 2007/06/12 15:29:23 moneta Exp $
// Author: Lorenzo Moneta 12/06/07

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2007  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// helper functions used internally by TF1

#ifndef ROOT_TF1Helper
#define ROOT_TF1Helper 

#include "TF1.h"

namespace ROOT { 

   namespace TF1Helper { 

      double IntegralError(TF1 * func, double a, double b, double eps); 

  /**
     function class representing the derivative with respect a parameter of a given TF1  
  */
      class GradientParFunction { 

      public: 

         GradientParFunction(int ipar, TF1 * f)  : 
            fPar(ipar),
            fFunc(f)
         {}
         
         double operator() (double * x, double *) const 
         { 
            // evaluate gradient vector of functions at point x
            return fFunc->GradientPar(fPar,x);  
         }
         
      private: 
         
         unsigned int fPar; 
         mutable TF1 * fFunc;
      }; 


   } // end namespace TF1Helper

} // end namespace TF1

#endif
