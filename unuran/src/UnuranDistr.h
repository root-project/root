// @(#)root/unuran:$Id$
// Author: L. Moneta Wed Sep 27 11:22:07 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class UnuranDistr

#ifndef ROOT_Math_UnuranDistr
#define ROOT_Math_UnuranDistr

#include "unuran.h"
#include <iostream>

#include <cmath>

/** 
   UnuranDistr 
   Provides free function based on TF1 to be called by unuran 
*/ 


template<class Function> 
struct UnuranDistr {

   /// evaluate the probal
   static double Pdf(double x, const UNUR_DISTR * dist) {  
      const Function * func = reinterpret_cast<const Function *> (  unur_distr_get_extobj(dist) ); 
      return func->operator()(x);      
   }   

   static double Dpdf(double x,  const UNUR_DISTR * dist) { 
      const Function * func = reinterpret_cast<const Function *> (  unur_distr_get_extobj(dist) ); 
      return func->Derivative(x);            
   }

   static double Cdf(double x,  const UNUR_DISTR * dist) { 
      const Function * func = reinterpret_cast<const Function *> (  unur_distr_get_extobj(dist) ); 
      return func->Cdf(x);            
   }

}; 

/**
   free functions for multidimensional functions
   needed bby UNURAN
*/
template<class Function> 
struct UnuranDistrMulti {

   /// evaluate the probality density function
   static double Pdf(const double * x, UNUR_DISTR * dist) {  
      const Function * func = reinterpret_cast<const Function *> (  unur_distr_get_extobj(dist) ); 
      assert( func != 0); 
      return func->operator()(x);      
   }   


   static int Dpdf(double * grad, const double * x,  UNUR_DISTR * dist) { 
      const Function * func = reinterpret_cast<const Function *> (  unur_distr_get_extobj(dist) ); 
      func->Gradient(x,grad);
      return 0; 
   }
   
   // provides the gradient componets separatly
   static double Pdpdf(const double * x, int coord, UNUR_DISTR * dist) { 
      const Function * func = reinterpret_cast<const Function *> (  unur_distr_get_extobj(dist) ); 
      return func->Gradient(x,coord);
   }

   static double Logpdf(const double * x, UNUR_DISTR * dist) {  
//       const Function * func = reinterpret_cast<const Function *> (  unur_distr_get_extobj(dist) ); 
//       assert( func != 0); 
//       double y  =  func->operator()(x);      
// //       if ( y < std::numeric_limits<double>::min() ) 
      return  std::log( Pdf(x,dist) ); 
   }   

   static int Dlogpdf(double * grad, const double * x,  UNUR_DISTR * dist) {    
      int dim = unur_distr_get_dim(dist);
      double pdf = Pdf(x,dist); 
      int ret = Dpdf(grad,x,dist); 
      for (int i = 0; i < dim; ++i) { 
         grad[i] /= pdf; 
      }
      return ret; 
   }

   static double Pdlogpdf(const double * x, int coord, UNUR_DISTR * dist) { 
      return  Pdpdf(x,coord,dist)/ Pdf(x,dist);
   }

}; 



#endif /* ROOT_Math_UnuranDistr */
