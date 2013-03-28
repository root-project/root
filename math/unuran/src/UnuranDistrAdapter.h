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
   Free functions adapter needed by UNURAN for onedimensional continuous distributions
*/ 

#include "TUnuranContDist.h"
#include "TUnuranMultiContDist.h"
#include "TUnuranDiscrDist.h"

struct ContDist {

   typedef TUnuranContDist Distribution; 

   /// evaluate the probality density function
   static double Pdf(double x, const UNUR_DISTR * dist) {  
      const Distribution * func = reinterpret_cast<const Distribution *> (  unur_distr_get_extobj(dist) ); 
      return func->Pdf(x);      
   }   
   /// evaluate the derivative of the pdf
   static double Dpdf(double x,  const UNUR_DISTR * dist) { 
      const Distribution * func = reinterpret_cast<const Distribution *> (  unur_distr_get_extobj(dist) ); 
      return func->DPdf(x);            
   }

   /// evaluate the Cumulative distribution function, integral of the pdf
   static double Cdf(double x,  const UNUR_DISTR * dist) { 
      const Distribution * func = reinterpret_cast<const Distribution *> (  unur_distr_get_extobj(dist) ); 
      return func->Cdf(x);            
   }

}; 

/**
   Free functions adapter needed by UNURAN for multidimensional cont distribution
*/
struct MultiDist {

   typedef TUnuranMultiContDist Distribution; 

   /// evaluate the probality density function
   static double Pdf(const double * x, UNUR_DISTR * dist) {  
      const Distribution * func = reinterpret_cast<const Distribution *> (  unur_distr_get_extobj(dist) ); 
      return func->Pdf(x);      
   }   

   // evaluate the gradient vector of the pdf
   static int Dpdf(double * grad, const double * x,  UNUR_DISTR * dist) { 
      const Distribution * func = reinterpret_cast<const Distribution *> (  unur_distr_get_extobj(dist) ); 
      func->Gradient(x,grad);
      return 0; 
   }
   
   // provides the gradient components separatly (partial derivatives)
   static double Pdpdf(const double * x, int coord, UNUR_DISTR * dist) { 
      const Distribution * func = reinterpret_cast<const Distribution *> (  unur_distr_get_extobj(dist) ); 
      return func->Derivative(x,coord);
   }

}; 


/**
   Free functions adapter needed by UNURAN for one-dimensional discrete distribution
*/
struct DiscrDist {

   typedef TUnuranDiscrDist Distribution; 


   /// evaluate the probality mesh function
   static double Pmf(int x, const UNUR_DISTR * dist) {  
      const Distribution * func = reinterpret_cast<const Distribution *> (  unur_distr_get_extobj(dist) ); 
      return func->Pmf(x);      
   }   

   /// evaluate the cumulative function
   static double Cdf(int x,  const UNUR_DISTR * dist) { 
      const Distribution * func = reinterpret_cast<const Distribution *> (  unur_distr_get_extobj(dist) ); 
      return func->Cdf(x);            
   }

}; 



#endif /* ROOT_Math_UnuranDistr */
