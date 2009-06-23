// @(#)root/mathmore:$Id$
// Author: L. Moneta 2009

// Implementation file for all the MinimizerVariableTransormation's 
// (implementations taken from minuit2 classes) 


#include "Math/MinimizerVariableTransformation.h"
#include <cmath>
#include <limits>

namespace ROOT { 

   namespace Math { 

// implementations for the class SinVariableTransformation

double SinVariableTransformation::Int2ext(double value, double lower, double upper) const {
   // transformation from  internal (unlimited) to external values (limited by lower/upper )
   return lower + 0.5*(upper - lower)*(std::sin(value) + 1.);
}

double SinVariableTransformation::Ext2int(double value, double lower, double upper) const {
   // transformation from external (limited by lower/upper )  to internal (unlimited) values given the lower/upper limits
   
   double piby2 = 2.*std::atan(1.);
   static const double eps = std::numeric_limits<double>::epsilon(); 
   double distnn = 8.*std::sqrt(eps);
   double vlimhi = piby2 - distnn;
   double vlimlo = -piby2 + distnn;
   
   double yy = 2.*(value - lower)/(upper - lower) - 1.;
   double yy2 = yy*yy;
   if(yy2 > (1. - 8 * eps) ) {
      if(yy < 0.) {
         // lower limit
         //       std::cout<<"SinVariableTransformation warning: is at its lower allowed limit. "<<value<<std::endl;
         return vlimlo;
      } else {
         // upper limit
         //       std::cout<<"SinVariableTransformation warning: is at its upper allowed limit."<<std::endl;
         return vlimhi;
      }
      
   } else {
         return std::asin(yy); 
   }
}

double SinVariableTransformation::DInt2Ext(double value, double lower, double upper) const {
   // return the derivative of the internal to external transformation (Int2Ext) : d Int2Ext / d Int 
   return 0.5*((upper - lower)*std::cos(value));
}  

// sqrt up 
// implementations for the class SqrtUpVariableTransformation


   double SqrtLowVariableTransformation::Int2ext(double value, double lower, double) const {
   /// internal to external transformation 
   double val = lower - 1. + std::sqrt( value*value + 1.);
   return val; 
}


double SqrtLowVariableTransformation::Ext2int(double value, double lower, double ) const {
   // external to internal transformation
   double yy = value - lower + 1.; 
   double yy2 = yy*yy; 
   if (yy2 < 1. ) 
      return 0; 
   else 
      return std::sqrt( yy2 -1); 
}

double SqrtLowVariableTransformation::DInt2Ext(double value, double, double) const {
   // derivative of internal to external transofrmation   :  d (Int2Ext) / d Int  
   double val = value/( std::sqrt( value*value + 1.) );
   return val; 
}

// sqrt up 
// implementations for the class SqrtUpVariableTransformation

double SqrtUpVariableTransformation::Int2ext(double value, double upper, double) const {
   // internal to external transformation
   double val = upper + 1. - std::sqrt( value*value + 1.);
   return val; 
}


double SqrtUpVariableTransformation::Ext2int(double value, double upper, double ) const {
   // external to internal transformation  
   double yy = upper - value + 1.; 
   double arg = yy*yy - 1; 
   return ( arg < 0 ) ? 0 : std::sqrt(arg); 
}


double SqrtUpVariableTransformation::DInt2Ext(double value, double, double) const {
   // derivative of internal to external transofrmation :  d Ext / d Int   
   double val = - value/( std::sqrt( value*value + 1.) );
   return val; 
}


   } // end namespace Math

} // end namespace ROOT

