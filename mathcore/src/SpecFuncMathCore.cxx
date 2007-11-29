// @(#)root/mathcore:$Id$
// Authors: Andras Zsenei & Lorenzo Moneta   06/2005 

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

#if defined(__sun) || defined(__sgi) || defined(_WIN32) || defined(_AIX) || defined(__alpha) 
#define NOT_HAVE_TGAMMA
#endif


#include "SpecFuncCephes.h"


#include <cmath>

// use cephes for functions which are also in C99
#define USE_CEPHES

// platforms not implemening C99
// #if defined(__sun) || defined(__sgi) || defined(_WIN32) || defined(_AIX) || defined(__alpha) 
// #define USE_CEPHES
// #endif


namespace ROOT {
namespace Math {





// (26.x.21.2) complementary error function

double erfc(double x) {
   
   
#ifdef USE_CEPHES
   // use cephes implementation   
   return ROOT::Math::Cephes::erfc(x);
#else
   return ::erfc(x);
#endif
   
}


// (26.x.21.1) error function

double erf(double x) {
   
   
#ifdef USE_CEPHES
   return ROOT::Math::Cephes::erf(x);
#else
   return ::erf(x);
#endif
   
   
}




double lgamma(double z) {
   
#ifdef USE_CEPHES
   return ROOT::Math::Cephes::lgam(z);
#else
   return ::lgamma(z);
#endif
   
}




// (26.x.18) gamma function

double tgamma(double x) {
      
#ifdef USE_CEPHES
   return ROOT::Math::Cephes::gamma(x);
#else
   return ::tgamma(x);
#endif
   
}

double inc_gamma( double a, double x) { 
   return ROOT::Math::Cephes::igam(a,x); 
} 

double inc_gamma_c( double a, double x) { 
   return ROOT::Math::Cephes::igamc(a,x); 
} 


// [5.2.1.3] beta function
// (26.x.19)

double beta(double x, double y) {   
   return std::exp(lgamma(x)+lgamma(y)-lgamma(x+y));   
}

double inc_beta( double x, double a, double b) { 
   return ROOT::Math::Cephes::incbet(a,b,x); 
} 




} // namespace Math
} // namespace ROOT





