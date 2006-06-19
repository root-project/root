// @(#)root/mathcore:$Name:  $:$Id: SpecFuncMathCore.cxx,v 1.1 2005/09/18 17:33:47 brun Exp $
// Authors: Andras Zsenei & Lorenzo Moneta   06/2005 

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/





//#include "MathCore/SpecFunc.h"
//#include "SpecFunc.h"
#if defined(__sun) || defined(__sgi) || defined(_WIN32) || defined(_AIX) || defined(__alpha) 
#define NOT_HAVE_TGAMMA
#endif


#include <cmath>

namespace ROOT {
namespace Math {





// (26.x.21.2) complementary error function

double erfc(double x) {
   
   
#ifdef NOT_HAVE_TGAMMA
   // copied from TMath for those platforms which do not have a
   // C99 compliant compiler
   
   // Compute the complementary error function erfc(x).
   // Erfc(x) = (2/sqrt(pi)) Integral(exp(-t^2))dt between x and infinity
   //
   //--- Nve 14-nov-1998 UU-SAP Utrecht
   
   // The parameters of the Chebyshev fit
   const double a1 = -1.26551223,   a2 = 1.00002368,
   a3 =  0.37409196,   a4 = 0.09678418,
   a5 = -0.18628806,   a6 = 0.27886807,
   a7 = -1.13520398,   a8 = 1.48851587,
   a9 = -0.82215223,  a10 = 0.17087277;
   
   double v = 1.0; // The return value
   double z = std::fabs(x);
   
   if (z <= 0) return v; // erfc(0)=1
   
   double t = 1.0/(1.0+0.5*z);
   
   v = t*std::exp((-z*z) +a1+t*(a2+t*(a3+t*(a4+t*(a5+t*(a6+t*(a7+t*(a8+t*(a9+t*a10)))))))));
   
   if (x < 0) v = 2.0-v; // erfc(-x)=2-erfc(x)
   
   return v;
#else
   return ::erfc(x);
#endif
   
}


// (26.x.21.1) error function

double erf(double x) {
   
   
#ifdef NOT_HAVE_TGAMMA
   return (1.0-ROOT::Math::erfc(x));
#else
   return ::erf(x);
#endif
   
   
}




double lgamma(double z) {
   
#ifdef NOT_HAVE_TGAMMA
   // copied from TMath for those platforms which do not have a
   // C99 compliant compiler 
   
   // Computation of ln[gamma(z)] for all z>0.
   //
   // C.Lanczos, SIAM Journal of Numerical Analysis B1 (1964), 86.
   //
   // The accuracy of the result is better than 2e-10.
   //
   //--- Nve 14-nov-1998 UU-SAP Utrecht
   
   if (z<=0) return 0;
   
   // Coefficients for the series expansion
   double c[7] = { 2.5066282746310005, 76.18009172947146, -86.50532032941677
      ,24.01409824083091,  -1.231739572450155, 0.1208650973866179e-2
      ,-0.5395239384953e-5};
   
   double x   = z;
   double y   = x;
   double tmp = x+5.5;
   tmp = (x+0.5)*std::log(tmp)-tmp;
   double ser = 1.000000000190015;
   for (int i=1; i<7; i++) {
      y   += 1.0;
      ser += c[i]/y;
   }
   double v = tmp+std::log(c[0]*ser/x);
   return v;
   
#else
   return ::lgamma(z);
#endif
   
}






// (26.x.18) gamma function

double tgamma(double x) {
   
   
#ifdef NOT_HAVE_TGAMMA
   return std::exp(ROOT::Math::lgamma(x));
#else
   return ::tgamma(x);
#endif
   
}



// [5.2.1.3] beta function
// (26.x.19)

double beta(double x, double y) {
   
   return std::exp(lgamma(x)+lgamma(y)-lgamma(x+y));
   
}




} // namespace Math
} // namespace ROOT





