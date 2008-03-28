// special functions taken from Cephes library 
//  see:  http://www.netlib.org/cephes
// 
// Copyright 1985, 1987, 2000 by Stephen L. Moshier
// 
//  granted permission from the author to be used in MathCore
//  




#ifndef ROOT_Math_SpecFunCephes
#define ROOT_Math_SpecFunCephes

namespace ROOT { 
   namespace Math { 

      namespace Cephes { 


//---
/* the machine roundoff error */
#define kMACHEP  1.11022302462515654042363166809e-16

/* largest argument for TMath::Exp() */
#define kMAXLOG  709.782712893383973096206318587

/* smallest argument for TMath::Exp() without underflow */
#define kMINLOG  -708.396418532264078748994506896

/* the maximal number that pow(x,x-0.5) has no overflow */
/* we use a (very) conservative portable bound          */
#define kMAXSTIR  108.116855767857671821730036754

#define kMAXLGM 2.556348e305


/** 
    incomplete complementary gamma function
 *  igamc(a, x) = 1 - igam(a, x)
*/
double igamc( double a, double x );

/* incomplete gamma function*/
double  igam( double a, double x );

/* Logarithm of gamma function */
double lgam( double x );

/* gamma function*/
double gamma( double x );

/* beta function*/
double beta(double z, double w);

/* evaluation of incomplete beta */
double incbet( double aa, double bb, double xx );

/* Continued fraction expansion #1
 * for incomplete beta integral
 * used when xx < (aa-1)/(aa+bb-2)
 * (and bb*xx > 1 or xx > 0.95) 
*/
double incbcf( double a, double b, double x );


/* Continued fraction expansion #2
 * for incomplete beta integral
 * used when xx > (aa-1)/(aa+bb-2)
 * (and bb*xx > 1 or xx > 0.95) 
 */
double incbd( double a, double b, double x );


/* Power series for incomplete beta integral.
   Use when b*x is small and x not too close to 1.  */

double pseries( double a, double b, double x );


/* error function */
double erf( double a );

/* complementary error function */
double erfc( double a );


// inverse function

/* normal quantile */ 
double ndtri (double y); 

/* normal quantile */ 
double ndtri (double y); 

/* inverse of incomplete gamma */ 
double igami (double a, double y); 

/* inverse of incomplete beta */ 
double incbi (double a, double b, double y); 


} // end namespace Cephes

/* routines for efficient polynomial evaluation*/
double Polynomialeval(double x, double* a, unsigned int N);
double Polynomial1eval(double x, double* a, unsigned int N);


} // end namespace Math
} // end namespace ROOT


#endif /* SpecFun */

