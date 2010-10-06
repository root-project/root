//
//
// gamma and related functions from Cephes library
// see:  http://www.netlib.org/cephes
// 
// Copyright 1985, 1987, 2000 by Stephen L. Moshier
// 
//

#include "SpecFuncCephes.h"
#include "Math/Math.h"


#include <cmath>

#include <limits> 



namespace ROOT { 
namespace Math { 

namespace Cephes { 


static double kBig = 4.503599627370496e15;
static double kBiginv =  2.22044604925031308085e-16;

/* log( sqrt( 2*pi ) ) */
static double LS2PI  =  0.91893853320467274178;


// incomplete gamma function (complement integral) 
//  igamc(a,x)   =   1 - igam(a,x)
//
//                            inf.
//                              -
//                     1       | |  -t  a-1
//               =   -----     |   e   t   dt.
//                    -      | |
//                   | (a)    -
//                             x
//
//

// In this implementation both arguments must be positive.
// The integral is evaluated by either a power series or
// continued fraction expansion, depending on the relative
// values of a and x.

double igamc( double a, double x ) 
{

   double ans, ax, c, yc, r, t, y, z;
   double pk, pkm1, pkm2, qk, qkm1, qkm2;

   // LM: for negative values returns 0.0
   // This is correct if a is a negative integer since Gamma(-n) = +/- inf
   if (a <= 0)  return 0.0;  

   if (x <= 0) return 1.0;

   if( (x < 1.0) || (x < a) )
      return( 1.0 - igam(a,x) );

   ax = a * std::log(x) - x - lgam(a);
   if( ax < -kMAXLOG )
      return( 0.0 );

   ax = std::exp(ax);

/* continued fraction */
   y = 1.0 - a;
   z = x + y + 1.0;
   c = 0.0;
   pkm2 = 1.0;
   qkm2 = x;
   pkm1 = x + 1.0;
   qkm1 = z * x;
   ans = pkm1/qkm1;

   do
   { 
      c += 1.0;
      y += 1.0;
      z += 2.0;
      yc = y * c;
      pk = pkm1 * z  -  pkm2 * yc;
      qk = qkm1 * z  -  qkm2 * yc;
      if(qk)
      {
         r = pk/qk;
         t = std::abs( (ans - r)/r );
         ans = r;
      }
      else
         t = 1.0;
      pkm2 = pkm1;
      pkm1 = pk;
      qkm2 = qkm1;
      qkm1 = qk;
      if( std::abs(pk) > kBig )
      {
         pkm2 *= kBiginv;
         pkm1 *= kBiginv;
         qkm2 *= kBiginv;
         qkm1 *= kBiginv;
      }
   }
   while( t > kMACHEP );

   return( ans * ax );
}



/* left tail of incomplete gamma function:
 *
 *          inf.      k
 *   a  -x   -       x
 *  x  e     >   ----------
 *           -     -
 *          k=0   | (a+k+1)
 *
 */

double igam( double a, double x )
{
   double ans, ax, c, r;

   // LM: for negative values returns 1.0 instead of zero
   // This is correct if a is a negative integer since Gamma(-n) = +/- inf
   if (a <= 0)  return 1.0;  

   if (x <= 0)  return 0.0;

   if( (x > 1.0) && (x > a ) )
      return( 1.0 - igamc(a,x) );

/* Compute  x**a * exp(-x) / gamma(a)  */
   ax = a * std::log(x) - x - lgam(a);
   if( ax < -kMAXLOG )
      return( 0.0 );

   ax = std::exp(ax);

/* power series */
   r = a;
   c = 1.0;
   ans = 1.0;

   do
   {
      r += 1.0;
      c *= x/r;
      ans += c;
   }
   while( c/ans > kMACHEP );

   return( ans * ax/a );
}

/*---------------------------------------------------------------------------*/

/* Logarithm of gamma function */
/* A[]: Stirling's formula expansion of log gamma
 * B[], C[]: log gamma function between 2 and 3
 */

static double A[] = {
   8.11614167470508450300E-4,
   -5.95061904284301438324E-4,
   7.93650340457716943945E-4,
   -2.77777777730099687205E-3,
   8.33333333333331927722E-2
};

static double B[] = {
   -1.37825152569120859100E3,
   -3.88016315134637840924E4,
   -3.31612992738871184744E5,
   -1.16237097492762307383E6,
   -1.72173700820839662146E6,
   -8.53555664245765465627E5
};
   
static double C[] = {
/* 1.00000000000000000000E0, */
   -3.51815701436523470549E2,
   -1.70642106651881159223E4,
   -2.20528590553854454839E5,
   -1.13933444367982507207E6,
   -2.53252307177582951285E6,
   -2.01889141433532773231E6
};

double lgam( double x )
{
   double p, q, u, w, z;
   int i;

   int sgngam = 1;

   if (x >= std::numeric_limits<double>::infinity())
      return(std::numeric_limits<double>::infinity());

   if( x < -34.0 )
   {
      q = -x;
      w = lgam(q); 
      p = std::floor(q);
      if( p==q )//_unur_FP_same(p,q)
         return (std::numeric_limits<double>::infinity());
      i = (int) p;
      if( (i & 1) == 0 )
         sgngam = -1;
      else
         sgngam = 1;
      z = q - p;
      if( z > 0.5 )
      {
         p += 1.0;
         z = p - q;
      }
      z = q * std::sin( ROOT::Math::Pi() * z );
      if( z == 0 )
         return (std::numeric_limits<double>::infinity());
/*	z = log(ROOT::Math::Pi()) - log( z ) - w;*/
      z = std::log(ROOT::Math::Pi()) - std::log( z ) - w;
      return( z );
   }

   if( x < 13.0 )
   {
      z = 1.0;
      p = 0.0;
      u = x;
      while( u >= 3.0 )
      {
         p -= 1.0;
         u = x + p;
         z *= u;
      }
      while( u < 2.0 )
      {
         if( u == 0 )
            return (std::numeric_limits<double>::infinity());
         z /= u;
         p += 1.0;
         u = x + p;
      }
      if( z < 0.0 )
      {
         sgngam = -1;
         z = -z;
      }
      else
         sgngam = 1;
      if( u == 2.0 )
         return( std::log(z) );
      p -= 2.0;
      x = x + p;
      p = x * Polynomialeval(x, B, 5 ) / Polynomial1eval( x, C, 6);
      return( std::log(z) + p );
   }

   if( x > kMAXLGM )
      return( sgngam * std::numeric_limits<double>::infinity() );

   q = ( x - 0.5 ) * std::log(x) - x + LS2PI;
   if( x > 1.0e8 )
      return( q );

   p = 1.0/(x*x);
   if( x >= 1000.0 )
      q += ((   7.9365079365079365079365e-4 * p
                - 2.7777777777777777777778e-3) *p
            + 0.0833333333333333333333) / x;
   else
      q += Polynomialeval( p, A, 4 ) / x;
   return( q );
}

/*---------------------------------------------------------------------------*/
static double P[] = {
   1.60119522476751861407E-4,
   1.19135147006586384913E-3,
   1.04213797561761569935E-2,
   4.76367800457137231464E-2,
   2.07448227648435975150E-1,
   4.94214826801497100753E-1,
   9.99999999999999996796E-1
};
static double Q[] = {
   -2.31581873324120129819E-5,
   5.39605580493303397842E-4,
   -4.45641913851797240494E-3,
   1.18139785222060435552E-2,
   3.58236398605498653373E-2,
   -2.34591795718243348568E-1,
   7.14304917030273074085E-2,
   1.00000000000000000320E0
};

/* Stirling's formula for the gamma function */
static double STIR[5] = {
   7.87311395793093628397E-4,
   -2.29549961613378126380E-4,
   -2.68132617805781232825E-3,
   3.47222221605458667310E-3,
   8.33333333333482257126E-2,
};

#define SQTPI   std::sqrt(2*ROOT::Math::Pi())        /* sqrt(2*pi) */
/* Stirling formula for the gamma function */
static double stirf( double x)
{
   double y, w, v;

   w = 1.0/x;
   w = 1.0 + w * Polynomialeval( w, STIR, 4 );
   y = exp(x);

/*   #define kMAXSTIR kMAXLOG/log(kMAXLOG)  */

   if( x > kMAXSTIR )
   { /* Avoid overflow in pow() */
      v = pow( x, 0.5 * x - 0.25 );
      y = v * (v / y);
   }
   else
   {
      y = pow( x, x - 0.5 ) / y;
   }
   y = SQTPI * y * w;
   return( y );
}

double gamma( double x )
{
   double p, q, z;
   int i;

   int sgngam = 1;

   if (x >=std::numeric_limits<double>::infinity())
      return(x);

   q = std::abs(x);

   if( q > 33.0 )
   {
      if( x < 0.0 )
      {
         p = std::floor(q);
         if( p == q )
         {
            return( sgngam * std::numeric_limits<double>::infinity());
         }
         i = (int) p;
         if( (i & 1) == 0 )
            sgngam = -1;
         z = q - p;
         if( z > 0.5 )
         {
            p += 1.0;
            z = q - p;
         }
         z = q * std::sin( ROOT::Math::Pi() * z );
         if( z == 0 )
         {
            return( sgngam * std::numeric_limits<double>::infinity());
         }
         z = std::abs(z);
         z = ROOT::Math::Pi()/(z * stirf(q) );
      }
      else
      {
         z = stirf(x);
      }
      return( sgngam * z );
   }

   z = 1.0;
   while( x >= 3.0 )
   {
      x -= 1.0;
      z *= x;
   }

   while( x < 0.0 )
   {
      if( x > -1.E-9 )
         goto small;
      z /= x;
      x += 1.0;
   }

   while( x < 2.0 )
   {
      if( x < 1.e-9 )
         goto small;
      z /= x;
      x += 1.0;
   }

   if( x == 2.0 )
      return(z);

   x -= 2.0;
   p = Polynomialeval( x, P, 6 );
   q = Polynomialeval( x, Q, 7 );
   return( z * p / q );

small:
   if( x == 0 )
      return( std::numeric_limits<double>::infinity() );
   else
      return( z/((1.0 + 0.5772156649015329 * x) * x) );
}

/*---------------------------------------------------------------------------*/

//#define kMAXLGM 2.556348e305

/*---------------------------------------------------------------------------*/
/* implementation based on cephes log gamma */
double beta(double z, double w)
{
   return std::exp(ROOT::Math::Cephes::lgam(z) + ROOT::Math::Cephes::lgam(w)- ROOT::Math::Cephes::lgam(z+w) );
}


/*---------------------------------------------------------------------------*/
/* inplementation of the incomplete beta function */
/**
 * DESCRIPTION:
 *
 * Returns incomplete beta integral of the arguments, evaluated
 * from zero to x.  The function is defined as
 *
 *                  x
 *     -            -
 *    | (a+b)      | |  a-1     b-1
 *  -----------    |   t   (1-t)   dt.
 *   -     -     | |
 *  | (a) | (b)   -
 *                 0
 *
 * The domain of definition is 0 <= x <= 1.  In this
 * implementation a and b are restricted to positive values.
 * The integral from x to 1 may be obtained by the symmetry
 * relation
 *
 *    1 - incbet( a, b, x )  =  incbet( b, a, 1-x ).
 *
 * The integral is evaluated by a continued fraction expansion
 * or, when b*x is small, by a power series.
 *
 * ACCURACY:
 *
 * Tested at uniformly distributed random points (a,b,x) with a and b
 * in "domain" and x between 0 and 1.
 *                                        Relative error
 * arithmetic   domain     # trials      peak         rms
 *    IEEE      0,5         10000       6.9e-15     4.5e-16
 *    IEEE      0,85       250000       2.2e-13     1.7e-14
 *    IEEE      0,1000      30000       5.3e-12     6.3e-13
 *    IEEE      0,10000    250000       9.3e-11     7.1e-12
 *    IEEE      0,100000    10000       8.7e-10     4.8e-11
 * Outputs smaller than the IEEE gradual underflow threshold
 * were excluded from these statistics.
 *
 * ERROR MESSAGES:
 *   message         condition      value returned
 * incbet domain      x<0, x>1          0.0
 * incbet underflow                     0.0
 *
 *  Cephes Math Library, Release 2.8:  June, 2000
 * Copyright 1984, 1995, 2000 by Stephen L. Moshier
 */


double incbet( double aa, double bb, double xx )
{
   double a, b, t, x, xc, w, y;
   int flag;

   if( aa <= 0.0 || bb <= 0.0 )
      return( 0.0 );

   // LM: changed: for X > 1 return 1.
   if  (xx <= 0.0)  return( 0.0 );
   if ( xx >= 1.0)  return( 1.0 );

   flag = 0;

/* - to test if that way is better for large b/  (comment out from Cephes version)
   if( (bb * xx) <= 1.0 && xx <= 0.95)
   {
   t = pseries(aa, bb, xx);
   goto done;
   }

**/
   w = 1.0 - xx;

/* Reverse a and b if x is greater than the mean. */
/* aa,bb > 1 -> sharp rise at x=aa/(aa+bb) */
   if( xx > (aa/(aa+bb)) )
   {
      flag = 1;
      a = bb;
      b = aa;
      xc = xx;
      x = w;
   }
   else
   {
      a = aa;
      b = bb;
      xc = w;
      x = xx;
   }

   if( flag == 1 && (b * x) <= 1.0 && x <= 0.95)
   {
      t = pseries(a, b, x);
      goto done;
   }

/* Choose expansion for better convergence. */
   y = x * (a+b-2.0) - (a-1.0);
   if( y < 0.0 )
      w = incbcf( a, b, x );
   else
      w = incbd( a, b, x ) / xc;

/* Multiply w by the factor
   a      b   _             _     _
   x  (1-x)   | (a+b) / ( a | (a) | (b) ) .   */

   y = a * std::log(x);
   t = b * std::log(xc);
   if( (a+b) < kMAXSTIR && std::abs(y) < kMAXLOG && std::abs(t) < kMAXLOG )
   {
      t = pow(xc,b);
      t *= pow(x,a);
      t /= a;
      t *= w;
      t *= ROOT::Math::Cephes::gamma(a+b) / (ROOT::Math::Cephes::gamma(a) * ROOT::Math::Cephes::gamma(b));
      goto done;
   }
/* Resort to logarithms.  */
   y += t + lgam(a+b) - lgam(a) - lgam(b);
   y += std::log(w/a);
   if( y < kMINLOG )
      t = 0.0;
   else
      t = std::exp(y);

done:

   if( flag == 1 )
   {
      if( t <= kMACHEP )
         t = 1.0 - kMACHEP;
      else
         t = 1.0 - t;
   }
   return( t );
}
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/

/* Continued fraction expansion #1
 * for incomplete beta integral
 */

double incbcf( double a, double b, double x )
{
   double xk, pk, pkm1, pkm2, qk, qkm1, qkm2;
   double k1, k2, k3, k4, k5, k6, k7, k8;
   double r, t, ans, thresh;
   int n;

   k1 = a;
   k2 = a + b;
   k3 = a;
   k4 = a + 1.0;
   k5 = 1.0;
   k6 = b - 1.0;
   k7 = k4;
   k8 = a + 2.0;

   pkm2 = 0.0;
   qkm2 = 1.0;
   pkm1 = 1.0;
   qkm1 = 1.0;
   ans = 1.0;
   r = 1.0;
   n = 0;
   thresh = 3.0 * kMACHEP;
   do
   {
	
      xk = -( x * k1 * k2 )/( k3 * k4 );
      pk = pkm1 +  pkm2 * xk;
      qk = qkm1 +  qkm2 * xk;
      pkm2 = pkm1;
      pkm1 = pk;
      qkm2 = qkm1;
      qkm1 = qk;

      xk = ( x * k5 * k6 )/( k7 * k8 );
      pk = pkm1 +  pkm2 * xk;
      qk = qkm1 +  qkm2 * xk;
      pkm2 = pkm1;
      pkm1 = pk;
      qkm2 = qkm1;
      qkm1 = qk;

      if( qk !=0 )
         r = pk/qk;
      if( r != 0 )
      {
         t = std::abs( (ans - r)/r );
         ans = r;
      }
      else
         t = 1.0;

      if( t < thresh )
         goto cdone;

      k1 += 1.0;
      k2 += 1.0;
      k3 += 2.0;
      k4 += 2.0;
      k5 += 1.0;
      k6 -= 1.0;
      k7 += 2.0;
      k8 += 2.0;

      if( (std::abs(qk) + std::abs(pk)) > kBig )
      {
         pkm2 *= kBiginv;
         pkm1 *= kBiginv;
         qkm2 *= kBiginv;
         qkm1 *= kBiginv;
      }
      if( (std::abs(qk) < kBiginv) || (std::abs(pk) < kBiginv) )
      {
         pkm2 *= kBig;
         pkm1 *= kBig;
         qkm2 *= kBig;
         qkm1 *= kBig;
      }
   }
   while( ++n < 300 );

cdone:
   return(ans);
}


/*---------------------------------------------------------------------------*/

/* Continued fraction expansion #2
 * for incomplete beta integral
 */

double incbd( double a, double b, double x )
{
   double xk, pk, pkm1, pkm2, qk, qkm1, qkm2;
   double k1, k2, k3, k4, k5, k6, k7, k8;
   double r, t, ans, z, thresh;
   int n;

   k1 = a;
   k2 = b - 1.0;
   k3 = a;
   k4 = a + 1.0;
   k5 = 1.0;
   k6 = a + b;
   k7 = a + 1.0;;
   k8 = a + 2.0;

   pkm2 = 0.0;
   qkm2 = 1.0;
   pkm1 = 1.0;
   qkm1 = 1.0;
   z = x / (1.0-x);
   ans = 1.0;
   r = 1.0;
   n = 0;
   thresh = 3.0 * kMACHEP;
   do
   {
	
      xk = -( z * k1 * k2 )/( k3 * k4 );
      pk = pkm1 +  pkm2 * xk;
      qk = qkm1 +  qkm2 * xk;
      pkm2 = pkm1;
      pkm1 = pk;
      qkm2 = qkm1;
      qkm1 = qk;

      xk = ( z * k5 * k6 )/( k7 * k8 );
      pk = pkm1 +  pkm2 * xk;
      qk = qkm1 +  qkm2 * xk;
      pkm2 = pkm1;
      pkm1 = pk;
      qkm2 = qkm1;
      qkm1 = qk;

      if( qk != 0 )
         r = pk/qk;
      if( r != 0 )
      {
         t = std::abs( (ans - r)/r );
         ans = r;
      }
      else
         t = 1.0;

      if( t < thresh )
         goto cdone;

      k1 += 1.0;
      k2 -= 1.0;
      k3 += 2.0;
      k4 += 2.0;
      k5 += 1.0;
      k6 += 1.0;
      k7 += 2.0;
      k8 += 2.0;

      if( (std::abs(qk) + std::abs(pk)) > kBig )
      {
         pkm2 *= kBiginv;
         pkm1 *= kBiginv;
         qkm2 *= kBiginv;
         qkm1 *= kBiginv;
      }
      if( (std::abs(qk) < kBiginv) || (std::abs(pk) < kBiginv) )
      {
         pkm2 *= kBig;
         pkm1 *= kBig;
         qkm2 *= kBig;
         qkm1 *= kBig;
      }
   }
   while( ++n < 300 );
cdone:
   return(ans);
}


/*---------------------------------------------------------------------------*/

/* Power series for incomplete beta integral.
   Use when b*x is small and x not too close to 1.  */

double pseries( double a, double b, double x )
{
   double s, t, u, v, n, t1, z, ai;

   ai = 1.0 / a;
   u = (1.0 - b) * x;
   v = u / (a + 1.0);
   t1 = v;
   t = u;
   n = 2.0;
   s = 0.0;
   z = kMACHEP * ai;
   while( std::abs(v) > z )
   {
      u = (n - b) * x / n;
      t *= u;
      v = t / (a + n);
      s += v; 
      n += 1.0;
   }
   s += t1;
   s += ai;

   u = a * log(x);
   if( (a+b) < kMAXSTIR && std::abs(u) < kMAXLOG )
   {
      t = gamma(a+b)/(gamma(a)*gamma(b));
      s = s * t * pow(x,a);
   }
   else
   {
      t = lgam(a+b) - lgam(a) - lgam(b) + u + std::log(s);
      if( t < kMINLOG )
         s = 0.0;
      else
         s = std::exp(t);
   }
   return(s);
}

/*---------------------------------------------------------------------------*/


/*---------------------------------------------------------------------------*/
/* for evaluation of error function */
/*---------------------------------------------------------------------------*/

static double erfP[] = {
   2.46196981473530512524E-10,
   5.64189564831068821977E-1,
   7.46321056442269912687E0,
   4.86371970985681366614E1,
   1.96520832956077098242E2,
   5.26445194995477358631E2,
   9.34528527171957607540E2,
   1.02755188689515710272E3,
   5.57535335369399327526E2
};
static double erfQ[] = {
/* 1.00000000000000000000E0,*/
   1.32281951154744992508E1,
   8.67072140885989742329E1,
   3.54937778887819891062E2,
   9.75708501743205489753E2,
   1.82390916687909736289E3,
   2.24633760818710981792E3,
   1.65666309194161350182E3,
   5.57535340817727675546E2
};
static double erfR[] = {
   5.64189583547755073984E-1,
   1.27536670759978104416E0,
   5.01905042251180477414E0,
   6.16021097993053585195E0,
   7.40974269950448939160E0,
   2.97886665372100240670E0
};
static double erfS[] = {
/* 1.00000000000000000000E0,*/
   2.26052863220117276590E0,
   9.39603524938001434673E0,
   1.20489539808096656605E1,
   1.70814450747565897222E1,
   9.60896809063285878198E0,
   3.36907645100081516050E0
};
static double erfT[] = {
   9.60497373987051638749E0,
   9.00260197203842689217E1,
   2.23200534594684319226E3,
   7.00332514112805075473E3,
   5.55923013010394962768E4
};
static double erfU[] = {
/* 1.00000000000000000000E0,*/
   3.35617141647503099647E1,
   5.21357949780152679795E2,
   4.59432382970980127987E3,
   2.26290000613890934246E4,
   4.92673942608635921086E4
};

/*---------------------------------------------------------------------------*/
/* complementary error function */
/* For small x, erfc(x) = 1 - erf(x); otherwise rational */
/* approximations are computed. */
 

double erfc( double a )
{
   double p,q,x,y,z;


   if( a < 0.0 )
      x = -a;
   else
      x = a;

   if( x < 1.0 )
      return( 1.0 - ROOT::Math::Cephes::erf(a) );

   z = -a * a;

   if( z < -kMAXLOG )
   {
   under:
      if( a < 0 )
         return( 2.0 );
      else
         return( 0.0 );
   }

   z = exp(z);

   if( x < 8.0 )
   {
      p = Polynomialeval( x, erfP, 8 );
      q = Polynomial1eval( x, erfQ, 8 );
   }
   else
   {
      p = Polynomialeval( x, erfR, 5 );
      q = Polynomial1eval( x, erfS, 6 );
   }
   y = (z * p)/q;

   if( a < 0 )
      y = 2.0 - y;

   if( y == 0 )
      goto under;

   return(y);
}

/*---------------------------------------------------------------------------*/
/* error function */
/* For 0 <= |x| < 1, erf(x) = x * P4(x**2)/Q5(x**2); otherwise */
/* erf(x) = 1 - erfc(x). */

double erf( double x)
{
   double y, z;

   if( std::abs(x) > 1.0 )
      return( 1.0 - ROOT::Math::Cephes::erfc(x) );
   z = x * x;
   y = x * Polynomialeval( z, erfT, 4 ) / Polynomial1eval( z, erfU, 5 );
   return( y );

}

} // end namespace Cephes


/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/* Routines used within this implementation */


/*
 * calculates a value of a polynomial of the form:
 * a[0]x^N+a[1]x^(N-1) + ... + a[N]
*/
double Polynomialeval(double x, double* a, unsigned int N)
{
   if (N==0) return a[0];
   else
   {
      double pom = a[0];
      for (unsigned int i=1; i <= N; i++)
         pom = pom *x + a[i];
      return pom;
   }
}

/*
 * calculates a value of a polynomial of the form:
 * x^N+a[0]x^(N-1) + ... + a[N-1]
*/
double Polynomial1eval(double x, double* a, unsigned int N)
{
   if (N==0) return a[0];
   else
   {
      double pom = x + a[0];
      for (unsigned int i=1; i < N; i++)
         pom = pom *x + a[i];
      return pom;
   }
}


} // end namespace Math
} // end namespace ROOT

