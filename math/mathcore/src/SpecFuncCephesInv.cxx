// inverse of gamma and beta from Cephes library
// see:  http://www.netlib.org/cephes
// 
// Copyright 1985, 1987, 2000 by Stephen L. Moshier



#include "Math/Error.h"

#include "SpecFuncCephes.h"

#include <cmath>

#include <limits> 

namespace ROOT { 
namespace Math { 

namespace Cephes { 



/*							
 *
 *	Inverse of Normal distribution function
 *
 *
 *
 * SYNOPSIS:
 *
 * double x, y, ndtri();
 *
 * x = ndtri( y );
 *
 *
 *
 * DESCRIPTION:
 *
 * Returns the argument, x, for which the area under the
 * Gaussian probability density function (integrated from
 * minus infinity to x) is equal to y.
 *
 *
 * For small arguments 0 < y < exp(-2), the program computes
 * z = sqrt( -2.0 * log(y) );  then the approximation is
 * x = z - log(z)/z  - (1/z) P(1/z) / Q(1/z).
 * There are two rational functions P/Q, one for 0 < y < exp(-32)
 * and the other for y up to exp(-2).  For larger arguments,
 * w = y - 0.5, and  x/sqrt(2pi) = w + w**3 R(w**2)/S(w**2)).
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain        # trials      peak         rms
 *    DEC      0.125, 1         5500       9.5e-17     2.1e-17
 *    DEC      6e-39, 0.135     3500       5.7e-17     1.3e-17
 *    IEEE     0.125, 1        20000       7.2e-16     1.3e-16
 *    IEEE     3e-308, 0.135   50000       4.6e-16     9.8e-17
 *
 *
 * ERROR MESSAGES:
 *
 *   message         condition    value returned
 * ndtri domain       x <= 0        -MAXNUM
 * ndtri domain       x >= 1         MAXNUM
 *
 */

/*
Cephes Math Library Release 2.8:  June, 2000
Copyright 1984, 1987, 1989, 2000 by Stephen L. Moshier
*/


static double s2pi = 2.50662827463100050242E0;
  
static double P0[5] = {
-5.99633501014107895267E1,
 9.80010754185999661536E1,
-5.66762857469070293439E1,
 1.39312609387279679503E1,
-1.23916583867381258016E0,
};
static double Q0[8] = {
 1.95448858338141759834E0,
 4.67627912898881538453E0,
 8.63602421390890590575E1,
-2.25462687854119370527E2,
 2.00260212380060660359E2,
-8.20372256168333339912E1,
 1.59056225126211695515E1,
-1.18331621121330003142E0,
};
static double P1[9] = {
 4.05544892305962419923E0,
 3.15251094599893866154E1,
 5.71628192246421288162E1,
 4.40805073893200834700E1,
 1.46849561928858024014E1,
 2.18663306850790267539E0,
-1.40256079171354495875E-1,
-3.50424626827848203418E-2,
-8.57456785154685413611E-4,
};
static double Q1[8] = {
 1.57799883256466749731E1,
 4.53907635128879210584E1,
 4.13172038254672030440E1,
 1.50425385692907503408E1,
 2.50464946208309415979E0,
-1.42182922854787788574E-1,
-3.80806407691578277194E-2,
-9.33259480895457427372E-4,
};
static double P2[9] = {
  3.23774891776946035970E0,
  6.91522889068984211695E0,
  3.93881025292474443415E0,
  1.33303460815807542389E0,
  2.01485389549179081538E-1,
  1.23716634817820021358E-2,
  3.01581553508235416007E-4,
  2.65806974686737550832E-6,
  6.23974539184983293730E-9,
};
static double Q2[8] = {
  6.02427039364742014255E0,
  3.67983563856160859403E0,
  1.37702099489081330271E0,
  2.16236993594496635890E-1,
  1.34204006088543189037E-2,
  3.28014464682127739104E-4,
  2.89247864745380683936E-6,
  6.79019408009981274425E-9,
};
double ndtri( double y0 )
{
   double x, y, z, y2, x0, x1;
   int code;
   if( y0 <= 0.0 )
      return( - std::numeric_limits<double>::infinity() );
   if( y0 >= 1.0 )
      return( + std::numeric_limits<double>::infinity() );
   code = 1;
   y = y0;
   if( y > (1.0 - 0.13533528323661269189) ) 
   {
      y = 1.0 - y;
      code = 0;
   }
   if( y > 0.13533528323661269189 )
   {
      y = y - 0.5;
      y2 = y * y;
      x = y + y * (y2 * Polynomialeval( y2, P0, 4)/ Polynomial1eval( y2, Q0, 8 ));
      x = x * s2pi; 
      return(x);
   }
   x = std::sqrt( -2.0 * std::log(y) );
   x0 = x - std::log(x)/x;
   z = 1.0/x;
   if( x < 8.0 ) 
      x1 = z * Polynomialeval( z, P1, 8 )/ Polynomial1eval ( z, Q1, 8 );
   else
      x1 = z * Polynomialeval( z, P2, 8 )/ Polynomial1eval( z, Q2, 8 );
   x = x0 - x1;
   if( code != 0 )
      x = -x;
   return( x );
}




/*							
 *
 *      Inverse of complemented imcomplete gamma integral
 *
 *
 *
 * SYNOPSIS:
 *
 * double a, x, p, igami();
 *
 * x = igami( a, p );
 *
 * DESCRIPTION:
 *
 * Given p, the function finds x such that
 *
 *  igamc( a, x ) = p.
 *
 * Starting with the approximate value
 *
 *         3
 *  x = a t
 *
 *  where
 *
 *  t = 1 - d - ndtri(p) sqrt(d)
 * 
 * and
 *
 *  d = 1/9a,
 *
 * the routine performs up to 10 Newton iterations to find the
 * root of igamc(a,x) - p = 0.
 *
 * ACCURACY:
 *
 * Tested at random a, p in the intervals indicated.
 *
 *                a        p                      Relative error:
 * arithmetic   domain   domain     # trials      peak         rms
 *    IEEE     0.5,100   0,0.5       100000       1.0e-14     1.7e-15
 *    IEEE     0.01,0.5  0,0.5       100000       9.0e-14     3.4e-15
 *    IEEE    0.5,10000  0,0.5        20000       2.3e-13     3.8e-14
 */
/*
Cephes Math Library Release 2.8:  June, 2000
Copyright 1984, 1987, 1995, 2000 by Stephen L. Moshier
*/

double igami( double a, double y0 )
{
   double x0, x1, x, yl, yh, y, d, lgm, dithresh;
   int i, dir;

   // check the domain
   if (a<= 0) { 
      MATH_ERROR_MSG("Cephes::igami","Wrong domain for parameter a (must be > 0)"); 
      return 0; 
   }
   if (y0 <= 0) { 
      //if (y0<0) MATH_ERROR_MSG("Cephes::igami","Wrong domain for y (must be in [0,1])"); 
      return std::numeric_limits<double>::infinity();
   }
   if (y0 >= 1) { 
      //if (y0>1) MATH_ERROR_MSG("Cephes::igami","Wrong domain for y (must be in [0,1])"); 
      return 0; 
   }
      

/* bound the solution */
   static double kMAXNUM = std::numeric_limits<double>::max(); 
   x0 = kMAXNUM;
   yl = 0;
   x1 = 0;
   yh = 1.0;
   dithresh = 5.0 * kMACHEP;

/* approximation to inverse function */
   d = 1.0/(9.0*a);
   y = ( 1.0 - d - ndtri(y0) * std::sqrt(d) );
   x = a * y * y * y;

   lgm = lgam(a);

   for( i=0; i<10; i++ )
   {
      if( x > x0 || x < x1 )
         goto ihalve;
      y = igamc(a,x);
      if( y < yl || y > yh )
         goto ihalve;
      if( y < y0 )
      {
         x0 = x;
         yl = y;
      }
      else
      {
         x1 = x;
         yh = y;
      }
/* compute the derivative of the function at this point */
      d = (a - 1.0) * std::log(x) - x - lgm;
      if( d < -kMAXLOG )
         goto ihalve;
      d = -std::exp(d);
/* compute the step to the next approximation of x */
      d = (y - y0)/d;
      if( std::fabs(d/x) < kMACHEP )
         goto done;
      x = x - d;
   }

/* Resort to interval halving if Newton iteration did not converge. */
ihalve:

   d = 0.0625;
   if( x0 == kMAXNUM )
   {
      if( x <= 0.0 )
         x = 1.0;
      while( x0 == kMAXNUM )
      {
         x = (1.0 + d) * x;
         y = igamc( a, x );
         if( y < y0 )
         {
            x0 = x;
            yl = y;
            break;
         }
         d = d + d;
      }
   }
   d = 0.5;
   dir = 0;

   for( i=0; i<400; i++ )
   {
      x = x1  +  d * (x0 - x1);
      y = igamc( a, x );
      lgm = (x0 - x1)/(x1 + x0);
      if( std::fabs(lgm) < dithresh )
         break;
      lgm = (y - y0)/y0;
      if( std::fabs(lgm) < dithresh )
         break;
      if( x <= 0.0 )
         break;
      if( y >= y0 )
      {
         x1 = x;
         yh = y;
         if( dir < 0 )
         {
            dir = 0;
            d = 0.5;
         }
         else if( dir > 1 )
            d = 0.5 * d + 0.5; 
         else
            d = (y0 - yl)/(yh - yl);
         dir += 1;
      }
      else
      {
         x0 = x;
         yl = y;
         if( dir > 0 )
         {
            dir = 0;
            d = 0.5;
         }
         else if( dir < -1 )
            d = 0.5 * d;
         else
            d = (y0 - yl)/(yh - yl);
         dir -= 1;
      }
   }

//    if( x == 0.0 )
//       mtherr( "igami", UNDERFLOW );

done:
   return( x );
}


/*							
 *
 *      Inverse of imcomplete beta integral
 *
 *
 *
 * SYNOPSIS:
 *
 * double a, b, x, y, incbi();
 *
 * x = incbi( a, b, y );
 *
 *
 *
 * DESCRIPTION:
 *
 * Given y, the function finds x such that
 *
 *  incbet( a, b, x ) = y .
 *
 * The routine performs interval halving or Newton iterations to find the
 * root of incbet(a,b,x) - y = 0.
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 *                x     a,b
 * arithmetic   domain  domain  # trials    peak       rms
 *    IEEE      0,1    .5,10000   50000    5.8e-12   1.3e-13
 *    IEEE      0,1   .25,100    100000    1.8e-13   3.9e-15
 *    IEEE      0,1     0,5       50000    1.1e-12   5.5e-15
 *    VAX       0,1    .5,100     25000    3.5e-14   1.1e-15
 * With a and b constrained to half-integer or integer values:
 *    IEEE      0,1    .5,10000   50000    5.8e-12   1.1e-13
 *    IEEE      0,1    .5,100    100000    1.7e-14   7.9e-16
 * With a = .5, b constrained to half-integer or integer values:
 *    IEEE      0,1    .5,10000   10000    8.3e-11   1.0e-11
 */

/*
Cephes Math Library Release 2.8:  June, 2000
Copyright 1984, 1996, 2000 by Stephen L. Moshier
*/


double incbi( double aa, double bb, double yy0 )
{
   double a, b, y0, d, y, x, x0, x1, lgm, yp, di, dithresh, yl, yh, xt;
   int i, rflg, dir, nflg;

   // check the domain
   if (aa<= 0) { 
      MATH_ERROR_MSG("Cephes::incbi","Wrong domain for parameter a (must be > 0)"); 
      return 0; 
   }
   if (bb<= 0) { 
      MATH_ERROR_MSG("Cephes::incbi","Wrong domain for parameter b (must be > 0)"); 
      return 0; 
   }


   i = 0;
   if( yy0 <= 0 )
      return(0.0);
   if( yy0 >= 1.0 )
      return(1.0);
   x0 = 0.0;
   yl = 0.0;
   x1 = 1.0;
   yh = 1.0;
   nflg = 0;

   if( aa <= 1.0 || bb <= 1.0 )
   {
      dithresh = 1.0e-6;
      rflg = 0;
      a = aa;
      b = bb;
      y0 = yy0;
      x = a/(a+b);
      y = incbet( a, b, x );
      goto ihalve;
   }
   else
   {
      dithresh = 1.0e-4;
   }
/* approximation to inverse function */

   yp = -ndtri(yy0);

   if( yy0 > 0.5 )
   {
      rflg = 1;
      a = bb;
      b = aa;
      y0 = 1.0 - yy0;
      yp = -yp;
   }
   else
   {
      rflg = 0;
      a = aa;
      b = bb;
      y0 = yy0;
   }

   lgm = (yp * yp - 3.0)/6.0;
   x = 2.0/( 1.0/(2.0*a-1.0)  +  1.0/(2.0*b-1.0) );
   d = yp * std::sqrt( x + lgm ) / x
      - ( 1.0/(2.0*b-1.0) - 1.0/(2.0*a-1.0) )
      * (lgm + 5.0/6.0 - 2.0/(3.0*x));
   d = 2.0 * d;
   if( d < kMINLOG )
   {
      x = 1.0;
      goto under;
   }
   x = a/( a + b * std::exp(d) );
   y = incbet( a, b, x );
   yp = (y - y0)/y0;
   if( std::fabs(yp) < 0.2 )
      goto newt;

/* Resort to interval halving if not close enough. */
ihalve:

   dir = 0;
   di = 0.5;
   for( i=0; i<100; i++ )
   {
      if( i != 0 )
      {
         x = x0  +  di * (x1 - x0);
         if( x == 1.0 )
            x = 1.0 - kMACHEP;
         if( x == 0.0 )
         {
            di = 0.5;
            x = x0  +  di * (x1 - x0);
            if( x == 0.0 )
               goto under;
         }
         y = incbet( a, b, x );
         yp = (x1 - x0)/(x1 + x0);
         if( std::fabs(yp) < dithresh )
            goto newt;
         yp = (y-y0)/y0;
         if( std::fabs(yp) < dithresh )
            goto newt;
      }
      if( y < y0 )
      {
         x0 = x;
         yl = y;
         if( dir < 0 )
         {
            dir = 0;
            di = 0.5;
         }
         else if( dir > 3 )
            di = 1.0 - (1.0 - di) * (1.0 - di);
         else if( dir > 1 )
            di = 0.5 * di + 0.5; 
         else
            di = (y0 - y)/(yh - yl);
         dir += 1;
         if( x0 > 0.75 )
         {
            if( rflg == 1 )
            {
               rflg = 0;
               a = aa;
               b = bb;
               y0 = yy0;
            }
            else
            {
               rflg = 1;
               a = bb;
               b = aa;
               y0 = 1.0 - yy0;
            }
            x = 1.0 - x;
            y = incbet( a, b, x );
            x0 = 0.0;
            yl = 0.0;
            x1 = 1.0;
            yh = 1.0;
            goto ihalve;
         }
      }
      else
      {
         x1 = x;
         if( rflg == 1 && x1 < kMACHEP )
         {
            x = 0.0;
            goto done;
         }
         yh = y;
         if( dir > 0 )
         {
            dir = 0;
            di = 0.5;
         }
         else if( dir < -3 )
            di = di * di;
         else if( dir < -1 )
            di = 0.5 * di;
         else
            di = (y - y0)/(yh - yl);
         dir -= 1;
      }
   }
   //mtherr( "incbi", PLOSS );
   if( x0 >= 1.0 )
   {
      x = 1.0 - kMACHEP;
      goto done;
   }
   if( x <= 0.0 )
   {
   under:
      //mtherr( "incbi", UNDERFLOW );
      x = 0.0;
      goto done;
   }

newt:

   if( nflg )
      goto done;
   nflg = 1;
   lgm = lgam(a+b) - lgam(a) - lgam(b);

   for( i=0; i<8; i++ )
   {
      /* Compute the function at this point. */
      if( i != 0 )
         y = incbet(a,b,x);
      if( y < yl )
      {
         x = x0;
         y = yl;
      }
      else if( y > yh )
      {
         x = x1;
         y = yh;
      }
      else if( y < y0 )
      {
         x0 = x;
         yl = y;
      }
      else
      {
         x1 = x;
         yh = y;
      }
      if( x == 1.0 || x == 0.0 )
         break;
      /* Compute the derivative of the function at this point. */
      d = (a - 1.0) * std::log(x) + (b - 1.0) * std::log(1.0-x) + lgm;
      if( d < kMINLOG )
         goto done;
      if( d > kMAXLOG )
         break;
      d = std::exp(d);
      /* Compute the step to the next approximation of x. */
      d = (y - y0)/d;
      xt = x - d;
      if( xt <= x0 )
      {
         y = (x - x0) / (x1 - x0);
         xt = x0 + 0.5 * y * (x - x0);
         if( xt <= 0.0 )
            break;
      }
      if( xt >= x1 )
      {
         y = (x1 - x) / (x1 - x0);
         xt = x1 - 0.5 * y * (x1 - x);
         if( xt >= 1.0 )
            break;
      }
      x = xt;
      if( std::fabs(d/x) < 128.0 * kMACHEP )
         goto done;
   }
/* Did not converge.  */
   dithresh = 256.0 * kMACHEP;
   goto ihalve;

done:

   if( rflg )
   {
      if( x <= kMACHEP )
         x = 1.0 - kMACHEP;
      else
         x = 1.0 - x;
   }
   return( x );
}

} // end namespace Cephes

} // end namespace Math
} // end namespace ROOT
