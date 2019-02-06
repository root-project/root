// @(#)root/mathcore:$Id$
// Author: L. Moneta Tue Nov 14 15:44:38 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// mathematical constants like Pi

#ifndef ROOT_Math_Math
#define ROOT_Math_Math

#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#endif

#include <cmath>

#if defined(__sun) || defined(_MSC_VER)
//Microsoft and solaris definition of cmath does not include math.h which has the definitions of numerical constants
#include <math.h>
#endif


#ifdef HAVE_NO_EXPM1
// needed to implement expm1
#include <limits>
#endif


#ifndef M_PI

#define M_PI       3.14159265358979323846264338328      // Pi
#endif

#ifndef M_PI_2
#define M_PI_2     1.57079632679489661923132169164      // Pi/2
#endif

#ifndef M_PI_4
#define M_PI_4     0.78539816339744830961566084582      // Pi/4
#endif

/**
   \namespace ROOT
   Namespace for new ROOT classes and functions
 */

namespace ROOT {

/**
\namespace Math
Namespace for new Math classes and functions.
See the \ref Math "Math Libraries" page for a detailed description.
*/

namespace Math {
// Enable Vc/VecCore template instantiations to replace std math functions.
//
// Vc declares `std::sqrt(Vc-type)`. To use this for Vc-`SCALAR`s, the call
// to `sqrt()` must only be resolved at the template instantiation time, when
// the Vc headers are guaranteed to be included, and thus its `sqrt()`
// overloads have been declared.
// The trick is to keep sqrt() dependent (on its argument type) by making it
// an unqualified name. The `std::` of `std::sqrt()` makes it a qualified
// name, so the code here has to use `sqrt()`, not `std::sqrt()`. To still
// find `std::sqrt()` we pull `std::sqrt()` into the surrounding namespace.
//
// We don't want to use 'using namespace std' because it would polute the including headers.
using std::atan2;
using std::cos;
using std::cosh;
using std::exp;
using std::floor;
using std::log;
using std::pow;
using std::sin;
using std::sinh;
using std::sqrt;
using std::tan;

/**
    Mathematical constants
*/
inline double Pi()
{
   return M_PI;
   }

   /**
       declarations for functions which are not implemented by some compilers
   */

   /// log(1+x) with error cancelatio when x is small
   inline double log1p(double x)
   {
#ifndef HAVE_NO_LOG1P
   return ::log1p(x);
#else
   // if log1p is not in c math library
  volatile double y;
  y = 1 + x;
  return std::log(y) - ((y-1)-x)/y ;  /* cancels errors with IEEE arithmetic */
#endif
}
/// exp(x) -1 with error cancellation when x is small
inline double expm1( double x) {
#ifndef HAVE_NO_EXPM1
   return ::expm1(x);
#else
   // compute using taylor expansion until difference is less than epsilon
   // use for values smaller than 0.5 (for larger (exp(x)-1 is fine
   if (std::abs(x) < 0.5)
   {
       // taylor series S = x + (1/2!) x^2 + (1/3!) x^3 + ...

      double i = 1.0;
      double sum = x;
      double term = x / 1.0;
      do {
         i++ ;
         term *= x/i;
         sum += term;
      }
      while (std::abs(term) > std::abs(sum) * std::numeric_limits<double>::epsilon() ) ;

      return sum ;
   }
   else
   {
      return std::exp(x) - 1;
   }
#endif
}

   } // end namespace Math

} // end namespace ROOT





#endif /* ROOT_Math_Math */
