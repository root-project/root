// @(#)root/mathcore:$Id$
// Authors: Andras Zsenei & Lorenzo Moneta   06/2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/



/**

Special mathematical functions.
The naming and numbering of the functions is taken from
<A HREF="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1687.pdf">
Matt Austern,
(Draft) Technical Report on Standard Library Extensions,
N1687=04-0127, September 10, 2004</A>

@author Created by Andras Zsenei on Mon Nov 8 2004

@defgroup SpecFunc Special functions
@ingroup MathCore
@ingroup MathMore

*/

#ifndef ROOT_Math_SpecFuncMathCore
#define ROOT_Math_SpecFuncMathCore


namespace ROOT {
namespace Math {


   /** @name Special Functions from MathCore */


   /**

   Error function encountered in integrating the normal distribution.

   \f[ erf(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2} dt  \f]

   For detailed description see
   <A HREF="http://mathworld.wolfram.com/Erf.html">
   Mathworld</A>.
   The implementation used is that of
   <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_7.html#SEC102">GSL</A>.
   This function is provided only for convenience,
   in case your standard C++ implementation does not support
   it. If it does, please use these standard version!

   @ingroup SpecFunc

   */
   // (26.x.21.1) error function

   double erf(double x);



   /**

   Complementary error function.

   \f[ erfc(x) = 1 - erf(x) = \frac{2}{\sqrt{\pi}} \int_{x}^{\infty} e^{-t^2} dt  \f]

   For detailed description see <A HREF="http://mathworld.wolfram.com/Erfc.html"> Mathworld</A>.
   The implementation used is that of  <A HREF="http://www.netlib.org/cephes">Cephes</A> from S. Moshier.

   @ingroup SpecFunc

   */
   // (26.x.21.2) complementary error function

   double erfc(double x);


   /**

   The gamma function is defined to be the extension of the
   factorial to real numbers.

   \f[ \Gamma(x) = \int_{0}^{\infty} t^{x-1} e^{-t} dt  \f]

   For detailed description see
   <A HREF="http://mathworld.wolfram.com/GammaFunction.html">
   Mathworld</A>.
   The implementation used is that of  <A HREF="http://www.netlib.org/cephes">Cephes</A> from S. Moshier.

   @ingroup SpecFunc

   */
   // (26.x.18) gamma function

   double tgamma(double x);


   /**

   Calculates the logarithm of the gamma function

   The implementation used is that of  <A HREF="http://www.netlib.org/cephes">Cephes</A> from S. Moshier.
   @ingroup SpecFunc

   */
   double lgamma(double x);


   /**
      Calculates the normalized (regularized) lower incomplete gamma function (lower integral)

      \f[ P(a, x) = \frac{ 1} {\Gamma(a) } \int_{0}^{x} t^{a-1} e^{-t} dt  \f]


      For a detailed description see
      <A HREF="http://mathworld.wolfram.com/RegularizedGammaFunction.html">
      Mathworld</A>.
      The implementation used is that of  <A HREF="http://www.netlib.org/cephes">Cephes</A> from S. Moshier.
      In this implementation both a and x must be positive. If a is negative 1.0 is returned for every x.
      This is correct only if a is negative integer.
      For a>0 and x<0  0 is returned (this is correct only for a>0 and x=0).

      @ingroup SpecFunc
   */
   double inc_gamma(double a, double x );

   /**
      Calculates the normalized (regularized) upper incomplete gamma function (upper integral)

      \f[ Q(a, x) = \frac{ 1} {\Gamma(a) } \int_{x}^{\infty} t^{a-1} e^{-t} dt  \f]


      For a detailed description see
      <A HREF="http://mathworld.wolfram.com/RegularizedGammaFunction.html">
      Mathworld</A>.
      The implementation used is that of  <A HREF="http://www.netlib.org/cephes">Cephes</A> from S. Moshier.
      In this implementation both a and x must be positive. If a is negative, 0 is returned for every x.
      This is correct only if a is negative integer.
      For a>0 and x<0  1 is returned (this is correct only for a>0 and x=0).

      @ingroup SpecFunc
   */
   double inc_gamma_c(double a, double x );

   /**

   Calculates the beta function.

   \f[ B(x,y) = \frac{\Gamma(x) \Gamma(y)}{\Gamma(x+y)} \f]

   for x>0 and y>0. For detailed description see
   <A HREF="http://mathworld.wolfram.com/BetaDistribution.html">
   Mathworld</A>.

   @ingroup SpecFunc

   */
   // [5.2.1.3] beta function

   double beta(double x, double y);


   /**

   Calculates the normalized (regularized) incomplete beta function.

   \f[ B(x, a, b ) =  \frac{ \int_{0}^{x} u^{a-1} (1-u)^{b-1} du } { B(a,b) } \f]

   for 0<=x<=1,  a>0,  and b>0. For detailed description see
   <A HREF="http://mathworld.wolfram.com/RegularizedBetaFunction.html">
   Mathworld</A>.
   The implementation used is that of  <A HREF="http://www.netlib.org/cephes">Cephes</A> from S. Moshier.

   @ingroup SpecFunc

   */

   double inc_beta( double x, double a, double b);




  /**

  Calculates the sine integral.

  \f[ Si(x) = - \int_{0}^{x} \frac{\sin t}{t} dt \f]

  For detailed description see
  <A HREF="http://mathworld.wolfram.com/SineIntegral.html">
  Mathworld</A>. The implementation used is that of
  <A HREF="https://cern-tex.web.cern.ch/cern-tex/shortwrupsdir/c336/top.html">
  CERNLIB</A>,
  based on Y.L. Luke, The special functions and their approximations, v.II, (Academic Press, New York l969) 325-326.


  @ingroup SpecFunc

  */

  double sinint(double x);




  /**

  Calculates the real part of the cosine integral Re(Ci).

  For x<0, the imaginary part is \f$\pi i\f$ and has to be added by the user,
  for x>0 the imaginary part of Ci(x) is 0.

  \f[ Ci(x) = - \int_{x}^{\infty} \frac{\cos t}{t} dt = \gamma + \ln x + \int_{0}^{x} \frac{\cos t - 1}{t} dt\f]

  For detailed description see
  <A HREF="http://mathworld.wolfram.com/CosineIntegral.html">
  Mathworld</A>. The implementation used is that of
  <A HREF="https://cern-tex.web.cern.ch/cern-tex/shortwrupsdir/c336/top.html">
  CERNLIB</A>,
  based on Y.L. Luke, The special functions and their approximations, v.II, (Academic Press, New York l969) 325-326.


  @ingroup SpecFunc

  */

  double cosint(double x);




} // namespace Math
} // namespace ROOT


#endif // ROOT_Math_SpecFuncMathCore
