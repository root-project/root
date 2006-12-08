// @(#)root/mathcore:$Name:  $:$Id: SpecFuncMathCore.h,v 1.1 2005/09/18 17:33:47 brun Exp $
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

*/



// TODO: 
// [5.2.1.18] Laguerre polynomials
// [5.2.1.22] spherical associated Legendre functions



#ifndef ROOT_Math_SpecFuncMathCore
#define ROOT_Math_SpecFuncMathCore


namespace ROOT {
namespace Math {




  /**

  Error function encountered in integrating the normal distribution.

  \f[ erf(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2} dt  \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/Erf.html">
  Mathworld</A>. The implementation used is that of 
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

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/Erfc.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_7.html#SEC103">GSL</A>.
  This function is provided only for convenience,
  in case your standard C++ implementation does not support
  it. If it does, please use these standard version!

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
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_7.html#SEC120">GSL</A>.
  This function is provided only for convenience,
  in case your standard C++ implementation does not support
  it. If it does, please use these standard version!

  @ingroup SpecFunc

  */
  // (26.x.18) gamma function

  double tgamma(double x);



  double lgamma(double x);


  /**
  
  Calculates the beta function.

  \f[ B(x,y) = \frac{\Gamma(x) \Gamma(y)}{\Gamma(x+y)} \f]

  for x>0 and y>0. For detailed description see 
  <A HREF="http://mathworld.wolfram.com/BetaDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_7.html#SEC120">GSL</A>.

  @ingroup SpecFunc

  */
  // [5.2.1.3] beta function

  double beta(double x, double y);



} // namespace Math
} // namespace ROOT


#endif // ROOT_Math_SpecFuncMathCore
