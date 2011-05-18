// @(#)root/mathmore:$Id$
// Authors: L. Moneta, A. Zsenei   08/2005 



 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2004 ROOT Foundation,  CERN/PH-SFT                   *
  *                                                                    *
  * This library is free software; you can redistribute it and/or      *
  * modify it under the terms of the GNU General Public License        *
  * as published by the Free Software Foundation; either version 2     *
  * of the License, or (at your option) any later version.             *
  *                                                                    *
  * This library is distributed in the hope that it will be useful,    *
  * but WITHOUT ANY WARRANTY; without even the implied warranty of     *
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU   *
  * General Public License for more details.                           *
  *                                                                    *
  * You should have received a copy of the GNU General Public License  *
  * along with this library (see file COPYING); if not, write          *
  * to the Free Software Foundation, Inc., 59 Temple Place, Suite      *
  * 330, Boston, MA 02111-1307 USA, or contact the author.             *
  *                                                                    *
  **********************************************************************/


#if defined(__CINT__) && !defined(__MAKECINT__)
// avoid to include header file when using CINT 
#ifndef _WIN32
#include "../lib/libMathMore.so"
#else
#include "../bin/libMathMore.dll"
#endif

#else


#ifndef ROOT_Math_QuantFuncMathMore
#define ROOT_Math_QuantFuncMathMore


namespace ROOT {
namespace Math {



  /** @defgroup QuantFunc Quantile Functions 
   *  @ingroup StatFunc 
   *
   *  Inverse functions of the cumulative distribution functions 
   *  and the inverse of the complement of the cumulative distribution functions 
   *  for various distributions.
   *  The functions with the extension <em>_quantile</em> calculate the
   *  inverse of the <em>_cdf</em> function, the 
   *  lower tail integral of the probability density function
   *  \f$D^{-1}(z)\f$ where
   *
   *  \f[ D(x) = \int_{-\infty}^{x} p(x') dx' \f]
   *
   *  while those with the <em>_quantile_c</em> extension calculate the 
   *  inverse of the <em>_cdf_c</em> functions, the upper tail integral of the probability 
   *  density function \f$D^{-1}(z) \f$ where
   *
   *  \f[ D(x) = \int_{x}^{+\infty} p(x') dx' \f]
   *
   * The implementation used is that of 
   * <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Random-Number-Distributions.html">GSL</A>.
   *
   * <strong>NOTE:</strong> In the old releases (< 5.14) the <em>_quantile</em> functions were called 
   * <em>_quant_inv</em> and the <em>_quantile_c</em> functions were called 
   * <em>_prob_inv</em>. 
   * These names are currently kept for backward compatibility, but 
   * their usage is deprecated.
   */

   /** @name Quantile Functions from MathMore 
   * The implementation used is that of 
   * <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Random-Number-Distributions.html">GSL</A>.
   */ 

  //@{


  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the upper tail of Student's t-distribution
  (#tdistribution_cdf_c). For detailed description see 
  <A HREF="http://mathworld.wolfram.com/Studentst-Distribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC305">GSL</A>.
  
  @ingroup QuantFunc

  */

  double tdistribution_quantile_c(double z, double r);




  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the lower tail of Student's t-distribution
  (#tdistribution_cdf). For detailed description see 
  <A HREF="http://mathworld.wolfram.com/Studentst-Distribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC305">GSL</A>.
  
  @ingroup QuantFunc

  */

  double tdistribution_quantile(double z, double r);


#ifdef HAVE_OLD_STAT_FUNC

  //@}
   /** @name Backward compatible functions */ 


   }
   inline double chisquared_quant_inv(double x, double r) {
      return chisquared_quantile  (x, r ); 
   }
   

   inline double gamma_quant_inv(double x, double alpha, double theta) {
      return gamma_quantile   (x, alpha, theta ); 
   }

   inline double tdistribution_prob_inv(double x, double r) {
      return tdistribution_quantile_c  (x, r ); 
   }

   inline double tdistribution_quant_inv(double x, double r) {
      return tdistribution_quantile    (x, r ); 
   }

#endif


} // namespace Math

namespace MathMore {



  /**

  Re-implementation in MathMore of the Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the lower tail of the \f$\chi^2\f$ distribution 
  with \f$r\f$ degrees of freedom (#chisquared_cdf). For detailed description see 
  <A HREF="http://mathworld.wolfram.com/Chi-SquaredDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC303">GSL</A>.

  @ingroup QuantFunc

  */

  double chisquared_quantile(double z, double r);




  /**

  Re-implementation in MathMore of the Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the lower tail of the gamma distribution
  (#gamma_cdf). For detailed description see 
  <A HREF="http://mathworld.wolfram.com/GammaDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC300">GSL</A>.
  
  @ingroup QuantFunc

  */

  double gamma_quantile(double z, double alpha, double theta);



} // end namespace MathMore
} // namespace ROOT



#endif // ROOT_Math_QuantFuncMathMore

#endif // if defined (__CINT__) && !defined(__MAKECINT__)
