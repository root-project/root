// @(#)root/mathmore:$Name:  $:$Id: ProbFuncInv.h,v 1.4 2006/12/06 17:53:47 moneta Exp $
// Authors: L. Moneta, A. Zsenei   08/2005 


// Authors: Andras Zsenei & Lorenzo Moneta   08/2005 


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


#ifndef ROOT_Math_ProbFuncInv
#define ROOT_Math_ProbFuncInv


namespace ROOT {
namespace Math {



  /** @name Quantile Functions
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
   * <bf>NOTE:</bf> In the old releases (< 5.14) the <em>_quantile</em> functions were called 
   * <em>_quant_inv</em> and the <em>_quantile_c</em> functions were called 
   * <em>_prob_inv</em>. 
   * These names are currently kept for backward compatibility, but 
   * their usage is deprecated.
   */

   
  //@{






  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the upper tail of the Breit-Wigner distribution (#breitwigner_cdf_c) 
  which is similar to the Cauchy distribution. For 
  detailed description see 
  <A HREF="http://mathworld.wolfram.com/CauchyDistribution.html">
  Mathworld</A>. It is evaluated using the same implementation of 
  #cauchy_quantile_c. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC294">GSL</A>.
  
  @ingroup StatFunc

  */

  double breitwigner_quantile_c(double z, double gamma);




  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the lower tail of the Breit_Wigner distribution (#breitwigner_cdf) 
  which is similar to the Cauchy distribution. For  
  detailed description see 
  <A HREF="http://mathworld.wolfram.com/CauchyDistribution.html">
  Mathworld</A>. It is evaluated using the same implementation of 
  #cauchy_quantile. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC294">GSL</A>.
  
  @ingroup StatFunc

  */

  double breitwigner_quantile(double z, double gamma);




  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the upper tail of the Cauchy distribution (#cauchy_cdf_c) 
  which is also called Lorentzian distribution. For 
  detailed description see 
  <A HREF="http://mathworld.wolfram.com/CauchyDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC294">GSL</A>.
  
  @ingroup StatFunc

  */

  double cauchy_quantile_c(double z, double b);




  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the lower tail of the Cauchy distribution (#cauchy_cdf) 
  which is also called Breit-Wigner or Lorentzian distribution. For 
  detailed description see 
  <A HREF="http://mathworld.wolfram.com/CauchyDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC294">GSL</A>.
  
  @ingroup StatFunc

  */

  double cauchy_quantile(double z, double b);




  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the upper tail of the \f$\chi^2\f$ distribution 
  with \f$r\f$ degrees of freedom (#chisquared_cdf_c). For detailed description see 
  <A HREF="http://mathworld.wolfram.com/Chi-SquaredDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC303">GSL</A>.

  @ingroup StatFunc

  */

  double chisquared_quantile_c(double z, double r);




  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the lower tail of the \f$\chi^2\f$ distribution 
  with \f$r\f$ degrees of freedom (#chisquared_cdf). For detailed description see 
  <A HREF="http://mathworld.wolfram.com/Chi-SquaredDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC303">GSL</A>.

  @ingroup StatFunc

  */

  double chisquared_quantile(double z, double r);




  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the upper tail of the exponential distribution
  (#exponential_cdf_c). For detailed description see 
  <A HREF="http://mathworld.wolfram.com/ExponentialDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC291">GSL</A>.
  
  @ingroup StatFunc

  */

  double exponential_quantile_c(double z, double lambda);




  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the lower tail of the exponential distribution
  (#exponential_cdf). For detailed description see 
  <A HREF="http://mathworld.wolfram.com/ExponentialDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC291">GSL</A>.
  
  @ingroup StatFunc

  */

  double exponential_quantile(double z, double lambda);

  

  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the lower tail of the f distribution
  (#fdistribution_cdf). For detailed description see 
  <A HREF="http://mathworld.wolfram.com/F-Distribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC291">GSL</A>.
  
  @ingroup StatFunc

  */
   double fdistribution_quantile(double z, double n, double m);  

  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the upper tail of the f distribution
  (#fdistribution_cdf_c). For detailed description see 
  <A HREF="http://mathworld.wolfram.com/F-Distribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC300">GSL</A>.
  
  @ingroup StatFunc
  */

   double fdistribution_quantile_c(double z, double n, double m);  


  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the upper tail of the gamma distribution
  (#gamma_cdf_c). For detailed description see 
  <A HREF="http://mathworld.wolfram.com/GammaDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC300">GSL</A>.
  
  @ingroup StatFunc

  */

  double gamma_quantile_c(double z, double alpha, double theta);




  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the lower tail of the gamma distribution
  (#gamma_cdf). For detailed description see 
  <A HREF="http://mathworld.wolfram.com/GammaDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC300">GSL</A>.
  
  @ingroup StatFunc

  */

  double gamma_quantile(double z, double alpha, double theta);




  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the upper tail of the normal (Gaussian) distribution
  (#gaussian_cdf_c). For detailed description see 
  <A HREF="http://mathworld.wolfram.com/NormalDistribution.html">
  Mathworld</A>. It can also be evaluated using #normal_quantile_c which will 
  call the same implementation. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC288">GSL</A>.

  @ingroup StatFunc

  */

  double gaussian_quantile_c(double z, double sigma);




  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the lower tail of the normal (Gaussian) distribution
  (#gaussian_cdf). For detailed description see 
  <A HREF="http://mathworld.wolfram.com/NormalDistribution.html">
  Mathworld</A>. It can also be evaluated using #normal_quantile which will 
  call the same implementation. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC288">GSL</A>.

  @ingroup StatFunc

  */

  double gaussian_quantile(double z, double sigma);




  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the upper tail of the lognormal distribution
  (#lognormal_cdf_c). For detailed description see 
  <A HREF="http://mathworld.wolfram.com/LogNormalDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC302">GSL</A>.
  
  @ingroup StatFunc

  */

  double lognormal_quantile_c(double x, double m, double s);




  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the lower tail of the lognormal distribution
  (#lognormal_cdf). For detailed description see 
  <A HREF="http://mathworld.wolfram.com/LogNormalDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC302">GSL</A>.
  
  @ingroup StatFunc

  */

  double lognormal_quantile(double x, double m, double s);




  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the upper tail of the normal (Gaussian) distribution
  (#normal_cdf_c). For detailed description see 
  <A HREF="http://mathworld.wolfram.com/NormalDistribution.html">
  Mathworld</A>. It can also be evaluated using #gaussian_quantile_c which will 
  call the same implementation. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC288">GSL</A>.

  @ingroup StatFunc

  */

  double normal_quantile_c(double z, double sigma);




  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the lower tail of the normal (Gaussian) distribution
  (#normal_cdf). For detailed description see 
  <A HREF="http://mathworld.wolfram.com/NormalDistribution.html">
  Mathworld</A>. It can also be evaluated using #gaussian_quantile which will 
  call the same implementation. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC288">GSL</A>.

  @ingroup StatFunc

  */

  double normal_quantile(double z, double sigma);




  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the upper tail of Student's t-distribution
  (#tdistribution_cdf_c). For detailed description see 
  <A HREF="http://mathworld.wolfram.com/Studentst-Distribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC305">GSL</A>.
  
  @ingroup StatFunc

  */

  double tdistribution_quantile_c(double z, double r);




  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the lower tail of Student's t-distribution
  (#tdistribution_cdf). For detailed description see 
  <A HREF="http://mathworld.wolfram.com/Studentst-Distribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC305">GSL</A>.
  
  @ingroup StatFunc

  */

  double tdistribution_quantile(double z, double r);




  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the upper tail of the uniform (flat) distribution
  (#uniform_cdf_c). For detailed description see 
  <A HREF="http://mathworld.wolfram.com/UniformDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC301">GSL</A>.
  
  @ingroup StatFunc

  */

  double uniform_quantile_c(double z, double a, double b);




  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the lower tail of the uniform (flat) distribution
  (#uniform_cdf). For detailed description see 
  <A HREF="http://mathworld.wolfram.com/UniformDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC301">GSL</A>.
  
  @ingroup StatFunc

  */

  double uniform_quantile(double z, double a, double b);


  /**
     
  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the upper tail of the beta distribution
  (#beta_cdf_c). 
  
  @ingroup StatFunc

  */
  double beta_quantile(double x, double a, double b);

  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the lower tail of the beta distribution
  (#beta_cdf). 

  @ingroup StatFunc

  */
  double beta_quantile_c(double x, double a, double b);



  //@}
   /** @name Backward compatible functions */ 


   inline double breitwigner_prob_inv(double x, double gamma) {
      return  breitwigner_quantile_c(x,gamma);
   }
   inline double breitwigner_quant_inv(double x, double gamma) { 
      return  breitwigner_quantile(x,gamma);
   }

   inline double cauchy_prob_inv(double x, double b) { 
      return cauchy_quantile_c(x,b);
   }
   inline double cauchy_quant_inv(double x, double b) {
      return cauchy_quantile  (x,b);
   }

   inline double exponential_prob_inv(double x, double lambda) { 
      return exponential_quantile_c(x, lambda );
   }
   inline double exponential_quant_inv(double x, double lambda) {
      return exponential_quantile  (x, lambda );
   }

   inline double gaussian_prob_inv(double x, double sigma) {
      return  gaussian_quantile_c( x, sigma );
   }
   inline double gaussian_quant_inv(double x, double sigma) { 
      return  gaussian_quantile  ( x, sigma );
   }

   inline double lognormal_prob_inv(double x, double m, double s) {
      return lognormal_quantile_c( x, m, s );   
   }
   inline double lognormal_quant_inv(double x, double m, double s) {
      return lognormal_quantile  ( x, m, s );   
   }

   inline double normal_prob_inv(double x, double sigma) {
      return  normal_quantile_c( x, sigma );
   }
   inline double normal_quant_inv(double x, double sigma) {
      return  normal_quantile  ( x, sigma );
   }

   inline double uniform_prob_inv(double x, double a, double b) { 
      return uniform_quantile_c( x, a, b ); 
   }
   inline double uniform_quant_inv(double x, double a, double b) {
      return uniform_quantile  ( x, a, b ); 
   }

   inline double chisquared_prob_inv(double x, double r) {
      return chisquared_quantile_c(x, r ); 
   }
   inline double chisquared_quant_inv(double x, double r) {
      return chisquared_quantile  (x, r ); 
   }
   
   inline double gamma_prob_inv(double x, double alpha, double theta) {
      return gamma_quantile_c (x, alpha, theta ); 
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



} // namespace Math
} // namespace ROOT



#endif // ROOT_Math_ProbFuncInv
