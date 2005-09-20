// @(#)root/mathmore:$Name:  $:$Id: ProbFuncInv.h,v 1.1 2005/09/18 17:33:47 brun Exp $
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

/**
   @defgroup StatFunc Statistical functions
*/

#ifndef ROOT_Math_ProbFuncInv
#define ROOT_Math_ProbFuncInv


namespace ROOT {
namespace Math {



  /** @name Inverses of the Cumulative Distribution Functions 
   *  Inverse functions of the cumulative distribution functions 
   *  of various distributions.
   *  The functions with the extension _quant_inv calculate the
   *  inverse of lower tail integral of the probability density function
   *  \f$D^{-1}(z)\f$ where
   *
   *  \f[ D(x) = \int_{-\infty}^{x} p(x') dx' \f]
   *
   *  while those with the _prob_inv extension calculate the 
   *  inverse of the upper tail integral of the probability 
   *  density function \f$D^{-1}(z) \f$ where
   *
   *  \f[ D(x) = \int_{x}^{+\infty} p(x') dx' \f]
   *
   */
  //@{






  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the upper tail of the Cauchy distribution (#breitwigner_prob) 
  which is also called Breit-Wigner or Lorentzian distribution. For 
  detailed description see 
  <A HREF="http://mathworld.wolfram.com/CauchyDistribution.html">
  Mathworld</A>. It can also be evaluated using #cauchy_prob_inv which 
  will call the same implementation. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC294">GSL</A>.
  
  @ingroup StatFunc

  */

  double breitwigner_prob_inv(double z, double gamma);




  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the lower tail of the Cauchy distribution (#breitwigner_quant) 
  which is also called Breit-Wigner or Lorentzian distribution. For 
  detailed description see 
  <A HREF="http://mathworld.wolfram.com/CauchyDistribution.html">
  Mathworld</A>. It can also be evaluated using #cauchy_quant_inv which 
  will call the same implementation. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC294">GSL</A>.
  
  @ingroup StatFunc

  */

  double breitwigner_quant_inv(double z, double gamma);




  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the upper tail of the Cauchy distribution (#cauchy_prob) 
  which is also called Breit-Wigner or Lorentzian distribution. For 
  detailed description see 
  <A HREF="http://mathworld.wolfram.com/CauchyDistribution.html">
  Mathworld</A>. It can also be evaluated using #breitwigner_prob_inv which 
  will call the same implementation. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC294">GSL</A>.
  
  @ingroup StatFunc

  */

  double cauchy_prob_inv(double z, double gamma);




  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the lower tail of the Cauchy distribution (#cauchy_quant) 
  which is also called Breit-Wigner or Lorentzian distribution. For 
  detailed description see 
  <A HREF="http://mathworld.wolfram.com/CauchyDistribution.html">
  Mathworld</A>. It can also be evaluated using #breitwigner_quant_inv which 
  will call the same implementation. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC294">GSL</A>.
  
  @ingroup StatFunc

  */

  double cauchy_quant_inv(double z, double gamma);




  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the upper tail of the \f$\chi^2\f$ distribution 
  with \f$r\f$ degrees of freedom (#chisquared_prob). For detailed description see 
  <A HREF="http://mathworld.wolfram.com/Chi-SquaredDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC303">GSL</A>.

  @ingroup StatFunc

  */

  double chisquared_prob_inv(double z, double r);




  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the lower tail of the \f$\chi^2\f$ distribution 
  with \f$r\f$ degrees of freedom (#chisquared_quant). For detailed description see 
  <A HREF="http://mathworld.wolfram.com/Chi-SquaredDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC303">GSL</A>.

  @ingroup StatFunc

  */

  double chisquared_quant_inv(double z, double r);




  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the upper tail of the exponential distribution
  (#exponential_prob). For detailed description see 
  <A HREF="http://mathworld.wolfram.com/ExponentialDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC291">GSL</A>.
  
  @ingroup StatFunc

  */

  double exponential_prob_inv(double z, double lambda);




  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the lower tail of the exponential distribution
  (#exponential_quant). For detailed description see 
  <A HREF="http://mathworld.wolfram.com/ExponentialDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC291">GSL</A>.
  
  @ingroup StatFunc

  */

  double exponential_quant_inv(double z, double lambda);




  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the upper tail of the gamma distribution
  (#gamma_prob). For detailed description see 
  <A HREF="http://mathworld.wolfram.com/GammaDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC300">GSL</A>.
  
  @ingroup StatFunc

  */

  double gamma_prob_inv(double z, double alpha, double theta);




  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the lower tail of the gamma distribution
  (#gamma_quant). For detailed description see 
  <A HREF="http://mathworld.wolfram.com/GammaDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC300">GSL</A>.
  
  @ingroup StatFunc

  */

  double gamma_quant_inv(double z, double alpha, double theta);




  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the upper tail of the normal (Gaussian) distribution
  (#gaussian_prob). For detailed description see 
  <A HREF="http://mathworld.wolfram.com/NormalDistribution.html">
  Mathworld</A>. It can also be evaluated using #normal_prob_inv which will 
  call the same implementation. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC288">GSL</A>.

  @ingroup StatFunc

  */

  double gaussian_prob_inv(double z, double sigma);




  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the lower tail of the normal (Gaussian) distribution
  (#gaussian_quant). For detailed description see 
  <A HREF="http://mathworld.wolfram.com/NormalDistribution.html">
  Mathworld</A>. It can also be evaluated using #normal_quant_inv which will 
  call the same implementation. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC288">GSL</A>.

  @ingroup StatFunc

  */

  double gaussian_quant_inv(double z, double sigma);




  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the upper tail of the lognormal distribution
  (#lognormal_prob). For detailed description see 
  <A HREF="http://mathworld.wolfram.com/LogNormalDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC302">GSL</A>.
  
  @ingroup StatFunc

  */

  double lognormal_prob_inv(double x, double m, double s);




  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the lower tail of the lognormal distribution
  (#lognormal_quant). For detailed description see 
  <A HREF="http://mathworld.wolfram.com/LogNormalDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC302">GSL</A>.
  
  @ingroup StatFunc

  */

  double lognormal_quant_inv(double x, double m, double s);




  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the upper tail of the normal (Gaussian) distribution
  (#normal_prob). For detailed description see 
  <A HREF="http://mathworld.wolfram.com/NormalDistribution.html">
  Mathworld</A>. It can also be evaluated using #gaussian_prob_inv which will 
  call the same implementation. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC288">GSL</A>.

  @ingroup StatFunc

  */

  double normal_prob_inv(double z, double sigma);




  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the lower tail of the normal (Gaussian) distribution
  (#normal_quant). For detailed description see 
  <A HREF="http://mathworld.wolfram.com/NormalDistribution.html">
  Mathworld</A>. It can also be evaluated using #gaussian_quant_inv which will 
  call the same implementation. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC288">GSL</A>.

  @ingroup StatFunc

  */

  double normal_quant_inv(double z, double sigma);




  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the upper tail of Student's t-distribution
  (#tdistribution_prob). For detailed description see 
  <A HREF="http://mathworld.wolfram.com/Studentst-Distribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC305">GSL</A>.
  
  @ingroup StatFunc

  */

  double tdistribution_prob_inv(double z, double r);




  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the lower tail of Student's t-distribution
  (#tdistribution_quant). For detailed description see 
  <A HREF="http://mathworld.wolfram.com/Studentst-Distribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC305">GSL</A>.
  
  @ingroup StatFunc

  */

  double tdistribution_quant_inv(double z, double r);




  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the upper tail of the uniform (flat) distribution
  (#uniform_prob). For detailed description see 
  <A HREF="http://mathworld.wolfram.com/UniformDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC301">GSL</A>.
  
  @ingroup StatFunc

  */

  double uniform_prob_inv(double z, double a, double b);




  /**

  Inverse (\f$D^{-1}(z)\f$) of the cumulative distribution 
  function of the lower tail of the uniform (flat) distribution
  (#uniform_quant). For detailed description see 
  <A HREF="http://mathworld.wolfram.com/UniformDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC301">GSL</A>.
  
  @ingroup StatFunc

  */

  double uniform_quant_inv(double z, double a, double b);





  //@}



} // namespace Math
} // namespace ROOT



#endif // ROOT_Math_ProbFuncInv
