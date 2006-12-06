// @(#)root/mathmore:$Name:  $:$Id: ProbFuncMathMore.h,v 1.1 2005/09/18 17:33:47 brun Exp $
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


#ifndef ROOT_Math_ProbFuncMathMore
#define ROOT_Math_ProbFuncMathMore

namespace ROOT {
namespace Math {


  /** @name Cumulative Distribution Functions (CDF)
   *  Cumulative distribution functions of various distributions.
   *  The functions with the extension <em>_cdf</em> calculate the
   *  lower tail integral of the probability density function
   *
   *  \f[ D(x) = \int_{-\infty}^{x} p(x') dx' \f]
   *
   *  while those with the <em>_cdf_c</em> extension calculate the complement of 
   *  cumulative distribution function, called in statistics the survival 
   *  function. 
   *  It corresponds to the upper tail integral of the 
   *  probability density function
   *
   *  \f[ D(x) = \int_{x}^{+\infty} p(x') dx' \f]
   *
   * 
   * <bf>NOTE:</bf> In the old releases (< 5.14) the <em>_cdf</em> functions were called 
   * <em>_quant</em> and the <em>_cdf_c</em> functions were called 
   * <em>_prob</em>. 
   * These names are currently kept for backward compatibility, but 
   * their usage is deprecated.
   */
  //@{






  /**

  Complement of the cumulative distribution function of the \f$\chi^2\f$ distribution 
  with \f$r\f$ degrees of freedom (upper tail).

  \f[ D_{r}(x) = \int_{x}^{+\infty} \frac{1}{\Gamma(r/2) 2^{r/2}} x'^{r/2-1} e^{-x'/2} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/Chi-SquaredDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC303">GSL</A>.
  
  @ingroup StatFunc

  */

  double chisquared_cdf_c(double x, double r, double x0 = 0);




  /**

  Cumulative distribution function of the \f$\chi^2\f$ distribution 
  with \f$r\f$ degrees of freedom (lower tail).

  \f[ D_{r}(x) = \int_{-\infty}^{x} \frac{1}{\Gamma(r/2) 2^{r/2}} x'^{r/2-1} e^{-x'/2} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/Chi-SquaredDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC303">GSL</A>.
  
  @ingroup StatFunc

  */

  double chisquared_cdf(double x, double r, double x0 = 0);





  /**

  Complement of the cumulative distribution function of the F-distribution 
  (upper tail).

  \f[ D_{n,m}(x) = \int_{x}^{+\infty} \frac{\Gamma(\frac{n+m}{2})}{\Gamma(\frac{n}{2}) \Gamma(\frac{m}{2})} n^{n/2} m^{m/2} x'^{n/2 -1} (m+nx')^{-(n+m)/2} dx' \f] 

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/F-Distribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC304">GSL</A>.
  
  @ingroup StatFunc

  */

  double fdistribution_cdf_c(double x, double n, double m, double x0 = 0);




  /**

  Cumulative distribution function of the F-distribution 
  (lower tail).

  \f[ D_{n,m}(x) = \int_{-\infty}^{x} \frac{\Gamma(\frac{n+m}{2})}{\Gamma(\frac{n}{2}) \Gamma(\frac{m}{2})} n^{n/2} m^{m/2} x'^{n/2 -1} (m+nx')^{-(n+m)/2} dx' \f] 

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/F-Distribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC304">GSL</A>.
  
  @ingroup StatFunc

  */

  double fdistribution_cdf(double x, double n, double m, double x0 = 0);




  /**

  Complement of the cumulative distribution function of the gamma distribution 
  (upper tail).

  \f[ D(x) = \int_{x}^{+\infty} {1 \over \Gamma(\alpha) \theta^{\alpha}} x'^{\alpha-1} e^{-x'/\theta} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/GammaDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC300">GSL</A>.
  
  @ingroup StatFunc

  */

  double gamma_cdf_c(double x, double alpha, double theta, double x0 = 0);
 



  /**

  Cumulative distribution function of the gamma distribution 
  (lower tail).

  \f[ D(x) = \int_{-\infty}^{x} {1 \over \Gamma(\alpha) \theta^{\alpha}} x'^{\alpha-1} e^{-x'/\theta} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/GammaDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC300">GSL</A>.
  
  @ingroup StatFunc

  */

  double gamma_cdf(double x, double alpha, double theta, double x0 = 0);
 


  /**

  Complement of the cumulative distribution function of Student's  
  t-distribution (upper tail).

  \f[ D_{r}(x) = \int_{x}^{+\infty} \frac{\Gamma(\frac{r+1}{2})}{\sqrt{r \pi}\Gamma(\frac{r}{2})} \left( 1+\frac{x'^2}{r}\right)^{-(r+1)/2} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/Studentst-Distribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC305">GSL</A>.
  
  @ingroup StatFunc

  */

  double tdistribution_cdf_c(double x, double r, double x0 = 0);




  /**

  Cumulative distribution function of Student's  
  t-distribution (lower tail).

  \f[ D_{r}(x) = \int_{-\infty}^{x} \frac{\Gamma(\frac{r+1}{2})}{\sqrt{r \pi}\Gamma(\frac{r}{2})} \left( 1+\frac{x'^2}{r}\right)^{-(r+1)/2} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/Studentst-Distribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC305">GSL</A>.
  
  @ingroup StatFunc

  */

  double tdistribution_cdf(double x, double r, double x0 = 0);





  //@}

   /** @name Backward compatible functions */ 
   // this I will maintain since is commonly used in physics
   inline double chisquared_prob(double x, double r, double x0 = 0) {
      return chisquared_cdf_c(x, r, x0); 
   }
   inline double chisquared_quant(double x, double r, double x0 = 0) {
      return chisquared_cdf  (x, r, x0); 
   }
   
   inline double fdistribution_prob(double x, double n, double m, double x0 = 0) {
      return fdistribution_cdf_c  (x, n, m, x0); 
   }
   inline double fdistribution_quant(double x, double n, double m, double x0 = 0) {
      return fdistribution_cdf    (x, n, m, x0); 
   }

   inline double gamma_prob(double x, double alpha, double theta, double x0 = 0) {
      return gamma_cdf_c (x, alpha, theta, x0); 
   }
   inline double gamma_quant(double x, double alpha, double theta, double x0 = 0) {
      return gamma_cdf   (x, alpha, theta, x0); 
   }

   inline double tdistribution_prob(double x, double r, double x0 = 0) {
      return tdistribution_cdf_c  (x, r, x0); 
   }

   inline double tdistribution_quant(double x, double r, double x0 = 0) {
      return tdistribution_cdf    (x, r, x0); 
   }


} // namespace Math
} // namespace ROOT


#endif // ROOT_Math_ProbFuncMathMore
