// @(#)root/mathcore:$Name:  $:$Id: ProbFuncMathCore.h,v 1.3 2006/12/06 17:51:13 moneta Exp $
// Authors: L. Moneta, A. Zsenei   06/2005 

#ifndef ROOT_Math_ProbFuncMathCore
#define ROOT_Math_ProbFuncMathCore

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
   * <strong>NOTE:</strong> In the old releases (< 5.14) the <em>_cdf</em> functions were called 
   * <em>_quant</em> and the <em>_cdf_c</em> functions were called 
   * <em>_prob</em>. 
   * These names are currently kept for backward compatibility, but 
   * their usage is deprecated.
   *
   *
   *   Additional CDF's are provided in the 
   *  <A HREF="../../MathMore/html/group__StatFunc.html">MathMore</A> library. 
   *
   */
  //@{





  /**

  Complement of the cumulative distribution function (upper tail) of the Breit_Wigner 
  distribution and it is similar (just a different parameter definition) to the 
  Cauchy distribution (see #cauchy_cdf_c )

  \f[ D(x) = \int_{x}^{+\infty} \frac{1}{\pi} \frac{\frac{1}{2} \Gamma}{x'^2 + (\frac{1}{2} \Gamma)^2} dx' \f]

  
  @ingroup StatFunc

  */
  double breitwigner_cdf_c(double x, double gamma, double x0 = 0);


  /**

  Cumulative distribution function (lower tail) of the Breit_Wigner 
  distribution and it is similar (just a different parameter definition) to the 
  Cauchy distribution (see #cauchy_cdf )

  \f[ D(x) = \int_{-\infty}^{x} \frac{1}{\pi} \frac{b}{x'^2 + (\frac{1}{2} \Gamma)^2} dx' \f]
 
  
  @ingroup StatFunc

  */
  double breitwigner_cdf(double x, double gamma, double x0 = 0);



  /**

  Complement of the cumulative distribution function (upper tail) of the 
  Cauchy distribution which is also Lorentzian distribution.
  It is similar (just a different parameter definition) to the 
  Breit_Wigner distribution (see #breitwigner_cdf_c )

  \f[ D(x) = \int_{x}^{+\infty} \frac{1}{\pi} \frac{ b }{ (x'-m)^2 + b^2} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/CauchyDistribution.html">
  Mathworld</A>. 
  
  @ingroup StatFunc

  */
  double cauchy_cdf_c(double x, double b, double x0 = 0);




  /**

  Cumulative distribution function (lower tail) of the 
  Cauchy distribution which is also Lorentzian distribution.
  It is similar (just a different parameter definition) to the 
  Breit_Wigner distribution (see #breitwigner_cdf )

  \f[ D(x) = \int_{-\infty}^{x} \frac{1}{\pi} \frac{ b }{ (x'-m)^2 + b^2} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/CauchyDistribution.html">
  Mathworld</A>. 

  
  @ingroup StatFunc

  */
  double cauchy_cdf(double x, double b, double x0 = 0);




  /**
   \if later

  Cumulative distribution function of the \f$\chi^2\f$ distribution 
  with \f$r\f$ degrees of freedom (upper tail).

  \f[ D_{r}(x) = \int_{x}^{+\infty} \frac{1}{\Gamma(r/2) 2^{r/2}} x'^{r/2-1} e^{-x'/2} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/Chi-SquaredDistribution.html">
  Mathworld</A>. 
  
  @ingroup StatFunc


   \endif
  */

  //double chisquared_prob(double x, double r);




  /**

   \if later
  Cumulative distribution function of the \f$\chi^2\f$ distribution 
  with \f$r\f$ degrees of freedom (lower tail).

  \f[ D_{r}(x) = \int_{-\infty}^{x} \frac{1}{\Gamma(r/2) 2^{r/2}} x'^{r/2-1} e^{-x'/2} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/Chi-SquaredDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC303">GSL</A>.
  
  @ingroup StatFunc
  
  \endif
  */

  // double chisquared_quant(double x, double r);




  /**

  Complement of the cumulative distribution function of the exponential distribution 
  (upper tail).

  \f[ D(x) = \int_{x}^{+\infty} \lambda e^{-\lambda x'} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/ExponentialDistribution.html">
  Mathworld</A>. 
  
  @ingroup StatFunc

  */

  double exponential_cdf_c(double x, double lambda, double x0 = 0);



  /**

  Cumulative distribution function of the exponential distribution 
  (lower tail).

  \f[ D(x) = \int_{-\infty}^{x} \lambda e^{-\lambda x'} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/ExponentialDistribution.html">
  Mathworld</A>. 
  
  @ingroup StatFunc

  */


  double exponential_cdf(double x, double lambda, double x0 = 0);



  /**
  \if later

  Complement of the cumulative distribution function of the F-distribution 
  (upper tail).

  \f[ D_{n,m}(x) = \int_{x}^{+\infty} \frac{\Gamma(\frac{n+m}{2})}{\Gamma(\frac{n}{2}) \Gamma(\frac{m}{2})} n^{n/2} m^{m/2} x'^{n/2 -1} (m+nx')^{-(n+m)/2} dx' \f] 

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/F-Distribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC304">GSL</A>.
  
  @ingroup StatFunc
  \endif

  */

  /* double fdistribution_prob(double x, double n, double m); */




  /**
  \if later
  Cumulative distribution function of the F-distribution 
  (lower tail).

  \f[ D_{n,m}(x) = \int_{-\infty}^{x} \frac{\Gamma(\frac{n+m}{2})}{\Gamma(\frac{n}{2}) \Gamma(\frac{m}{2})} n^{n/2} m^{m/2} x'^{n/2 -1} (m+nx')^{-(n+m)/2} dx' \f] 

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/F-Distribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC304">GSL</A>.
  
  @ingroup StatFunc

  \endif
  */

  // double fdistribution_quant(double x, double n, double m);




  /**
     \if later

  Cumulative distribution function of the gamma distribution 
  (upper tail).

  \f[ D(x) = \int_{x}^{+\infty} {1 \over \Gamma(\alpha) \theta^{\alpha}} x'^{\alpha-1} e^{-x'/\theta} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/GammaDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC300">GSL</A>.
  
  @ingroup StatFunc

  \endif
  */

  // double gamma_prob(double x, double alpha, double theta);
 



  /**
  \if later

  Cumulative distribution function of the gamma distribution 
  (lower tail).

  \f[ D(x) = \int_{-\infty}^{x} {1 \over \Gamma(\alpha) \theta^{\alpha}} x'^{\alpha-1} e^{-x'/\theta} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/GammaDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC300">GSL</A>.
  
  @ingroup StatFunc

  \endif
  */

  //double gamma_quant(double x, double alpha, double theta);



  /**

  Complement of the cumulative distribution function of the normal (Gaussian) 
  distribution (upper tail).

  \f[ D(x) = \int_{x}^{+\infty} {1 \over \sqrt{2 \pi \sigma^2}} e^{-x'^2 / 2\sigma^2} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/NormalDistribution.html">
  Mathworld</A>. It can also be evaluated using #normal_cdf_c which will 
  call the same implementation. 

  @ingroup StatFunc

  */

  double gaussian_cdf_c(double x, double sigma, double x0 = 0);



  /**

  Cumulative distribution function of the normal (Gaussian) 
  distribution (lower tail).

  \f[ D(x) = \int_{-\infty}^{x} {1 \over \sqrt{2 \pi \sigma^2}} e^{-x'^2 / 2\sigma^2} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/NormalDistribution.html">
  Mathworld</A>. It can also be evaluated using #normal_quant which will 
  call the same implementation. 

  @ingroup StatFunc
 
  */

  double gaussian_cdf(double x, double sigma, double x0 = 0);



  /**

  Complement of the cumulative distribution function of the lognormal distribution 
  (upper tail).

  \f[ D(x) = \int_{x}^{+\infty} {1 \over x' \sqrt{2 \pi s^2} } e^{-(\ln{x'} - m)^2/2 s^2} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/LogNormalDistribution.html">
  Mathworld</A>. 
  
  @ingroup StatFunc

  */

  double lognormal_cdf_c(double x, double m, double s, double x0 = 0);




  /**

  Cumulative distribution function of the lognormal distribution 
  (lower tail).

  \f[ D(x) = \int_{-\infty}^{x} {1 \over x' \sqrt{2 \pi s^2} } e^{-(\ln{x'} - m)^2/2 s^2} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/LogNormalDistribution.html">
  Mathworld</A>. 
  
  @ingroup StatFunc

  */

  double lognormal_cdf(double x, double m, double s, double x0 = 0);




  /**

  Complement of the cumulative distribution function of the normal (Gaussian) 
  distribution (upper tail).

  \f[ D(x) = \int_{x}^{+\infty} {1 \over \sqrt{2 \pi \sigma^2}} e^{-x'^2 / 2\sigma^2} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/NormalDistribution.html">
  Mathworld</A>. It can also be evaluated using #gaussian_prob which will 
  call the same implementation. 

  @ingroup StatFunc

  */

  double normal_cdf_c(double x, double sigma, double x0 = 0);



  /**

  Cumulative distribution function of the normal (Gaussian) 
  distribution (lower tail).

  \f[ D(x) = \int_{-\infty}^{x} {1 \over \sqrt{2 \pi \sigma^2}} e^{-x'^2 / 2\sigma^2} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/NormalDistribution.html">
  Mathworld</A>. It can also be evaluated using #gaussian_quant which will 
  call the same implementation. 

  @ingroup StatFunc
 
  */

  double normal_cdf(double x, double sigma, double x0 = 0);




  /**
     \if later

  Cumulative distribution function of Student's  
  t-distribution (upper tail).

  \f[ D_{r}(x) = \int_{x}^{+\infty} \frac{\Gamma(\frac{r+1}{2})}{\sqrt{r \pi}\Gamma(\frac{r}{2})} \left( 1+\frac{x'^2}{r}\right)^{-(r+1)/2} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/Studentst-Distribution.html">
  Mathworld</A>. 
  
  @ingroup StatFunc

   \endif
  */

  //double tdistribution_prob(double x, double r);




  /**
    \if later
  Cumulative distribution function of Student's  
  t-distribution (lower tail).

  \f[ D_{r}(x) = \int_{-\infty}^{x} \frac{\Gamma(\frac{r+1}{2})}{\sqrt{r \pi}\Gamma(\frac{r}{2})} \left( 1+\frac{x'^2}{r}\right)^{-(r+1)/2} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/Studentst-Distribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC305">GSL</A>.
  
  @ingroup StatFunc

    \endif
  */

  //double tdistribution_quant(double x, double r);




  /**

  Complement of the cumulative distribution function of the uniform (flat)  
  distribution (upper tail).

  \f[ D(x) = \int_{x}^{+\infty} {1 \over (b-a)} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/UniformDistribution.html">
  Mathworld</A>. 
  
  @ingroup StatFunc

  */

  double uniform_cdf_c(double x, double a, double b, double x0 = 0);




  /**

  Cumulative distribution function of the uniform (flat)  
  distribution (lower tail).

  \f[ D(x) = \int_{-\infty}^{x} {1 \over (b-a)} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/UniformDistribution.html">
  Mathworld</A>. 
  
  @ingroup StatFunc

  */

  double uniform_cdf(double x, double a, double b, double x0 = 0);






  //@}
   /** @name Backward compatible functions */ 


   inline double breitwigner_prob(double x, double gamma, double x0 = 0) {
      return  breitwigner_cdf_c(x,gamma,x0);
   }
   inline double breitwigner_quant(double x, double gamma, double x0 = 0) { 
      return  breitwigner_cdf(x,gamma,x0);
   }

   inline double cauchy_prob(double x, double b, double x0 = 0) { 
      return cauchy_cdf_c(x,b,x0);
   }
   inline double cauchy_quant(double x, double b, double x0 = 0) {
      return cauchy_cdf  (x,b,x0);
   }

   inline double exponential_prob(double x, double lambda, double x0 = 0) { 
      return exponential_cdf_c(x, lambda, x0 );
   }
   inline double exponential_quant(double x, double lambda, double x0 = 0) {
      return exponential_cdf  (x, lambda, x0 );
   }

   inline double gaussian_prob(double x, double sigma, double x0 = 0) {
      return  gaussian_cdf_c( x, sigma, x0 );
   }
   inline double gaussian_quant(double x, double sigma, double x0 = 0) { 
      return  gaussian_cdf  ( x, sigma, x0 );
   }

   inline double lognormal_prob(double x, double m, double s, double x0 = 0) {
      return lognormal_cdf_c( x, m, s, x0 );   
   }
   inline double lognormal_quant(double x, double m, double s, double x0 = 0) {
      return lognormal_cdf  ( x, m, s, x0 );   
   }

   inline double normal_prob(double x, double sigma, double x0 = 0) {
      return  normal_cdf_c( x, sigma, x0 );
   }
   inline double normal_quant(double x, double sigma, double x0 = 0) {
      return  normal_cdf  ( x, sigma, x0 );
   }

   inline double uniform_prob(double x, double a, double b, double x0 = 0) { 
      return uniform_cdf_c( x, a, b, x0 ); 
   }
   inline double uniform_quant(double x, double a, double b, double x0 = 0) {
      return uniform_cdf  ( x, a, b, x0 ); 
   }



} // namespace Math
} // namespace ROOT


#endif // ROOT_Math_ProbFuncMathCore
