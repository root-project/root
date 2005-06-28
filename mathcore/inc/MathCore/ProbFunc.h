// @(#)root/mathcore:$Name:  $:$Id: ProbFunc.h,v 1.1 2005/06/24 18:54:24 brun Exp $
// Authors: L. Moneta, A. Zsenei   06/2005 

#ifndef ROOT_Math_ProbFunc
#define ROOT_Math_ProbFunc

namespace ROOT {
namespace Math {


  /** @name Cumulative Distribution Functions (CDF)
   *  Cumulative distribution functions of various distributions.
   *  The functions with the extension _quant calculate the
   *  lower tail integral of the probability density function
   *
   *  \f[ D(x) = \int_{-\infty}^{x} p(x') dx' \f]
   *
   *  while those with the _prob extension calculate the 
   *  upper tail integral of the probability density function
   *
   *  \f[ D(x) = \int_{x}^{+\infty} p(x') dx' \f]
   *
   */
  //@{





  /**

  Cumulative distribution function (upper tail) of the Cauchy 
  distribution which is also called Breit-Wigner or Lorentzian
  distribution.

  \f[ D(x) = \int_{x}^{+\infty} \frac{1}{\pi} \frac{\frac{1}{2} \Gamma}{x'^2 + (\frac{1}{2} \Gamma)^2} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/CauchyDistribution.html">
  Mathworld</A>. It can also be evaluated using #cauchy_prob which 
  will call the same implementation. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC294">GSL</A>.
  
  @ingroup StatFunc

  */

  double breitwigner_prob(double x, double gamma);




  /**

  Cumulative distribution function (lower tail) of the 
  Cauchy distribution which is also called Breit-Wigner 
  or Lorentzian distribution.

  \f[ D(x) = \int_{-\infty}^{x} \frac{1}{\pi} \frac{\frac{1}{2} \Gamma}{x'^2 + (\frac{1}{2} \Gamma)^2} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/CauchyDistribution.html">
  Mathworld</A>. It can also be evaluated using #cauchy_quant which 
  will call the same implementation. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC294">GSL</A>.
  
  @ingroup StatFunc

  */

  double breitwigner_quant(double x, double gamma);




  /**

  Cumulative distribution function (upper tail) of the Cauchy 
  distribution which is also called Breit-Wigner or Lorentzian
  distribution.

  \f[ D(x) = \int_{x}^{+\infty} \frac{1}{\pi} \frac{\frac{1}{2} \Gamma}{x'^2 + (\frac{1}{2} \Gamma)^2} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/CauchyDistribution.html">
  Mathworld</A>. It can also be evaluated using #breitwigner_prob which 
  will call the same implementation. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC294">GSL</A>.
  
  @ingroup StatFunc

  */

  double cauchy_prob(double x, double gamma);




  /**

  Cumulative distribution function (lower tail) of the Cauchy 
  distribution which is also called Breit-Wigner or Lorentzian
  distribution.

  \f[ D(x) = \int_{-\infty}^{x} \frac{1}{\pi} \frac{\frac{1}{2} \Gamma}{x'^2 + (\frac{1}{2} \Gamma)^2} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/CauchyDistribution.html">
  Mathworld</A>. It can also be evaluated using #breitwigner_quant which 
  will call the same implementation. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC294">GSL</A>.
  
  @ingroup StatFunc

  */

  double cauchy_quant(double x, double gamma);




  /**

  Cumulative distribution function of the \f$\chi^2\f$ distribution 
  with \f$r\f$ degrees of freedom (upper tail).

  \f[ D_{r}(x) = \int_{x}^{+\infty} \frac{1}{\Gamma(r/2) 2^{r/2}} x'^{r/2-1} e^{-x'/2} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/Chi-SquaredDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC303">GSL</A>.
  
  @ingroup StatFunc

  */

  //double chisquared_prob(double x, double r);




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

  //double chisquared_quant(double x, double r);




  /**

  Cumulative distribution function of the exponential distribution 
  (upper tail).

  \f[ D(x) = \int_{x}^{+\infty} \lambda e^{-\lambda x'} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/ExponentialDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC291">GSL</A>.
  
  @ingroup StatFunc

  */

  double exponential_prob(double x, double lambda);




  /**

  Cumulative distribution function of the exponential distribution 
  (lower tail).

  \f[ D(x) = \int_{-\infty}^{x} \lambda e^{-\lambda x'} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/ExponentialDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC291">GSL</A>.
  
  @ingroup StatFunc

  */

  double exponential_quant(double x, double lambda);




  /**

  Cumulative distribution function of the F-distribution 
  (upper tail).

  \f[ D_{n,m}(x) = \int_{x}^{+\infty} \frac{\Gamma(\frac{n+m}{2})}{\Gamma(\frac{n}{2}) \Gamma(\frac{m}{2})} n^{n/2} m^{m/2} x'^{n/2 -1} (m+nx')^{-(n+m)/2} dx' \f] 

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/F-Distribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC304">GSL</A>.
  
  @ingroup StatFunc

  */

  //double fdistribution_prob(double x, double n, double m);




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

  //double fdistribution_quant(double x, double n, double m);




  /**

  Cumulative distribution function of the gamma distribution 
  (upper tail).

  \f[ D(x) = \int_{x}^{+\infty} {1 \over \Gamma(\alpha) \theta^{\alpha}} x'^{\alpha-1} e^{-x'/\theta} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/GammaDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC300">GSL</A>.
  
  @ingroup StatFunc

  */

  //double gamma_prob(double x, double alpha, double theta);
 



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

  //double gamma_quant(double x, double alpha, double theta);
 



  /**

  Cumulative distribution function of the normal (Gaussian) 
  distribution (upper tail).

  \f[ D(x) = \int_{x}^{+\infty} {1 \over \sqrt{2 \pi \sigma^2}} e^{-x'^2 / 2\sigma^2} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/NormalDistribution.html">
  Mathworld</A>. It can also be evaluated using #normal_prob which will 
  call the same implementation. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC288">GSL</A>.

  @ingroup StatFunc

  */

  double gaussian_prob(double x, double sigma);



  /**

  Cumulative distribution function of the normal (Gaussian) 
  distribution (lower tail).

  \f[ D(x) = \int_{-\infty}^{x} {1 \over \sqrt{2 \pi \sigma^2}} e^{-x'^2 / 2\sigma^2} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/NormalDistribution.html">
  Mathworld</A>. It can also be evaluated using #normal_quant which will 
  call the same implementation. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC288">GSL</A>.

  @ingroup StatFunc
 
  */

  double gaussian_quant(double x, double sigma);




  /**

  Cumulative distribution function of the lognormal distribution 
  (upper tail).

  \f[ D(x) = \int_{x}^{+\infty} {1 \over x' \sqrt{2 \pi s^2} } e^{-(\ln{x'} - m)^2/2 s^2} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/LogNormalDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC302">GSL</A>.
  
  @ingroup StatFunc

  */

  double lognormal_prob(double x, double m, double s);




  /**

  Cumulative distribution function of the lognormal distribution 
  (lower tail).

  \f[ D(x) = \int_{-\infty}^{x} {1 \over x' \sqrt{2 \pi s^2} } e^{-(\ln{x'} - m)^2/2 s^2} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/LogNormalDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC302">GSL</A>.
  
  @ingroup StatFunc

  */

  double lognormal_quant(double x, double m, double s);




  /**

  Cumulative distribution function of the normal (Gaussian) 
  distribution (upper tail).

  \f[ D(x) = \int_{x}^{+\infty} {1 \over \sqrt{2 \pi \sigma^2}} e^{-x'^2 / 2\sigma^2} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/NormalDistribution.html">
  Mathworld</A>. It can also be evaluated using #gaussian_prob which will 
  call the same implementation. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC288">GSL</A>.

  @ingroup StatFunc

  */

  double normal_prob(double x, double sigma);



  /**

  Cumulative distribution function of the normal (Gaussian) 
  distribution (lower tail).

  \f[ D(x) = \int_{-\infty}^{x} {1 \over \sqrt{2 \pi \sigma^2}} e^{-x'^2 / 2\sigma^2} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/NormalDistribution.html">
  Mathworld</A>. It can also be evaluated using #gaussian_quant which will 
  call the same implementation. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC288">GSL</A>.

  @ingroup StatFunc
 
  */

  double normal_quant(double x, double sigma);




  /**

  Cumulative distribution function of Student's  
  t-distribution (upper tail).

  \f[ D_{r}(x) = \int_{x}^{+\infty} \frac{\Gamma(\frac{r+1}{2})}{\sqrt{r \pi}\Gamma(\frac{r}{2})} \left( 1+\frac{x'^2}{r}\right)^{-(r+1)/2} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/Studentst-Distribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC305">GSL</A>.
  
  @ingroup StatFunc

  */

  //double tdistribution_prob(double x, double r);




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

  //double tdistribution_quant(double x, double r);




  /**

  Cumulative distribution function of the uniform (flat)  
  distribution (upper tail).

  \f[ D(x) = \int_{x}^{+\infty} {1 \over (b-a)} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/UniformDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC301">GSL</A>.
  
  @ingroup StatFunc

  */

  double uniform_prob(double x, double a, double b);




  /**

  Cumulative distribution function of the uniform (flat)  
  distribution (lower tail).

  \f[ D(x) = \int_{-\infty}^{x} {1 \over (b-a)} dx' \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/UniformDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_19.html#SEC301">GSL</A>.
  
  @ingroup StatFunc

  */

  double uniform_quant(double x, double a, double b);






  //@}




} // namespace Math
} // namespace ROOT


#endif // ROOT_Math_ProbFunc
