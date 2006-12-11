// @(#)root/mathcore:$Name:  $:$Id: PdfFuncMathCore.h,v 1.1 2006/12/08 16:42:52 moneta Exp $
// Authors: Andras Zsenei & Lorenzo Moneta   06/2005 

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/



/**

Probability density functions, cumulative distribution functions 
and their inverses for various statistical distributions (continuous and discrete).
Whenever possible the conventions followed are those of the
CRC Concise Encyclopedia of Mathematics, Second Edition
(or <A HREF="http://mathworld.wolfram.com/">Mathworld</A>).
By convention the distributions are centered around 0, so for
example in the case of a Gaussian there is no parameter mu. The
user must calculate the shift himself if he wishes. 

MathCore provides the majority of the probability density funcitons and a sub-set of the 
cumulative distributions. Additional distributions and all the inverses cumulative distributions 
(quantiles) are provided by <A HREF="../../MathMore/html/group__StatFunc.html">MathMore</A> library. 


@author Created by Andras Zsenei on Wed Nov 17 2004

@defgroup StatFunc Statistical functions

*/






#ifndef ROOT_Math_PdfFuncMathCore
#define ROOT_Math_PdfFuncMathCore




namespace ROOT {
namespace Math {



  /** @name Probability Density Functions (PDF)
   *  Probability density functions of various statistical distributions 
   *  (continuous and discrete).
   *  The probability density function returns the probability that 
   *  the variate has the value x. 
   *  In statistics the PDF is also called the frequency function.
   *  
   *  Additional PDF's are provided in the 
   *  <A HREF="../../MathMore/html/group__StatFunc.html">MathMore</A> library. 
   * 
   */
  //@{

  /**
    
  Probability density function of the binomial distribution.

  \f[ p(k) = \frac{n!}{k! (n-k)!} p^k (1-p)^{n-k} \f]

  for \f$ 0 \leq k \leq n \f$. For detailed description see 
  <A HREF="http://mathworld.wolfram.com/BinomialDistribution.html">
  Mathworld</A>. 
  
  @ingroup StatFunc

  */

  double binomial_pdf(unsigned int k, double p, unsigned int n);




  /**

  Probability density function of Breit-Wigner distribution, which is similar, just 
  a different definition of the parameters, to the Cauchy distribution 
  (see  #cauchy_pdf )

  \f[ p(x) = \frac{1}{\pi} \frac{\frac{1}{2} \Gamma}{x^2 + (\frac{1}{2} \Gamma)^2} \f]

  
  @ingroup StatFunc

  */

  double breitwigner_pdf(double x, double gamma, double x0 = 0);




  /**

  Probability density function of the Cauchy distribution which is also
  called Lorentzian distribution.

  
  \f[ p(x) = \frac{1}{\pi} \frac{ b }{ (x-m)^2 + b^2} \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/CauchyDistribution.html">
  Mathworld</A>. It is also related to the #breitwigner_pdf which 
  will call the same implementation.
  
  @ingroup StatFunc

  */

  double cauchy_pdf(double x, double b = 1, double x0 = 0);




  /**

  Probability density function of the \f$\chi^2\f$ distribution with \f$r\f$ 
  degrees of freedom.

  \f[ p_r(x) = \frac{1}{\Gamma(r/2) 2^{r/2}} x^{r/2-1} e^{-x/2} \f]

  for \f$x \geq 0\f$. For detailed description see 
  <A HREF="http://mathworld.wolfram.com/Chi-SquaredDistribution.html">
  Mathworld</A>. 
  
  @ingroup StatFunc

  */

  double chisquared_pdf(double x, double r, double x0 = 0);




  /**

  Probability density function of the exponential distribution.

  \f[ p(x) = \lambda e^{-\lambda x} \f]

  for x>0. For detailed description see 
  <A HREF="http://mathworld.wolfram.com/ExponentialDistribution.html">
  Mathworld</A>. 

  
  @ingroup StatFunc

  */

  double exponential_pdf(double x, double lambda, double x0 = 0);




  /**

  Probability density function of the F-distribution.

  \f[ p_{n,m}(x) = \frac{\Gamma(\frac{n+m}{2})}{\Gamma(\frac{n}{2}) \Gamma(\frac{m}{2})} n^{n/2} m^{m/2} x^{n/2 -1} (m+nx)^{-(n+m)/2} \f]

  for x>=0. For detailed description see 
  <A HREF="http://mathworld.wolfram.com/F-Distribution.html">
  Mathworld</A>. 
  
  @ingroup StatFunc

  */


  double fdistribution_pdf(double x, double n, double m, double x0 = 0);




  /**

  Probability density function of the gamma distribution.

  \f[ p(x) = {1 \over \Gamma(\alpha) \theta^{\alpha}} x^{\alpha-1} e^{-x/\theta} \f]

  for x>0. For detailed description see 
  <A HREF="http://mathworld.wolfram.com/GammaDistribution.html">
  Mathworld</A>. 
  
  @ingroup StatFunc

  */

  double gamma_pdf(double x, double alpha, double theta, double x0 = 0);




  /**

  Probability density function of the normal (Gaussian) distribution.

  \f[ p(x) = {1 \over \sqrt{2 \pi \sigma^2}} e^{-x^2 / 2\sigma^2} \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/NormalDistribution.html">
  Mathworld</A>. It can also be evaluated using #normal_pdf which will 
  call the same implementation. 

  @ingroup StatFunc
 
  */

  double gaussian_pdf(double x, double sigma, double x0 = 0);





  /**

  Probability density function of the lognormal distribution.

  \f[ p(x) = {1 \over x \sqrt{2 \pi s^2} } e^{-(\ln{x} - m)^2/2 s^2} \f]

  for x>0. For detailed description see 
  <A HREF="http://mathworld.wolfram.com/LogNormalDistribution.html">
  Mathworld</A>. 

  
  @ingroup StatFunc

  */

  double lognormal_pdf(double x, double m, double s, double x0 = 0);




  /**

  Probability density function of the normal (Gaussian) distribution.

  \f[ p(x) = {1 \over \sqrt{2 \pi \sigma^2}} e^{-x^2 / 2\sigma^2} \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/NormalDistribution.html">
  Mathworld</A>. It can also be evaluated using #gaussian_pdf which will call the same 
  implementation. 

  @ingroup StatFunc
 
  */

  double normal_pdf(double x, double sigma, double x0 = 0);




  /**

  Probability density function of the Poisson distribution.

  \f[ p(n) = \frac{\mu^n}{n!} e^{- \mu} \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/PoissonDistribution.html">
  Mathworld</A>. 
  
  @ingroup StatFunc

  */

  double poisson_pdf(unsigned int n, double mu);




  /**

  Probability density function of Student's t-distribution.

  \f[ p_{r}(x) = \frac{\Gamma(\frac{r+1}{2})}{\sqrt{r \pi}\Gamma(\frac{r}{2})} \left( 1+\frac{x^2}{r}\right)^{-(r+1)/2}  \f]

  for \f$k \geq 0\f$. For detailed description see 
  <A HREF="http://mathworld.wolfram.com/Studentst-Distribution.html">
  Mathworld</A>. 
  
  @ingroup StatFunc

  */

  double tdistribution_pdf(double x, double r, double x0 = 0);




  /**

  Probability density function of the uniform (flat) distribution.

  \f[ p(x) = {1 \over (b-a)} \f]

  if \f$a \leq x<b\f$ and 0 otherwise. For detailed description see 
  <A HREF="http://mathworld.wolfram.com/UniformDistribution.html">
  Mathworld</A>. 
  
  @ingroup StatFunc

  */

  double uniform_pdf(double x, double a, double b, double x0 = 0);










} // namespace Math
} // namespace ROOT



#endif // ROOT_Math_PdfFunc
