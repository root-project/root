// @(#)root/mathcore:$Id$
// Authors: Andras Zsenei & Lorenzo Moneta   06/2005 

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/



/**

Probability density functions, cumulative distribution functions 
and their inverses (quantiles) for various statistical distributions (continuous and discrete).
Whenever possible the conventions followed are those of the
CRC Concise Encyclopedia of Mathematics, Second Edition
(or <A HREF="http://mathworld.wolfram.com/">Mathworld</A>).
By convention the distributions are centered around 0, so for
example in the case of a Gaussian there is no parameter mu. The
user must calculate the shift himself if he wishes. 

MathCore provides the majority of the probability density functions, of the 
cumulative distributions and of the quantiles (inverses of the cumulatives). 
Additional distributions are also provided by the
<A HREF="../../MathMore/html/group__StatFunc.html">MathMore</A> library. 


@defgroup StatFunc Statistical functions

*/



#if defined(__CINT__) && !defined(__MAKECINT__)
// avoid to include header file when using CINT 
#ifndef _WIN32
#include "../lib/libMathCore.so"
#else
#include "../bin/libMathCore.dll"
#endif

#else


#ifndef ROOT_Math_PdfFuncMathCore
#define ROOT_Math_PdfFuncMathCore




namespace ROOT {
namespace Math {



  /** @defgroup PdfFunc Probability Density Functions (PDF) from MathCore      
   *   @ingroup StatFunc
   *  Probability density functions of various statistical distributions 
   *  (continuous and discrete).
   *  The probability density function returns the probability that 
   *  the variate has the value x. 
   *  In statistics the PDF is also called the frequency function.
   *  
   * 
   */

   /** @name Probability Density Functions from MathCore 
   *   Additional PDF's are provided in the MathMore library
   *   (see PDF functions from MathMore)   
   */ 

  //@{

  /**
     
  Probability density function of the beta distribution.
  
  \f[ p(x) = \frac{\Gamma (a + b) } {\Gamma(a)\Gamma(b) } x ^{a-1} (1 - x)^{b-1} \f]

  for \f$0 \leq x \leq 1 \f$. For detailed description see 
  <A HREF="http://mathworld.wolfram.com/BetaDistribution.html">
  Mathworld</A>. 
  
  @ingroup PdfFunc

  */

  double beta_pdf(double x, double a, double b);


  /**
    
  Probability density function of the binomial distribution.

  \f[ p(k) = \frac{n!}{k! (n-k)!} p^k (1-p)^{n-k} \f]

  for \f$ 0 \leq k \leq n \f$. For detailed description see 
  <A HREF="http://mathworld.wolfram.com/BinomialDistribution.html">
  Mathworld</A>. 
  
  @ingroup PdfFunc

  */

  double binomial_pdf(unsigned int k, double p, unsigned int n);


  /**
    
  Probability density function of the negative binomial distribution.

  \f[ p(k) = \frac{(k+n-1)!}{k! (n-1)!} p^{n} (1-p)^{k} \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/NegativeBinomialDistribution.html">
  Mathworld</A> (where $k \to x$ and $n \to r$).
  The distribution in <A HREF="http://en.wikipedia.org/wiki/Negative_binomial_distribution">
  Wikipedia</A> is defined with a $p$ corresponding to $1-p$ in this case.

  
  @ingroup PdfFunc

  */

  double negative_binomial_pdf(unsigned int k, double p, double n);



  /**

  Probability density function of Breit-Wigner distribution, which is similar, just 
  a different definition of the parameters, to the Cauchy distribution 
  (see  #cauchy_pdf )

  \f[ p(x) = \frac{1}{\pi} \frac{\frac{1}{2} \Gamma}{x^2 + (\frac{1}{2} \Gamma)^2} \f]

  
  @ingroup PdfFunc

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
  
  @ingroup PdfFunc

  */

  double cauchy_pdf(double x, double b = 1, double x0 = 0);




  /**

  Probability density function of the \f$\chi^2\f$ distribution with \f$r\f$ 
  degrees of freedom.

  \f[ p_r(x) = \frac{1}{\Gamma(r/2) 2^{r/2}} x^{r/2-1} e^{-x/2} \f]

  for \f$x \geq 0\f$. For detailed description see 
  <A HREF="http://mathworld.wolfram.com/Chi-SquaredDistribution.html">
  Mathworld</A>. 
  
  @ingroup PdfFunc

  */

  double chisquared_pdf(double x, double r, double x0 = 0);




  /**

  Probability density function of the exponential distribution.

  \f[ p(x) = \lambda e^{-\lambda x} \f]

  for x>0. For detailed description see 
  <A HREF="http://mathworld.wolfram.com/ExponentialDistribution.html">
  Mathworld</A>. 

  
  @ingroup PdfFunc

  */

  double exponential_pdf(double x, double lambda, double x0 = 0);




  /**

  Probability density function of the F-distribution.

  \f[ p_{n,m}(x) = \frac{\Gamma(\frac{n+m}{2})}{\Gamma(\frac{n}{2}) \Gamma(\frac{m}{2})} n^{n/2} m^{m/2} x^{n/2 -1} (m+nx)^{-(n+m)/2} \f]

  for x>=0. For detailed description see 
  <A HREF="http://mathworld.wolfram.com/F-Distribution.html">
  Mathworld</A>. 
  
  @ingroup PdfFunc

  */


  double fdistribution_pdf(double x, double n, double m, double x0 = 0);




  /**

  Probability density function of the gamma distribution.

  \f[ p(x) = {1 \over \Gamma(\alpha) \theta^{\alpha}} x^{\alpha-1} e^{-x/\theta} \f]

  for x>0. For detailed description see 
  <A HREF="http://mathworld.wolfram.com/GammaDistribution.html">
  Mathworld</A>. 
  
  @ingroup PdfFunc

  */

  double gamma_pdf(double x, double alpha, double theta, double x0 = 0);




  /**

  Probability density function of the normal (Gaussian) distribution.

  \f[ p(x) = {1 \over \sqrt{2 \pi \sigma^2}} e^{-x^2 / 2\sigma^2} \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/NormalDistribution.html">
  Mathworld</A>. It can also be evaluated using #normal_pdf which will 
  call the same implementation. 

  @ingroup PdfFunc
 
  */

  double gaussian_pdf(double x, double sigma = 1, double x0 = 0);



   /**

   Probability density function of the Landau distribution:
  \f[ p(x) = \frac{1}{\xi} \phi (\lambda) \f]
   with
   \f[  \phi(\lambda) = \frac{1}{2 \pi i}\int_{c-i\infty}^{c+i\infty} e^{\lambda s + s \log{s}} ds\f]
   where \f$\lambda = (x-x_0)/\xi\f$. For a detailed description see 
   K.S. K&ouml;lbig and B. Schorr, A program package for the Landau distribution, 
   <A HREF="http://dx.doi.org/10.1016/0010-4655(84)90085-7">Computer Phys. Comm. 31 (1984) 97-111</A>
   <A HREF="http://dx.doi.org/10.1016/j.cpc.2008.03.002">[Erratum-ibid. 178 (2008) 972]</A>. 
   The same algorithms as in 
   <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/g110/top.html">
   CERNLIB</A> (DENLAN)  is used 
   
   @param x The argument \f$x\f$ 
   @param xi The width parameter \f$\xi\f$ 
   @param x0 The location parameter \f$x_0\f$ 
   
   @ingroup PdfFunc
   
   */

   double landau_pdf(double x, double xi = 1, double x0 = 0); 



  /**

  Probability density function of the lognormal distribution.

  \f[ p(x) = {1 \over x \sqrt{2 \pi s^2} } e^{-(\ln{x} - m)^2/2 s^2} \f]

  for x>0. For detailed description see 
  <A HREF="http://mathworld.wolfram.com/LogNormalDistribution.html">
  Mathworld</A>. 
  @param s scale parameter (not the sigma of the distribution which is not even defined)
  @param x0  location parameter, corresponds approximatly to the most probable value. For x0 = 0, sigma = 1, the x_mpv = -0.22278
  
  @ingroup PdfFunc

  */

  double lognormal_pdf(double x, double m, double s, double x0 = 0);




  /**

  Probability density function of the normal (Gaussian) distribution.

  \f[ p(x) = {1 \over \sqrt{2 \pi \sigma^2}} e^{-x^2 / 2\sigma^2} \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/NormalDistribution.html">
  Mathworld</A>. It can also be evaluated using #gaussian_pdf which will call the same 
  implementation. 

  @ingroup PdfFunc
 
  */

  double normal_pdf(double x, double sigma =1, double x0 = 0);


  /**

  Probability density function of the Poisson distribution.

  \f[ p(n) = \frac{\mu^n}{n!} e^{- \mu} \f]

  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/PoissonDistribution.html">
  Mathworld</A>. 
  
  @ingroup PdfFunc

  */

  double poisson_pdf(unsigned int n, double mu);




  /**

  Probability density function of Student's t-distribution.

  \f[ p_{r}(x) = \frac{\Gamma(\frac{r+1}{2})}{\sqrt{r \pi}\Gamma(\frac{r}{2})} \left( 1+\frac{x^2}{r}\right)^{-(r+1)/2}  \f]

  for \f$k \geq 0\f$. For detailed description see 
  <A HREF="http://mathworld.wolfram.com/Studentst-Distribution.html">
  Mathworld</A>. 
  
  @ingroup PdfFunc

  */

  double tdistribution_pdf(double x, double r, double x0 = 0);




  /**

  Probability density function of the uniform (flat) distribution.

  \f[ p(x) = {1 \over (b-a)} \f]

  if \f$a \leq x<b\f$ and 0 otherwise. For detailed description see 
  <A HREF="http://mathworld.wolfram.com/UniformDistribution.html">
  Mathworld</A>. 
  
  @ingroup PdfFunc

  */

  double uniform_pdf(double x, double a, double b, double x0 = 0);



} // namespace Math
} // namespace ROOT



#endif // ROOT_Math_PdfFunc

#endif // if defined (__CINT__) && !defined(__MAKECINT__)
