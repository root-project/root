// @(#)root/mathmore:$Name:  $:$Id: ProbFuncMathMore.h,v 1.2 2006/12/06 17:53:47 moneta Exp $
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

/**

Probability density functions, cumulative distribution functions 
and their inverses of the different distributions.
Whenever possible the conventions followed are those of the
CRC Concise Encyclopedia of Mathematics, Second Edition
(or <A HREF="http://mathworld.wolfram.com/">Mathworld</A>).
By convention the distributions are centered around 0, so for
example in the case of a Gaussian there is no parameter mu. The
user must calculate the shift himself if he wishes.


@author Created by Andras Zsenei on Wed Nov 17 2004

@defgroup StatFunc Statistical functions

*/


#ifndef ROOT_Math_PdfFuncMathMore
#define ROOT_Math_PdfFuncMathMore

namespace ROOT {
namespace Math {

  /** @name Probability Density Functions (PDF)
   *  Probability density functions of various distributions.
   *  The probability density function returns the probability that 
   *  the variate has the value x. 
   *  In statistics the PDF is called also as the frequency function.
   *   
   */

  //@{

  /**
     
  Probability density function of the beta distribution.
  
  \f[ p(x) = \frac{\Gamma (a + b) } {\Gamma(a)\Gamma(b) } x ^{a-1} (1 - x)^{b-1} \f]

  for \f$0 \leq x \leq 1 \f$. For detailed description see 
  <A HREF="http://mathworld.wolfram.com/BetaDistribution.html">
  Mathworld</A>. The implementation used is that of 
  <A HREF="http://www.gnu.org/software/gsl/manual/html_node/The-Beta-Distribution.html">GSL</A>.
  
  @ingroup StatFunc

  */

  double beta_pdf(double x, double a, double b);



   /**

   Probability density function of the Landau distribution.
   
   \f[  p(x) = \frac{1}{2 \pi i}\int_{c-i\infty}^{c+i\infty} e^{x s + s \log{s}} ds\f]
   
   Where s = (x-x0)/sigma. For detailed description see 
   <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/g110/top.html">
   CERNLIB</A>. 
   
   @ingroup StatFunc
   
   */

   double landau_pdf(double x, double sigma = 1, double x0 = 0.); 

  /**

  Multinomial distribution probability density function

  http://mathworld.wolfram.com/MultinomialDistribution.html

  */

  //double multinomial_pdf(const size_t k, const double p[], const unsigned int n[]);


  //@}


} // namespace Math
} // namespace ROOT


#endif // ROOT_Math_ProbFuncMathMore
