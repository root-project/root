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


#ifndef ROOT_Math_PdfFuncMathMore
#define ROOT_Math_PdfFuncMathMore

namespace ROOT { 
   namespace Math { 


  /**

  Probability density function of the non central \f$\chi^2\f$ distribution with \f$r\f$ 
  degrees of freedom and the noon-central parameter \f$\lambda\f$ 

  \f[ p_r(x) = \frac{1}{\Gamma(r/2) 2^{r/2}} x^{r/2-1} e^{-x/2} \f]

  for \f$x \geq 0\f$. 
  For detailed description see 
  <A HREF="http://mathworld.wolfram.com/NoncentralChi-SquaredDistribution.html">
  Mathworld</A>. 
  
  @ingroup PdfFunc

  */

  double noncentral_chisquared_pdf(double x, double r, double lambda);

   }  //end namespace Math
} // end namespace ROOT


#endif  // ROOT_Math_PdfFuncMathMore
