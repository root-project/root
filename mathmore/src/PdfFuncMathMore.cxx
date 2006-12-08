// @(#)root/mathmore:$Name:  $:$Id: PdfFuncMathMore.cxx,v 1.1 2006/12/07 11:07:03 moneta Exp $
// Authors: Andras Zsenei & Lorenzo Moneta   06/2005 


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



#include <cmath>
#include "Math/ProbFuncMathMore.h"
#include "gsl/gsl_randist.h"



namespace ROOT {
namespace Math {



   
   double beta_pdf(double x, double a, double b ) {
      return gsl_ran_beta_pdf(x,a,b); 
   }
   
   
   double landau_pdf(double x, double sigma, double x0 ) {
      
      double s = (x - x0)/sigma; 
      return gsl_ran_landau_pdf(s); 
            
   }
   
   


} // namespace Math
} // namespace ROOT





