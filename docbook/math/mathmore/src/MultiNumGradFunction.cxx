// @(#)root/mathmore:$Id$
// Author: L. Moneta Wed Dec 20 14:36:31 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
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

// implementation file for class MultiNumGradFunction

#include "Math/MultiNumGradFunction.h"
#include <limits>
#include <cmath>
#include <algorithm>    // needed for std::max on Solaris

#ifndef ROOT_Math_Derivator
#include "Math/Derivator.h"
#endif


namespace ROOT { 

   namespace Math { 


double MultiNumGradFunction::fgEps = 0.001; 

double MultiNumGradFunction::DoDerivative (const double * x, unsigned int icoord  ) const { 
      // calculate derivative using mathcore derivator class 
   // step size can be changes using SetDerivPrecision()

   static double kPrecision = std::sqrt ( std::numeric_limits<double>::epsilon() );
   double x0 = x[icoord];
   double step = std::max( fgEps* std::abs(x0), 8.0*kPrecision*(std::abs(x0) + kPrecision) );
   return ROOT::Math::Derivator::Eval(*fFunc, x, icoord, step); 
}  

void MultiNumGradFunction::SetDerivPrecision(double eps) { fgEps = eps; }

double MultiNumGradFunction::GetDerivPrecision( ) { return fgEps; }


   } // end namespace Math

} // end namespace ROOT
