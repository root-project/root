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

// Implementation file for class GSLRootHelper
// 
// Created by: moneta  at Sun Nov 14 21:34:15 2004
// 
// Last update: Sun Nov 14 21:34:15 2004
// 

#include "Math/GSLRootHelper.h"

#include "gsl/gsl_roots.h"

namespace ROOT {
namespace Math {
   
   namespace GSLRootHelper { 
      
      
      int TestInterval(double xlow, double xup, double epsAbs, double epsRel) { 
         
         return gsl_root_test_interval( xlow, xup, epsAbs, epsRel); 
      } 
      
      int TestDelta(double x1, double x0, double epsAbs, double epsRel) { 
         // be careful is inverted with respect to GSL (don't know why ) 
         return gsl_root_test_delta( x1, x0, epsRel, epsAbs); 
      } 
      
      int TestResidual(double f, double epsAbs) { 
         
         return gsl_root_test_residual( f, epsAbs); 
      } 
      
   }
   
} // namespace Math
} // namespace ROOT
