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

// Header file for class GSLRootHelper
// 
// Created by: moneta  at Sun Nov 14 21:34:15 2004
// 
// Last update: Sun Nov 14 21:34:15 2004
// 
#ifndef ROOT_Math_GSLRootHelper
#define ROOT_Math_GSLRootHelper


namespace ROOT {
namespace Math {

  
  /** 
      Helper functions to test convergence of Root-Finding algorithms. 
      Used by ROOT::Math::RootFinder class (see there for the doc)
  */


   namespace GSLRootHelper {

     int TestInterval(double xlow, double xup, double epsAbs, double epsRel); 

     int TestDelta(double x1, double x0, double epsAbs, double epsRel); 

     int  TestResidual(double f,  double epsAbs);
   
   }

} // namespace Math
} // namespace ROOT


#endif /* ROOT_Math_GSLRootHelper */
