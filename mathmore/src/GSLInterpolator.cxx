// @(#)root/mathmore:$Name:  $:$Id: GSLInterpolator.cxx,v 1.1 2005/09/08 07:14:56 brun Exp $
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

// Implementation file for class GSLInterpolator
// 
// Created by: moneta  at Sun Nov 28 08:54:48 2004
// 
// Last update: Sun Nov 28 08:54:48 2004
// 

#include "GSLInterpolator.h"

#include <cassert>

namespace ROOT {
namespace Math {


GSLInterpolator::GSLInterpolator (const Interpolation::Type type, const std::vector<double> & x, const std::vector<double> & y) 
{

  const gsl_interp_type* T = 0 ;
  switch ( type )
    {
    case ROOT::Math::Interpolation::LINEAR          : 
      T = gsl_interp_linear; 
      fName = "Linear";
      break ;
    case ROOT::Math::Interpolation::POLYNOMIAL       :
      T = gsl_interp_polynomial; 
      fName = "Polynomial";
      break ;
      // to be implemented : dpened on GSL linear algebra
//     case ROOT::Math::Interpolation::CSPLINE         :
//       T = gsl_interp_cspline ;          
//       fName = "Cspline";
//       break ;
//     case ROOT::Math::Interpolation::CSPLINE_PERIODIC :
//       T = gsl_interp_cspline_periodic  ; 
//       fName = "Cspline_Periodic";
//       break ;
    case ROOT::Math::Interpolation::AKIMA            :
      T = gsl_interp_akima; 
      fName = "Akima";
      break ;
    case ROOT::Math::Interpolation::AKIMA_PERIODIC   :
      T = gsl_interp_akima_periodic; 
      fName = "Akima_Periodic";
      break ;
    default :
      //  T = gsl_interp_cspline   
      T = gsl_interp_akima; 
      fName = "Akima";
      break ;
    }
    
  // allocate objects

  size_t size = std::min( x.size(), y.size() );

  fSpline = gsl_spline_alloc( T, size); 
  // should check here the return Status 
  gsl_spline_init( fSpline , &x.front() , &y.front() , size ) ;

  fAccel  = gsl_interp_accel_alloc() ; 

  //  if (fSpline == 0 || fAccel == 0) 
  //  throw std::exception();
  assert (fSpline != 0); 
  assert (fAccel != 0); 
}

GSLInterpolator::~GSLInterpolator() 
{
  // free gsl objects
  gsl_spline_free(fSpline); 
  gsl_interp_accel_free( fAccel);
}

GSLInterpolator::GSLInterpolator(const GSLInterpolator &) 
{
}

GSLInterpolator & GSLInterpolator::operator = (const GSLInterpolator &rhs) 
{
   if (this == &rhs) return *this;  // time saving self-test

   return *this;
}


} // namespace Math
} // namespace ROOT
