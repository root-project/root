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


GSLInterpolator::GSLInterpolator (unsigned int size, Interpolation::Type type) :
   fResetNErrors(true),
   fAccel(0),
   fSpline(0)
{
   // constructor given type and vectors of (x,y) points

   switch ( type )
   {
      case ROOT::Math::Interpolation::kLINEAR          :
         fInterpType = gsl_interp_linear;
         break ;
      case ROOT::Math::Interpolation::kPOLYNOMIAL       :
         fInterpType = gsl_interp_polynomial;
         break ;
         // depened on GSL linear algebra
      case ROOT::Math::Interpolation::kCSPLINE         :
         fInterpType = gsl_interp_cspline ;
         break ;
      case ROOT::Math::Interpolation::kCSPLINE_PERIODIC :
         fInterpType = gsl_interp_cspline_periodic  ;
         break ;
      case ROOT::Math::Interpolation::kAKIMA            :
         fInterpType = gsl_interp_akima;
         break ;
      case ROOT::Math::Interpolation::kAKIMA_PERIODIC   :
         fInterpType = gsl_interp_akima_periodic;
         break ;
      default :
         // cspline
         fInterpType = gsl_interp_cspline;
         break ;
   }
   // allocate objects

   if (size >= fInterpType->min_size)
      fSpline = gsl_spline_alloc( fInterpType, size);

}

bool  GSLInterpolator::Init(unsigned int size, const double *x, const double * y) {
   // initialize interpolation object with the given data
   // if given size is different a new interpolator object is created
   if (fSpline == 0)
      fSpline = gsl_spline_alloc( fInterpType, size);

   else {
      gsl_interp * interp = fSpline->interp;
      if (size != interp->size) {
         //  free and reallocate a new object
         gsl_spline_free(fSpline);
         fSpline = gsl_spline_alloc( fInterpType, size);

      }
   }
   if (!fSpline) return false;

   int iret = gsl_spline_init( fSpline , x , y , size );
   if (iret != 0) return false;

   if(fAccel==0)
      fAccel = gsl_interp_accel_alloc() ;
   else
      gsl_interp_accel_reset(fAccel);

   //  if (fSpline == 0 || fAccel == 0)
   //  throw std::exception();
   assert (fSpline != 0);
   assert (fAccel != 0);
   // reset counter for error messages
   fResetNErrors = true;
   return true;
}

GSLInterpolator::~GSLInterpolator()
{
   // free gsl objects
   if (fSpline != 0) gsl_spline_free(fSpline);
   if (fAccel != 0) gsl_interp_accel_free( fAccel);
}

GSLInterpolator::GSLInterpolator(const GSLInterpolator &)
{
   // dummy copy ctr
}

GSLInterpolator & GSLInterpolator::operator = (const GSLInterpolator &rhs)
{
   // dummy assignment operator
   if (this == &rhs) return *this;  // time saving self-test

   return *this;
}


} // namespace Math
} // namespace ROOT
