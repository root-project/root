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

// Header file for class GSLInterpolator
// 
// Created by: moneta  at Fri Nov 26 15:31:41 2004
// 
// Last update: Fri Nov 26 15:31:41 2004
// 
#ifndef ROOT_Math_GSLInterpolator
#define ROOT_Math_GSLInterpolator

#include <vector>
#include <string>
#include <cassert>

#include "Math/InterpolationTypes.h"

#include "gsl/gsl_interp.h"
#include "gsl/gsl_spline.h"

#include "gsl/gsl_errno.h"
#include "Math/Error.h"

namespace ROOT {
namespace Math {


   /**
   Interpolation class based on GSL interpolation functions
    @ingroup Interpolation
    */
   
   class GSLInterpolator {
      
   public: 

      GSLInterpolator(unsigned int ndata, Interpolation::Type type);

      GSLInterpolator(const Interpolation::Type type, const std::vector<double> & x, const std::vector<double> & y ); 
      virtual ~GSLInterpolator(); 
      
   private:
         // usually copying is non trivial, so we make this unaccessible
         GSLInterpolator(const GSLInterpolator &); 
      GSLInterpolator & operator = (const GSLInterpolator &); 
      
   public: 

      bool Init(unsigned int ndata, const double *x, const double * y); 
         
      double Eval( double x ) const
      {
         assert(fAccel);
         double y = 0; 
         int ierr = gsl_spline_eval_e(fSpline, x, fAccel, &y );  
         if (ierr) MATH_WARN_MSG("GSLInterpolator::Eval",gsl_strerror(ierr) )
         return y;
      }
      
      double Deriv( double x ) const 
      {
         assert(fAccel);
         double deriv = 0; 
         int ierr = gsl_spline_eval_deriv_e(fSpline, x, fAccel, &deriv );  
         if (ierr) MATH_WARN_MSG("GSLInterpolator::Deriv",gsl_strerror(ierr) )
         return deriv;
      }
      
      double Deriv2( double x ) const {  
         assert(fAccel);
         double deriv2 = 0; 
         int ierr = gsl_spline_eval_deriv2_e(fSpline, x, fAccel, &deriv2 );  
         if (ierr) MATH_WARN_MSG("GSLInterpolator::Deriv2",gsl_strerror(ierr) )
         return deriv2;
      }
      
      double Integ( double a, double b) const { 
         if ( a > b) return -Integ(b,a);  // gsl will report an error in this case
         assert(fAccel);
         double result = 0; 
         int ierr = gsl_spline_eval_integ_e(fSpline, a, b, fAccel, &result );  
         if (ierr) MATH_WARN_MSG("GSLInterpolator::Integ",gsl_strerror(ierr) )
         return result;
      }
      
      std::string Name() { 
         return fInterpType->name; 
      }
      
      
   protected: 
         
         
   private: 
         
      gsl_interp_accel * fAccel; 
      gsl_spline * fSpline; 
      const gsl_interp_type * fInterpType;
      
   }; 
   
} // namespace Math
} // namespace ROOT


#endif /* ROOT_Math_GSLInterpolator */
