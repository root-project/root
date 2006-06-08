// @(#)root/mathmore:$Name:  $:$Id: GSLInterpolator.h,v 1.2 2005/09/18 20:41:25 brun Exp $
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

#include "Math/InterpolationTypes.h"

#include "gsl/gsl_interp.h"
#include "gsl/gsl_spline.h"


namespace ROOT {
namespace Math {


  /**
     Interpolation class based on GSL interpolation functions
     @ingroup Interpolation
   */

  class GSLInterpolator {

  public: 
    GSLInterpolator(const Interpolation::Type type, const std::vector<double> & x, const std::vector<double> & y ); 
    virtual ~GSLInterpolator(); 

  private:
    // usually copying is non trivial, so we make this unaccessible
    GSLInterpolator(const GSLInterpolator &); 
    GSLInterpolator & operator = (const GSLInterpolator &); 
    
  public: 

    double Eval( double x ) const
    {
      return gsl_spline_eval(fSpline, x, fAccel ); 
    }

    double Deriv( double x ) const 
    {
      return gsl_spline_eval_deriv(fSpline, x, fAccel );  
    }
    
    double Deriv2( double x ) const {  
      return gsl_spline_eval_deriv2(fSpline, x, fAccel );  
    }
    
    double Integ( double a, double b) const { 
      return gsl_spline_eval_integ(fSpline, a, b, fAccel );  
    }

    std::string Name() { 
      //return gsl_interp_name(fInterp); 
      // Name not impl for gsl_spline objects
      return fName;
    }


  protected: 


  private: 

    std::string  fName;
    gsl_interp_accel * fAccel; 
    gsl_spline * fSpline; 

}; 

} // namespace Math
} // namespace ROOT


#endif /* ROOT_Math_GSLInterpolator */
