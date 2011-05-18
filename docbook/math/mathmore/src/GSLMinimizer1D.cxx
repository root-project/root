// @(#)root/mathmore:$Id$
// Authors: L. Moneta, A. Zsenei   08/2005
 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2004 moneta,  CERN/PH-SFT                            *
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

// Implementation file for class GSLMinimizer1D
// 
// Created by: moneta  at Wed Dec  1 15:04:51 2004
// 
// Last update: Wed Dec  1 15:04:51 2004
// 

#include <assert.h>

#include "Math/GSLMinimizer1D.h"
#include "Math/Error.h"

#include "GSLFunctionWrapper.h"
#include "GSL1DMinimizerWrapper.h"


#include "gsl/gsl_min.h"
#include "gsl/gsl_errno.h"

#include <iostream> 
#include <cmath>

namespace ROOT { 

namespace Math { 


GSLMinimizer1D::GSLMinimizer1D(Minim1D::Type type) : 
    fXmin(0), fXlow(0), fXup(0), fMin(0), fLow(0), fUp(0), 
    fIter(0), fStatus(-1), fIsSet(false), 
    fMinimizer(0), fFunction(0)
{
   // construct a minimizer passing the algorithm type as an enumeration

   const gsl_min_fminimizer_type* T = 0 ;
   switch ( type )
   {
   case Minim1D::kGOLDENSECTION          : 
      T = gsl_min_fminimizer_goldensection; 
      break ;
   case Minim1D::kBRENT       :
      T = gsl_min_fminimizer_brent; 
      break ;
   default :
      // default case is brent 
      T = gsl_min_fminimizer_brent; 
      break ;
   }

   fMinimizer = new GSL1DMinimizerWrapper(T); 
   fFunction  = new GSLFunctionWrapper();

}

GSLMinimizer1D::~GSLMinimizer1D() 
{
   // destructor: clean up minimizer and function pointers 

   if (fMinimizer) delete fMinimizer;
   if (fFunction)  delete  fFunction;
}

GSLMinimizer1D::GSLMinimizer1D(const GSLMinimizer1D &): IMinimizer1D()
{
   // dummy copy ctr
}

GSLMinimizer1D & GSLMinimizer1D::operator = (const GSLMinimizer1D &rhs) 
{
   // dummy operator = 
   if (this == &rhs) return *this;  // time saving self-test
   return *this;
}

void GSLMinimizer1D::SetFunction(  GSLFuncPointer f, void * p, double xmin, double xlow, double xup) { 
   // set the function to be minimized 
   assert(fFunction);
   assert(fMinimizer);
   fXlow = xlow; 
   fXup = xup;
   fXmin = xmin;
   fFunction->SetFuncPointer( f ); 
   fFunction->SetParams( p ); 

#ifdef DEBUG
   std::cout << " [ "<< xlow << " , " << xup << " ]" << std::endl;
#endif

   int status = gsl_min_fminimizer_set( fMinimizer->Get(), fFunction->GetFunc(), xmin, xlow, xup);
   if (status != GSL_SUCCESS) 
      std::cerr <<"GSLMinimizer1D: Error:  Interval [ "<< xlow << " , " << xup << " ] does not contain a minimum" << std::endl; 


   fIsSet = true; 
   fStatus = -1;
   return;
}

int GSLMinimizer1D::Iterate() {
   // perform an iteration and update values 
   if (!fIsSet) {
      std::cerr << "GSLMinimizer1D- Error: Function has not been set in Minimizer" << std::endl;
      return -1; 
   }
 
   int status =  gsl_min_fminimizer_iterate(fMinimizer->Get());
   // update values
   fXmin = gsl_min_fminimizer_x_minimum(fMinimizer->Get() );
   fMin = gsl_min_fminimizer_f_minimum(fMinimizer->Get() );
   // update interval values
   fXlow =  gsl_min_fminimizer_x_lower(fMinimizer->Get() ); 
   fXup =  gsl_min_fminimizer_x_upper(fMinimizer->Get() );
   fLow =  gsl_min_fminimizer_f_lower(fMinimizer->Get() ); 
   fUp =  gsl_min_fminimizer_f_upper(fMinimizer->Get() );
   return status;
}

double GSLMinimizer1D::XMinimum() const { 
   // return x value at function minimum
   return fXmin;
}

double GSLMinimizer1D::XLower() const { 
   // return lower x value of bracketing interval
   return fXlow; 
}

double GSLMinimizer1D::XUpper() const { 
   // return upper x value of bracketing interval
   return fXup;
}

double GSLMinimizer1D::FValMinimum() const { 
   // return function value at minimum
   return fMin;
}

double GSLMinimizer1D::FValLower() const { 
   // return function value at x lower
   return fLow; 
}

double GSLMinimizer1D::FValUpper() const { 
   // return function value at x upper
   return fUp;
}

const char * GSLMinimizer1D::Name() const {
   // return name of minimization algorithm
   return gsl_min_fminimizer_name(fMinimizer->Get() ); 
}

bool GSLMinimizer1D::Minimize (int maxIter, double absTol, double relTol) 
{ 
   // find the minimum via multiple iterations
   fStatus = -1; 
   int iter = 0; 
   int status = 0; 
   do { 
      iter++;
      status = Iterate();
      if (status != GSL_SUCCESS) { 
         MATH_ERROR_MSG("GSLMinimizer1D::Minimize","error returned when performing an iteration");
         fStatus = status; 
         return false; 
      }

#ifdef DEBUG
      std::cout << "Min1D - iteration " << iter << " interval : [ " << fXlow << "  , " << fXup << " ]  min = " << fXmin
                << " fmin " << fMin << " f(a) " << fLow << " f(b) " << fUp << std::endl;
#endif

  
      status =  TestInterval(fXlow, fXup, absTol, relTol); 
      if (status == GSL_SUCCESS) { 
         fIter = iter;
         fStatus = status;
         return true; 
      }
   }
   while (status == GSL_CONTINUE && iter < maxIter); 
   if (status == GSL_CONTINUE) { 
      double tol = std::abs(fXup-fXlow);
      MATH_INFO_MSGVAL("GSLMinimizer1D::Minimize","exceeded max iterations, reached tolerance is not sufficient",tol);
   }
   fStatus = status; 
   return false;
}


int GSLMinimizer1D::TestInterval( double xlow, double xup, double epsAbs, double epsRel) { 
// static function to test interval 
   return gsl_min_test_interval(xlow, xup, epsAbs, epsRel);
}

} // end namespace Math

} // end namespace ROOT

