// @(#)root/mathmore:$Name:  $:$Id: Integrator.h,v 1.2 2006/06/16 10:34:08 moneta Exp $
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

// Implementation file for class Minimizer1D
// 
// Created by: moneta  at Wed Dec  1 15:04:51 2004
// 
// Last update: Wed Dec  1 15:04:51 2004
// 

#include <assert.h>

#include "Math/Minimizer1D.h"

#include "GSLFunctionWrapper.h"
#include "GSL1DMinimizer.h"


#include "gsl/gsl_min.h"
#include "gsl/gsl_errno.h"

#include <iostream> 

namespace ROOT { 

namespace Math { 


Minimizer1D::Minimizer1D(Minim1D::Type type) : 
   fIsSet(false)
{
   // construct a minimizer passing the algorithm type as an enumeration

   const gsl_min_fminimizer_type* T = 0 ;
   switch ( type )
   {
   case Minim1D::GOLDENSECTION          : 
      T = gsl_min_fminimizer_goldensection; 
      break ;
   case Minim1D::BRENT       :
      T = gsl_min_fminimizer_brent; 
      break ;
   default :
      // default case is brent 
      T = gsl_min_fminimizer_brent; 
      break ;
   }

   fMinimizer = new GSL1DMinimizer(T); 
   fFunction  = new GSLFunctionWrapper();

}

Minimizer1D::~Minimizer1D() 
{
   // destructor: clean up minimizer and function pointers 

   if (fMinimizer) delete fMinimizer;
   if (fFunction)  delete  fFunction;
}

Minimizer1D::Minimizer1D(const Minimizer1D &) 
{
   // dummy copy ctr
}

Minimizer1D & Minimizer1D::operator = (const Minimizer1D &rhs) 
{
   // dummy operator = 
   if (this == &rhs) return *this;  // time saving self-test
   return *this;
}

void Minimizer1D::SetFunction(  GSLFuncPointer f, void * p, double xmin, double xlow, double xup) { 
   // set the funciton to be minimized 
   assert(fFunction);
   assert(fMinimizer);
   fXlow = xlow; 
   fXup = xup;
   fXmin = xmin;
   fFunction->SetFuncPointer( f ); 
   fFunction->SetParams( p ); 
   std::cout << " [ "<< xlow << " , " << xup << " ]" << std::endl;

   int status = gsl_min_fminimizer_set( fMinimizer->Get(), fFunction->GetFunc(), xmin, xlow, xup);
   if (status != GSL_SUCCESS) 
      std::cerr <<"Minimizer1D: Error:  Interval [ "<< xlow << " , " << xup << " ] does not contain a minimum" << std::endl; 


   fIsSet = true; 
   return;
}

int Minimizer1D::Iterate() {
   // perform an iteration and update values 
   if (!fIsSet) {
      std::cerr << "Minimizer1D- Error: Function has not been set in Minimizer" << std::endl;
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

double Minimizer1D::XMinimum() const { 
   // return x value at function minimum
   return fXmin;
}

double Minimizer1D::XLower() const { 
   // return lower x value of bracketing interval
   return fXlow; 
}

double Minimizer1D::XUpper() const { 
   // return upper x value of bracketing interval
   return fXup;
}

double Minimizer1D::FValMinimum() const { 
   // return function value at minimum
   return fMin;
}

double Minimizer1D::FValLower() const { 
   // return function value at x lower
   return fLow; 
}

double Minimizer1D::FValUpper() const { 
   // return function value at x upper
   return fUp;
}

const char * Minimizer1D::Name() const {
   // return name of minimization algorithm
   return gsl_min_fminimizer_name(fMinimizer->Get() ); 
}

int Minimizer1D::Minimize (int maxIter, double absTol, double relTol) 
{ 
   // find the minimum via multiple iterations
   int iter = 0; 
   int status = 0; 
   do { 
      iter++;
      try {
         status = Iterate();
      }
      catch ( std::exception &e) { 
         //std::cerr << "Minimization failed : " << e.what() << std::endl; 
         //throw mathlib::MathlibException("Minimize: Cannot perform iterations");
         return -1; 
      }
  
      status =  TestInterval(fXlow, fXup, absTol, relTol); 
      if (status == GSL_SUCCESS) { 
         fIter = iter;
         return status; 
      }
   }
   while (status == GSL_CONTINUE && iter < maxIter); 
   return status;
}


int Minimizer1D::TestInterval( double xlow, double xup, double epsAbs, double epsRel) { 
// static function to test interval 
   return gsl_min_test_interval(xlow, xup, epsAbs, epsRel);
}

} // end namespace Math

} // end namespace ROOT

