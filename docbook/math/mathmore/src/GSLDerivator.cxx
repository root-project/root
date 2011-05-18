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

// Implementation file for class GSLDerivator
// 
// Created by: moneta  at Sat Nov 13 14:46:00 2004
// 
// Last update: Sat Nov 13 14:46:00 2004
// 

#include "GSLDerivator.h"

#include "GSLFunctionWrapper.h"
// for GSL greater then 1.5
#include "gsl/gsl_deriv.h"
// for OLD GSL versions
//#include "gsl/gsl_diff.h"

#include <iostream>

namespace ROOT {
namespace Math {



double GSLDerivator::EvalCentral( double x, double h) {    
   // Central evaluation using previously set function  
   if ( !fFunction.IsValid() ) { 
      std::cerr << "GSLDerivator: Error : The function has not been specified" << std::endl;
      fStatus = -1; 
      return 0; 
   }
   fStatus =  gsl_deriv_central(  fFunction.GetFunc(), x, h, &fResult, &fError); 
   return fResult;
}

double GSLDerivator::EvalForward( double x, double h) {
   // Forward evaluation using previously set function  
   if ( !fFunction.IsValid() ) { 
      std::cerr << "GSLDerivator: Error : The function has not been specified" << std::endl;
      fStatus = -1; 
      return 0; 
   }
   fStatus =  gsl_deriv_forward(  fFunction.GetFunc(), x, h, &fResult, &fError); 
   return fResult;
}

double GSLDerivator::EvalBackward( double x, double h) { 
   // Backward evaluation using previously set function  
   if ( !fFunction.IsValid() ) { 
      std::cerr << "GSLDerivator: Error : The function has not been specified" << std::endl;
      fStatus = -1; 
      return 0; 
   }
   fStatus =  gsl_deriv_backward(  fFunction.GetFunc(), x, h, &fResult, &fError); 
   return fResult;
}

// static methods not requiring the function
double GSLDerivator::EvalCentral(const IGenFunction & f, double x, double h) { 
   // Central evaluation using given function 
   GSLFunctionWrapper gslfw; 
   double result, error = 0; 
   gslfw.SetFunction(f); 
   gsl_deriv_central(  gslfw.GetFunc(), x, h, &result, &error);
   return result;
}

double GSLDerivator::EvalForward(const IGenFunction & f, double x, double h) { 
   // Forward evaluation using given function 
   GSLFunctionWrapper gslfw; 
   double result, error = 0; 
   gslfw.SetFunction(f); 
   gsl_deriv_forward(  gslfw.GetFunc(), x, h, &result, &error);
   return result;
}

double GSLDerivator::EvalBackward(const IGenFunction & f, double x, double h) { 
   // Backward evaluation using given function 
   GSLFunctionWrapper gslfw; 
   double result, error = 0; 
   gslfw.SetFunction(f); 
   gsl_deriv_backward(  gslfw.GetFunc(), x, h, &result, &error);
   return result;
}


double GSLDerivator::Result() const { return fResult; }

double GSLDerivator::Error() const { return fError; }

int GSLDerivator::Status() const { return fStatus; }

// fill GSLFunctionWrapper with the pointer to the function

void  GSLDerivator::SetFunction( GSLFuncPointer  fp, void * p) {  
  fFunction.SetFuncPointer( fp ); 
  fFunction.SetParams ( p ); 
}


void  GSLDerivator::SetFunction(const IGenFunction &f) {  
  fFunction.SetFunction(f); 
}

} // namespace Math
} // namespace ROOT
