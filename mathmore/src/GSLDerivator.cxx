// @(#)root/mathmore:$Name:  $:$Id: GSLDerivator.cxxv 1.0 2005/06/23 12:00:00 moneta Exp $
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

namespace ROOT {
namespace Math {



GSLDerivator::GSLDerivator(const IGenFunction &f) 
{
  // allocate GSLFunctionWrapper
   fFunction = new GSLFunctionWrapper(); 
   SetFunction(f);
}

GSLDerivator::GSLDerivator(const GSLFuncPointer &f) 
{
  // allocate GSLFunctionWrapper
   fFunction = new GSLFunctionWrapper(); 
   SetFunction(f);
}

GSLDerivator::~GSLDerivator() 
{
  if (fFunction) delete fFunction;
}


GSLDerivator::GSLDerivator(const GSLDerivator &) 
{
}

GSLDerivator & GSLDerivator::operator = (const GSLDerivator &rhs) 
{
   if (this == &rhs) return *this;  // time saving self-test

   return *this;
}

double GSLDerivator::EvalCentral( double x, double h) { 
  fStatus =  gsl_deriv_central(  fFunction->GetFunc(), x, h, &fResult, &fError); 
  //fStatus =  gsl_diff_central(  fFunction->GetFunc(), x, &fResult, &fError); 
  return fResult;
}

double GSLDerivator::EvalForward( double x, double h) { 
  fStatus =  gsl_deriv_forward(  fFunction->GetFunc(), x, h, &fResult, &fError); 
  //fStatus =  gsl_diff_forward(  fFunction->GetFunc(), x, &fResult, &fError); 
  return fResult;
}

double GSLDerivator::EvalBackward( double x, double h) { 
  fStatus =  gsl_deriv_backward(  fFunction->GetFunc(), x, h, &fResult, &fError); 
  //fStatus =  gsl_diff_backward(  fFunction->GetFunc(), x, &fResult, &fError); 
  return fResult;
}


double GSLDerivator::Result() const { return fResult; }

double GSLDerivator::Error() const { return fError; }

int GSLDerivator::Status() const { return fStatus; }

// fill GSLFunctionWrapper with the pointer to the function

void  GSLDerivator::FillGSLFunction( GSLFuncPointer  fp, void * p) {  
  fFunction->SetFuncPointer( fp ); 
  fFunction->SetParams ( p ); 
}


void  GSLDerivator::FillGSLFunction(const IGenFunction &f) {  
  fFunction->SetFunction(f); 
}

} // namespace Math
} // namespace ROOT
