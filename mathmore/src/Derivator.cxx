// @(#)root/mathmore:$Name:  $:$Id: Derivator.cxx,v 1.3 2006/06/16 10:34:08 moneta Exp $
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

#include "Math/IFunction.h"
#include "Math/Derivator.h"
#include "GSLDerivator.h"


// for GSL greater then 1.5
#include "gsl/gsl_deriv.h"
// for OLD GSL versions
//#include "gsl/gsl_diff.h"

namespace ROOT {
namespace Math {


   
Derivator::Derivator(const IGenFunction &f) 
{
   // allocate GSLDerivator
   fDerivator = new GSLDerivator(f);  
}

Derivator::Derivator(const GSLFuncPointer &f) 
{
   // allocate GSLDerivator
   fDerivator = new GSLDerivator(f);  
}

Derivator::~Derivator() 
{
   if (fDerivator) delete fDerivator;
}


Derivator::Derivator(const Derivator &) 
{
}

Derivator & Derivator::operator = (const Derivator &rhs) 
{
   if (this == &rhs) return *this;  // time saving self-test
   
   return *this;
}


void Derivator::SetFunction(const IGenFunction &f) {
   fDerivator->SetFunction(f);
}

void Derivator::SetFunction( const GSLFuncPointer &f) {
   fDerivator->SetFunction(f);
}


double Derivator::Eval(const IGenFunction & f, double x, double h ) const {
   return fDerivator->Eval(f, x, h);
}

double Derivator::EvalCentral(const IGenFunction & f, double x, double h) const {
   return fDerivator->EvalCentral(f, x, h);
}

double Derivator::EvalForward(const IGenFunction & f, double x, double h) const {
   return fDerivator->EvalForward(f, x, h);
} 

double Derivator::EvalBackward(const IGenFunction & f, double x, double h) const {
   return fDerivator->EvalBackward(f, x, h);
}


double Derivator::Eval( double x, double h) const { 
   return fDerivator->EvalCentral(x, h);
}

double Derivator::EvalCentral( double x, double h) const { 
   return fDerivator->EvalCentral(x, h);
}

double Derivator::EvalForward( double x, double h) const { 
   return fDerivator->EvalForward(x, h);
}

double Derivator::EvalBackward( double x, double h) const { 
   return fDerivator->EvalBackward(x, h);
}


double Derivator::Result() const { return fDerivator->Result(); }

double Derivator::Error() const { return fDerivator->Error(); }

int Derivator::Status() const { return fDerivator->Status(); }



} // namespace Math
} // namespace ROOT
