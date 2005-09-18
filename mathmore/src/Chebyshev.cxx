// @(#)root/mathmore:$Name:  $:$Id: Chebyshev.cxx,v 1.1 2005/09/08 07:14:56 brun Exp $
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

// Implementation file for class Chebyshev
// 
// Created by: moneta  at Thu Dec  2 14:51:15 2004
// 
// Last update: Thu Dec  2 14:51:15 2004
// 

#include <assert.h>

#include "Math/Chebyshev.h"
#include "Math/IGenFunction.h"
#include "GSLFunctionWrapper.h"
#include "GSLChebSeries.h"

#include "gsl/gsl_chebyshev.h"


namespace ROOT {
namespace Math {


Chebyshev::Chebyshev(const ROOT::Math::IGenFunction & f, double a, double b, size_t n) : 
  fOrder(n) , fSeries(0), fFunction(0)
{
  fSeries = new GSLChebSeries(n); 
  GSLFunctionAdapter<ROOT::Math::IGenFunction> adapter;
  const void * p = &f; 
  Initialize(  &adapter.F, const_cast<void *>(p), a, b );     
}

// constructor with GSL function
Chebyshev::Chebyshev(GSLFuncPointer f, void * params, double a, double b, size_t n) :
  fOrder(n) , fSeries(0), fFunction(0)
{
  fSeries = new GSLChebSeries(n); 
  Initialize(  f, params, a, b ); 
}

Chebyshev::~Chebyshev() 
{
  if (fFunction) delete fFunction;
  if (fSeries) delete fSeries;
}

Chebyshev::Chebyshev(size_t n) : 
  fOrder(n) , fSeries(0), fFunction(0)
{
  fSeries = new GSLChebSeries(n); 
}

// cannot copy series because don't know original function
Chebyshev::Chebyshev(const Chebyshev & /*cheb */ )  
{
}

Chebyshev & Chebyshev::operator = (const Chebyshev &rhs) 
{
   if (this == &rhs) return *this;  // time saving self-test

   return *this;
}

void Chebyshev::Initialize( GSLFuncPointer f, void * params, double a, double b) { 
  // delete previous existing one
  assert(fSeries); 
  if (fFunction) delete fFunction;
  
  fFunction = new GSLFunctionWrapper(); 
  fFunction->SetFuncPointer( f ); 
  fFunction->SetParams( params ); 
  
  // check for errors here ???
  gsl_cheb_init( fSeries->get(), fFunction->GetFunc(), a, b); 
}

double Chebyshev::operator() ( double x ) const { 
  return gsl_cheb_eval(fSeries->get(), x);
} 

std::pair<double, double>  Chebyshev::EvalErr( double x) const { 
  double result, error; 
  gsl_cheb_eval_err(fSeries->get(), x, &result, &error);
  return std::make_pair( result, error); 
}

double Chebyshev::operator() ( double x, size_t n) const {
  return gsl_cheb_eval_n(fSeries->get(), n, x);
}

std::pair<double, double>  Chebyshev::EvalErr( double x, size_t n) const { 
  double result, error; 
  gsl_cheb_eval_n_err(fSeries->get(), n, x, &result, &error);
  return std::make_pair( result, error); 
}


// need to return auto_ptr because copying is not supported. 
std::auto_ptr<Chebyshev>  Chebyshev::Deriv() { 

  Chebyshev * deriv = new Chebyshev(fOrder); 
  
  // check for errors ? 
  gsl_cheb_calc_deriv( (deriv->fSeries)->get(), fSeries->get() );
  std::auto_ptr<Chebyshev> pDeriv(deriv);
  return pDeriv;  
}
  
std::auto_ptr<Chebyshev>  Chebyshev::Integral() { 

  Chebyshev * integ = new Chebyshev(fOrder); 
  
  // check for errors ? 
  gsl_cheb_calc_integ( (integ->fSeries)->get(), fSeries->get() );
  std::auto_ptr<Chebyshev> pInteg(integ);
  return pInteg;  
}

} // namespace Math
} // namespace ROOT
