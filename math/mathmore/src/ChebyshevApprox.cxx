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

// Implementation file for class ChebyshevApprox
//
// Created by: moneta  at Thu Dec  2 14:51:15 2004
//
// Last update: Thu Dec  2 14:51:15 2004
//


#include "Math/IFunction.h"

#include "Math/ChebyshevApprox.h"
#include "GSLFunctionWrapper.h"
#include "GSLChebSeries.h"

#include "gsl/gsl_chebyshev.h"

#include <cassert>


namespace ROOT {
namespace Math {


ChebyshevApprox::ChebyshevApprox(const ROOT::Math::IGenFunction & f, double a, double b, size_t n) :
   fOrder(n) , fSeries(0), fFunction(0)
{
   // constructor from function (IGenFunction type) and interval [a,b] and series size n
   fSeries = new GSLChebSeries(n);
   GSLFunctionAdapter<ROOT::Math::IGenFunction> adapter;
   const void * p = &f;
   Initialize(  &adapter.F, const_cast<void *>(p), a, b );
}

// constructor with GSL function
ChebyshevApprox::ChebyshevApprox(GSLFuncPointer f, void * params, double a, double b, size_t n) :
fOrder(n) , fSeries(0), fFunction(0)
{
   // constructor from function (GSL type) and interval [a,b] and series size n
   fSeries = new GSLChebSeries(n);
   Initialize(  f, params, a, b );
}

ChebyshevApprox::~ChebyshevApprox()
{
   // desctructor (clean up resources)
   if (fFunction) delete fFunction;
   if (fSeries) delete fSeries;
}

ChebyshevApprox::ChebyshevApprox(size_t n) :
fOrder(n) , fSeries(0), fFunction(0)
{
   // constructor passing only size (need to initialize setting the function afterwards)
   fSeries = new GSLChebSeries(n);
}

ChebyshevApprox::ChebyshevApprox(const ChebyshevApprox & /*cheb */ )
{
   // cannot copy series because don't know original function
}

ChebyshevApprox & ChebyshevApprox::operator = (const ChebyshevApprox &rhs)
{
   // dummy assignment
   if (this == &rhs) return *this;  // time saving self-test

   return *this;
}

void ChebyshevApprox::Initialize( GSLFuncPointer f, void * params, double a, double b) {
   // initialize by passing a function and interval [a,b]
   // delete previous existing function pointer
   assert(fSeries != 0);
   if (fFunction) delete fFunction;

   fFunction = new GSLFunctionWrapper();
   fFunction->SetFuncPointer( f );
   fFunction->SetParams( params );

   // check for errors here ???
   gsl_cheb_init( fSeries->get(), fFunction->GetFunc(), a, b);
}

double ChebyshevApprox::operator() ( double x ) const {
   // evaluate the approximation
   return gsl_cheb_eval(fSeries->get(), x);
}

std::pair<double, double>  ChebyshevApprox::EvalErr( double x) const {
   // evaluate returning result and error
   double result, error;
   gsl_cheb_eval_err(fSeries->get(), x, &result, &error);
   return std::make_pair( result, error);
}

double ChebyshevApprox::operator() ( double x, size_t n) const {
   // evaluate at most order n ( truncate the series)
   return gsl_cheb_eval_n(fSeries->get(), n, x);
}

std::pair<double, double>  ChebyshevApprox::EvalErr( double x, size_t n) const {
   // evaluate at most order n ( truncate the series) returning resutl + error
   double result, error;
   gsl_cheb_eval_n_err(fSeries->get(), n, x, &result, &error);
   return std::make_pair( result, error);
}

ChebyshevApprox *ChebyshevApprox::Deriv()
{
   // calculate derivative. Returns pointer to a new series
   ChebyshevApprox *deriv = new ChebyshevApprox(fOrder);

   // check for errors ?
   gsl_cheb_calc_deriv((deriv->fSeries)->get(), fSeries->get());
   return deriv;
}

ChebyshevApprox *ChebyshevApprox::Integral()
{
   // integral (return pointer)
   ChebyshevApprox *integ = new ChebyshevApprox(fOrder);

   // check for errors ?
   gsl_cheb_calc_integ((integ->fSeries)->get(), fSeries->get());
   return integ;
}

} // namespace Math
} // namespace ROOT
