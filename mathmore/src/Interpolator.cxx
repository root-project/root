// @(#)root/mathmore:$Name:  $:$Id: Interpolator.cxx,v 1.1 2005/09/08 07:14:56 brun Exp $
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

// Implementation file for class Interpolator using GSL
//
// Created by: moneta  at Fri Nov 26 15:00:25 2004
//
// Last update: Fri Nov 26 15:00:25 2004
//

#include "MathMore/Interpolator.h"
#include "GSLInterpolator.h"


namespace ROOT {
namespace Math {


Interpolator::Interpolator(const std::vector<double> & x, const std::vector<double> & y, Interpolation::Type type)
{
  // allocate GSL interpolation object

  fInterp = new GSLInterpolator(type, x, y);
}

Interpolator::~Interpolator()
{
  if (fInterp) delete fInterp;
}

Interpolator::Interpolator(const Interpolator &)
{
}

Interpolator & Interpolator::operator = (const Interpolator &rhs)
{
   if (this == &rhs) return *this;  // time saving self-test

   return *this;
}

double Interpolator::Eval( double x ) const
{
  return fInterp->Eval(x);
}

double Interpolator::Deriv( double x ) const
{
  return fInterp->Deriv(x);
}

double Interpolator::Deriv2( double x ) const {
  return fInterp->Deriv2(x);
}

double Interpolator::Integ( double a, double b) const {
  return fInterp->Integ(a,b);
}

std::string  Interpolator::TypeGet() const {
  return fInterp->Name();
}



} // namespace Math
} // namespace ROOT
