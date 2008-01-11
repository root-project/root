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

// Implementation file for class Interpolator using GSL
//
// Created by: moneta  at Fri Nov 26 15:00:25 2004
//
// Last update: Fri Nov 26 15:00:25 2004
//

#include "Math/Interpolator.h"
#include "GSLInterpolator.h"


namespace ROOT {
namespace Math {

Interpolator::Interpolator(unsigned int ndata, Interpolation::Type type ) { 
   // allocate GSL interpolaiton object 
   fInterp = new GSLInterpolator(ndata, type);
}

Interpolator::Interpolator(const std::vector<double> & x, const std::vector<double> & y, Interpolation::Type type)
{
   // allocate and initialize GSL interpolation object with data

   size_t size = std::min( x.size(), y.size() );
   
   fInterp = new GSLInterpolator(size, type);

   fInterp->Init(size, &x.front(), &y.front() );

}


Interpolator::~Interpolator()
{
   // destructor (delete underlined obj)
   if (fInterp) delete fInterp;
}

Interpolator::Interpolator(const Interpolator &)
{
}

Interpolator & Interpolator::operator = (const Interpolator &rhs)
{
   // dummy (private) assignment 
   if (this == &rhs) return *this;  // time saving self-test
   
   return *this;
}

bool Interpolator::SetData(unsigned int ndata, const double * x, const double *y) { 
   // set the interpolation data
   return fInterp->Init(ndata, x, y); 
}
bool Interpolator::SetData(const std::vector<double> & x, const std::vector<double> &y) { 
   // set the interpolation data
   size_t size = std::min( x.size(), y.size() );
   return fInterp->Init(size, &x.front(), &y.front()); 
}


double Interpolator::Eval( double x ) const
{
   // forward evaluation
   return fInterp->Eval(x);
}

double Interpolator::Deriv( double x ) const
{
   // forward deriv evaluation   
   return fInterp->Deriv(x);
}

double Interpolator::Deriv2( double x ) const {
   // forward deriv evaluation   
   return fInterp->Deriv2(x);
}

double Interpolator::Integ( double a, double b) const {
   // forward integ evaluation
   return fInterp->Integ(a,b);
}

std::string  Interpolator::TypeGet() const {
   // forward name request
   return fInterp->Name();
}
std::string  Interpolator::Type() const {
   // forward name request
   return fInterp->Name();
}



} // namespace Math
} // namespace ROOT
