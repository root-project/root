// @(#)root/mathmore:$Name:  $:$Id: RootFinderAlgorithms.cxx,v 1.1 2005/09/08 07:14:56 brun Exp $
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

// Implementation file for class GSLRootFinderAlgorithms
// 
// Created by: moneta  at Sun Nov 14 14:07:50 2004
// 
// Last update: Sun Nov 14 14:07:50 2004
// 

#include "Math/RootFinderAlgorithms.h"
#include "GSLRootFSolver.h"
#include "GSLRootFdFSolver.h"

#include "gsl/gsl_roots.h"

namespace ROOT {
namespace Math {


namespace Roots { 


Bisection::Bisection() 
{
  GSLRootFSolver * s = new GSLRootFSolver( gsl_root_fsolver_bisection ); 
  SetSolver(s); 
}

Bisection::~Bisection() 
{
  FreeSolver();  
}

Bisection::Bisection(const Bisection &) : GSLRootFinder()
{
}

Bisection & Bisection::operator = (const Bisection &rhs) 
{
   if (this == &rhs) return *this;  // time saving self-test
   return *this;
}

// falsepos method

FalsePos::FalsePos() 
{
  GSLRootFSolver * s = new GSLRootFSolver( gsl_root_fsolver_falsepos ); 
  SetSolver(s); 
}

FalsePos::~FalsePos() 
{
  FreeSolver();  
}

FalsePos::FalsePos(const FalsePos &) : GSLRootFinder()
{
}

FalsePos & FalsePos::operator = (const FalsePos &rhs) 
{
   if (this == &rhs) return *this;  // time saving self-test
   return *this;
}

// Brent method

Brent::Brent() 
{
  GSLRootFSolver * s = new GSLRootFSolver( gsl_root_fsolver_brent ); 
  SetSolver(s); 
}

Brent::~Brent() 
{
  FreeSolver();  
}

Brent::Brent(const Brent &) : GSLRootFinder()
{
}

Brent & Brent::operator = (const Brent &rhs) 
{
   if (this == &rhs) return *this;  // time saving self-test
   return *this;
}


//---------------------------------------------------------------------
// algorithms with Derivatives 
//--------------------------------------------------------------------

// Newton

Newton::Newton() 
{
  GSLRootFdFSolver * s = new GSLRootFdFSolver( gsl_root_fdfsolver_newton ); 
  SetSolver(s); 
}

Newton::~Newton() 
{
  FreeSolver();  
}

Newton::Newton(const Newton &) : GSLRootFinderDeriv()
{
}

Newton & Newton::operator = (const Newton &rhs) 
{
   if (this == &rhs) return *this;  // time saving self-test
   return *this;
}

// Secant

Secant::Secant() 
{
  GSLRootFdFSolver * s = new GSLRootFdFSolver( gsl_root_fdfsolver_secant ); 
  SetSolver(s); 
}

Secant::~Secant() 
{
  FreeSolver();  
}

Secant::Secant(const Secant &) : GSLRootFinderDeriv()
{
}

Secant & Secant::operator = (const Secant &rhs) 
{
   if (this == &rhs) return *this;  // time saving self-test
   return *this;
}

// Steffenson

Steffenson::Steffenson() 
{
  GSLRootFdFSolver * s = new GSLRootFdFSolver( gsl_root_fdfsolver_steffenson ); 
  SetSolver(s); 
}

Steffenson::~Steffenson() 
{
  FreeSolver();  
}

Steffenson::Steffenson(const Steffenson &) : GSLRootFinderDeriv()
{
}

Steffenson & Steffenson::operator = (const Steffenson &rhs) 
{
   if (this == &rhs) return *this;  // time saving self-test
   return *this;
}



} // end namespace GSLRoots

} // namespace Math
} // namespace ROOT
