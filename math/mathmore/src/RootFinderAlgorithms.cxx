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
   // Bisection constructor
   GSLRootFSolver * s = new GSLRootFSolver( gsl_root_fsolver_bisection ); 
   SetSolver(s); 
}

Bisection::~Bisection() 
{
   // destructor
   FreeSolver();  
}

Bisection::Bisection(const Bisection &) : GSLRootFinder()
{
  // dummy copy ctr
}

Bisection & Bisection::operator = (const Bisection &rhs) 
{
   // dummy (private) operator=
   if (this == &rhs) return *this;  // time saving self-test
   return *this;
}


// falsepos method

FalsePos::FalsePos() 
{
   // FalsePos constructor
   GSLRootFSolver * s = new GSLRootFSolver( gsl_root_fsolver_falsepos ); 
   SetSolver(s); 
}

FalsePos::~FalsePos() 
{
   // destructor
   FreeSolver();  
}

FalsePos::FalsePos(const FalsePos &) : GSLRootFinder()
{
  // dummy copy ctr
}

FalsePos & FalsePos::operator = (const FalsePos &rhs) 
{
   // dummy (private) operator=
   if (this == &rhs) return *this;  // time saving self-test
   return *this;
}

// Brent method

Brent::Brent() 
{
   // Brent constructor
   GSLRootFSolver * s = new GSLRootFSolver( gsl_root_fsolver_brent ); 
   SetSolver(s); 
}

Brent::~Brent() 
{
   // destructor
   FreeSolver();  
}

Brent::Brent(const Brent &) : GSLRootFinder()
{
  // dummy copy ctr
}

Brent & Brent::operator = (const Brent &rhs) 
{
   // dummy (private) operator=
   if (this == &rhs) return *this;  // time saving self-test
   return *this;
}


//---------------------------------------------------------------------
// algorithms with Derivatives 
//--------------------------------------------------------------------

// Newton

Newton::Newton() 
{
   // Newton constructor
   GSLRootFdFSolver * s = new GSLRootFdFSolver( gsl_root_fdfsolver_newton ); 
   SetSolver(s); 
}

Newton::~Newton() 
{
   // destructor
   FreeSolver();  
}

Newton::Newton(const Newton &) : GSLRootFinderDeriv()
{
  // dummy copy ctr
}

Newton & Newton::operator = (const Newton &rhs) 
{
   // dummy (private) operator=
   if (this == &rhs) return *this;  // time saving self-test
   return *this;
}

// Secant

Secant::Secant() 
{
   // Secant constructor
   GSLRootFdFSolver * s = new GSLRootFdFSolver( gsl_root_fdfsolver_secant ); 
   SetSolver(s); 
}

Secant::~Secant() 
{
   // destructor
   FreeSolver();  
}

Secant::Secant(const Secant &) : GSLRootFinderDeriv()
{
  // dummy copy ctr
}

Secant & Secant::operator = (const Secant &rhs) 
{
   // dummy (private) operator=
   if (this == &rhs) return *this;  // time saving self-test
   return *this;
}

// Steffenson

Steffenson::Steffenson() 
{
   // Steffenson constructor
   GSLRootFdFSolver * s = new GSLRootFdFSolver( gsl_root_fdfsolver_steffenson ); 
   SetSolver(s); 
}

Steffenson::~Steffenson() 
{
   // destructor
   FreeSolver();  
}

Steffenson::Steffenson(const Steffenson &) : GSLRootFinderDeriv()
{
  // dummy copy ctr
}

Steffenson & Steffenson::operator = (const Steffenson &rhs) 
{
   // dummy (private) operator=
   if (this == &rhs) return *this;  // time saving self-test
   return *this;
}



} // end namespace GSLRoots

} // namespace Math
} // namespace ROOT
