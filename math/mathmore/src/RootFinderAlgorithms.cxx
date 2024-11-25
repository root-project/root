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


} // end namespace GSLRoots

} // namespace Math
} // namespace ROOT
