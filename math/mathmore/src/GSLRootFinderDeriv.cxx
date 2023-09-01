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

// Implementation file for class GSLRootFinderDeriv
//
// Created by: moneta  at Sun Nov 21 16:26:03 2004
//
// Last update: Sun Nov 21 16:26:03 2004
//

#include "Math/IFunction.h"
#include "Math/Error.h"
#include "Math/GSLRootFinderDeriv.h"
#include "Math/GSLRootHelper.h"
#include "GSLRootFdFSolver.h"
#include "GSLFunctionWrapper.h"

#include "gsl/gsl_roots.h"
#include "gsl/gsl_errno.h"

#include <cmath>

namespace ROOT {
namespace Math {


GSLRootFinderDeriv::GSLRootFinderDeriv() :
   fFunction(0), fS(0),
   fRoot(0), fPrevRoot(0),
   fIter(0), fStatus(-1),
   fValidPoint(false)
{
   // create function wrapper
   fFunction = new GSLFunctionDerivWrapper();
}

GSLRootFinderDeriv::~GSLRootFinderDeriv()
{
   // delete function wrapper
   if (fFunction) delete fFunction;
}

GSLRootFinderDeriv::GSLRootFinderDeriv(const GSLRootFinderDeriv &) : IRootFinderMethod()
{
}

GSLRootFinderDeriv & GSLRootFinderDeriv::operator = (const GSLRootFinderDeriv &rhs)
{
   // private operator=
   if (this == &rhs) return *this;  // time saving self-test

   return *this;
}




bool GSLRootFinderDeriv::SetFunction(  GSLFuncPointer f, GSLFuncPointer df, GSLFdFPointer Fdf, void * p, double xstart) {
   fStatus = -1;
   // set Function with signature as GSL
   fRoot = xstart;
   fFunction->SetFuncPointer( f );
   fFunction->SetDerivPointer( df );
   fFunction->SetFdfPointer( Fdf );
   fFunction->SetParams( p );
   int status = gsl_root_fdfsolver_set( fS->Solver(), fFunction->GetFunc(), fRoot);
   if (status == GSL_SUCCESS)
      fValidPoint = true;
   else
      fValidPoint = false;

   return fValidPoint;

}

void GSLRootFinderDeriv::SetSolver(GSLRootFdFSolver * s ) {
   // set solver
   fS = s;
}

void GSLRootFinderDeriv::FreeSolver( ) {
   // free the gsl solver
   if (fS) delete fS;
}

int GSLRootFinderDeriv::Iterate() {
   // iterate........

   if (!fFunction->IsValid() ) {
      MATH_ERROR_MSG("GSLRootFinderDeriv::Iterate"," Function is not valid");
      return -1;
   }
   if (!fValidPoint ) {
      MATH_ERROR_MSG("GSLRootFinderDeriv::Iterate"," Estimated point is not valid");
      return -2;
   }


   int status = gsl_root_fdfsolver_iterate(fS->Solver());
   // update Root
   fPrevRoot = fRoot;
   fRoot =  gsl_root_fdfsolver_root(fS->Solver() );
   return status;
}

double GSLRootFinderDeriv::Root() const {
   // return cached value
   return fRoot;
}

const char * GSLRootFinderDeriv::Name() const {
   // get name from GSL
   return gsl_root_fdfsolver_name(fS->Solver() );
}

bool GSLRootFinderDeriv::Solve (int maxIter, double absTol, double relTol)
{
   // solve for roots
   fStatus = -1;
   int iter = 0;
   int status = 0;
   do {
      iter++;

      status = Iterate();
      if (status != GSL_SUCCESS) {
         MATH_ERROR_MSG("GSLRootFinderDeriv::Solve","error returned when performing an iteration");
         fStatus = status;
         return false;
      }
      status = GSLRootHelper::TestDelta(fRoot, fPrevRoot, absTol, relTol);
      if (status == GSL_SUCCESS) {
         fIter = iter;
         fStatus = status;
         return true;
      }

      //     std::cout << "iteration " << iter << " Root " << fRoot << " prev Root " <<
      //       fPrevRoot << std::endl;
   }
   while (status == GSL_CONTINUE && iter < maxIter);

   if (status == GSL_CONTINUE) {
      double tol = std::abs(fRoot-fPrevRoot);
      MATH_INFO_MSGVAL("GSLRootFinderDeriv::Solve","exceeded max iterations, reached tolerance is not sufficient",tol);
   }

   fStatus = status;
   return false;
}


} // namespace Math
} // namespace ROOT
