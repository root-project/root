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

// Implementation file for class GSLRootFinder
//
// Created by: moneta  at Sun Nov 14 11:27:11 2004
//
// Last update: Sun Nov 14 11:27:11 2004
//

#include "Math/IFunction.h"
#include "Math/GSLRootFinder.h"
#include "Math/GSLRootHelper.h"
#include "Math/Error.h"

#include "GSLRootFSolver.h"
#include "GSLFunctionWrapper.h"

#include "gsl/gsl_roots.h"
#include "gsl/gsl_errno.h"
#include <cmath>


namespace ROOT {
namespace Math {


GSLRootFinder::GSLRootFinder() :
   fFunction(0), fS(0),
   fRoot(0), fXlow(0), fXup(0),
   fIter(0), fStatus(-1),
   fValidInterval(false)
{
   // create function wrapper
   fFunction = new GSLFunctionWrapper();
}

GSLRootFinder::~GSLRootFinder()
{
   // delete function wrapper
   if (fFunction) delete fFunction;
}

GSLRootFinder::GSLRootFinder(const GSLRootFinder &): IRootFinderMethod()
{
}

GSLRootFinder & GSLRootFinder::operator = (const GSLRootFinder &rhs)
{
   // dummy operator=
   if (this == &rhs) return *this;  // time saving self-test

   return *this;
}

bool GSLRootFinder::SetFunction(  GSLFuncPointer f, void * p, double xlow, double xup) {
   // set from GSL function
   fXlow = xlow;
   fXup = xup;
   fFunction->SetFuncPointer( f );
   fFunction->SetParams( p );

   int status = gsl_root_fsolver_set( fS->Solver(), fFunction->GetFunc(), xlow, xup);
   if (status == GSL_SUCCESS)
      fValidInterval = true;
   else
      fValidInterval = false;

   return fValidInterval;
}

bool GSLRootFinder::SetFunction( const IGenFunction & f, double xlow, double xup) {
   // set from IGenFunction
   fStatus  = -1; // invalid the status
   fXlow = xlow;
   fXup = xup;
   fFunction->SetFunction( f );
   int status = gsl_root_fsolver_set( fS->Solver(), fFunction->GetFunc(), xlow, xup);
   if (status == GSL_SUCCESS)
      fValidInterval = true;
   else
      fValidInterval = false;

   return fValidInterval;
}

void GSLRootFinder::SetSolver(GSLRootFSolver * s ) {
   // set type of solver
   fS = s;
}

void GSLRootFinder::FreeSolver( ) {
   // free resources
   if (fS) delete fS;
}

int GSLRootFinder::Iterate() {
   // iterate
   int status = 0;
   if (!fFunction->IsValid() ) {
      MATH_ERROR_MSG("GSLRootFinder::Iterate"," Function is not valid");
      status = -1;
      return status;
   }
   if (!fValidInterval ) {
      MATH_ERROR_MSG("GSLRootFinder::Iterate"," Interval is not valid");
      status = -2;
      return status;
   }

   status =  gsl_root_fsolver_iterate(fS->Solver());

   // update Root
   fRoot = gsl_root_fsolver_root(fS->Solver() );
   // update interval
   fXlow =  gsl_root_fsolver_x_lower(fS->Solver() );
   fXup =  gsl_root_fsolver_x_upper(fS->Solver() );

   //std::cout << "iterate .." << fRoot << " status " << status << " interval "
   //          << fXlow << "  " << fXup << std::endl;

   return status;
}

double GSLRootFinder::Root() const {
   // return cached value
   return fRoot;
}
/**
double GSLRootFinder::XLower() const {
   return fXlow;
}

double GSLRootFinder::XUpper() const {
   return fXup;
}
*/
const char * GSLRootFinder::Name() const {
   // get GSL name
   return gsl_root_fsolver_name(fS->Solver() );
}

bool GSLRootFinder::Solve (int maxIter, double absTol, double relTol)
{
   // find the roots by iterating
   fStatus = -1;
   int status = 0;
   int iter = 0;
   do {
      iter++;
      status = Iterate();
      //std::cerr << "RF: iteration " << iter << " status = " << status << std::endl;
      if (status != GSL_SUCCESS) {
         MATH_ERROR_MSG("GSLRootFinder::Solve","error returned when performing an iteration");
         fStatus = status;
         return false;
      }
      status =  GSLRootHelper::TestInterval(fXlow, fXup, absTol, relTol);
      if (status == GSL_SUCCESS) {
         fIter = iter;
         fStatus = status;
         return true;
      }
   }
   while (status == GSL_CONTINUE && iter < maxIter);
   if (status == GSL_CONTINUE) {
      double tol = std::abs(fXup-fXlow);
      MATH_INFO_MSGVAL("GSLRootFinder::Solve","exceeded max iterations, reached tolerance is not sufficient",tol);
   }
   fStatus = status;
   return false;
}




} // namespace Math
} // namespace ROOT
