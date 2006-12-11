// @(#)root/mathmore:$Name:  $:$Id: GSLRootFinder.cxx,v 1.4 2006/11/17 18:26:50 moneta Exp $
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

#include "GSLRootFSolver.h"
#include "GSLFunctionWrapper.h"

#include "gsl/gsl_roots.h"
#include "gsl/gsl_errno.h"


namespace ROOT {
namespace Math {


GSLRootFinder::GSLRootFinder() 
{
   // create function wrapper
   fFunction = new GSLFunctionWrapper(); 
}

GSLRootFinder::~GSLRootFinder() 
{
   // delete function wrapper
   if (fFunction) delete fFunction;
}

GSLRootFinder::GSLRootFinder(const GSLRootFinder &) 
{
}

GSLRootFinder & GSLRootFinder::operator = (const GSLRootFinder &rhs) 
{
   // dummy operator=
   if (this == &rhs) return *this;  // time saving self-test
   
   return *this;
}

void GSLRootFinder::SetFunction(  GSLFuncPointer f, void * p, double xlow, double xup) { 
   // set from GSL function
   fXlow = xlow; 
   fXup = xup;
   fFunction->SetFuncPointer( f ); 
   fFunction->SetParams( p ); 
   gsl_root_fsolver_set( fS->Solver(), fFunction->GetFunc(), xlow, xup); 
}

void GSLRootFinder::SetFunction( const IGenFunction & f, double xlow, double xup) {
   // set from IGenFunction
   fXlow = xlow; 
   fXup = xup;
   fFunction->SetFunction( f );  
   gsl_root_fsolver_set( fS->Solver(), fFunction->GetFunc(), xlow, xup); 
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
   if (!fFunction->IsValid() ) {
      std::cerr << "GSLRootFinder - Error: Function is not valid" << std::endl;
      return -1; 
   }
   int status =  gsl_root_fsolver_iterate(fS->Solver());
   // update Root 
   fRoot = gsl_root_fsolver_root(fS->Solver() );
   // update interval
   fXlow =  gsl_root_fsolver_x_lower(fS->Solver() ); 
   fXup =  gsl_root_fsolver_x_upper(fS->Solver() );
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

int GSLRootFinder::Solve (int maxIter, double absTol, double relTol) 
{ 
   // find the roots by iterating
   int iter = 0; 
   int status = 0; 
   do { 
      iter++; 
      status = Iterate();
      if (status != GSL_SUCCESS) return status; 
      status =  GSLRootHelper::TestInterval(fXlow, fXup, absTol, relTol); 
      if (status == GSL_SUCCESS) { 
         fIter = iter;
         return status; 
      }
   }
   while (status == GSL_CONTINUE && iter < maxIter); 
   return status;
}




} // namespace Math
} // namespace ROOT
