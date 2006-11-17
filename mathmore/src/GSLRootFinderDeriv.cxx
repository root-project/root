// @(#)root/mathmore:$Name:  $:$Id: GSLRootFinderDeriv.cxx,v 1.4 2006/06/19 08:44:08 moneta Exp $
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

#include "Math/GSLRootFinderDeriv.h"
#include "Math/GSLRootHelper.h"
#include "GSLRootFdFSolver.h"
#include "GSLFunctionWrapper.h"

#include "gsl/gsl_roots.h"
#include "gsl/gsl_errno.h"

#include <iostream>

namespace ROOT {
namespace Math {


GSLRootFinderDeriv::GSLRootFinderDeriv() 
{ 
   // create function wrapper
   fFunction = new GSLFunctionDerivWrapper(); 
}

GSLRootFinderDeriv::~GSLRootFinderDeriv() 
{
   // delete function wrapper
   if (fFunction) delete fFunction;
}

GSLRootFinderDeriv::GSLRootFinderDeriv(const GSLRootFinderDeriv &) 
{
}

GSLRootFinderDeriv & GSLRootFinderDeriv::operator = (const GSLRootFinderDeriv &rhs) 
{
   // private operator=
   if (this == &rhs) return *this;  // time saving self-test
   
   return *this;
}




void GSLRootFinderDeriv::SetFunction(  GSLFuncPointer f, GSLFuncPointer df, GSLFdFPointer Fdf, void * p, double Root) {
   // set Function with signature as GSL
   fRoot = Root;
   fFunction->SetFuncPointer( f ); 
   fFunction->SetDerivPointer( df ); 
   fFunction->SetFdfPointer( Fdf ); 
   fFunction->SetParams( p ); 
   gsl_root_fdfsolver_set( fS->Solver(), fFunction->GetFunc(), Root); 
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
   
   //     // function values (for debugging)
   //     const gsl_function_fdf * func = fFunction->getFunc(); 
   
   //     double f = GSL_FN_FDF_EVAL_F(func, fRoot); 
   //     //double df = func->df(fRoot,func->params);
   //     double df =  GSL_FN_FDF_EVAL_DF(func, fRoot); 
   //     std::cout << " r = " << fRoot << " f(r) = " << f << " df = " << df;
   //     GSL_FN_FDF_EVAL_F_DF(func,fRoot, &f, &df); 
   //     std::cout << " Fdf = " << f << "  " << df << std::endl;
   
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

int GSLRootFinderDeriv::Solve (int maxIter, double absTol, double relTol) 
{ 
   // solve for roots 
   int iter = 0; 
   int status = 0; 
   do { 
      iter++; 
      
      status = Iterate();
      status = GSLRootHelper::TestDelta(fRoot, fPrevRoot, absTol, relTol);
      if (status == GSL_SUCCESS) { 
         fIter = iter;
         return status; 
      }
      
      //     std::cout << "iteration " << iter << " Root " << fRoot << " prev Root " << 
      //       fPrevRoot << std::endl;
   }
   while (status == GSL_CONTINUE && iter < maxIter); 
   return status;
}


} // namespace Math
} // namespace ROOT
