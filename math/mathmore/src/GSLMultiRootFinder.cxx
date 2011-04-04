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

// Implementation file for class GSLMultiRootFinder
// 
// Created by: moneta  at Sun Nov 14 11:27:11 2004
// 
// Last update: Sun Nov 14 11:27:11 2004
// 

#include "Math/IFunction.h"
#include "Math/GSLMultiRootFinder.h"
#include "GSLMultiRootSolver.h"
#include "Math/Error.h"

#include "gsl/gsl_multiroots.h"
#include "gsl/gsl_errno.h"
#include <cmath>
#include <iomanip>

#include <algorithm>
#include <functional>
#include <ctype.h>   // need to use c version of tolower defined here


namespace ROOT {
namespace Math {

   // default values 

   int gDefaultMaxIter = 100;
   double gDefaultAbsTolerance = 1.E-6;
   double gDefaultRelTolerance = 1.E-10;

// impelmentation of static methods 
void GSLMultiRootFinder::SetDefaultTolerance(double abstol, double reltol ) {
   // set default tolerance
      gDefaultAbsTolerance = abstol; 
      if (reltol > 0) gDefaultRelTolerance = reltol; 
}
void GSLMultiRootFinder::SetDefaultMaxIterations(int maxiter) { 
   // set default max iter
   gDefaultMaxIter = maxiter;
}

GSLMultiRootFinder::GSLMultiRootFinder(EType type) : 
   fIter(0), fStatus(-1), fPrintLevel(0),
   fType(type), fUseDerivAlgo(false),
   fSolver(0) 
{
   // constructor for non derivative type
   fFunctions.reserve(2);
}

GSLMultiRootFinder::GSLMultiRootFinder(EDerivType type) : 
   fIter(0), fStatus(-1), fPrintLevel(0),
   fType(type), fUseDerivAlgo(true),
   fSolver(0) 
{
   // constructor for non derivative type
   fFunctions.reserve(2);
}

GSLMultiRootFinder::GSLMultiRootFinder(const char * name) : 
   fIter(0), fStatus(-1), fPrintLevel(0),
   fType(0), fUseDerivAlgo(false),
   fSolver(0) 
{
   // constructor for a string
   fFunctions.reserve(2);
   SetType(name);
}

GSLMultiRootFinder::~GSLMultiRootFinder() 
{
   // delete function wrapper
   ClearFunctions();
   if (fSolver) delete fSolver;
}

GSLMultiRootFinder::GSLMultiRootFinder(const GSLMultiRootFinder &) 
{
}

GSLMultiRootFinder & GSLMultiRootFinder::operator = (const GSLMultiRootFinder &rhs) 
{
   // dummy operator=
   if (this == &rhs) return *this;  // time saving self-test
   
   return *this;
}

void GSLMultiRootFinder::SetType(const char * name) {
   // set type using a string
   std::pair<bool,int> type = GetType(name);
   fUseDerivAlgo = type.first; 
   fType = type.second;
}


int GSLMultiRootFinder::AddFunction(const ROOT::Math::IMultiGenFunction & func) { 
   // add a new function in the vector
   ROOT::Math::IMultiGenFunction * f = func.Clone(); 
   if (!f) return 0;
   fFunctions.push_back(f);
   return fFunctions.size();
}

void GSLMultiRootFinder::ClearFunctions() {
   // clear the function list
   for (unsigned int i = 0; i < fFunctions.size(); ++i) {
      if (fFunctions[i] != 0 ) delete fFunctions[i]; 
      fFunctions[i] = 0;
   }
   fFunctions.clear();
}

void GSLMultiRootFinder::Clear() {
   // clear the function list and the solver 
   ClearFunctions(); 
   if (fSolver) Clear(); 
   fSolver = 0; 
}


const double * GSLMultiRootFinder::X() const { 
   // return x
   return (fSolver != 0) ? fSolver->X() : 0;
}
const double * GSLMultiRootFinder::Dx() const { 
   // return x
   return (fSolver != 0) ? fSolver->Dx() : 0; 
}
const double * GSLMultiRootFinder::FVal() const { 
   // return x
   return (fSolver != 0) ? fSolver->FVal() : 0;
}
const char * GSLMultiRootFinder::Name() const {
   // get GSL name 
   return (fSolver != 0) ? fSolver->Name().c_str() : ""; 
}

// bool GSLMultiRootFinder::AddFunction( const ROOT::Math::IMultiGenFunction & func) { 
//    // clone and add function to the list 
//    // If using a derivative algorithm the function is checked if it implements
//    // the gradient interface. If this is not the case the type is set to non-derivatibe algo
//    ROOT::Math::IGenMultiFunction * f = func.Clone(); 
//    if (f != 0) return false; 
//    if (fUseDerivAlgo)  {
//       bool gradFunc = (dynamic_cast<ROOT::Math::IMultiGradFunction *> (f) != 0 );
//       if (!gradFunc)  { 
//          MATH_ERROR_MSG("GSLMultiRootFinder::AddFunction","Function does not provide gradient interface");
//          MATH_WARN_MSG("GSLMultiRootFinder::AddFunction","clear the function list");         
//          ClearFunctions(); 
//          return false; 
//       }
//    }
//    fFunctions.push_back(f);
//    return true; 
// }

   const gsl_multiroot_fsolver_type * GetGSLType(GSLMultiRootFinder::EType type) { 
     //helper functions to find GSL type
   switch(type)
      {
      case ROOT::Math::GSLMultiRootFinder::kHybridS:
         return gsl_multiroot_fsolver_hybrids;
      case ROOT::Math::GSLMultiRootFinder::kHybrid:
         return gsl_multiroot_fsolver_hybrid;
      case ROOT::Math::GSLMultiRootFinder::kDNewton:
         return gsl_multiroot_fsolver_dnewton;
      case ROOT::Math::GSLMultiRootFinder::kBroyden:
         return gsl_multiroot_fsolver_broyden;
      default: 
         return gsl_multiroot_fsolver_hybrids;
      }
   return 0;
}

const gsl_multiroot_fdfsolver_type * GetGSLDerivType(GSLMultiRootFinder::EDerivType type) { 
//helper functions to find GSL deriv type
   switch(type)
   {
   case ROOT::Math::GSLMultiRootFinder::kHybridSJ :
      return gsl_multiroot_fdfsolver_hybridsj; 
   case ROOT::Math::GSLMultiRootFinder::kHybridJ :
      return gsl_multiroot_fdfsolver_hybridj; 
   case ROOT::Math::GSLMultiRootFinder::kNewton :
      return gsl_multiroot_fdfsolver_newton; 
   case ROOT::Math::GSLMultiRootFinder::kGNewton :
      return gsl_multiroot_fdfsolver_gnewton; 
   default:
      return gsl_multiroot_fdfsolver_hybridsj; 
   }
   return 0; // cannot happen
}

std::pair<bool,int> GSLMultiRootFinder::GetType(const char * name) { 
   if (name == 0) return std::make_pair<bool,int>(false, -1); 
   std::string aname = name; 
   std::transform(aname.begin(), aname.end(), aname.begin(), (int(*)(int)) tolower ); 

   if (aname.find("hybridsj") != std::string::npos) return std::make_pair(true,  kHybridSJ);
   if (aname.find("hybridj") != std::string::npos) return std::make_pair(true,  kHybridJ); 
   if (aname.find("hybrids") != std::string::npos) return std::make_pair(false,  kHybridS); 
   if (aname.find("hybrid") != std::string::npos) return std::make_pair(false,  kHybrid); 
   if (aname.find("gnewton") != std::string::npos) return std::make_pair(true,  kGNewton);
   if (aname.find("dnewton") != std::string::npos) return std::make_pair(false,  kDNewton);
   if (aname.find("newton") != std::string::npos) return std::make_pair(true,  kNewton);
   if (aname.find("broyden") != std::string::npos) return std::make_pair(false,  kBroyden);
   MATH_INFO_MSG("GSLMultiRootFinder::GetType","Unknow algorithm - use default one");
   return std::make_pair(false, -1);   
} 

bool GSLMultiRootFinder::Solve (const double * x, int maxIter, double absTol, double relTol) 
{ 
   fIter = 0;
   // create the solvers - delete previous existing solver 
   if (fSolver) delete fSolver; 
   fSolver = 0; 

   if (fFunctions.size() == 0) {
      MATH_ERROR_MSG("GSLMultiRootFinder::Solve","Function list is empty");
      fStatus = -1;
      return false;
   }

   if (fUseDerivAlgo) { 
      EDerivType type = (EDerivType) fType; 
      if (!fSolver) fSolver = new GSLMultiRootDerivSolver( GetGSLDerivType(type), Dim() );
   }
   else { 
      EType type = (EType) fType; 
      if (!fSolver) fSolver = new GSLMultiRootSolver( GetGSLType(type), Dim() );
   }


   // first set initial values and function
   assert(fSolver != 0);
   bool ret = fSolver->InitSolver( fFunctions, x);
   if (!ret) { 
      MATH_ERROR_MSG("GSLMultiRootFinder::Solve","Error initializing the solver");
      fStatus = -2;
      return false;
   }

   if (maxIter == 0) maxIter = gDefaultMaxIter;
   if (absTol <= 0) absTol = gDefaultAbsTolerance;
   if (relTol <= 0) relTol = gDefaultRelTolerance;

   if (fPrintLevel >= 1) 
      std::cout << "GSLMultiRootFinder::Solve:" << Name() << " max iterations " <<   maxIter << " and  tolerance " <<  absTol << std::endl;

   // find the roots by iterating
   fStatus = 0;
   int status = 0;
   int iter = 0; 
   do { 
      iter++; 
      status = fSolver->Iterate();

      if (fPrintLevel >= 2) {
         std::cout << "GSLMultiRootFinder::Solve - iteration # " <<   iter << " status = " << status << std::endl;
         PrintState(); 
      }
      // act in case of error 
      if (status == GSL_EBADFUNC) { 
         MATH_ERROR_MSG("GSLMultiRootFinder::Solve","The iteration encountered a singolar point due to a bad function value");
         fStatus = status;
         break;
      }
      if (status == GSL_ENOPROG) { 
         MATH_ERROR_MSG("GSLMultiRootFinder::Solve","The iteration is not making any progress");
         fStatus = status;
         break;
      }
      if (status != GSL_SUCCESS) { 
         MATH_ERROR_MSG("GSLMultiRootFinder::Solve","Uknown iteration error - exit");
         fStatus = status;
         break;
      }

      // test also residual
      status =  fSolver->TestResidual(absTol); 


      // should test also the Delta ??
      int status2 =  fSolver->TestDelta(absTol, relTol); 
      if (status2 == GSL_SUCCESS) { 
         MATH_INFO_MSG("GSLMultiRootFinder::Solve","The iteration converged");
      }
   }
   while (status == GSL_CONTINUE && iter < maxIter);
   if (status == GSL_CONTINUE) { 
      MATH_INFO_MSGVAL("GSLMultiRootFinder::Solve","exceeded max iterations, reached tolerance is not sufficient",absTol);
   }
   if (status == GSL_SUCCESS) { 
      if (fPrintLevel>=1) {          // print the result
         MATH_INFO_MSG("GSLMultiRootFinder::Solve","The iteration converged");
         std::cout << "GSL Algorithm used is :  " << fSolver->Name() << std::endl;
         std::cout << "Number of iterations  =  " << iter<< std::endl;
         
         PrintState(); 
      }
   }
   fIter = iter;
   fStatus = status; 
   return  (fStatus == GSL_SUCCESS);

}

void GSLMultiRootFinder::PrintState(std::ostream & os) {    
   // print current state 
   if (!fSolver) return;
   double ndigits = std::log10( double( Dim() ) );
   int wi = int(ndigits)+1;
   const double * xtmp = fSolver->X(); 
   const double * ftmp = fSolver->FVal(); 
   os << "Root values     = "; 
   for (unsigned int i = 0; i< Dim(); ++i) 
      os << "x[" << std::setw(wi) << i << "] = " << std::setw(12) << xtmp[i] << "   ";
   os << std::endl;
   os << "Function values = "; 
   for (unsigned int i = 0; i< Dim(); ++i) 
      os << "f[" << std::setw(wi) << i << "] = " << std::setw(12) << ftmp[i] << "   ";
   os << std::endl; 
}



} // namespace Math
} // namespace ROOT
