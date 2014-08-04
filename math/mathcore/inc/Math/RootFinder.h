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

// Header file for class RootFinder
//
// Created by: moneta  at Sun Nov 14 16:59:55 2004
//
// Last update: Sun Nov 14 16:59:55 2004
//
#ifndef ROOT_Math_RootFinder
#define ROOT_Math_RootFinder


#ifndef ROOT_Math_IFunctionfwd
#include "Math/IFunctionfwd.h"
#endif

#ifndef ROOT_Math_IRootFinderMethod
#include "Math/IRootFinderMethod.h"
#endif


/**
   @defgroup RootFinders One-dimensional Root-Finding algorithms
   Various implementation esists in MathCore and MathMore
   The user interacts with a proxy class ROOT::Math::RootFinder which creates behing
   the chosen algorithms which are implemented using the ROOT::Math::IRootFinderMethod interface

   @ingroup NumAlgo
*/


namespace ROOT {
   namespace Math {


//_____________________________________________________________________________________
      /**
         User Class to find the Root of one dimensional functions.
         The GSL Methods are implemented in MathMore and they are loaded automatically
         via the plug-in manager

         The possible types of Root-finding algorithms are:
         <ul>
         <li>Root Bracketing Algorithms which do not require function derivatives
         <ol>
         <li>RootFinder::kBRENT  (default method implemented in MathCore)
         <li>RootFinder::kGSL_BISECTION
         <li>RootFinder::kGSL_FALSE_POS
         <li>RootFinder::kGSL_BRENT
         </ol>
         <li>Root Finding Algorithms using Derivatives
         <ol>
         <li>RootFinder::kGSL_NEWTON
         <li>RootFinder::kGSL_SECANT
         <li>RootFinder::kGSL_STEFFENSON
         </ol>
         </ul>

         This class does not cupport copying

         @ingroup RootFinders

      */

      class RootFinder {

      public:

         enum EType { kBRENT,                                   // Methods from MathCore
                     kGSL_BISECTION, kGSL_FALSE_POS, kGSL_BRENT, // GSL Normal
                     kGSL_NEWTON, kGSL_SECANT, kGSL_STEFFENSON   // GSL Derivatives
         };

         /**
            Construct a Root-Finder algorithm
         */
         RootFinder(RootFinder::EType type = RootFinder::kBRENT);
         virtual ~RootFinder();

      private:
         // usually copying is non trivial, so we make this unaccessible
         RootFinder(const RootFinder & ) {}
         RootFinder & operator = (const RootFinder & rhs)
         {
            if (this == &rhs) return *this;  // time saving self-test
            return *this;
         }

      public:

         bool SetMethod(RootFinder::EType type = RootFinder::kBRENT);

         /**
            Provide to the solver the function and the initial search interval [xlow, xup]
            for algorithms not using derivatives (bracketing algorithms)
            The templated function f must be of a type implementing the \a operator() method,
            <em>  double  operator() (  double  x ) </em>
            Returns non zero if interval is not valid (i.e. does not contains a root)
         */

         bool SetFunction( const IGenFunction & f, double xlow, double xup) {
            return fSolver->SetFunction( f, xlow, xup);
         }


         /**
            Provide to the solver the function and an initial estimate of the root,
            for algorithms using derivatives.
            The templated function f must be of a type implementing the \a operator()
            and the \a Gradient() methods.
            <em>  double  operator() (  double  x ) </em>
            Returns non zero if starting point is not valid
         */

         bool  SetFunction( const IGradFunction & f, double xstart) {
            return fSolver->SetFunction( f, xstart);
         }

         template<class Function, class Derivative>
         bool Solve(Function &f, Derivative &d, double start,
                   int maxIter = 100, double absTol = 1E-8, double relTol = 1E-10);

         template<class Function>
         bool Solve(Function &f, double min, double max,
                   int maxIter = 100, double absTol = 1E-8, double relTol = 1E-10);

         /**
             Compute the roots iterating until the estimate of the Root is within the required tolerance returning
             the iteration Status
         */
         bool Solve( int maxIter = 100, double absTol = 1E-8, double relTol = 1E-10) {
            return fSolver->Solve( maxIter, absTol, relTol );
         }

         /**
             Return the number of iteration performed to find the Root.
         */
         int Iterations() const {
            return fSolver->Iterations();
         }

         /**
            Perform a single iteration and return the Status
         */
         int Iterate() {
            return fSolver->Iterate();
         }

         /**
            Return the current and latest estimate of the Root
         */
         double Root() const {
            return fSolver->Root();
         }

         /**
            Return the status of the last estimate of the Root
            = 0 OK, not zero failure
         */
         int Status() const {
            return fSolver->Status();
         }


         /**
            Return the current and latest estimate of the lower value of the Root-finding interval (for bracketing algorithms)
         */
/*   double XLower() const {  */
/*     return fSolver->XLower();  */
/*   } */

         /**
            Return the current and latest estimate of the upper value of the Root-finding interval (for bracketing algorithms)
         */
/*   double XUpper() const {  */
/*     return  fSolver->XUpper();  */
/*   } */

         /**
            Get Name of the Root-finding solver algorithm
         */
         const char * Name() const {
            return fSolver->Name();
         }


      protected:


      private:

         IRootFinderMethod* fSolver;   // type of algorithm to be used


      };

   } // namespace Math
} // namespace ROOT


#ifndef ROOT_Math_WrappedFunction
#include "Math/WrappedFunction.h"
#endif

#ifndef ROOT_Math_Functor
#include "Math/Functor.h"
#endif

template<class Function, class Derivative>
bool ROOT::Math::RootFinder::Solve(Function &f, Derivative &d, double start,
                                  int maxIter, double absTol, double relTol)
{
   if (!fSolver) return false;
   ROOT::Math::GradFunctor1D wf(f, d);
   bool ret = fSolver->SetFunction(wf, start);
   if (!ret) return false;
   return Solve(maxIter, absTol, relTol);
}

template<class Function>
bool ROOT::Math::RootFinder::Solve(Function &f, double min, double max,
                                  int maxIter, double absTol, double relTol)
{
   if (!fSolver) return false;
   ROOT::Math::WrappedFunction<Function &> wf(f);
   bool ret = fSolver->SetFunction(wf, min, max);
   if (!ret) return false;
   return Solve(maxIter, absTol, relTol);
}

#endif /* ROOT_Math_RootFinder */
