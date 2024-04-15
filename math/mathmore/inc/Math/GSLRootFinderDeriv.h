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

// Header file for class GSLRootFinderDeriv
//
// Created by: moneta  at Sun Nov 21 16:26:03 2004
//
// Last update: Sun Nov 21 16:26:03 2004
//
#ifndef ROOT_Math_GSL_RootFinderDeriv
#define ROOT_Math_GSL_RootFinderDeriv


#include "Math/GSLFunctionAdapter.h"

#include "Math/IFunctionfwd.h"
#include "Math/IFunction.h"

#include "Math/IRootFinderMethod.h"

#include <iostream>

namespace ROOT {
namespace Math {


   class GSLRootFdFSolver;
   class GSLFunctionDerivWrapper;


//_____________________________________________________________________________________
   /**
      Base class for GSL Root-Finding algorithms for one dimensional functions which use function derivatives.
      For finding the roots users should not use this class directly but instantiate the derived classes,
      for example  ROOT::Math::Roots::Newton for using the Newton algorithm.
      All the classes defining the alhorithms are defined in the header Math/RootFinderAlgorithm.h
      They possible types implementing root bracketing algorithms which use function
      derivatives are:
      <ul>
         <li>ROOT::Math::Roots::Newton
         <li>ROOT::Math::Roots::Secant
         <li>ROOT::Math::Roots::Steffenson
     </ul>

      See also those classes  for the documentation.
      See the GSL <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Root-Finding-Algorithms-using-Derivatives.html"> online manual</A> for
      information on the GSL Root-Finding algorithms

      @ingroup RootFinders
   */


class GSLRootFinderDeriv: public IRootFinderMethod {

public:
   GSLRootFinderDeriv();
   ~GSLRootFinderDeriv() override;

private:
   // usually copying is non trivial, so we make this unaccessible
   GSLRootFinderDeriv(const GSLRootFinderDeriv &);
   GSLRootFinderDeriv & operator = (const GSLRootFinderDeriv &);

public:



#if defined(__MAKECINT__) || defined(G__DICTIONARY)
   bool SetFunction( const IGenFunction & , double , double ) override {
      std::cerr <<"GSLRootFinderDeriv - Error : Algorithm requirs derivatives" << std::endl;
      return false;
   }
#endif

   bool SetFunction( const IGradFunction & f, double xstart) override {
      const void * p = &f;
      return SetFunction(  &GSLFunctionAdapter<IGradFunction>::F, &GSLFunctionAdapter<IGradFunction>::Df, &GSLFunctionAdapter<IGradFunction>::Fdf, const_cast<void *>(p), xstart );
   }


   typedef double ( * GSLFuncPointer ) ( double, void *);
   typedef void ( * GSLFdFPointer ) ( double, void *, double *, double *);
   bool SetFunction( GSLFuncPointer f, GSLFuncPointer df, GSLFdFPointer fdf, void * p, double Root );

   using IRootFinderMethod::SetFunction;

   /// iterate (return GSL_SUCCESS in case of successful iteration)
   int Iterate() override;

   double Root() const override;

   /// Find the root (return false if failed)
   bool Solve( int maxIter = 100, double absTol = 1E-8, double relTol = 1E-10) override;

   /// Return number of iterations
   int Iterations() const override {
      return fIter;
   }

   /// Return the status of last root finding
   int Status() const override { return fStatus; }

   const char * Name() const override;

protected:

   void SetSolver (  GSLRootFdFSolver * s );

   void FreeSolver();

private:

   GSLFunctionDerivWrapper * fFunction;
   GSLRootFdFSolver * fS;

   mutable double fRoot;
   mutable double fPrevRoot;
   int fIter;
   int fStatus;
   bool fValidPoint;

};

} // namespace Math
} // namespace ROOT


#endif /* ROOT_Math_GSL_RootFinderDeriv */
