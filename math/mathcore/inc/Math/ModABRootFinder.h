// @(#)root/mathcore:$Id$
// Authors: Nedelcho Ganchovski    03/03/2026

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2026  CERN                                           *
 * All rights reserved.                                               *
 *                                                                    *
 * For the licensing terms see $ROOTSYS/LICENSE.                      *
 * For the list of contributors see $ROOTSYS/README/CREDITS.          *
 *                                                                    *
 **********************************************************************/

// Header for the RootFinder
//
// Created by: Nedelcho Ganchovski  : Wed March 03 2026
//

#ifndef ROOT_Math_ModABRootFinder
#define ROOT_Math_ModABRootFinder

#include "Math/IFunction.h"

#include "Math/IRootFinderMethod.h"

namespace ROOT {
namespace Math {

//___________________________________________________________________________________________
/**
   Class for finding the root of a one dimensional function using the ModAB algorithm.
   It is based on the Modified Anderson-Björck method (2022 Ganchovski & Traykov) that
   adaptively switches between bisection and regula-falsi with
   side-correction, yielding superlinear convergence on well-behaved
   functions while retaining the robustness of bisection.

   This class is implemented from TF1::GetX() method.

   @ingroup RootFinders

 */

class ModABRootFinder : public IRootFinderMethod {
public:
   /** Default Constructor. */
   ModABRootFinder();

   /** Default Destructor. */
   ~ModABRootFinder() override {}

   /** Set function to solve and the interval in where to look for the root.

       \@param f Function to be minimized.
       \@param xlow Lower bound of the search interval.
       \@param xup Upper bound of the search interval.
   */
   using IRootFinderMethod::SetFunction;
   bool SetFunction(const ROOT::Math::IGenFunction &f, double xlow, double xup) override;

   /** Returns the X value corresponding to the function value fy for (xmin<x<xmax).
       Method:
       Modified Anderson-Björck method is applied on the bracketed interval.

       \@param maxIter maximum number of iterations.
       \@param absTol desired absolute error in the minimum position.
       \@param absTol desired relative error in the minimum position.
   */
   bool Solve(int maxIter = 100, double absTol = 1E-8, double relTol = 1E-10) override;

   /** Returns root value. Need to call first Solve(). */
   double Root() const override { return fRoot; }

   /** Returns status of last estimate. If = 0 is OK */
   int Status() const override { return fStatus; }

   /** Return number of iteration used to find minimum */
   int Iterations() const override { return fNIter; }

   /** Return name of root finder algorithm ("BrentRootFinder"). */
   const char *Name() const override;

   // static function used to modify the default parameters

   /** set number of default Npx used at construction time (when SetNpx is not called)
       Default value is 100
    */

private:
   const IGenFunction *fFunction; ///< Pointer to the function.
   int fNIter;                    ///< Number of iterations needed for the last estimation.
   int fStatus;                   ///< Status of code of the last estimate
   double fXMin;                  ///< Lower bound of the search interval.
   double fXMax;                  ///< Upper bound of the search interval
   double fRoot;                  ///< Current estimation of the function root.
};

} // namespace Math
} // namespace ROOT

#endif /* ROOT_Math_BrentRootFinder */