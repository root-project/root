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

namespace ROOT::Math {

//___________________________________________________________________________________________
/**
   Class for finding the root of a one dimensional function using the ModAB algorithm.
   It is based on the Modified Anderson-Björck method (2022 Ganchovski & Traykov) that
   adaptively switches between bisection and regula-falsi with
   side-correction, yielding superlinear convergence on well-behaved
   functions while retaining the robustness of bisection.
   @ingroup RootFinders
 */

class ModABRootFinder : public IRootFinderMethod {
public:
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
       \@param relTol desired relative error in the minimum position.
   */
   bool Solve(int maxIter = 100, double absTol = 1E-8, double relTol = 1E-10) override;

   /** Returns root value. Need to call first Solve(). */
   double Root() const override { return fRoot; }

   /** Returns status of last estimate. If = 0 is OK */
   int Status() const override { return fStatus; }

   /** Return number of iteration used to find minimum */
   int Iterations() const override { return fNIter; }

   /** Return name of root finder algorithm ("ModABRootFinder"). */
   const char *Name() const override;

private:
   const IGenFunction *fFunction = nullptr; // Pointer to the function.
   int fNIter = 0;                          // Number of iterations needed for the last estimation.
   int fStatus = -1;                        // Status of code of the last estimate
   double fXMin = 0.;                       // Lower bound of the search interval.
   double fXMax = 0.;                       // Upper bound of the search interval
   double fRoot = 0.;                       // Current estimation of the function root.
};
} // namespace ROOT::Math

#endif /* ROOT_Math_ModABRootFinder */
