// @(#)root/mathcore:$Id$
// Authors: David Gonzalez Maline    01/2008

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header for the IRootFinderMethod interface
//
// Created by: David Gonzalez Maline  : Fri Jan 25 2008
//

#ifndef ROOT_Math_IRootFinderMethod
#define ROOT_Math_IRootFinderMethod

#include "Math/Error.h"

#include "Math/IFunctionfwd.h"

namespace ROOT {
namespace Math {

//___________________________________________________________________________________________
/**
   Interface for finding function roots of one-dimensional functions

   @ingroup RootFinders

 */

class IRootFinderMethod {
public:
   /** Default Destructor. */
   virtual ~IRootFinderMethod() {}

   /** Default Constructor. */
   IRootFinderMethod() {}

   // Common functionality

   /** Sets the function for algorithms using derivatives.  */
   virtual bool SetFunction(const ROOT::Math::IGradFunction&, double)
   {
      MATH_ERROR_MSG("SetFunction", "This method must be used with a Root Finder algorithm using derivatives");
      return false;
   }

   /** Sets the function for the rest of the algorithms.
       The parameters set the interval where the root has to be calculated. */
   virtual bool SetFunction(const ROOT::Math::IGenFunction& , double , double )
   {
      MATH_ERROR_MSG("SetFunction", "Algorithm requires derivatives");
      return false;
   }

   /** Returns the previously calculated root. */
   virtual double Root() const = 0;

   /** Returns the status of the previous estimate */
   virtual int Status() const = 0;

   // Methods to be Implemented in the derived classes

   /** Stimates the root for the function.
       \@param maxIter maximum number of iterations.
       \@param absTol desired absolute error in the minimum position.
       \@param absTol desired relative error in the minimum position.
   */
   virtual bool Solve(int maxIter = 100, double absTol = 1E-8, double relTol = 1E-10) = 0;

   /** Return name of root finder algorithm */
   virtual const char* Name() const = 0;

   /** This method is  implemented only by the GSLRootFinder
       and GSLRootFinderDeriv classes and will return an error if it's not one of them. */
   virtual int Iterate() {
      MATH_ERROR_MSG("Iterate", "This method must be used with a Root Finder algorithm wrapping the GSL Library");
      return -1;
   }

   /** Return number of iterations used to find the root
       Must be implemented by derived classes
   */
   virtual int Iterations() const { return -1; }

};

} // namespace Math
} // namespace ROOT


#endif /* ROOT_Math_IRootFinderMethod */
