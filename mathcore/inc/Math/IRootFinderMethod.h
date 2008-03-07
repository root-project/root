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

#ifndef ROOT_Math_Error
#include "Math/Error.h"
#endif

#include <Math/IFunction.h>

namespace ROOT {
namespace Math {

class IRootFinderMethod {
public:
   virtual ~IRootFinderMethod() {}
   IRootFinderMethod() {}
   
   // Common functionality
   virtual int SetFunction(const ROOT::Math::IGradFunction&, double) // for algorithms using derivatives only!
   {
      MATH_ERROR_MSG("SetFunction", "This method must be used with a Root Finder algorithm using derivatives");
      return -1;
   }
   virtual int SetFunction(const ROOT::Math::IGenFunction& , double , double )  // for the rest of algorithms...
   {
      MATH_ERROR_MSG("SetFunction", "Algorithm requires derivatives");
      return -1;
   }
   virtual double Root() const = 0;

   // Methods to be Implemented in the derived classes
   virtual int Solve(int maxIter = 100, double absTol = 1E-3, double relTol = 1E-6) = 0;
   virtual const char* Name() const = 0;
   
   // To accomplish with the GSLRootFinder and GSLRootFinderDeriv classes
   // They will return an error if it's not one of them.
   virtual int Iterate() {
      MATH_ERROR_MSG("Iterate", "This method must be used with a Root Finder algorithm wrapping the GSL Library");
      return -1;
   }
   virtual int Iterations() const {
      MATH_ERROR_MSG("Iterations", "This method must be used with a Root Finder algorithm wrapping the GSL Library");
      return -1;
   }

};

}
}


#endif
