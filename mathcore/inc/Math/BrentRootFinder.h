// @(#)root/mathcore:$Id$
// Authors: David Gonzalez Maline    01/2008 

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header for the RootFinder
// 
// Created by: David Gonzalez Maline  : Wed Jan 21 2008
// 

#ifndef ROOT_Math_BrentRootFinder
#define ROOT_Math_BrentRootFinder

#include <Math/IFunction.h>
#include <Math/IRootFinderMethod.h>

namespace ROOT {
namespace Math {

   class BrentRootFinder: public IRootFinderMethod {
   public:
      virtual ~BrentRootFinder();
      BrentRootFinder();
      
      using IRootFinderMethod::SetFunction;
      int SetFunction(const ROOT::Math::IGenFunction& f, double xlow, double xup);
      
      int Solve(int maxIter = 100, double absTol = 1E-3, double relTol = 1E-6);
      double Root() const;
      
      const char* Name() const;
      
   protected:
      const IGenFunction* fFunction;
      double fXMin, fXMax;
      double fRoot;
   };
   
}
}

#endif
