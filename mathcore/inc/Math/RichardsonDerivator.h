// @(#)root/mathcore:$Id$
// Authors: David Gonzalez Maline    01/2008 

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for RichardsonDerivator
// 
// Created by: David Gonzalez Maline  : Mon Feb 4 2008
// 

#include <Math/IFunction.h>

namespace ROOT {
namespace Math {

class RichardsonDerivator {
public:
   ~RichardsonDerivator();
   RichardsonDerivator();
   
   // Implementing VirtualIntegrator Interface
   void SetRelTolerance (double);
   double Error () const;

   // Implementing VirtualIntegratorOneDim Interface
   double Derivative1 (double x);
   double Derivative2 (double x);
   double Derivative3 (double x);

   void SetFunction (const IGenFunction &, double xmin, double xmax);

protected:
   double fEpsilon;
   double fLastError;
   const IGenFunction* fFunction;
   bool fFunctionCopied;
   double fXMin, fXMax;

};

}
}
