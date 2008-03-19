// @(#)root/mathcore:$Id$
// Authors: David Gonzalez Maline    01/2008 

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

#include "Math/RichardsonDerivator.h"
#include <cmath>

#ifndef ROOT_Math_Error
#include "Math/Error.h"
#endif

namespace ROOT {
namespace Math {

RichardsonDerivator::RichardsonDerivator()
{
   // Default Constructor.

   fEpsilon = 0.001;
   fLastError = 0;
   fFunctionCopied = false;
   fFunction = 0;
}

RichardsonDerivator::~RichardsonDerivator()
{
   if ( fFunction != 0 && fFunctionCopied )
      delete fFunction;
}

void RichardsonDerivator::SetRelTolerance (double eps)
{
   if(eps< 1e-10 || eps > 1e-2) {
      MATH_WARN_MSG("RichardsonDerivator::SetRelTolerance","parameter esp out of allowed range[1e-10,1e-2], reset to 0.001");
      eps = 0.001;
   }

   fEpsilon = eps;
}

double RichardsonDerivator::Error() const
{   return fLastError;  }

void RichardsonDerivator::SetFunction (const IGenFunction & function, double xlow, double xup)
{
   fFunction = &function;

   if (xlow >= xup) 
   {
      double tmp = xlow;
      xlow = xup; 
      xup = tmp;
   }
   fXMin = xlow;
   fXMax = xup;
}

double RichardsonDerivator::Derivative1 (double x)
{
   const double kC1 = 1e-15;

   double h = fEpsilon*(fXMax-fXMin);

   double xx[1];
   xx[0] = x+h;     double f1 = (*fFunction)(xx);
   xx[0] = x;       double fx = (*fFunction)(xx);
   xx[0] = x-h;     double f2 = (*fFunction)(xx);
   
   xx[0] = x+h/2;   double g1 = (*fFunction)(xx);
   xx[0] = x-h/2;   double g2 = (*fFunction)(xx);

   //compute the central differences
   double h2    = 1/(2.*h);
   double d0    = f1 - f2;
   double d2    = 2*(g1 - g2);
   fLastError   = kC1*h2*fx;  //compute the error
   double deriv = h2*(4*d2 - d0)/3.;
   return deriv;
}

double RichardsonDerivator::Derivative2 (double x)
{
   const double kC1 = 2*1e-15;

   double h = fEpsilon*(fXMax-fXMin);

   double xx[1];
   xx[0] = x+h;     double f1 = (*fFunction)(xx);
   xx[0] = x;       double f2 = (*fFunction)(xx);
   xx[0] = x-h;     double f3 = (*fFunction)(xx);

   xx[0] = x+h/2;   double g1 = (*fFunction)(xx);
   xx[0] = x-h/2;   double g3 = (*fFunction)(xx);

   //compute the central differences
   double hh    = 1/(h*h);
   double d0    = f3 - 2*f2 + f1;
   double d2    = 4*g3 - 8*f2 +4*g1;
   fLastError   = kC1*hh*f2;  //compute the error
   double deriv = hh*(4*d2 - d0)/3.;
   return deriv;
}

double RichardsonDerivator::Derivative3 (double x)
{
   const double kC1 = 1e-15;

   double h = fEpsilon*(fXMax-fXMin);

   double xx[1];
   xx[0] = x+2*h;   double f1 = (*fFunction)(xx);
   xx[0] = x+h;     double f2 = (*fFunction)(xx);
   xx[0] = x-h;     double f3 = (*fFunction)(xx);
   xx[0] = x-2*h;   double f4 = (*fFunction)(xx);
   xx[0] = x;       double fx = (*fFunction)(xx);
   xx[0] = x+h/2;   double g2 = (*fFunction)(xx);
   xx[0] = x-h/2;   double g3 = (*fFunction)(xx);

   //compute the central differences
   double hhh  = 1/(h*h*h);
   double d0   = 0.5*f1 - f2 +f3 - 0.5*f4;
   double d2   = 4*f2 - 8*g2 +8*g3 - 4*f3;
   fLastError    = kC1*hhh*fx;   //compute the error
   double deriv = hhh*(4*d2 - d0)/3.;
   return deriv;
}

} // end namespace Math
   
} // end namespace ROOT
