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
#include <limits>

#ifndef ROOT_Math_Error
#include "Math/Error.h"
#endif

namespace ROOT {
namespace Math {

RichardsonDerivator::RichardsonDerivator(double h) : 
   fFunctionCopied(false),
   fStepSize(h),
   fLastError(0),
   fFunction(0)
{
   // Default Constructor.
}
RichardsonDerivator::RichardsonDerivator(const ROOT::Math::IGenFunction & f, double h, bool copyFunc) : 
   fFunctionCopied(copyFunc),
   fStepSize(h),
   fLastError(0),
   fFunction(0)
{
   // Constructor from a function and step size
   if (copyFunc) fFunction = f.Clone(); 
   else fFunction = &f;
}


RichardsonDerivator::~RichardsonDerivator()
{
   // destructor
   if ( fFunction != 0 && fFunctionCopied )
      delete fFunction;
}

RichardsonDerivator::RichardsonDerivator(const RichardsonDerivator & rhs) 
{
    // copy constructor
    // copy constructor (deep copy or not depending on fFunctionCopied)
   fStepSize = rhs.fStepSize;
   fLastError = rhs.fLastError;
   fFunctionCopied = rhs.fFunctionCopied; 
   SetFunction(*rhs.fFunction);   
 }

RichardsonDerivator &  RichardsonDerivator::operator= ( const RichardsonDerivator & rhs) 
{
   // Assignment operator
   if (&rhs == this) return *this;
   fFunctionCopied = rhs.fFunctionCopied;
   fStepSize = rhs.fStepSize;
   fLastError = rhs.fLastError;
   SetFunction(*rhs.fFunction);
   return *this;
}

void RichardsonDerivator::SetFunction(const ROOT::Math::IGenFunction & f)
{
   // set function
   if (fFunctionCopied) {
      if (fFunction) delete fFunction;
      fFunction = f.Clone(); 
   }
   else fFunction = &f;
}

double RichardsonDerivator::Derivative1 (double x)
{
   const double kC1 = 1.E-15;

   double h = fStepSize; 

   double xx[1];
   xx[0] = x+h;     double f1 = (*fFunction)(xx);
   //xx[0] = x;       double fx = (*fFunction)(xx); // not needed
   xx[0] = x-h;     double f2 = (*fFunction)(xx);
   
   xx[0] = x+h/2;   double g1 = (*fFunction)(xx);
   xx[0] = x-h/2;   double g2 = (*fFunction)(xx);

   //compute the central differences
   double h2    = 1/(2.*h);
   double d0    = f1 - f2;
   double d2    = g1 - g2;
   double deriv = h2*(8*d2 - d0)/3.;
   // compute the error ( to be improved ) this is just a simple truncation error
   fLastError   = kC1*h2*0.5*(f1+f2);  //compute the error

   return deriv;
}

double RichardsonDerivator::Derivative2 (double x)
{
   const double kC1 = 2*1e-15;

   double h = fStepSize;

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

   double h = fStepSize;

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
