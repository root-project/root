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
   // Destructor: Removes function if needed.

   if ( fFunction != 0 && fFunctionCopied )
      delete fFunction;
}

void RichardsonDerivator::SetRelTolerance (double eps)
{
   // Set the desired relative Error

   if(eps< 1e-10 || eps > 1e-2) {
      MATH_WARN_MSG("RichardsonDerivator::SetRelTolerance","parameter esp out of allowed range[1e-10,1e-2], reset to 0.001");
      eps = 0.001;
   }

   fEpsilon = eps;
}

double RichardsonDerivator::Error() const
{
   // Returns the estimate of the absolute Error of the last derivative
   // calculation
   
   return fLastError;
}

void RichardsonDerivator::SetFunction (const IGenFunction & function, double xlow, double xup)
{
   // Set derivation function (flag control if function must be copied inside)

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
   // Returns the first derivative of the function at point x,
   // computed by Richardson's extrapolation method (use 2 derivative estimates
   // to compute a third, more accurate estimation)
   // first, derivatives with steps h and h/2 are computed by central difference formulas
   //Begin_Latex
   // D(h) = #frac{f(x+h) - f(x-h)}{2h}
   //End_Latex
   // the final estimate Begin_Latex D = #frac{4D(h/2) - D(h)}{3} End_Latex
   //  "Numerical Methods for Scientists and Engineers", H.M.Antia, 2nd edition"
   //
   // the argument eps may be specified to control the step size (precision).
   // the step size is taken as eps*(xmax-xmin).
   // the default value (0.001) should be good enough for the vast majority
   // of functions. Give a smaller value if your function has many changes
   // of the second derivative in the function range.
   //
   // Getting the error via TF1::DerivativeError:
   //   (total error = roundoff error + interpolation error)
   // the estimate of the roundoff error is taken as follows:
   //Begin_Latex
   //    err = k#sqrt{f(x)^{2} + x^{2}deriv^{2}}#sqrt{#sum ai^{2}},
   //End_Latex
   // where k is the double precision, ai are coefficients used in
   // central difference formulas
   // interpolation error is decreased by making the step size h smaller.

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
   // Returns the second derivative of the function at point x,
   // computed by Richardson's extrapolation method (use 2 derivative estimates
   // to compute a third, more accurate estimation)
   // first, derivatives with steps h and h/2 are computed by central difference formulas
   //Begin_Latex
   //    D(h) = #frac{f(x+h) - 2f(x) + f(x-h)}{h^{2}}
   //End_Latex
   // the final estimate Begin_Latex D = #frac{4D(h/2) - D(h)}{3} End_Latex
   //  "Numerical Methods for Scientists and Engineers", H.M.Antia, 2nd edition"
   //
   // the argument eps may be specified to control the step size (precision).
   // the step size is taken as eps*(xmax-xmin).
   // the default value (0.001) should be good enough for the vast majority
   // of functions. Give a smaller value if your function has many changes
   // of the second derivative in the function range.
   //
   // Getting the error via TF1::DerivativeError:
   //   (total error = roundoff error + interpolation error)
   // the estimate of the roundoff error is taken as follows:
   //Begin_Latex
   //    err = k#sqrt{f(x)^{2} + x^{2}deriv^{2}}#sqrt{#sum ai^{2}},
   //End_Latex
   // where k is the double precision, ai are coefficients used in
   // central difference formulas
   // interpolation error is decreased by making the step size h smaller.

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
   // Returns the third derivative of the function at point x,
   // computed by Richardson's extrapolation method (use 2 derivative estimates
   // to compute a third, more accurate estimation)
   // first, derivatives with steps h and h/2 are computed by central difference formulas
   //Begin_Latex
   //    D(h) = #frac{f(x+2h) - 2f(x+h) + 2f(x-h) - f(x-2h)}{2h^{3}}
   //End_Latex
   // the final estimate Begin_Latex D = #frac{4D(h/2) - D(h)}{3} End_Latex
   //  "Numerical Methods for Scientists and Engineers", H.M.Antia, 2nd edition"
   //
   // the argument eps may be specified to control the step size (precision).
   // the step size is taken as eps*(xmax-xmin).
   // the default value (0.001) should be good enough for the vast majority
   // of functions. Give a smaller value if your function has many changes
   // of the second derivative in the function range.
   //
   // Getting the error via TF1::DerivativeError:
   //   (total error = roundoff error + interpolation error)
   // the estimate of the roundoff error is taken as follows:
   //Begin_Latex
   //    err = k#sqrt{f(x)^{2} + x^{2}deriv^{2}}#sqrt{#sum ai^{2}},
   //End_Latex
   // where k is the double precision, ai are coefficients used in
   // central difference formulas
   // interpolation error is decreased by making the step size h smaller.

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

}
}
