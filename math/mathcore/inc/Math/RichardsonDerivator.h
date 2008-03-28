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

#ifndef ROOT_Math_RichardsonDerivator
#define ROOT_Math_RichardsonDerivator

#include <Math/IFunction.h>

namespace ROOT {
namespace Math {

//___________________________________________________________________________________________
/**
   User class for performing function integration. 

   It will use the Richards Method for function derivation in a given interval. 
   This class is implemented from TF1::Derivate{,2,3}().

   @ingroup Derivation
  
 */

class RichardsonDerivator {
public:

   /** Destructor: Removes function if needed. */
   ~RichardsonDerivator();

   /** Default Constructor. */
   RichardsonDerivator();
   
   // Implementing VirtualIntegrator Interface

   /** Set the desired relative Error. */
   void SetRelTolerance (double);

   /** Returns the estimate of the absolute Error of the last derivative calculation. */
   double Error () const;

   // Implementing VirtualIntegratorOneDim Interface

   /**
      Returns the first derivative of the function at point x,
      computed by Richardson's extrapolation method (use 2 derivative estimates
      to compute a third, more accurate estimation)
      first, derivatives with steps h and h/2 are computed by central difference formulas
     Begin_Latex
      D(h) = #frac{f(x+h) - f(x-h)}{2h}
     End_Latex
      the final estimate Begin_Latex D = #frac{4D(h/2) - D(h)}{3} End_Latex
       "Numerical Methods for Scientists and Engineers", H.M.Antia, 2nd edition"
     
      the argument eps may be specified to control the step size (precision).
      the step size is taken as eps*(xmax-xmin).
      the default value (0.001) should be good enough for the vast majority
      of functions. Give a smaller value if your function has many changes
      of the second derivative in the function range.
     
      Getting the error via TF1::DerivativeError:
        (total error = roundoff error + interpolation error)
      the estimate of the roundoff error is taken as follows:
     Begin_Latex
         err = k#sqrt{f(x)^{2} + x^{2}deriv^{2}}#sqrt{#sum ai^{2}},
     End_Latex
      where k is the double precision, ai are coefficients used in
      central difference formulas
      interpolation error is decreased by making the step size h smaller.
   */
   double Derivative1 (double x);

   /**
      Returns the second derivative of the function at point x,
      computed by Richardson's extrapolation method (use 2 derivative estimates
      to compute a third, more accurate estimation)
      first, derivatives with steps h and h/2 are computed by central difference formulas
     Begin_Latex
         D(h) = #frac{f(x+h) - 2f(x) + f(x-h)}{h^{2}}
     End_Latex
      the final estimate Begin_Latex D = #frac{4D(h/2) - D(h)}{3} End_Latex
       "Numerical Methods for Scientists and Engineers", H.M.Antia, 2nd edition"
     
      the argument eps may be specified to control the step size (precision).
      the step size is taken as eps*(xmax-xmin).
      the default value (0.001) should be good enough for the vast majority
      of functions. Give a smaller value if your function has many changes
      of the second derivative in the function range.
     
      Getting the error via TF1::DerivativeError:
        (total error = roundoff error + interpolation error)
      the estimate of the roundoff error is taken as follows:
     Begin_Latex
         err = k#sqrt{f(x)^{2} + x^{2}deriv^{2}}#sqrt{#sum ai^{2}},
     End_Latex
      where k is the double precision, ai are coefficients used in
      central difference formulas
      interpolation error is decreased by making the step size h smaller.
   */
   double Derivative2 (double x);

   /**
      Returns the third derivative of the function at point x,
      computed by Richardson's extrapolation method (use 2 derivative estimates
      to compute a third, more accurate estimation)
      first, derivatives with steps h and h/2 are computed by central difference formulas
     Begin_Latex
         D(h) = #frac{f(x+2h) - 2f(x+h) + 2f(x-h) - f(x-2h)}{2h^{3}}
     End_Latex
      the final estimate Begin_Latex D = #frac{4D(h/2) - D(h)}{3} End_Latex
       "Numerical Methods for Scientists and Engineers", H.M.Antia, 2nd edition"
     
      the argument eps may be specified to control the step size (precision).
      the step size is taken as eps*(xmax-xmin).
      the default value (0.001) should be good enough for the vast majority
      of functions. Give a smaller value if your function has many changes
      of the second derivative in the function range.
     
      Getting the error via TF1::DerivativeError:
        (total error = roundoff error + interpolation error)
      the estimate of the roundoff error is taken as follows:
     Begin_Latex
         err = k#sqrt{f(x)^{2} + x^{2}deriv^{2}}#sqrt{#sum ai^{2}},
     End_Latex
      where k is the double precision, ai are coefficients used in
      central difference formulas
      interpolation error is decreased by making the step size h smaller.
   */
   double Derivative3 (double x);

   /** Set function to solve and the interval in where to look for the root. 

       \@param f Function to be minimized.
       \@param xlow Lower bound of the search interval.
       \@param xup Upper bound of the search interval.
   */
   void SetFunction (const IGenFunction &, double xmin, double xmax);

protected:
   double fEpsilon;
   double fLastError;
   const IGenFunction* fFunction;
   bool fFunctionCopied;
   double fXMin, fXMax;

};

} // end namespace Math
   
} // end namespace ROOT

#endif /* ROOT_Math_RichardsonDerivator */

