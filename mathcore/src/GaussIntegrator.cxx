// @(#)root/mathcore:$Id$
// Authors: David Gonzalez Maline    01/2008 

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

#include "Math/GaussIntegrator.h"
#include <cmath>

namespace ROOT {
namespace Math {

bool GaussIntegratorOneDim::fgAbsValue = false;

GaussIntegratorOneDim::GaussIntegratorOneDim()
{
   // Default Constructor.

   fEpsilon = 1e-12;
   fLastResult = fLastError = 0;
   fUsedOnce = false;
   fFunctionCopied = false;
   fFunction = 0;
}

GaussIntegratorOneDim::~GaussIntegratorOneDim()
{
   // Deletes the function if it was previously copied.

   if ( fFunctionCopied && fFunction != 0   )
      delete fFunction;
}

void GaussIntegratorOneDim::AbsValue(bool flag)
{
   // Static function: set the fgAbsValue flag.
   // By default TF1::Integral uses the original function value to compute the integral
   // However, TF1::Moment, CentralMoment require to compute the integral
   // using the absolute value of the function.
   
   fgAbsValue = flag;
}

double GaussIntegratorOneDim::Integral(double a, double b)
{
   // Return Integral of function between a and b.
   //
   //   based on original CERNLIB routine DGAUSS by Sigfried Kolbig
   //   converted to C++ by Rene Brun
   //
   // This function computes, to an attempted specified accuracy, the value
   // of the integral.
   //Begin_Latex
   //   I = #int^{B}_{A} f(x)dx
   //End_Latex
   // Usage:
   //   In any arithmetic expression, this function has the approximate value
   //   of the integral I.
   //   - A, B: End-points of integration interval. Note that B may be less
   //           than A.
   //   - params: Array of function parameters. If 0, use current parameters.
   //   - epsilon: Accuracy parameter (see Accuracy).
   //
   //Method:
   //   For any interval [a,b] we define g8(a,b) and g16(a,b) to be the 8-point
   //   and 16-point Gaussian quadrature approximations to
   //Begin_Latex
   //   I = #int^{b}_{a} f(x)dx
   //End_Latex
   //   and define
   //Begin_Latex
   //   r(a,b) = #frac{#||{g_{16}(a,b)-g_{8}(a,b)}}{1+#||{g_{16}(a,b)}}
   //End_Latex
   //   Then,
   //Begin_Latex
   //   G = #sum_{i=1}^{k}g_{16}(x_{i-1},x_{i})
   //End_Latex
   //   where, starting with x0 = A and finishing with xk = B,
   //   the subdivision points xi(i=1,2,...) are given by
   //Begin_Latex
   //   x_{i} = x_{i-1} + #lambda(B-x_{i-1})
   //End_Latex
   //   Begin_Latex #lambdaEnd_Latex is equal to the first member of the
   //   sequence 1,1/2,1/4,... for which r(xi-1, xi) < EPS.
   //   If, at any stage in the process of subdivision, the ratio
   //Begin_Latex
   //   q = #||{#frac{x_{i}-x_{i-1}}{B-A}}
   //End_Latex
   //   is so small that 1+0.005q is indistinguishable from 1 to
   //   machine accuracy, an error exit occurs with the function value
   //   set equal to zero.
   //
   // Accuracy:
   //   Unless there is severe cancellation of positive and negative values of
   //   f(x) over the interval [A,B], the relative error may be considered as
   //   specifying a bound on the <I>relative</I> error of I in the case
   //   |I|&gt;1, and a bound on the absolute error in the case |I|&lt;1. More
   //   precisely, if k is the number of sub-intervals contributing to the
   //   approximation (see Method), and if
   //Begin_Latex
   //   I_{abs} = #int^{B}_{A} #||{f(x)}dx
   //End_Latex
   //   then the relation
   //Begin_Latex
   //   #frac{#||{G-I}}{I_{abs}+k} < EPS
   //End_Latex
   //   will nearly always be true, provided the routine terminates without
   //   printing an error message. For functions f having no singularities in
   //   the closed interval [A,B] the accuracy will usually be much higher than
   //   this.
   //
   // Error handling:
   //   The requested accuracy cannot be obtained (see Method).
   //   The function value is set equal to zero.
   //
   // Note 1:
   //   Values of the function f(x) at the interval end-points A and B are not
   //   required. The subprogram may therefore be used when these values are
   //   undefined.
   //

   const double kHF = 0.5;
   const double kCST = 5./1000;

   double x[12] = { 0.96028985649753623,  0.79666647741362674,
                      0.52553240991632899,  0.18343464249564980,
                      0.98940093499164993,  0.94457502307323258,
                      0.86563120238783174,  0.75540440835500303,
                      0.61787624440264375,  0.45801677765722739,
                      0.28160355077925891,  0.09501250983763744};

   double w[12] = { 0.10122853629037626,  0.22238103445337447,
                      0.31370664587788729,  0.36268378337836198,
                      0.02715245941175409,  0.06225352393864789,
                      0.09515851168249278,  0.12462897125553387,
                      0.14959598881657673,  0.16915651939500254,
                      0.18260341504492359,  0.18945061045506850};

   double h, aconst, bb, aa, c1, c2, u, s8, s16, f1, f2;
   double xx[1];
   int i;

   if ( fFunction == 0 )
   {
      MATH_ERROR_MSG("ROOT::Math::GausIntegratorOneDim", "A function must be set first!");
      return 0.0;
   }

   h = 0;
   if (b == a) return h;
   aconst = kCST/std::abs(b-a);
   bb = a;
CASE1:
   aa = bb;
   bb = b;
CASE2:
   c1 = kHF*(bb+aa);
   c2 = kHF*(bb-aa);
   s8 = 0;
   for (i=0;i<4;i++) {
      u     = c2*x[i];
      xx[0] = c1+u;
      f1    = (*fFunction)(xx);
      if (fgAbsValue) f1 = std::abs(f1);
      xx[0] = c1-u;
      f2    = (*fFunction) (xx);
      if (fgAbsValue) f2 = std::abs(f2);
      s8   += w[i]*(f1 + f2);
   }
   s16 = 0;
   for (i=4;i<12;i++) {
      u     = c2*x[i];
      xx[0] = c1+u;
      f1    = (*fFunction) (xx);
      if (fgAbsValue) f1 = std::abs(f1);
      xx[0] = c1-u;
      f2    = (*fFunction) (xx);
      if (fgAbsValue) f2 = std::abs(f2);
      s16  += w[i]*(f1 + f2);
   }
   s16 = c2*s16;
   if (std::abs(s16-c2*s8) <= fEpsilon*(1. + std::abs(s16))) {
      h += s16;
      if(bb != b) goto CASE1;
   } else {
      bb = c1;
      if(1. + aconst*std::abs(c2) != 1) goto CASE2;
      h = s8;  //this is a crude approximation (cernlib function returned 0 !)
   }

   fUsedOnce = true;
   fLastResult = h;
   fLastError = std::abs(s16-c2*s8);

   return h;
}
   

void GaussIntegratorOneDim::SetRelTolerance (double eps)
{
   // Set the desired relative Error

   fEpsilon = eps;
}

void GaussIntegratorOneDim::SetAbsTolerance (double)
{
   // This method is not implemented.

   MATH_ERROR_MSG("ROOT::Math::GausIntegratorOneDim", "There is no Absolute Tolerance!");
}

double GaussIntegratorOneDim::Result () const
{
   // Return  the Result of the last Integral calculation.

   if (!fUsedOnce)
      MATH_ERROR_MSG("ROOT::Math::GausIntegratorOneDim", "You must calculate the result at least once!");

   return fLastResult;
}

double GaussIntegratorOneDim::Error() const
{
   // Return the estimate of the absolute Error of the last Integral calculation.

   return fLastError;
}

int GaussIntegratorOneDim::Status() const
{
   // This method is not implemented.

   return 0;
}

void GaussIntegratorOneDim::SetFunction (const IGenFunction & function, bool copy)
{
   // Set integration function (flag control if function must be copied inside)

   if ( copy )
      fFunction = function.Clone();
   else
      fFunction = &function;

   fFunctionCopied = copy;
}


double GaussIntegratorOneDim::Integral ()
{
   // This method is not implemented.

   return 0.0;
}

double GaussIntegratorOneDim::IntegralUp (double /*a*/)
{
   // This method is not implemented.

   return 0.0;
}

double GaussIntegratorOneDim::IntegralLow (double /*b*/)
{
   // This method is not implemented.

   return 0.0;
}

double GaussIntegratorOneDim::Integral (const std::vector< double > &/*pts*/)
{
   // This method is not implemented.

   return 0.0;
}

double GaussIntegratorOneDim::IntegralCauchy (double /*a*/, double /*b*/, double /*c*/)
{
   // This method is not implemented.

   return 0.0;
}


}
}
