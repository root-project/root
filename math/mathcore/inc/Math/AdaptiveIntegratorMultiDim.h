// @(#)root/mathcore:$Id$
// Author: M. Slawinska   08/2007

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2007 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header source file for class AdaptiveIntegratorMultiDim


#ifndef ROOT_Math_AdaptiveIntegratorMultiDim
#define ROOT_Math_AdaptiveIntegratorMultiDim

#include "Math/IFunctionfwd.h"

#include "Math/VirtualIntegrator.h"

namespace ROOT {
namespace Math {

/** \class AdaptiveIntegratorMultiDim
    \ingroup Integration

Class for adaptive quadrature integration in multi-dimensions using rectangular regions.
Algorithm from  A.C. Genz, A.A. Malik, An adaptive algorithm for numerical integration over
an N-dimensional rectangular region, J. Comput. Appl. Math. 6 (1980) 295-302.

Converted/adapted by R.Brun to C++ from Fortran CERNLIB routine RADMUL (D120)
The new code features many changes compared to the Fortran version.

Control  parameters are:

  - \f$ minpts \f$: Minimum number of function evaluations requested. Must not exceed maxpts.
            if minpts < 1 minpts is set to \f$ 2^n +2n(n+1) +1 \f$ where n is the function dimension
  - \f$ maxpts \f$: Maximum number of function evaluations to be allowed.
            \f$ maxpts >= 2^n +2n(n+1) +1 \f$
            if \f$ maxpts<minpts \f$, \f$ maxpts \f$ is set to \f$ 10minpts \f$
  - \f$ epstol \f$, \f$ epsrel \f$ : Specified relative and  absolute accuracy.

The integral will stop if the relative error is less than relative tolerance OR the
absolute error is less than the absolute tolerance

The class computes in addition to the integral of the function in the desired interval:

  - an estimation of the relative accuracy of the result.
  - number of function evaluations performed.
  - status code:
      0. Normal exit.  . At least minpts and at most maxpts calls to the function were performed.
      1. maxpts is too small for the specified accuracy eps.
         The result and relerr contain the values obtainable for the
         specified value of maxpts.
      2. size is too small for the specified number MAXPTS of function evaluations.
      3. n<2 or n>15

### Method:

An integration rule of degree seven is used together with a certain
strategy of subdivision.
For a more detailed description of the method see References.

### Notes:

  1..Multi-dimensional integration is time-consuming. For each rectangular
     subregion, the routine requires function evaluations.
     Careful programming of the integrand might result in substantial saving
     of time.
  2..Numerical integration usually works best for smooth functions.
     Some analysis or suitable transformations of the integral prior to
     numerical work may contribute to numerical efficiency.

### References:

  1. A.C. Genz and A.A. Malik, Remarks on algorithm 006:
     An adaptive algorithm for numerical integration over
     an N-dimensional rectangular region, J. Comput. Appl. Math. 6 (1980) 295-302.
  2. A. van Doren and L. de Ridder, An adaptive algorithm for numerical
     integration over an n-dimensional cube, J.Comput. Appl. Math. 2 (1976) 207-217.

*/

class AdaptiveIntegratorMultiDim : public VirtualIntegratorMultiDim {

public:

   /**
      Construct given optionally tolerance (absolute and relative), maximum number of function evaluation (maxpts)  and
      size of the working array.
      The integration will stop when the absolute error is less than the absolute tolerance OR when the relative error is less
      than the relative tolerance. The absolute tolerance by default is not used (it is equal to zero).
      The size of working array represents the number of sub-division used for calculating the integral.
      Higher the dimension, larger sizes are required for getting the same accuracy.
      The size must be larger than
      \f$ (2n + 3) (1 + maxpts/(2^n + 2n(n + 1) + 1))/2) \f$.
      For smaller value passed, the minimum allowed will be used
   */
   explicit
   AdaptiveIntegratorMultiDim(double absTol = 0.0, double relTol = 1E-9, unsigned int maxpts = 100000, unsigned int size = 0);

   /**
      Construct with a reference to the integrand function and given optionally
      tolerance (absolute and relative), maximum number of function evaluation (maxpts)  and
      size of the working array.
   */
   explicit
   AdaptiveIntegratorMultiDim(const IMultiGenFunction &f, double absTol = 0.0, double relTol = 1E-9,  unsigned int maxcall = 100000, unsigned int size = 0);

   /**
      destructor (no operations)
    */
   virtual ~AdaptiveIntegratorMultiDim() {}


   /**
      evaluate the integral with the previously given function between xmin[] and xmax[]
   */
   double Integral(const double* xmin, const double * xmax) {
      return DoIntegral(xmin,xmax, false);
   }


   /// evaluate the integral passing a new function
   double Integral(const IMultiGenFunction &f, const double* xmin, const double * xmax);

   /// set the integration function (must implement multi-dim function interface: IBaseFunctionMultiDim)
   void SetFunction(const IMultiGenFunction &f);

   /// return result of integration
   double Result() const { return fResult; }

   /// return integration error
   double Error() const { return fError; }

   /// return relative error
   double RelError() const { return fRelError; }

   /// return status of integration
   ///  - status = 0  successful integration
   ///  - status = 1
   ///    MAXPTS is too small for the specified accuracy EPS.
   ///    The result contain the values
   ///    obtainable for the specified value of MAXPTS.
   ///  - status = 2
   ///    size is too small for the specified number MAXPTS of function evaluations.
   ///  - status = 3
   ///    wrong dimension , N<2 or N > 15. Returned result and error are zero
   int Status() const { return fStatus; }

   /// return number of function evaluations in calculating the integral
   int NEval() const { return fNEval; }

   /// set relative tolerance
   void SetRelTolerance(double relTol);

   /// set absolute tolerance
   void SetAbsTolerance(double absTol);

   ///set workspace size
   void SetSize(unsigned int size) { fSize = size; }

   ///set min points
   void SetMinPts(unsigned int n) { fMinPts = n; }

   ///set max points
   void SetMaxPts(unsigned int n) { fMaxPts = n; }

   /// set the options
   void SetOptions(const ROOT::Math::IntegratorMultiDimOptions & opt);

   ///  get the option used for the integration
   ROOT::Math::IntegratorMultiDimOptions Options() const;

protected:

   // internal function to compute the integral (if absVal is true compute abs value of function integral
   double DoIntegral(const double* xmin, const double * xmax, bool absVal = false);

 private:

   unsigned int fDim;     // dimensionality of integrand
   unsigned int fMinPts;  // minimum number of function evaluation requested
   unsigned int fMaxPts;  // maximum number of function evaluation requested
   unsigned int fSize;    // max size of working array (explode with dimension)
   double fAbsTol;        // absolute tolerance
   double fRelTol;        // relative tolerance

   double fResult;        // last integration result
   double fError;         // integration error
   double fRelError;      // Relative error
   int    fNEval;         // number of function evaluation
   int fStatus;           // status of algorithm (error if not zero)

   const IMultiGenFunction* fFun;   // pointer to integrand function

};

}//namespace Math
}//namespace ROOT

#endif /* ROOT_Math_AdaptiveIntegratorMultiDim */
