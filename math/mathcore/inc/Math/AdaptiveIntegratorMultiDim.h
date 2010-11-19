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

#ifndef ROOT_Math_IFunctionfwd
#include "Math/IFunctionfwd.h"
#endif

#include "Math/VirtualIntegrator.h"

namespace ROOT {
namespace Math {


//__________________________________________________________________________________________
/**
   class for adaptive quadrature integration in multi-dimensions
   Algorithm from  A.C. Genz, A.A. Malik, An adaptive algorithm for numerical integration over 
   an N-dimensional rectangular region, J. Comput. Appl. Math. 6 (1980) 295-302.

   Converted/adapted by R.Brun to C++ from Fortran CERNLIB routine RADMUL (D120)
   The new code features many changes compared to the Fortran version.

   @ingroup Integration

  
*/

class AdaptiveIntegratorMultiDim : public VirtualIntegratorMultiDim {

public:

   /**
      construct given optionally tolerance (absolute and relative), maximum number of function evaluation (maxpts)  and 
      size of the working array. 
      The size of working array represents the number of sub-division used for calculating the integral. 
      Higher the dimension, larger sizes are required for getting the same accuracy.
      The size must be larger than  >= (2N + 3) * (1 + MAXPTS/(2**N + 2N(N + 1) + 1))/2). For smaller value passed, the 
      minimum allowed will be used
   */
   explicit 
   AdaptiveIntegratorMultiDim(double absTol = 1.E-6, double relTol = 1E-6, unsigned int maxpts = 100000, unsigned int size = 0);

   /**
      Construct with a reference to the integrand function and given optionally 
      tolerance (absolute and relative), maximum number of function evaluation (maxpts)  and 
      size of the working array. 
   */
   explicit
   AdaptiveIntegratorMultiDim(const IMultiGenFunction &f, double absTol = 1.E-9, double relTol = 1E-6,  unsigned int maxcall = 100000, unsigned int size = 0);

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

   unsigned int fDim;     // dimentionality of integrand
   unsigned int fMinPts;    // minimum number of function evaluation requested 
   unsigned int fMaxPts;    // maximum number of function evaluation requested 
   unsigned int fSize;    // max size of working array (explode with dimension)
   double fAbsTol;        // absolute tolerance
   double fRelTol;        // relative tolerance

   double fResult;        // last integration result 
   double fError;         // integration error 
   double fRelError;      // Relative error
   int    fNEval;        // number of function evaluation
   int fStatus;   // status of algorithm (error if not zero)

   const IMultiGenFunction* fFun;   // pointer to integrand function 

};

}//namespace Math
}//namespace ROOT

#endif /* ROOT_Math_AdaptiveIntegratorMultiDim */
