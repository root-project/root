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
      construct given optionally tolerance (absolute and relative) and maximum size of working array 
      The size of working array represents the number of sub-division used for calculating the integral. 
      Higher the dimension, larger sizes are required for getting the same accuracy. 
   */
   explicit 
   AdaptiveIntegratorMultiDim(double absTol = 1.E-6, double relTol = 1E-6, unsigned int size = 100000);

   /**
      construct with a reference to the integrand function and given optionally 
      tolerance (absolute and relative) and maximum size of working array.
   */
   explicit
   AdaptiveIntegratorMultiDim(const IMultiGenFunction &f, double absTol = 1.E-9, double relTol = 1E-6, unsigned int size = 100000);

   /**
      destructor (no operations)
    */
   virtual ~AdaptiveIntegratorMultiDim() {}


   /**
      evaluate the integral with the previously given function between xmin[] and xmax[]  
   */
   double Integral(const double* xmin, const double * xmax);


   /// evaluate the integral passing a new function
   double Integral(const IMultiGenFunction &f, const double* xmin, const double * xmax);

   /// set the integration function (must implement multi-dim funciton interface: IBaseFunctionMultiDim)
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
   unsigned int NEval() const { return fNEval; }
 
   /// set relative tolerance 
   void SetRelTolerance(double relTol);

   /// set absolute tolerance
   void SetAbsTolerance(double absTol);


 private:

   unsigned int fDim;     // dimentionality of integrand

   double fAbsTol;        // absolute tolerance
   double fRelTol;        // relative tolerance
   unsigned int fSize;    // max size of working array (explode with dimension)

   double fResult;        // last integration result 
   double fError;         // integration error 
   double fRelError;      // Relative error
   unsigned int  fNEval;  // number of function evaluation
   int fStatus;   // status of algorithm (error if not zero)

   const IMultiGenFunction* fFun;   // pointer to integrand function 

};

}//namespace Math
}//namespace ROOT

#endif /* ROOT_Math_AdaptiveIntegratorMultiDim */
