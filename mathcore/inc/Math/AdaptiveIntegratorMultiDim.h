// @(#)root/mathcore:$Id$
// Authors: M. Slawinska   08/2007 

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


/**
   class for adaptive quadrature integration in multi-dimensions
   Algorithm from  A.C. Genz, A.A. Malik, 
    1.A.C. Genz and A.A. Malik, An adaptive algorithm for numerical integration over 
    an N-dimensional rectangular region, J. Comput. Appl. Math. 6 (1980) 295-302.

   Converted/adapted by R.Brun to C++ from Fortran CERNLIB routine RADMUL (D120)
   The new code features many changes compared to the Fortran version.

   @ingroup Integration

  
 */

class AdaptiveIntegratorMultiDim : public VirtualIntegratorMultiDim {

public:
   // constructors
   explicit 
   AdaptiveIntegratorMultiDim(double absTol = 1.E-6, double relTol = 1E-6, unsigned int size = 100000);

   explicit
   AdaptiveIntegratorMultiDim(const IMultiGenFunction &f, double absTol = 1.E-9, double relTol = 1E-6, unsigned int size = 100000);


   virtual ~AdaptiveIntegratorMultiDim() {}

   //private:
   //   Integrator(const Integrator &);
   //   Integrator & operator=(const Integrator &);


   /**
      evaluate the integral with the previously given function between xmin[] and xmax[]  
   */
   double Integral(const double* xmin, const double * xmax);


   /// evaluate the integral passing a new function
   double Integral(const IMultiGenFunction &f, const double* xmin, const double * xmax);

   void SetFunction(const IMultiGenFunction &f);

   /// return result of integration 
   double Result() const { return fResult; }

   // return integration error 
   double Error() const { return fError; } 

   int Status() const { return fStatus; }

   // return number of function evaluations in calculating the integral 
   unsigned int NEval() const { return fNEval; }
 
   void SetRelTolerance(double relTol);
   void SetAbsTolerance(double absTol);


 private:

   unsigned int fDim; // dimentionality of integrand

   double fAbsTol;
   double fRelTol;
   unsigned int fSize;

   double fResult;
   double fError;
   unsigned int  fNEval;
   int fStatus;   // status of algorithm (error if not zero)

   const IMultiGenFunction* fFun;

};

}//namespace Math
}//namespace ROOT

#endif /* ROOT_Math_AdaptiveIntegratorMultiDim */
