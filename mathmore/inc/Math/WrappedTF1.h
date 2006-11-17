// @(#)root/fit:$Name:  $:$Id: inc/Fit/WrappedTF1.h,v 1.0 2006/01/01 12:00:00 moneta Exp $
// Author: L. Moneta Wed Sep  6 09:52:26 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class WrappedTFunction

#ifndef ROOT_Math_WrappedTF1
#define ROOT_Math_WrappedTF1


#ifndef ROOT_Math_IParamFunction
#include "Math/IParamFunction.h"
#endif


#include "TF1.h"

namespace ROOT { 

   namespace Math { 


/** 
   Class to Wrap a ROOT Function class (like TF1)  in a IParamFunction interface
   in order to be used in fitting
*/ 
class WrappedTF1 : public ROOT::Math::IParamGradFunction<ROOT::Math::OneDim> {

public: 

   typedef  ROOT::Math::IParamGradFunction<ROOT::Math::OneDim> BaseGradFunc; 
   typedef  ROOT::Math::IParamGradFunction<ROOT::Math::OneDim>::BaseFunc BaseFunc; 
 
   WrappedTF1() {}

   /** 
      constructor from a function pointer. A flag (default false) is specified if 
      class owns the pointer
   */ 
   WrappedTF1 ( TF1 & f  )  : 
      fFunc(&f) 
   {
      fFunc->InitArgs(fX, 0 );
   }

   /** 
      Destructor (no operations). Function pointer is not owned
   */ 
   virtual ~WrappedTF1 () {}

   /** 
      Copy constructor
   */ 
   WrappedTF1(const WrappedTF1 & rhs) :
      BaseFunc(),
      BaseGradFunc(),
      fFunc(rhs.fFunc) 
   {
      fFunc->InitArgs(fX, 0 );
   }

   /** 
      Assignment operator
   */ 
   WrappedTF1 & operator = (const WrappedTF1 & rhs) { 
      if (this == &rhs) return *this;  // time saving self-test
      fFunc = rhs.fFunc; 
      fFunc->InitArgs(fX, 0 );
   } 



   /** @name interface inherited from IFunction */

   /** 
       Clone the wrapper but not the original function
   */
   ROOT::Math::IGenFunction * Clone() const { 
      return new WrappedTF1(*fFunc); 
   } 


   /** @name interface inherited from IParamFunction */     

   /// access the parameter values
   const double * Parameters() const {
      return fFunc->GetParameters();   
   }

   /// set parameter values
   void SetParameters(const double * p) { 
      fFunc->SetParameters(p); 
      // need to re-initialize it
      fFunc->InitArgs(fX, p );
   } 

   /// return number of parameters 
   unsigned int NPar() const { 
      return static_cast<unsigned int>(fFunc->GetNpar() );
   }

   /// return parameter name
   std::string ParameterName(unsigned int i) const { 
      return std::string(fFunc->GetParName(i)); 
   } 

   /// evaluate function passing coordinates x and vector of parameters
   double operator() (const double * x, const double * p ) { 
      fFunc->InitArgs(x,p);  // needed for interpreted functions 
      return fFunc->EvalPar(x,p); 
   }

   /// evaluate integral between x1   and x2 
   //double Integral(double * x1, double * x2) const;

   using BaseGradFunc::operator();



private: 


   /// evaluate function using parameter values cached in the TF1 
   double DoEval (double x) const { 
      // no need to InitArg
      fX[0] = x; 
      return fFunc->EvalPar(fX,0); 
   }

   /// return the function derivatives w.r.t. x 
   double DoDerivative( double  x  ) const { 
      static const double kEps = 0.001;
      return  fFunc->Derivative(x,0,kEps); 
   }

   /// evaluate the derivative of the function with respect to the parameters
   void  DoParameterGradient(double x, double * grad ) const { 
      static const double kEps = 0.001;
      fFunc->GradientPar(&x,grad,kEps); 
   }


   // pointer to ROOT function
   TF1 * fFunc; 
   mutable double fX[1]; 
}; 

   } // end namespace Fit

} // end namespace ROOT


#endif /* ROOT_Fit_WrappedTF1 */
