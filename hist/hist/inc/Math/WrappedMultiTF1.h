// @(#)root/mathmore:$Id$
// Author: L. Moneta Wed Sep  6 09:52:26 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class WrappedTFunction

#ifndef ROOT_Math_WrappedMultiTF1
#define ROOT_Math_WrappedMultiTF1


#ifndef ROOT_Math_IParamFunction
#include "Math/IParamFunction.h"
#endif

#ifndef ROOT_TF1
#include "TF1.h"
#endif

namespace ROOT { 

   namespace Math { 


/** 
   Class to Wrap a ROOT Function class (like TF1)  in a IParamMultiFunction interface
   of multi-dimensions to be used in the ROOT::Math numerical algorithm
   The parameter are stored in this wrapper class, so the TF1 parameter values are not used for evaluating the function. 
   This allows for the copy of the wrapper function without the need to copy the TF1. 
   This wrapper class does not own the TF1 pointer, so it assumes it exists during the wrapper lifetime. 

   @ingroup CppFunctions
*/ 
class WrappedMultiTF1 : public ROOT::Math::IParamMultiGradFunction {

public: 

   typedef  ROOT::Math::IParamMultiGradFunction        BaseParamFunc; 
   typedef  ROOT::Math::IParamMultiFunction::BaseFunc  BaseFunc; 
 

   /** 
      constructor from a function pointer to a TF1 
      If dim = 0 dimension is taken from TF1::GetNdim(). 
      IN case of multi-dimensional function created using directly TF1 object the dimension 
      returned by TF1::GetNdim is always 1. The user must then pass the correct value of dim
   */ 
   WrappedMultiTF1 (TF1 & f, unsigned int dim = 0 );  

   /** 
      Destructor (no operations). Function pointer is not owned
   */ 
   virtual ~WrappedMultiTF1 () {}

   /** 
      Copy constructor
   */ 
   WrappedMultiTF1(const WrappedMultiTF1 & rhs); 

   /** 
      Assignment operator
   */ 
   WrappedMultiTF1 & operator = (const WrappedMultiTF1 & rhs); 


   /** @name interface inherited from IFunction */

   /** 
       Clone the wrapper but not the original function
   */
   IMultiGenFunction * Clone() const { 
      return new WrappedMultiTF1(*this); 
   } 

   /// function dimension
   unsigned int NDim() const { 
      return fDim;
   }


   /** @name interface inherited from IParamFunction */     

   /// get the parameter values (return values cached inside, those inside TF1 might be different) 
   const double * Parameters() const {
      return  (fParams.size() > 0) ? &fParams.front() : 0;  
   }

   /// set parameter values (only the cached one in this class,leave unchanges those of TF1)
   void SetParameters(const double * p) { 
      std::copy(p,p+fParams.size(),fParams.begin());
   } 

   /// return number of parameters 
   unsigned int NPar() const {
      return fParams.size();
   }

   /// return parameter name (from TF1)
   std::string ParameterName(unsigned int i) const { 
      return std::string(fFunc->GetParName(i)); 
   } 

   /// evaluate the derivative of the function with respect to the parameters
   void  ParameterGradient(const double * x, const double * par, double * grad ) const;

   /// precision value used for calculating the derivative step-size 
   /// h = eps * |x|. The default is 0.001, give a smaller in case function changes rapidly
   static void SetDerivPrecision(double eps); 

   /// get precision value used for calculating the derivative step-size 
   static double GetDerivPrecision();


private: 

   /// evaluate function passing coordinates x and vector of parameters
   double DoEvalPar (const double * x, const double * p ) const { 
      if (fFunc->GetMethodCall() )  fFunc->InitArgs(x,p);  // needed for interpreted functions 
      return fFunc->EvalPar(x,p); 
   }

   /// evaluate the partial derivative with respect to the parameter
   double DoParameterDerivative(const double * x, const double * p, unsigned int ipar) const; 


   bool fLinear;                 // flag for linear functions 
   TF1 * fFunc;                   // pointer to ROOT function
   unsigned int fDim;             // cached value of dimension
   std::vector<double> fParams;   // cached vector with parameter values

   static double fgEps;          // epsilon used in derivative calculation h ~ eps |p|
}; 

   } // end namespace Fit

} // end namespace ROOT


#endif /* ROOT_Fit_WrappedMultiTF1 */
