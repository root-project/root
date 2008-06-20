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
class WrappedMultiTF1 : public ROOT::Math::IParamMultiFunction {

public: 

   typedef  ROOT::Math::IParamMultiFunction            BaseParamFunc; 
   typedef  ROOT::Math::IParamMultiFunction::BaseFunc  BaseFunc; 
 

   /** 
      constructor from a function pointer. 
      
   */ 
   WrappedMultiTF1 (TF1 & f )  : 
      fFunc(&f),
      fParams(f.GetParameters(),f.GetParameters()+f.GetNpar())
   { }

   /** 
      Destructor (no operations). Function pointer is not owned
   */ 
   virtual ~WrappedMultiTF1 () {}

   /** 
      Copy constructor
   */ 
   WrappedMultiTF1(const WrappedMultiTF1 & rhs) :
      BaseFunc(),
      BaseParamFunc(),
      fFunc(rhs.fFunc),
      fParams(rhs.fParams)
   {}

   /** 
      Assignment operator
   */ 
   WrappedMultiTF1 & operator = (const WrappedMultiTF1 & rhs) { 
      if (this == &rhs) return *this;  // time saving self-test
      fFunc = rhs.fFunc; 
      fParams = rhs.fParams;
      return *this;
   } 



   /** @name interface inherited from IFunction */

   /** 
       Clone the wrapper but not the original function
   */
   IMultiGenFunction * Clone() const { 
      return new WrappedMultiTF1(*this); 
   } 

   /// function dimension
   unsigned int NDim() const { 
      return fFunc->GetNdim(); 
   }


   /** @name interface inherited from IParamFunction */     

   /// get the parameter values (return values cachen inside, those inside TF1 might be different) 
   const double * Parameters() const {
      return &fParams.front();   
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


private: 

   /// evaluate function passing coordinates x and vector of parameters
   double DoEvalPar (const double * x, const double * p ) const { 
      if (fFunc->GetMethodCall() )  fFunc->InitArgs(x,p);  // needed for interpreted functions 
      return fFunc->EvalPar(x,p); 
   }


   TF1 * fFunc;                   // pointer to ROOT function
   std::vector<double> fParams;   // cached vector with parameter values

}; 

   } // end namespace Fit

} // end namespace ROOT


#endif /* ROOT_Fit_WrappedMultiTF1 */
