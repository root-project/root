// @(#)root/fit:$Name:  $:$Id: WrappedMultiTF1.h,v 1.1 2006/11/17 18:26:50 moneta Exp $
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


#include "TF1.h"

namespace ROOT { 

   namespace Math { 


/** 
   Class to Wrap a ROOT Function class (like TF1)  in a IParamFunction interface
   of multi-dimensions to be used in the ROOT::Math numerical algorithm

   @ingroup CppFunctions
*/ 
class WrappedMultiTF1 : public ROOT::Math::IParamFunction<ROOT::Math::MultiDim> {

public: 

   typedef  ROOT::Math::IParamFunction<ROOT::Math::MultiDim>            BaseParamFunc; 
   typedef  ROOT::Math::IParamFunction<ROOT::Math::MultiDim>::BaseFunc  BaseFunc; 
 

   /** 
      constructor from a function pointer. A flag (default false) is specified if 
      class owns the pointer
   */ 
   WrappedMultiTF1 (TF1 & f )  : 
      fFunc(&f) 
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
      fFunc(rhs.fFunc) 
   {}

   /** 
      Assignment operator
   */ 
   WrappedMultiTF1 & operator = (const WrappedMultiTF1 & rhs) { 
      if (this == &rhs) return *this;  // time saving self-test
      fFunc = rhs.fFunc; 
   } 



   /** @name interface inherited from IFunction */

   /** 
       Clone the wrapper but not the original function
   */
   IMultiGenFunction * Clone() const { 
      return new WrappedMultiTF1(*fFunc); 
   } 

   /// function dimension
   unsigned int NDim() const { 
      return fFunc->GetNdim(); 
   }


   /** @name interface inherited from IParamFunction */     

   /// access the parameter values
   const double * Parameters() const {
      return fFunc->GetParameters();   
   }

   /// set parameter values
   void SetParameters(const double * p) { 
      fFunc->SetParameters(p); 
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

   using BaseFunc::operator();



private: 


   /// evaluate function using parameter values cached in the TF1 
   double DoEval (const double * x) const { 
      fFunc->InitArgs(x, 0 );
      return fFunc->EvalPar(x,0); 
   }

//    /// return the function derivatives w.r.t. x 
//    double DoDerivative(const double * x, unsigned int icoord   ) const { 
//       std::cerr << "WrappedMultiTF1:: gradient for multidim functions not implemented" << std::endl;
//    }

//    /// evaluate the derivative of the function with respect to the parameters
//    void  DoParameterGradient(const double * x, double * grad ) const { 
//       static const double kEps = 0.001;
//       fFunc->GradientPar(x,grad,kEps); 
//    }


   // pointer to ROOT function
   TF1 * fFunc; 

}; 

   } // end namespace Fit

} // end namespace ROOT


#endif /* ROOT_Fit_WrappedMultiTF1 */
