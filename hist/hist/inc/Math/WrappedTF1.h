// @(#)root/mathmore:$Id$
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

#ifndef ROOT_TF1
#include "TF1.h"
#endif
#include <cmath>

namespace ROOT { 

   namespace Math { 


/** 
   Class to Wrap a ROOT Function class (like TF1)  in a IParamFunction interface
   of one dimensions to be used in the ROOT::Math numerical algorithms
   The parameter are stored in this wrapper class, so  the TF1 parameter values are not used for evaluating the function. 
   We use TF1 only for the function evaluation. 
   This allows for the copy of the wrapper function without the need to copy the TF1. 
   The wrapper does not own the TF1 pointer, so it assumes it exists during the wrapper lifetime

   @ingroup CppFunctions
*/ 
class WrappedTF1 : public ROOT::Math::IParamGradFunction {

public: 

   typedef  ROOT::Math::IParamGradFunction BaseGradFunc; 
   typedef  ROOT::Math::IParamGradFunction::BaseFunc BaseFunc; 
 
   WrappedTF1() {}

   /** 
      constructor from a function pointer. 
   */ 
   WrappedTF1 ( TF1 & f  )  : 
      fLinear(false), 
      fPolynomial(false),
      fFunc(&f), 
      fParams(f.GetParameters(),f.GetParameters()+f.GetNpar())

   {
      // init the pointers for CINT
      if (fFunc->GetMethodCall() )  fFunc->InitArgs(fX, &fParams.front() );
      // distinguish case of polynomial functions and linear functions
      if (fFunc->GetNumber() >= 300 && fFunc->GetNumber() < 310) { 
         fLinear = true; 
         fPolynomial = true; 
      }
      // check that in case function is linear the linear terms are not zero
      if (fFunc->IsLinear() ) { 
         unsigned int ip = 0; 
         fLinear = true;
         while (fLinear && ip < fParams.size() )  { 
            fLinear &= (fFunc->GetLinearPart(ip) != 0) ; 
            ip++;
         }
      }

   }

   /** 
      Destructor (no operations). TF1 Function pointer is not owned
   */ 
   virtual ~WrappedTF1 () {}

   /** 
      Copy constructor
   */ 
   WrappedTF1(const WrappedTF1 & rhs) :
      BaseFunc(),
      BaseGradFunc(),
      fLinear(rhs.fLinear), 
      fPolynomial(rhs.fPolynomial),
      fFunc(rhs.fFunc), 
      fParams(rhs.fParams)
   {
      fFunc->InitArgs(fX,&fParams.front()  );
   }

   /** 
      Assignment operator
   */ 
   WrappedTF1 & operator = (const WrappedTF1 & rhs) { 
      if (this == &rhs) return *this;  // time saving self-test
      fLinear = rhs.fLinear;  
      fPolynomial = rhs.fPolynomial; 
      fFunc = rhs.fFunc; 
      fFunc->InitArgs(fX, &fParams.front() );
      fParams = rhs.fParams;
      return *this;
   } 


   /** @name interface inherited from IFunction */

   /** 
       Clone the wrapper but not the original function
   */
   ROOT::Math::IGenFunction * Clone() const { 
      return  new WrappedTF1(*this); 
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

   /// return parameter name (this is stored in TF1)
   std::string ParameterName(unsigned int i) const { 
      return std::string(fFunc->GetParName(i)); 
   } 


   using BaseGradFunc::operator();

   /// evaluate the derivative of the function with respect to the parameters
   void  ParameterGradient(double x, const double * par, double * grad ) const { 
      if (!fLinear) { 
         // need to set parameter values
         fFunc->SetParameters( par );
         static const double kEps = 0.001;
         fFunc->GradientPar(&x,grad,kEps);
      }
      else { 
         unsigned int np = NPar();
         for (unsigned int i = 0; i < np; ++i) 
            grad[i] = DoParameterDerivative(x, par, i);
      }
   }


private: 


   /// evaluate function passing coordinates x and vector of parameters
   double DoEvalPar (double x, const double * p ) const { 
      fX[0] = x;  
      if (fFunc->GetMethodCall() ) fFunc->InitArgs(fX,p);  // needed for interpreted functions 
      return fFunc->EvalPar(fX,p); 
   }

   /// evaluate function using the cached parameter values of this class (not of TF1)
   /// re-implement for better efficiency
   double DoEval (double x) const { 
      // no need to call InitArg for interpreted functions (done in ctor)
      // use EvalPar since it is much more efficient than Eval
      fX[0] = x;  
      return fFunc->EvalPar(fX,&fParams.front()); 
   }

   /// return the function derivatives w.r.t. x 
   double DoDerivative( double  x  ) const { 
      static const double kEps = 0.001;
      // parameter are passed as non-const in Derivative
      double * p = const_cast<double *>(&fParams.front() );
      return  fFunc->Derivative(x,p,kEps); 
   }

   /// evaluate the derivative of the function with respect to the parameters
   double  DoParameterDerivative(double x, const double * p, unsigned int ipar ) const { 
      // not very efficient - use ParameterGradient
      if (! fLinear ) {  
         std::vector<double> grad(NPar());
         ParameterGradient(x, p, &grad[0] ); 
         return grad[ipar]; 
      }
      else if (fPolynomial) { 
         // case of polynomial function (no parameter dependency)  
         return std::pow(x, static_cast<int>(ipar) );  
      }
      else { 
         // case of general linear function (bbuilt with ++ )
         const TFormula * df = dynamic_cast<const TFormula*>( fFunc->GetLinearPart(ipar) );
         assert(df != 0); 
         fX[0] = x; 
         // hack since TFormula::EvalPar is not const
         return (const_cast<TFormula*> ( df) )->EvalPar( fX ) ; // derivatives should not depend on parameters since func is linear 
      }
   }



   bool fLinear;                 // flag for linear functions 
   bool fPolynomial;             // flag for polynomial functions 
   TF1 * fFunc;                  // pointer to ROOT function
   mutable double fX[1];         //! cached vector for x value (needed for TF1::EvalPar signature) 
   std::vector<double> fParams;  //  cached vector with parameter values
}; 

   } // end namespace Fit

} // end namespace ROOT


#endif /* ROOT_Fit_WrappedTF1 */
