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
   The parameter are stored in the WrappedFunction so we don't rely on the TF1 state values. 
   We use TF1 only for the function evaluation
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
      fFunc->InitArgs(fX, &fParams.front() );
      // distinguish case of polynomial functions and linear functions
      if (fFunc->GetNumber() >= 300 && fFunc->GetNumber() < 310) { 
         fLinear = true; 
         fPolynomial = true; 
      }
      // check that in case function is linear the linear terms are not zero
      if (fFunc->IsLinear() ) { 
         unsigned int ip = 0; 
         fLinear = true;
         while (fLinear)  { 
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

   /// access the parameter values
   const double * Parameters() const {
      return &fParams.front(); 
      //return fFunc->GetParameters();   
   }

   /// set parameter values
   void SetParameters(const double * p) { 
      std::copy(p,p+fParams.size(),fParams.begin());
//       fFunc->SetParameters(p); 
//       // need to re-initialize it
//       fFunc->InitArgs(fX, p );
   } 

   /// return number of parameters 
   unsigned int NPar() const { 
      return fParams.size();
      //return static_cast<unsigned int>(fFunc->GetNpar() );
   }

   /// return parameter name (from TF1)
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

   /// evaluate the derivative of the function with respect to the parameters
   void  ParameterGradient(double x, double * grad ) const { 
      if (!fLinear) { 
         fFunc->SetParameters(&fParams.front() );
         static const double kEps = 0.001;
         fFunc->GradientPar(&x,grad,kEps);
      }
      else { 
         unsigned int np = NPar();
         for (unsigned int i = 0; i < np; ++i) 
            grad[i] = DoParameterDerivative(x, i);
      }
   }


private: 


   /// evaluate function using parameter values cached in the TF1 
   double DoEval (double x) const { 
      // no need to InitArg (done in ctor)
      fX[0] = x; 
      return fFunc->EvalPar(fX,&fParams.front()); 
      //return fFunc->EvalPar(fX,0); 
   }

   /// return the function derivatives w.r.t. x 
   double DoDerivative( double  x  ) const { 
      static const double kEps = 0.001;
      // parameter are passed as non-const in Derivative
      double * p = const_cast<double *>(&fParams.front() );
      return  fFunc->Derivative(x,p,kEps); 
   }

   /// evaluate the derivative of the function with respect to the parameters
   double  DoParameterDerivative(double x, unsigned int ipar ) const { 
      // not very efficient - use ParameterGradient
      if (! fLinear ) {  
         std::vector<double> grad(NPar());
         ParameterGradient(x, &grad[0] ); 
         return grad[ipar]; 
      }
      else if (fPolynomial) { 
         // case of polynomial function 
         return std::pow(x, static_cast<int>(ipar) );  
      }
      else { 
         // case of general linear function (bbuilt with ++ )
         const TFormula * df = dynamic_cast<const TFormula*>( fFunc->GetLinearPart(ipar) );
         assert(df != 0); 
         fX[0] = x; 
         // hack since evalpar is not const
         return (const_cast<TFormula*> ( df) )->EvalPar( fX ) ; // derivatives should not depend on parameters since func is linear 
      }
   }


   // pointer to ROOT function
   bool fLinear;      // linear function 
   bool fPolynomial;    // polynomial function
   TF1 * fFunc; 
   mutable double fX[1]; 
   std::vector<double> fParams;
}; 

   } // end namespace Fit

} // end namespace ROOT


#endif /* ROOT_Fit_WrappedTF1 */
