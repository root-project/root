// @(#)root/mathcore:$Id$
// Authors: M. Slawinska   08/2007 

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2007 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header source file for class IntegratorMultiDim


#ifndef ROOT_Math_IntegratorMultiDim
#define ROOT_Math_IntegratorMultiDim


#ifndef ROOT_Math_IFunctionfwd
#include "Math/IFunctionfwd.h"
#endif

#ifndef ROOT_Math_IntegrationTypes
#include "Math/AllIntegrationTypes.h"
#endif


#ifndef ROOT_Math_VirtualIntegrator
#include "Math/VirtualIntegrator.h"
#endif

#ifndef __CINT__

#ifndef ROOT_Math_WrappedFunction
#include "Math/WrappedFunction.h"
#endif

#endif

namespace ROOT {
namespace Math {

//___________________________________________________________________________________________
/**
   User class for performing multidimensional integration 

   By default uses adaptive multi-dimensional integration using the algorithm from Genz Mallik
   implemented in the class ROOT::Math::AdaptiveIntegratorMultiDim otherwise it can uses via the 
   plug-in manager the MC integration class (ROOT::Math::GSLMCIntegration) from MathMore. 

   @ingroup Integration

  
 */

class IntegratorMultiDim {

public:


  
    /** Generic constructor of multi dimensional Integrator. By default uses the Adaptive integration method 

       @param type   integration type (adaptive, MC methods, etc..)
       @param absTol desired absolute Error
       @param relTol desired relative Error
       @param size maximum number of sub-intervals
    */
   explicit 
   IntegratorMultiDim(IntegrationMultiDim::Type type = IntegrationMultiDim::kADAPTIVE, double absTol = 1.E-9, double relTol = 1E-6, unsigned int ncall = 100000) { 
       fIntegrator = CreateIntegrator(type, absTol, relTol, ncall); 
   }
   
    /** Generic Constructor of multi dimensional Integrator passing a function. By default uses the adaptive integration method 

       @param f      integration function (multi-dim interface) 
       @param type   integration type (adaptive, MC methods, etc..)
       @param absTol desired absolute Error
       @param relTol desired relative Error
       @param ncall  number of function calls (apply only to MC integratioon methods)
    */
   explicit
   IntegratorMultiDim(const IMultiGenFunction &f, IntegrationMultiDim::Type type = IntegrationMultiDim::kADAPTIVE, double absTol = 1.E-9, double relTol = 1E-6, unsigned int ncall = 100000) { 
      fIntegrator = CreateIntegrator(type, absTol, relTol, ncall); 
      SetFunction(f);            
   }
   
    /** Template Constructor of multi dimensional Integrator passing a generic function. By default uses the adaptive integration method 

       @param f      integration function (generic function implementin operator()(const double *) 
       @param dim    function dimension 
       @param type   integration type (adaptive, MC methods, etc..)
       @param absTol desired absolute Error
       @param relTol desired relative Error
       @param ncall  number of function calls (apply only to MC integratioon methods)
    */
#ifdef LATER
   template<class Function>
   IntegratorMultiDim(const Function & f, unsigned int dim, IntegrationMultiDim::Type type = IntegrationMultiDim::kADAPTIVE, double absTol = 1.E-9, double relTol = 1E-6, unsigned int ncall = 100000) { 
      fIntegrator = CreateIntegrator(type, absTol, relTol, ncall); 
      SetFunction(f, dim); 
   }
#endif

   /// destructor
   virtual ~IntegratorMultiDim() { 
      if (fIntegrator) delete fIntegrator;
   }

   // disable copy constructur and assignment operator 

private:
   IntegratorMultiDim(const IntegratorMultiDim &) {}
   IntegratorMultiDim & operator=(const IntegratorMultiDim &) { return *this; }

public:


   /**
      evaluate the integral with the previously given function between xmin[] and xmax[]  
   */
   double Integral(const double* xmin, const double * xmax) {
      return fIntegrator == 0 ? 0 : fIntegrator->Integral(xmin,xmax);
   }

   /// evaluate the integral passing a new function
   double Integral(const IMultiGenFunction &f, const double* xmin, const double * xmax) {
      SetFunction(f);
      return Integral(xmin,xmax);
   }

   /// evaluate the integral passing a new generic function
   template<class Function>
   double Integral(Function f, unsigned int dim, const double* xmin, const double * xmax) {
      SetFunction(f,dim);
      return Integral(xmin, xmax);
   }


   /** 
       set integration function using a generic function implementing the operator()(double *x)
       The dimension of the function is in this case required 
   */
   template <class Function> 
   void SetFunction(const Function & f, unsigned int dim) { 
      ROOT::Math::WrappedMultiFunction<Function> wf(f, dim); 
      SetFunction(wf);
   }
   void SetFunction(const IMultiGenFunction &f) { 
      if (fIntegrator) fIntegrator->SetFunction(f);
   }

   /// return result of last integration 
   double Result() const { return fIntegrator == 0 ? 0 : fIntegrator->Result(); }

   /// return integration error 
   double Error() const { return fIntegrator == 0 ? 0 : fIntegrator->Error(); } 

   ///  return the Error Status of the last Integral calculation
   int Status() const { return fIntegrator == 0 ? -1 : fIntegrator->Status(); }

   // return number of function evaluations in calculating the integral 
   //unsigned int NEval() const { return fNEval; }
 
   /// set the relative tolerance
   void SetRelTolerance(double relTol) { if (fIntegrator) fIntegrator->SetRelTolerance(relTol); }

   /// set absolute tolerance
   void SetAbsTolerance(double absTol)  { if (fIntegrator) fIntegrator->SetAbsTolerance(absTol); }

   /// return a pointer to integrator object 
   VirtualIntegratorMultiDim * GetIntegrator() { return fIntegrator; }  

protected:

   VirtualIntegratorMultiDim * CreateIntegrator(IntegrationMultiDim::Type type , double absTol, double relTol, unsigned int ncall);

 private:

   VirtualIntegratorMultiDim * fIntegrator;     // pointer to multi-dimensional integrator base class


};

}//namespace Math
}//namespace ROOT

#endif /* ROOT_Math_IntegratorMultiDim */
