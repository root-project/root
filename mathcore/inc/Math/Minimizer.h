// @(#)root/mathcore:$Id$
// Author: L. Moneta Fri Sep 22 15:06:47 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class Minimizer

#ifndef ROOT_Math_Minimizer
#define ROOT_Math_Minimizer

#ifndef ROOT_Math_IFunctionfwd
#include "Math/IFunctionfwd.h"
#endif


#include <vector> 

#include <limits> 

//#define DEBUG
#ifdef DEBUG
#include <iostream> 
#endif

namespace ROOT { 
   

   namespace Math { 

/**
   @defgroup MultiMin Multi-dimensional Minimization
   @ingroup NumAlgo

   Classes implementing algorithms for multi-dimensional minimization 
 */



//_______________________________________________________________________________
/** 
   Abstract Minimizer class, defining  the interface for the various minimizer
   (like Minuit2, Minuit, GSL, etc..) 
   Plug-in's exist in ROOT to be able to instantiate the derived classes like 
   ROOT::Math::GSLMinimizer or ROOT::Math::Minuit2Minimizer via the 
   plug-in manager.

   Provides interface for setting the function to be minimized. 
   The function must  implemente the multi-dimensional generic interface
   ROOT::Math::IBaseFunctionMultiDim. 
   If the function provides gradient calculation 
   (implements the ROOT::Math::IGradientFunctionMultiDim interface) this will be 
   used by the Minimizer. 

   It Defines also interface for setting the initial values for the function variables (which are the parameters in 
   of the model function in case of solving for fitting) and especifying their limits. 

   It defines the interface to set and retrieve basic minimization parameters 
   (for specific Minimizer parameters one must use the derived classes). 

   Then it defines the interface to retrieve the result of minimization ( minimum X values, function value, 
   gradient, error on the mimnimum, etc...)

   @ingroup MultiMin
*/
 
class Minimizer {

public: 

   typedef ROOT::Math::IMultiGenFunction IObjFunction; 
   typedef ROOT::Math::IMultiGradFunction IGradObjFunction; 

   /** 
      Default constructor
   */ 
   Minimizer () : 
#ifndef DEBUG
      fDebug(0), 
#else
      fDebug(3),
#endif 
      fStrategy(1),
      fMaxCalls(0), 
      fMaxIter(0),
      fTol(1.E-6), 
      fUp(1.)
   {} 

   /** 
      Destructor (no operations)
   */ 
   virtual ~Minimizer ()  {}  




private:
   // usually copying is non trivial, so we make this unaccessible

   /** 
      Copy constructor
   */ 
   Minimizer(const Minimizer &) {} 

   /** 
      Assignment operator
   */ 
   Minimizer & operator = (const Minimizer & rhs)  {
      if (this == &rhs) return *this;  // time saving self-test
      return *this;
   }

public: 
   
   /// reset for consecutive minimizations - implement if needed 
   virtual void Clear() {}

   /// set the function to minimize
   virtual void SetFunction(const IObjFunction & func) = 0; 

   /// set a function to minimize using gradient 
   virtual void SetFunction(const IGradObjFunction & func) = 0;
   

   /// add variables  . Return number of variables succesfully added
   template<class VariableIterator> 
   int SetVariables(const VariableIterator & begin, const VariableIterator & end) { 
      unsigned int ivar = 0; 
      for ( VariableIterator vitr = begin; vitr != end; ++vitr) { 
#ifdef DEBUG
         std::cout << "adding variable " << ivar << "  " << vitr->Name(); 
         if (vitr->IsDoubleBound() ) std::cout << " bounded to [ " <<  vitr->LowerLimit() << " , " <<  vitr->UpperLimit() << " ] ";
         std::cout << std::endl; 
#endif
         bool iret = false; 
         if (vitr->IsFixed() )
            iret = SetFixedVariable(ivar,  vitr->Name(), vitr->Value() ); 
         else if (vitr->IsDoubleBound() )
            iret = SetLimitedVariable(ivar,  vitr->Name(), vitr->Value(), vitr->StepSize(), vitr->LowerLimit(), vitr->UpperLimit() );
         else if (vitr->HasLowerLimit() )
            iret = SetLowerLimitedVariable(ivar,  vitr->Name(), vitr->Value(), vitr->StepSize(), vitr->LowerLimit() );
         else if (vitr->HasUpperLimit() )
            iret = SetUpperLimitedVariable(ivar,  vitr->Name(), vitr->Value(), vitr->StepSize(), vitr->UpperLimit() );
         else 
            iret = SetVariable( ivar, vitr->Name(), vitr->Value(), vitr->StepSize() ); 

         if (iret) ivar++; 
#ifdef DEBUG
         if (iret) 
            std::cout << "Added variable " << vitr->Name() << " val = " << vitr->Value() << " step " << vitr->StepSize() 
                      << std::endl; 
         else 
            std::cout << "Failed to Add variable " << vitr->Name() << std::endl; 
#endif

      }
      return ivar; 
   }
   /// set free variable 
   virtual bool SetVariable(unsigned int ivar, const std::string & name, double val, double step) = 0; 
   /// set lower limit variable  (override if minimizer supports them )
   virtual bool SetLowerLimitedVariable(unsigned int  ivar , const std::string & name , double val , double step , double lower ) { 
      return SetLimitedVariable(ivar, name, val, step, lower, std::numeric_limits<double>::infinity() );  
   } 
   /// set upper limit variable (override if minimizer supports them )
   virtual bool SetUpperLimitedVariable(unsigned int ivar , const std::string & name , double val , double step , double upper ) { 
      return SetLimitedVariable(ivar, name, val, step, - std::numeric_limits<double>::infinity(), upper );  
   } 
   /// set upper/lower limited variable (override if minimizer supports them )
   virtual bool SetLimitedVariable(unsigned int ivar , const std::string & name , double val , double step , double /* lower */, double /* upper */) { 
      return SetVariable(ivar, name, val, step );  
   }
   /// set fixed variable (override if minimizer supports them )
   virtual bool SetFixedVariable(unsigned int ivar , const std::string & name , double val ) { 
      return SetLimitedVariable(ivar, name, val, 0., val, val); 
   }

   /// method to perform the minimization
   virtual  bool Minimize() = 0; 

   /// return minimum function value
   virtual double MinValue() const = 0; 

   /// return expected distance reached from the minimum
   virtual double Edm() const = 0; 

   /// return  pointer to X values at the minimum 
   virtual const double *  X() const = 0; 

   /// return pointer to gradient values at the minimum 
   virtual const double *  MinGradient() const = 0;  

   /// number of function calls to reach the minimum 
   virtual unsigned int NCalls() const = 0;    

   /// this is <= Function().NDim() which is the total 
   /// number of variables (free+ constrained ones) 
   virtual unsigned int NDim() const = 0;  

   /// number of free variables (real dimension of the problem) 
   /// this is <= Function().NDim() which is the total 
   virtual unsigned int NFree() const = 0;  

   /// minimizer provides error and error matrix
   virtual bool ProvidesError() const = 0; 

   /// return errors at the minimum 
   virtual const double * Errors() const = 0;

   /** return covariance matrices elements 
       if the variable is fixed the matrix is zero
       The ordering of the variables is the same as in errors
   */ 
   virtual double CovMatrix(unsigned int i, unsigned int j) const = 0;  

   /// minos error for variable i, return false if Minos failed or not supported 
   virtual bool GetMinosError(unsigned int /* i */, double & errLow, double & errUp) { 
      errLow = 0; errUp = 0; 
      return false; 
   }  

   /// return reference to the objective function
   ///virtual const ROOT::Math::IGenFunction & Function() const = 0; 


   /** minimizer configuration parameters **/

   /// set print level
   int PrintLevel() const { return fDebug; }

   ///  max number of function calls
   unsigned int MaxFunctionCalls() { return fMaxCalls; } 

   /// max iterations
   unsigned int MaxIterations() { return fMaxIter; } 

   /// absolute tolerance 
   double Tolerance() const { return  fTol; }

   /// strategy 
   int Strategy() const { return fStrategy; }

   /// return the statistical scale used for calculate the error
   /// is typically 1 for Chi2 minimizetion and 0.5 for likelihood's
   double ErrorUp() const { return fUp; } 

   /// set print level
   void SetPrintLevel(int level) { fDebug = level; }

   ///set maximum of function calls 
   void SetMaxFunctionCalls(unsigned int maxfcn) { if (maxfcn > 0) fMaxCalls = maxfcn; }

   /// set maximum iterations (one iteration can have many function calls) 
   void SetMaxIterations(unsigned int maxiter) { if (maxiter > 0) fMaxIter = maxiter; } 

   /// set the tolerance
   void SetTolerance(double tol) { fTol = tol; }

   ///set the strategy 
   void SetStrategy(int strategyLevel) { fStrategy = strategyLevel; }  

   /// set scale for calculating the errors
   void SetErrorUp(double up) { fUp = up; }



protected: 


//private: 

   // keep protected to be accessible by the derived classes 
 
   int fDebug;                  // print level
   int fStrategy;               // minimizer strategy
   unsigned int fMaxCalls;      // max number of funciton calls 
   unsigned int fMaxIter;       // max number or iterations used to find the minimum
   double fTol;                 // tolerance (absolute)
   double fUp;                  // error scale 

}; 

   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_Minimizer */
