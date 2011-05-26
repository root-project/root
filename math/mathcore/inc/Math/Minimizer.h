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

#ifndef ROOT_Math_IFunction
#include "Math/IFunction.h"
#endif

#ifndef ROOT_Math_MinimizerOptions
#include "Math/MinimizerOptions.h"
#endif


#include <vector> 
#include <string> 

#include <limits> 
#include <cmath>


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

   /** 
      Default constructor
   */ 
   Minimizer () : 
      fValidError(false),
      fDebug(MinimizerOptions::DefaultPrintLevel()), 
      fStrategy(MinimizerOptions::DefaultStrategy()), 
      fStatus(-1),
      fMaxCalls(MinimizerOptions::DefaultMaxFunctionCalls()), 
      fMaxIter(MinimizerOptions::DefaultMaxIterations()), 
      fTol(MinimizerOptions::DefaultTolerance()), 
      fPrec(MinimizerOptions::DefaultPrecision()), 
      fUp(MinimizerOptions::DefaultErrorDef() )
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
   virtual void SetFunction(const ROOT::Math::IMultiGenFunction & func) = 0; 

   /// set a function to minimize using gradient 
   virtual void SetFunction(const ROOT::Math::IMultiGradFunction & func) 
   {
      SetFunction(static_cast<const ::ROOT::Math::IMultiGenFunction &> (func));
   }
   

   /// add variables  . Return number of variables succesfully added
   template<class VariableIterator> 
   int SetVariables(const VariableIterator & begin, const VariableIterator & end) { 
      unsigned int ivar = 0; 
      for ( VariableIterator vitr = begin; vitr != end; ++vitr) { 
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

         // an error message should be eventually be reported in the virtual single SetVariable methods
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
   virtual bool SetLimitedVariable(unsigned int /* ivar */ , const std::string & /* name */ , double /*val */ , double /* step  */, double /* lower */, double /* upper */) { 
      return false;  
   }
   /// set fixed variable (override if minimizer supports them )
   virtual bool SetFixedVariable(unsigned int /* ivar */ , const std::string & /* name */ , double /* val */ ) { 
      return false; 
   }
   /// set the value of an existing variable 
   virtual bool SetVariableValue(unsigned int , double ) { return false;  }
   /// set the values of all existing variables (array must be dimensioned to the size of the existing parameters)
   virtual bool SetVariableValues(const double * x) { 
      bool ret = true; 
      unsigned int i = 0;
      while ( i <= NDim() && ret) { 
         SetVariableValue(i,x[i] ); i++; 
      }
      return ret; 
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

   /** 
       Fill the passed array with the  covariance matrix elements 
       if the variable is fixed or const the value is zero. 
       The array will be filled as cov[i *ndim + j]
       The ordering of the variables is the same as in errors and parameter value. 
       This is different from the direct interface of Minuit2 or TMinuit where the 
       values were obtained only to variable parameters
   */ 
   virtual bool GetCovMatrix(double * /*cov*/) const { return false; } 

   /** 
       Fill the passed array with the Hessian matrix elements 
       The Hessian matrix is the matrix of the second derivatives 
       and is the inverse of the covariance matrix
       If the variable is fixed or const the values for that variables are zero. 
       The array will be filled as h[i *ndim + j]
   */ 
   virtual bool GetHessianMatrix(double * /* h */) const { return false; }


   ///return status of covariance matrix 
   /// using Minuit convention {0 not calculated 1 approximated 2 made pos def , 3 accurate}
   /// Minimizer who implements covariance matrix calculation will re-implement the method
   virtual int CovMatrixStatus() const {  return 0; }

   /**
      return correlation coefficient between variable i and j.
      If the variable is fixed or const the return value is zero
    */
   virtual double Correlation(unsigned int i, unsigned int j ) const { 
      double tmp = CovMatrix(i,i) * CovMatrix(j,j);
      return ( tmp < 0) ? 0 : CovMatrix(i,j) / std::sqrt( tmp );  
   }

   /**
      return global correlation coefficient for variable i
      This is a number between zero and one which gives 
      the correlation between the i-th parameter  and that linear combination of all 
      other parameters which is most strongly correlated with i.
      Minimizer must overload method if implemented 
    */
   virtual double GlobalCC(unsigned int ) const { return -1; }

   /**
      minos error for variable i, return false if Minos failed or not supported 
      and the lower and upper errors are returned in errLow and errUp
      An extra flag  specifies if only the lower (runopt=-1) or the upper (runopt=+1) error calculation is run
      (This feature isnot yet implemented)
   */
   virtual bool GetMinosError(unsigned int /* i */, double & errLow, double & errUp, int  = 0) { 
      errLow = 0; errUp = 0; 
      return false; 
   }  

   /**
      perform a full calculation of the Hessian matrix for error calculation
    */
   virtual bool Hesse() { return false; }

   /**
      scan function minimum for variable i. Variable and function must be set before using Scan 
      Return false if an error or if minimizer does not support this functionality
    */
   virtual bool Scan(unsigned int /* i */, unsigned int & /* nstep */, double * /* x */, double * /* y */, 
                     double /*xmin */ = 0, double /*xmax*/ = 0) {
      return false; 
   }

   /**
      find the contour points (xi,xj) of the function for parameter i and j around the minimum
      The contour will be find for value of the function = Min + ErrorUp();
    */
   virtual bool Contour(unsigned int /* i */, unsigned int /* j */, unsigned int &/* np */, 
                        double * /* xi */, double * /* xj */) { 
      return false; 
   }

   /// return reference to the objective function
   ///virtual const ROOT::Math::IGenFunction & Function() const = 0; 

   /// print the result according to set level (implemented for TMinuit for mantaining Minuit-style printing)
   virtual void PrintResults() {}

   /// get name of variables (override if minimizer support storing of variable names)
   /// return an empty string if variable is not found
   virtual std::string VariableName(unsigned int ) const { return std::string();}  // return empty string 

   /// get index of variable given a variable given a name
   /// return -1 if variable is not found
   virtual int VariableIndex(const std::string &) const { return -1; }
      
   /** minimizer configuration parameters **/

   /// set print level
   int PrintLevel() const { return fDebug; }

   ///  max number of function calls
   unsigned int MaxFunctionCalls() const { return fMaxCalls; } 

   /// max iterations
   unsigned int MaxIterations() const { return fMaxIter; } 

   /// absolute tolerance 
   double Tolerance() const { return  fTol; }

   /// precision of minimizer in the evaluation of the objective function
   /// ( a value <=0 corresponds to the let the minimizer choose its default one)
   double Precision() const { return fPrec; }
   
   /// strategy 
   int Strategy() const { return fStrategy; }

   /// status code of minimizer 
   int Status() const { return fStatus; } 

   /// return the statistical scale used for calculate the error
   /// is typically 1 for Chi2 and 0.5 for likelihood minimization
   double ErrorDef() const { return fUp; } 

   ///return true if Minimizer has performed a detailed error validation (e.g. run Hesse for Minuit)
   bool IsValidError() const { return fValidError; }

   /// retrieve the minimizer options (implement derived class if needed)
   virtual MinimizerOptions  Options() const { 
      MinimizerOptions opt; 
      opt.SetPrintLevel(fDebug);
      opt.SetStrategy(fStrategy);
      opt.SetMaxFunctionCalls(fMaxCalls);
      opt.SetMaxIterations(fMaxIter);
      opt.SetTolerance(fTol);
      opt.SetPrecision(fPrec);
      opt.SetErrorDef(fUp);
      return opt;
   }

   /// set print level
   void SetPrintLevel(int level) { fDebug = level; }

   ///set maximum of function calls 
   void SetMaxFunctionCalls(unsigned int maxfcn) { if (maxfcn > 0) fMaxCalls = maxfcn; }

   /// set maximum iterations (one iteration can have many function calls) 
   void SetMaxIterations(unsigned int maxiter) { if (maxiter > 0) fMaxIter = maxiter; } 

   /// set the tolerance
   void SetTolerance(double tol) { fTol = tol; }

   /// set in the minimizer the objective function evaluation precision 
   /// ( a value <=0 means the minimizer will choose its optimal value automatically, i.e. default case)
   void SetPrecision(double prec) { fPrec = prec; }

   ///set the strategy 
   void SetStrategy(int strategyLevel) { fStrategy = strategyLevel; }  

   /// set scale for calculating the errors
   void SetErrorDef(double up) { fUp = up; }

   /// flag to check if minimizer needs to perform accurate error analysis (e.g. run Hesse for Minuit)
   void SetValidError(bool on) { fValidError = on; } 

   /// set all options in one go
   void SetOptions(const MinimizerOptions & opt) { 
      fDebug = opt.PrintLevel();
      fStrategy = opt.Strategy();
      fMaxCalls = opt.MaxFunctionCalls();
      fMaxIter = opt.MaxIterations();
      fTol = opt.Tolerance();
      fPrec = opt.Precision();
      fUp = opt.ErrorDef();
   }

   /// reset the defaut options (defined in MinimizerOptions)
   void SetDefaultOptions() { 
      fDebug = MinimizerOptions::DefaultPrintLevel();
      fStrategy = MinimizerOptions::DefaultStrategy();
      fMaxCalls = MinimizerOptions::DefaultMaxFunctionCalls();
      fMaxIter = MinimizerOptions::DefaultMaxIterations();
      fTol = MinimizerOptions::DefaultTolerance();
      fPrec = MinimizerOptions::DefaultPrecision();
      fUp = MinimizerOptions::DefaultErrorDef();
   }

protected: 


//private: 


   // keep protected to be accessible by the derived classes 
 

   bool fValidError;            // flag to control if errors have been validated (Hesse has been run in case of Minuit)
   int fDebug;                  // print level
   int fStrategy;               // minimizer strategy
   int fStatus;                 // status of minimizer    
   unsigned int fMaxCalls;      // max number of function calls 
   unsigned int fMaxIter;       // max number or iterations used to find the minimum
   double fTol;                 // tolerance (absolute)
   double fPrec;                // precision
   double fUp;                  // error scale 

}; 

   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_Minimizer */
