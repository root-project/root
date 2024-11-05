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

#include "Math/IFunction.h"
#include "Math/MinimizerOptions.h"

#include <ROOT/RSpan.hxx>

#include <string>
#include <limits>
#include <cmath>
#include <vector>
#include <functional>



namespace ROOT {

   namespace Fit {
      class ParameterSettings;
   }


   namespace Math {

/**
   @defgroup MultiMin Multi-dimensional Minimization
   @ingroup NumAlgo

   Classes implementing algorithms for multi-dimensional minimization
 */



//_______________________________________________________________________________
/**
   Abstract Minimizer class, defining  the interface for the various minimizer
   (like Minuit2, Minuit, GSL, etc..) in ROOT.
   Plug-in's exist in ROOT to be able to instantiate the derived classes without linking the library
   using the static function ROOT::Math::Factory::CreateMinimizer.

   Here is the list of all possible minimizers and their respective methods (algorithms) that can be instantiated:
   The name shown below can be used to create them. More documentation can be found in the respective class

   - Minuit   (class TMinuitMinimizer)
      - Migrad (default)
      - MigradImproved  (Migrad with adding a method to improve minimization when ends-up in a local minimum, see par. 6.3 of [Minuit tutorial on Function Minimization](https://seal.web.cern.ch/documents/minuit/mntutorial.pdf))
      - Simplex
      - Minimize (a combination of Simplex + Migrad)
      - Minimize
      - Scan
      - Seek

   - Minuit2 (class ROOT::Minuit2::Minuit2Minimizer)
     - Migrad (default)
     - Simplex
     - Minimize
     - Fumili (Fumili2)
     - Scan

   - Fumili (class TFumiliMinimizer)

   - GSLMultiMin (class ROOT::Math::GSLMinimizer) available when ROOT is built with `mathmore` support
     - BFGS2 (Default)
     - BFGS
     - ConjugateFR
     - ConjugatePR
     - SteepestDescent

   - GSLMultiFit (class ROOT::Math::GSLNLMinimizer) available when ROOT is built `mathmore` support

   - GSLSimAn  (class ROOT::Math::GSLSimAnMinimizer) available when ROOT is built with `mathmore` support

   - Genetic  (class ROOT::Math::GeneticMinimizer)

   - RMinimizer (class ROOT::Math::RMinimizer)  available when ROOT is built with `r` support
     - BFGS (default)
     - L-BFGS-S
     - Nelder-Mead
     - CG
     - and more methods, see the Details in the documentation of the function `optimix` of the [optmix R package](https://cran.r-project.org/web/packages/optimx/optimx.pdf)


   The Minimizer class provides the interface to perform the minimization including


   In addition to provide the API for function minimization (via ROOT::Math::Minimizer::Minimize) the Minimizer class  provides:
   - the interface for setting the function to be minimized. The objective function passed to the Minimizer must  implement the multi-dimensional generic interface
   ROOT::Math::IBaseFunctionMultiDim. If the function provides gradient calculation (e.g. implementing the ROOT::Math::IGradientFunctionMultiDim interface)
   the gradient will be used by the Minimizer class, when needed. There are convenient classes for the users to wrap their own functions in this required interface for minimization.
   These are the `ROOT::Math::Functor` class and the `ROOT::Math::GradFunctor` class for wrapping functions providing both evaluation and gradient. Some methods, like Fumili, Fumili2 and GSLMultiFit are
   specialized method for least-square and also likelihood minimizations. They require then that the given function implements in addition
   the `ROOT::Math::FitMethodFunction` interface.
   - The interface for setting the initial values for the function variables (which are the parameters in
   of the model function in case of solving for fitting) and specifying their limits.
   - The interface to set and retrieve basic minimization parameters. These parameter are controlled by the class `ROOT::Math::MinimizerOptions`.
   When no parameters are specified the default ones are used. Specific Minimizer options can also be passed via the `MinimizerOptions` class.
   For the list of the available option parameter one must look at the documentation of the corresponding derived class.
   - The interface to retrieve the result of minimization ( minimum X values, function value, gradient, error on the minimum, etc...)
   - The interface to perform a Scan, Hesse or a Contour plot (for the minimizers that support this, i.e. Minuit and Minuit2)

   An example on how to use this interface is the tutorial NumericalMinimization.C in the tutorials/fit directory.

   @ingroup MultiMin
*/

class Minimizer {

public:

   /// Default constructor.
   Minimizer () {}

   /// Destructor (no operations).
   virtual ~Minimizer ()  {}

   // usually copying is non trivial, so we delete this
   Minimizer(Minimizer const&) = delete;
   Minimizer &operator=(Minimizer const&) = delete;
   Minimizer(Minimizer &&) = delete;
   Minimizer &operator=(Minimizer &&) = delete;

   /// reset for consecutive minimization - implement if needed
   virtual void Clear() {}

   /// set the function to minimize
   virtual void SetFunction(const ROOT::Math::IMultiGenFunction & func) = 0;

   /// set the function implementing Hessian computation (re-implemented by Minimizer using it)
   virtual void SetHessianFunction(std::function<bool(std::span<const double>, double *)> ) {}

   /// add variables  . Return number of variables successfully added
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
   /// set a new free variable
   virtual bool SetVariable(unsigned int ivar, const std::string & name, double val, double step) = 0;
   /// set initial second derivatives
   virtual bool SetCovarianceDiag(std::span<const double> d2, unsigned int n);
   /// set initial covariance matrix
   virtual bool SetCovariance(std::span<const double> cov, unsigned int nrow);

   /// set a new lower limit variable  (override if minimizer supports them )
   virtual bool SetLowerLimitedVariable(unsigned int  ivar , const std::string & name , double val , double step , double lower ) {
      return SetLimitedVariable(ivar, name, val, step, lower, std::numeric_limits<double>::infinity() );
   }
   /// set a new upper limit variable (override if minimizer supports them )
   virtual bool SetUpperLimitedVariable(unsigned int ivar , const std::string & name , double val , double step , double upper ) {
      return SetLimitedVariable(ivar, name, val, step, - std::numeric_limits<double>::infinity(), upper );
   }
   virtual bool SetLimitedVariable(unsigned int ivar  , const std::string & name  , double val  , double  step ,
                                   double lower , double  upper );
   virtual bool SetFixedVariable(unsigned int  ivar  , const std::string &  name , double val  );
   virtual bool SetVariableValue(unsigned int ivar , double value);
   /// set the values of all existing variables (array must be dimensioned to the size of the existing parameters)
   virtual bool SetVariableValues(const double * x) {
      bool ret = true;
      unsigned int i = 0;
      while ( i <= NDim() && ret) {
         ret &= SetVariableValue(i,x[i] ); i++;
      }
      return ret;
   }
   virtual bool SetVariableStepSize(unsigned int ivar, double value );
   virtual bool SetVariableLowerLimit(unsigned int ivar, double lower);
   virtual bool SetVariableUpperLimit(unsigned int ivar, double upper);
   /// set the limits of an already existing variable
   virtual bool SetVariableLimits(unsigned int ivar, double lower, double upper) {
      return SetVariableLowerLimit(ivar,lower) && SetVariableUpperLimit(ivar,upper);
   }
   virtual bool FixVariable(unsigned int ivar);
   virtual bool ReleaseVariable(unsigned int ivar);
   virtual bool IsFixedVariable(unsigned int ivar) const;
   virtual bool GetVariableSettings(unsigned int ivar, ROOT::Fit::ParameterSettings & pars) const;

   /// set the initial range of an existing variable
   virtual bool SetVariableInitialRange(unsigned int /* ivar */, double /* mininitial */, double /* maxinitial */) {
     return false;
   }

   /// method to perform the minimization
   virtual  bool Minimize() = 0;

   /// return minimum function value
   virtual double MinValue() const = 0;

   /// return  pointer to X values at the minimum
   virtual const double *  X() const = 0;

   /// return expected distance reached from the minimum (re-implement if minimizer provides it
   virtual double Edm() const { return -1; }

   /// return pointer to gradient values at the minimum
   virtual const double *  MinGradient() const { return nullptr; }

   /// number of function calls to reach the minimum
   virtual unsigned int NCalls() const { return 0; }

   /// number of iterations to reach the minimum
   virtual unsigned int NIterations() const { return NCalls(); }

   /// this is <= Function().NDim() which is the total
   /// number of variables (free+ constrained ones)
   virtual unsigned int NDim() const = 0;

   /// number of free variables (real dimension of the problem)
   /// this is <= Function().NDim() which is the total
   /// (re-implement if minimizer supports bounded parameters)
   virtual unsigned int NFree() const { return NDim(); }

   /// minimizer provides error and error matrix
   virtual bool ProvidesError() const { return false; }

   /// return errors at the minimum
   virtual const double * Errors() const { return nullptr; }

   virtual double CovMatrix(unsigned int  ivar , unsigned int jvar ) const;
   virtual bool GetCovMatrix(double * covMat) const;
   virtual bool GetHessianMatrix(double * hMat) const;


   ///return status of covariance matrix
   /// using Minuit convention {0 not calculated 1 approximated 2 made pos def , 3 accurate}
   /// Minimizer who implements covariance matrix calculation will re-implement the method
   virtual int CovMatrixStatus() const {
      return 0;
   }

   /**
      return correlation coefficient between variable i and j.
      If the variable is fixed or const the return value is zero
    */
   virtual double Correlation(unsigned int i, unsigned int j ) const {
      double tmp = CovMatrix(i,i) * CovMatrix(j,j);
      return ( tmp < 0) ? 0 : CovMatrix(i,j) / std::sqrt( tmp );
   }

   virtual double GlobalCC(unsigned int ivar) const;

   virtual bool GetMinosError(unsigned int ivar , double & errLow, double & errUp, int option = 0);
   virtual bool Hesse();
   virtual bool Scan(unsigned int ivar , unsigned int & nstep , double * x , double * y ,
                     double xmin = 0, double xmax = 0);
   virtual bool Contour(unsigned int ivar , unsigned int jvar, unsigned int & npoints,
                        double *  xi , double * xj );

   /// return reference to the objective function
   ///virtual const ROOT::Math::IGenFunction & Function() const = 0;

   /// print the result according to set level (implemented for TMinuit for maintaining Minuit-style printing)
   virtual void PrintResults() {}

   virtual std::string VariableName(unsigned int ivar) const;

   virtual int VariableIndex(const std::string & name) const;

   /** minimizer configuration parameters **/

   /// set print level
   int PrintLevel() const { return fOptions.PrintLevel(); }

   ///  max number of function calls
   unsigned int MaxFunctionCalls() const { return fOptions.MaxFunctionCalls(); }

   /// max iterations
   unsigned int MaxIterations() const { return fOptions.MaxIterations(); }

   /// absolute tolerance
   double Tolerance() const { return  fOptions.Tolerance(); }

   /// precision of minimizer in the evaluation of the objective function
   /// ( a value <=0 corresponds to the let the minimizer choose its default one)
   double Precision() const { return fOptions.Precision(); }

   /// strategy
   int Strategy() const { return fOptions.Strategy(); }

   /// status code of minimizer
   int Status() const { return fStatus; }

   /// status code of Minos (to be re-implemented by the minimizers supporting Minos)
   virtual int MinosStatus() const { return -1; }

   /// return the statistical scale used for calculate the error
   /// is typically 1 for Chi2 and 0.5 for likelihood minimization
   double ErrorDef() const { return fOptions.ErrorDef(); }

   ///return true if Minimizer has performed a detailed error validation (e.g. run Hesse for Minuit)
   bool IsValidError() const { return fValidError; }

   /// retrieve the minimizer options (implement derived class if needed)
   virtual MinimizerOptions  Options() const {
      return fOptions;
   }

   /// set print level
   void SetPrintLevel(int level) { fOptions.SetPrintLevel(level); }

   ///set maximum of function calls
   void SetMaxFunctionCalls(unsigned int maxfcn) { if (maxfcn > 0) fOptions.SetMaxFunctionCalls(maxfcn); }

   /// set maximum iterations (one iteration can have many function calls)
   void SetMaxIterations(unsigned int maxiter) { if (maxiter > 0) fOptions.SetMaxIterations(maxiter); }

   /// set the tolerance
   void SetTolerance(double tol) { fOptions.SetTolerance(tol); }

   /// set in the minimizer the objective function evaluation precision
   /// ( a value <=0 means the minimizer will choose its optimal value automatically, i.e. default case)
   void SetPrecision(double prec) { fOptions.SetPrecision(prec); }

   ///set the strategy
   void SetStrategy(int strategyLevel) { fOptions.SetStrategy(strategyLevel); }

   /// set scale for calculating the errors
   void SetErrorDef(double up) { fOptions.SetErrorDef(up); }

   /// flag to check if minimizer needs to perform accurate error analysis (e.g. run Hesse for Minuit)
   void SetValidError(bool on) { fValidError = on; }

   /// set all options in one go
   void SetOptions(const MinimizerOptions & opt) {
      fOptions = opt;
   }

   /// set only the extra options
   void SetExtraOptions(const IOptions & extraOptions) { fOptions.SetExtraOptions(extraOptions); }

   /// reset the default options (defined in MinimizerOptions)
   void SetDefaultOptions() {
      fOptions.ResetToDefaultOptions();
   }

protected:

   // keep protected to be accessible by the derived classes

   bool fValidError = false;    ///< flag to control if errors have been validated (Hesse has been run in case of Minuit)
   MinimizerOptions fOptions;   ///< minimizer options
   int fStatus = -1;            ///< status of minimizer
};

   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_Minimizer */
