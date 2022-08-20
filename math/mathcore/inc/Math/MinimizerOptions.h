// @(#)root/mathcore:$Id$
// Author: L. Moneta Fri Aug 15 2008

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2008  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Math_MinimizerOptions
#define ROOT_Math_MinimizerOptions

#include <string>

#include <iostream>

namespace ROOT {


namespace Math {


class IOptions;

//_______________________________________________________________________________
/**
    Minimizer options

    @ingroup MultiMin

    Class defining the options for the minimizer.
    It contains also static methods for setting the default Minimizer option values
    that will be used by default by all Minimizer instances.
    To see the current default options do:

         ROOT::Math::MinimizerOptions::PrintDefault();

*/
class MinimizerOptions {

public:

   // static methods for setting and retrieving the default options

   /// Set the default Minimizer type and corresponding algorithms.
   /// Here is the list of the available minimizers and their corresponding algorithms.
   /// For some minimizers (e.g. Fumili) there are no specific algorithms available, then there is no need to specify it.
   ///
   /// \anchor ROOTMinimizers
   /// ### ROOT Minimizers
   ///
   /// - Minuit  Minimizer based on TMinuit, the legacy Minuit implementation. Here are the available algorithms:
   ///     - Migrad     default algorithm based on the variable metric minimizer
   ///     - Minimize   combination of Simplex and Migrad
   ///     - Simplex    minimization algorithm not using the gradient information
   ///     - Scan       brute function scan
   /// - Minuit2  New C++ implementation of Minuit (the recommended one)
   ///     - Migrad  (default)
   ///     - Minimize
   ///     - Simplex
   ///     - Fumili2    new implementation of Fumili integrated in Minuit2
   /// - Fumili  Minimizer using an approximation for the Hessian based on first derivatives of the model function (see TFumili). Works only for chi-squared and likelihood functions.
   /// - Linear  Linear minimizer (fitter) working only for linear functions (see TLinearFitter and TLinearMinimizer)
   /// - GSLMultiMin  Minimizer from GSL based on the ROOT::Math::GSLMinimizer. Available algorithms are:
   ///    - BFGS2  (default)
   ///    - BFGS
   ///    - ConjugateFR
   ///    - ConjugatePR
   ///    - SteepestDescent
   /// - GSLMultiFit Minimizer based on GSL for minimizing only non linear least-squared functions (using an approximation similar to Fumili). See ROOT::Math::GSLMultiFit.
   /// - GSLSimAn Simulated annealing minimizer from GSL (see ROOT::Math::GSLSimAnMinimizer). It is a stochastic minimization algorithm using only function values and not the gradient.
   /// - Genetic Genetic minimization algorithms (see TMVA::Genetic)
   ///
   static void SetDefaultMinimizer(const char * type, const char * algo = 0);

   /// Set the default level for computing the parameter errors.
   /// For example for 1-sigma parameter errors
   ///  - up = 1 for a chi-squared function
   ///  - up = 0.5 for a negative log-likelihood function
   ///
   /// The value will be used also by Minos when computing the confidence interval
   static void SetDefaultErrorDef( double up);

   /// Set the Minimization tolerance.
   /// The Default value for Minuit and Minuit2 is 0.01
   static void SetDefaultTolerance(double tol);

   /// Set the default Minimizer precision.
   /// (used only by MInuit and Minuit2)
   /// It is used to specify the numerical precision used for computing the
   /// objective function. It should be left to the default value found by the Minimizer
   /// (typically double precision)
   static void SetDefaultPrecision(double prec);

   /// Set the maximum number of function calls.
   static void SetDefaultMaxFunctionCalls(int maxcall);

   /// Set the maximum number of iterations.
   /// Used by the GSL minimizers and Genetic. Not used by Minuit,Minuit2.
   static void SetDefaultMaxIterations(int maxiter);

   /// Set the default strategy.
   /// The strategy is a parameter used only by Minuit and Minuit2.
   /// Possible values are:
   /// - `strat = 0` : rough approximation of Hessian using the gradient. Avoid computing the full Hessian matrix
   /// - `strat = 1` (default and recommended one) - Use Hessian approximation but compute full Hessian at the end of minimization if needed.
   /// - `strat = 2`  Perform several full Hessian computations during the minimization. Slower and not always working better than `strat=1`.
   static void SetDefaultStrategy(int strat);

   /// Set the default Print Level.
   /// Possible levels are from 0 (minimal printing) to 3 (maximum printing)
   static void SetDefaultPrintLevel(int level);

   /// Set additional minimizer options as pair of (string,value).
   /// Extra option defaults can be configured for a specific algorithm and
   /// then if a matching with the correct option name exists it will be used
   /// whenever creating a new minimizer instance.
   /// For example for changing the default number of steps of the Genetic minimizer from 100 to 500 do
   ///
   ///      auto extraOpt = ROOT::Math::MinimizerOptions::Default("Genetic")
   ///      extraOpts.SetValue("Steps",500);
   ///
   ///  and when creating the Genetic minimizer you will have the new value for the option:
   ///
   ///      auto gmin = ROOT::Math::Factory::CreateMinimizer("Genetic");
   ///      gmin->Options().Print();
   ///
   static void SetDefaultExtraOptions(const IOptions * extraoptions);


   static const std::string & DefaultMinimizerType();
   static const std::string & DefaultMinimizerAlgo();
   static double DefaultErrorDef();
   static double DefaultTolerance();
   static double DefaultPrecision();
   static int DefaultMaxFunctionCalls();
   static int DefaultMaxIterations();
   static int DefaultStrategy();
   static int DefaultPrintLevel();
   static IOptions * DefaultExtraOptions();

   /// Retrieve extra options for a given minimizer name.
   /// If the extra options already exist in a global map of (Minimizer name, options)
   /// it will return a reference to that options, otherwise it will create a new one
   /// and return the corresponding reference
   static ROOT::Math::IOptions & Default(const char * name);

   /// Find an extra options and return a nullptr if it is not existing.
   /// Same as above but it will not create a new one
   static ROOT::Math::IOptions * FindDefault(const char * name);

   /// Print all the default options including the extra one specific for a given minimizer name.
   /// If no minimizer name is given, all the extra default options, which have been set and configured will be printed 
   static void PrintDefault(const char * name = nullptr, std::ostream & os = std::cout);

public:

   // constructor using the default options
   MinimizerOptions();

   // destructor
   ~MinimizerOptions();

   // copy constructor
   MinimizerOptions(const MinimizerOptions & opt);

   /// assignment operators
   MinimizerOptions & operator=(const MinimizerOptions & opt);

   /** non-static methods for retrieving options */

   /// set print level
   int PrintLevel() const { return fLevel; }

   ///  max number of function calls
   unsigned int MaxFunctionCalls() const { return fMaxCalls; }

   /// max iterations
   unsigned int MaxIterations() const { return fMaxIter; }

   /// strategy
   int Strategy() const { return fStrategy; }

   /// absolute tolerance
   double Tolerance() const { return  fTolerance; }

   /// precision in the objective function calculation (value <=0 means left to default)
   double Precision() const { return  fPrecision; }

   /// error definition
   double ErrorDef() const { return  fErrorDef; }

   /// return extra options (NULL pointer if they are not present)
   const IOptions * ExtraOptions() const { return fExtraOptions; }

   /// type of minimizer
   const std::string & MinimizerType() const { return fMinimType; }

   /// type of algorithm
   const std::string & MinimizerAlgorithm() const { return fAlgoType; }

   /// print all the options
   void Print(std::ostream & os = std::cout) const;

   /** non-static methods for setting options */
   void ResetToDefaultOptions();

   /// set print level
   void SetPrintLevel(int level) { fLevel = level; }

   ///set maximum of function calls
   void SetMaxFunctionCalls(unsigned int maxfcn) { fMaxCalls = maxfcn; }

   /// set maximum iterations (one iteration can have many function calls)
   void SetMaxIterations(unsigned int maxiter) { fMaxIter = maxiter; }

   /// set the tolerance
   void SetTolerance(double tol) { fTolerance = tol; }

   /// set the precision
   void SetPrecision(double prec) { fPrecision = prec; }

   /// set the strategy
   void SetStrategy(int stra) { fStrategy = stra; }

   /// set error def
   void SetErrorDef(double err) { fErrorDef = err; }

   /// set minimizer type
   void SetMinimizerType(const char * type) { fMinimType = type; }

   /// set minimizer algorithm
   void SetMinimizerAlgorithm(const char *type) { fAlgoType = type; }

   /// set extra options (in this case pointer is cloned)
   void SetExtraOptions(const IOptions & opt);


private:

   int fLevel;               ///< debug print level
   int fMaxCalls;            ///< maximum number of function calls
   int fMaxIter;             ///< maximum number of iterations
   int fStrategy;            ///< minimizer strategy (used by Minuit)
   double fErrorDef;         ///< error definition (=1. for getting 1 sigma error for chi2 fits)
   double fTolerance;        ///< minimize tolerance to reach solution
   double fPrecision;        ///< precision of the objective function evaluation (value <=0 means left to default)
   std::string fMinimType;   ///< Minimizer type (Minuit, Minuit2, etc..
   std::string fAlgoType;    ///< Minimizer algorithmic specification (Migrad, Minimize, ...)

   // extra options
   ROOT::Math::IOptions *   fExtraOptions;  // extra options

};

   } // end namespace Math

} // end namespace ROOT

#endif
