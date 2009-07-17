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

namespace ROOT { 
   

   namespace Math { 

//_______________________________________________________________________________
/** 
    Minimizer options 

    @ingroup MultiMin
*/
class MinimizerOptions {

public:

   // static methods for setting and retrieving the default options 

   static void SetDefaultMinimizer(const char * type, const char * algo = 0);
   static void SetDefaultErrorDef( double up); 
   static void SetDefaultTolerance(double tol); 
   static void SetDefaultMaxFunctionCalls(int maxcall);
   static void SetDefaultMaxIterations(int maxiter);
   static void SetDefaultStrategy(int strat);
   static void SetDefaultPrintLevel(int level);

   static const std::string & DefaultMinimizerType();
   static const std::string & DefaultMinimizerAlgo(); 
   static double DefaultErrorDef();
   static double DefaultTolerance(); 
   static int DefaultMaxFunctionCalls(); 
   static int DefaultMaxIterations(); 
   static int DefaultStrategy(); 
   static int DefaultPrintLevel(); 


   // default options 
   MinimizerOptions() : 
      fLevel( MinimizerOptions::DefaultPrintLevel()),
      fMaxCalls( MinimizerOptions::DefaultMaxFunctionCalls() ), 
      fMaxIter( MinimizerOptions::DefaultMaxIterations() ), 
      fStrategy(  MinimizerOptions::DefaultStrategy() ), 
      fErrorDef(  MinimizerOptions::DefaultErrorDef() ), 
      fTolerance( MinimizerOptions::DefaultTolerance() ),
      fMinimType( MinimizerOptions::DefaultMinimizerType() ), 
      fAlgoType(  MinimizerOptions::DefaultMinimizerAlgo() )
   {}

   /** non-static methods for  retrivieng options */

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

   /// error definition 
   double ErrorDef() const { return  fErrorDef; }

   /// type of minimizer
   const std::string & MinimizerType() const { return fMinimType; }

   /// type of algorithm
   const std::string & MinimizerAlgorithm() const { return fAlgoType; }

   /** non-static methods for setting options */

   /// set print level
   void SetPrintLevel(int level) { fLevel = level; }

   ///set maximum of function calls 
   void SetMaxFunctionCalls(unsigned int maxfcn) { fMaxCalls = maxfcn; }

   /// set maximum iterations (one iteration can have many function calls) 
   void SetMaxIterations(unsigned int maxiter) { fMaxIter = maxiter; } 

   /// set the tolerance
   void SetTolerance(double tol) { fTolerance = tol; }

   /// set the strategy
   void SetStrategy(int stra) { fStrategy = stra; }

   /// set error def
   void SetErrorDef(double err) { fErrorDef = err; }

   /// set minimizer type
   void SetMinimizerType(const char * type) { fMinimType = type; }

   /// set minimizer algorithm
   void SetMinimizerAlgorithm(const char *type) { fAlgoType = type; }

private:

   int fLevel;               // debug print level 
   int fMaxCalls;            // maximum number of function calls
   int fMaxIter;             // maximum number of iterations
   int fStrategy;            // minimizer strategy (used by Minuit)
   double fErrorDef;         // error definition (=1. for getting 1 sigma error for chi2 fits)
   double fTolerance;        // minimize tolerance to reach solution
   std::string fMinimType;   // Minimizer type (Minuit, Minuit2, etc..
   std::string fAlgoType;    // Minimizer algorithmic specification (Migrad, Minimize, ...)
     
};

   } // end namespace Math

} // end namespace ROOT

#endif
