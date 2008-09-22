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
    Minimizer options structure

    @ingroup MultiMin
*/
struct MinimizerOptions {

   // static methods for setting and retrieving the default options 

   static void SetDefaultMinimizer(const std::string & type, const std::string & algo);
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
      MinimType( MinimizerOptions::DefaultMinimizerType() ), 
      AlgoType(  MinimizerOptions::DefaultMinimizerAlgo() ),
      ErrorDef(  MinimizerOptions::DefaultErrorDef() ), // no need for a static method here
      Tolerance( MinimizerOptions::DefaultTolerance() ),
      MaxFunctionCalls( MinimizerOptions::DefaultMaxFunctionCalls() ), 
      MaxIterations( MinimizerOptions::DefaultMaxIterations() ), 
      Strategy(  MinimizerOptions::DefaultStrategy() ), 
      PrintLevel( MinimizerOptions::DefaultPrintLevel())
   {}


   std::string MinimType;   // Minimizer type (Minuit, Minuit2, etc..
   std::string AlgoType;    // Minimizer algorithmic specification (Migrad, Minimize, ...)
   double ErrorDef;         // error definition (=1. for getting 1 sigma error for chi2 fits)
   double Tolerance;        // minimize tolerance to reach solution
   int MaxFunctionCalls;    // maximum number of function calls
   int MaxIterations;       // maximum number of iterations
   int Strategy;            // minimizer strategy (used by Minuit)
   int PrintLevel;          // debug print level 

     
};

   } // end namespace Math

} // end namespace ROOT

#endif
