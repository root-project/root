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
*/
class MinimizerOptions {

public:

   // static methods for setting and retrieving the default options 

   static void SetDefaultMinimizer(const char * type, const char * algo = 0);
   static void SetDefaultErrorDef( double up); 
   static void SetDefaultTolerance(double tol); 
   static void SetDefaultPrecision(double prec); 
   static void SetDefaultMaxFunctionCalls(int maxcall);
   static void SetDefaultMaxIterations(int maxiter);
   static void SetDefaultStrategy(int strat);
   static void SetDefaultPrintLevel(int level);

   static const std::string & DefaultMinimizerType();
   static const std::string & DefaultMinimizerAlgo(); 
   static double DefaultErrorDef();
   static double DefaultTolerance(); 
   static double DefaultPrecision(); 
   static int DefaultMaxFunctionCalls(); 
   static int DefaultMaxIterations(); 
   static int DefaultStrategy(); 
   static int DefaultPrintLevel(); 

   /// retrieve extra options - if not existing create a IOptions 
   static ROOT::Math::IOptions & Default(const char * name);

   // find extra options - return 0 if not existing 
   static ROOT::Math::IOptions * FindDefault(const char * name);

   /// print all the default options for the name given
   static void PrintDefault(const char * name, std::ostream & os = std::cout); 

public:

   // constructor using the default options 
   // pass optionally a pointer to the additional options
   // otehrwise look if they exist for this default minimizer
   // and in that case they are copied in the constructed instance
   MinimizerOptions(IOptions * extraOpts = 0);

   // destructor  
   ~MinimizerOptions();

   // copy constructor 
   MinimizerOptions(const MinimizerOptions & opt);

   /// assignment operators 
   MinimizerOptions & operator=(const MinimizerOptions & opt);

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

   /// precision in the objective funciton calculation (value <=0 means left to default)
   double Precision() const { return  fPrecision; }

   /// error definition 
   double ErrorDef() const { return  fErrorDef; }

   /// return extra options (NULL pointer if they are not present)
   IOptions * ExtraOptions() const { return fExtraOptions; }

   /// type of minimizer
   const std::string & MinimizerType() const { return fMinimType; }

   /// type of algorithm
   const std::string & MinimizerAlgorithm() const { return fAlgoType; }

   /// print all the options 
   void Print(std::ostream & os = std::cout) const; 

   /** non-static methods for setting options */

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
   void  SetExtraOptions(const IOptions & opt); 


private:

   int fLevel;               // debug print level 
   int fMaxCalls;            // maximum number of function calls
   int fMaxIter;             // maximum number of iterations
   int fStrategy;            // minimizer strategy (used by Minuit)
   double fErrorDef;         // error definition (=1. for getting 1 sigma error for chi2 fits)
   double fTolerance;        // minimize tolerance to reach solution
   double fPrecision;        // precision of the objective function evaluation (value <=0 means left to default)
   std::string fMinimType;   // Minimizer type (Minuit, Minuit2, etc..
   std::string fAlgoType;    // Minimizer algorithmic specification (Migrad, Minimize, ...)

   // extra options
   ROOT::Math::IOptions *   fExtraOptions;  // extra options 
     
};

   } // end namespace Math

} // end namespace ROOT

#endif
