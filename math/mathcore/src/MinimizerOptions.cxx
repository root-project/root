// @(#)root/mathcore:$Id$ 
// Author: L. Moneta Fri Aug 15 2008

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2008  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

#include "Math/MinimizerOptions.h"
#include "RConfigure.h"

namespace ROOT { 
   

namespace Math { 

// default minimizer options (static variable) 

#ifdef R__HAS_MINUIT2
   static std::string gDefaultMinimizer = "Minuit2";
#else 
   static std::string gDefaultMinimizer = "Minuit";
#endif
   static std::string gDefaultMinimAlgo = "Migrad";
   static double gDefaultErrorDef = 1.;
   static double gDefaultTolerance = 1.E-4; 
   static int  gDefaultMaxCalls = 0; // 0 means leave default values Deaf
   static int  gDefaultMaxIter  = 0; 
   static int  gDefaultStrategy  = 1; 
   static int  gDefaultPrintLevel  = 0; 



void MinimizerOptions::SetDefaultMinimizer(const char * type, const char * algo ) {   
   // set the default minimizer type and algorithm
   if (type) gDefaultMinimizer = std::string(type); 
   if (algo) gDefaultMinimAlgo = std::string(algo);
}
void MinimizerOptions::SetDefaultErrorDef(double up) {
   // set the default error definition 
   gDefaultErrorDef = up; 
}
void MinimizerOptions::SetDefaultTolerance(double tol) {
   // set the defult tolerance
   gDefaultTolerance = tol; 
}
void MinimizerOptions::SetDefaultMaxFunctionCalls(int maxcall) {
   // set the default maximum number of function calls
   gDefaultMaxCalls = maxcall; 
}
void MinimizerOptions::SetDefaultMaxIterations(int maxiter) {
   // set the default maximum number of iterations
   gDefaultMaxIter = maxiter; 
}
void MinimizerOptions::SetDefaultStrategy(int stra) {
   // set the default minimization strategy
   gDefaultStrategy = stra; 
}
void MinimizerOptions::SetDefaultPrintLevel(int level) {
   // set the default printing level 
   gDefaultPrintLevel = level; 
}
const std::string & MinimizerOptions::DefaultMinimizerType() { return gDefaultMinimizer; }
const std::string & MinimizerOptions::DefaultMinimizerAlgo() { return gDefaultMinimAlgo; }
double MinimizerOptions::DefaultErrorDef()         { return gDefaultErrorDef; }
double MinimizerOptions::DefaultTolerance()        { return gDefaultTolerance; }
int    MinimizerOptions::DefaultMaxFunctionCalls() { return gDefaultMaxCalls; }
int    MinimizerOptions::DefaultMaxIterations()    { return gDefaultMaxIter; }
int    MinimizerOptions::DefaultStrategy()         { return gDefaultStrategy; }
int    MinimizerOptions::DefaultPrintLevel()       { return gDefaultPrintLevel; }

     
} // end namespace Math

} // end namespace ROOT

