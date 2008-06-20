// @(#)root/mathcore:$Id$
// Author: L. Moneta Thu Sep 21 16:21:29 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class FitConfig

#include "Fit/FitConfig.h"

#include "Math/IParamFunction.h"
#include "Math/Util.h"

#include "Math/Minimizer.h"
#include "Math/Factory.h"

#include <cmath> 

#include <string> 
#include <sstream> 

#include "Math/Error.h"

//#define DEBUG
#ifdef DEBUG
#include <iostream>
#endif

namespace ROOT { 

namespace Fit { 


   static std::string fgDefaultMinimizer = "Minuit2";
   static std::string fgDefaultMinimAlgo = "Migrad";


FitConfig::FitConfig(unsigned int npar) : 
   fNormErrors(false),
   fSettings(std::vector<ParameterSettings>(npar) )  
{
   // constructor implementation

   // default minimizer type (ue static default values) 
   fMinimizerType = fgDefaultMinimizer; 
   fMinimAlgoType = fgDefaultMinimAlgo; 
}


FitConfig::~FitConfig() 
{
   // destructor implementation. No Op
}

void FitConfig::SetParamsSettings(unsigned int npar, const double *params ) { 
   // initialize fit config from parameter values
   if (params == 0) { 
      fSettings =  std::vector<ParameterSettings>(npar); 
      return; 
   }
   // if a vector of parameters is given
   fSettings.clear(); 
   fSettings.reserve(npar); 
   unsigned int i = 0; 
   const double * end = params+npar;
   for (const double * ipar = params; ipar !=  end; ++ipar) {  
      double val = *ipar; 
      double step = 0.3*std::fabs(val);   // step size is 30% of par value
      //double step = 2.0*std::fabs(val);   // step size is 30% of par value
      if (val ==  0) step  =  0.3; 
      
      fSettings.push_back( ParameterSettings("Par_" + ROOT::Math::Util::ToString(i), val, step ) ); 
#ifdef DEBUG
      std::cout << "FitConfig: add parameter " <<  func.ParameterName(i) << " val = " << val << std::endl;
#endif
      i++;
   } 
}

void FitConfig::SetParamsSettings(const ROOT::Math::IParamMultiFunction & func) { 
   // initialize from model function
   // set the parameters values from the function
   unsigned int npar = func.NPar(); 
   const double * begin = func.Parameters(); 
   if (begin == 0) { 
      fSettings =  std::vector<ParameterSettings>(npar); 
      return; 
   }

   fSettings.clear(); 
   fSettings.reserve(npar); 
   const double * end =  begin+npar; 
   unsigned int i = 0; 
   for (const double * ipar = begin; ipar !=  end; ++ipar) {  
      double val = *ipar; 
      double step = 0.3*std::fabs(val);   // step size is 30% of par value
      //double step = 2.0*std::fabs(val);   // step size is 30% of par value
      if (val ==  0) step  =  0.3; 
      
      fSettings.push_back( ParameterSettings(func.ParameterName(i), val, step ) ); 
#ifdef DEBUG
      std::cout << "FitConfig: add parameter " <<  func.ParameterName(i) << " val = " << val << std::endl;
#endif
      i++;
   } 

}

ROOT::Math::Minimizer * FitConfig::CreateMinimizer() { 
   // create minimizer according to the chosen configuration using the 
   // plug-in manager

   ROOT::Math::Minimizer * min = ROOT::Math::Factory::CreateMinimizer(fMinimizerType, fMinimAlgoType); 

   if (min == 0) { 
      std::string minim2 = "Minuit2";
      if (fMinimizerType != minim2 ) {
         std::string msg = "Could not create Minimizer " + fMinimizerType + " trying using minimizer " + minim2; 
         MATH_WARN_MSG("FitConfig::CreateMinimizer",msg.c_str());
         min = ROOT::Math::Factory::CreateMinimizer(minim2); 
         if (min == 0) { 
            MATH_ERROR_MSG("FitConfig::CreateMinimizer","Could not create the Minuit2 minimizer");
            return 0; 
         }
      }
      else {
         std::string msg = "Could not create the Minimizer " + fMinimizerType; 
         MATH_ERROR_MSG("FitConfig::CreateMinimizer",msg.c_str());
         return 0;
      }
   } 

   // set default max of function calls according to the number of parameters
   // formula from Minuit2 (adapted)
   if (fMinimizerOpts.MaxFunctionCalls() == 0) {  
      unsigned int npar =  fSettings.size();      
      int maxfcn = 1000 + 100*npar + 5*npar*npar;
      fMinimizerOpts.SetMaxFunctionCalls(maxfcn); 
   }


   // set default minimizer control parameters 
   min->SetPrintLevel( fMinimizerOpts.PrintLevel() ); 
   min->SetMaxFunctionCalls( fMinimizerOpts.MaxFunctionCalls() ); 
   min->SetMaxIterations( fMinimizerOpts.MaxIterations() ); 
   min->SetTolerance( fMinimizerOpts.Tolerance() ); 
   min->SetValidError( fMinimizerOpts.ParabErrors() );
   min->SetStrategy( fMinimizerOpts.Strategy() );


   return min; 
} 

void FitConfig::SetDefaultMinimizer(const std::string & type, const std::string & algo ) { 
   // set the default minimizer type and algorithm
   if (type != "") fgDefaultMinimizer = type; 
   if (algo != "") fgDefaultMinimAlgo = algo;
} 

   } // end namespace Fit

} // end namespace ROOT

