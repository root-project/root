// @(#)root/mathcore:$Id$
// Author: L. Moneta Thu Sep 21 16:21:29 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class FitConfig

#ifndef ROOT_Fit_FitConfig
#define ROOT_Fit_FitConfig


#ifndef ROOT_Fit_ParameterSettings
#include "Fit/ParameterSettings.h"
#endif

#ifndef ROOT_Fit_MinimizerControlParams
#include "Fit/MinimizerControlParams.h"
#endif

#ifndef ROOT_Math_IParamFunctionfwd
#include "Math/IParamFunctionfwd.h"
#endif


#include <vector>

namespace ROOT { 

   namespace Math { 

      class Minimizer;
   }

   namespace Fit { 

//___________________________________________________________________________________
/** 
   Class describing the configuration of the fit, options and parameter settings
   using the ROOT::Fit::ParameterSettings class 

   @ingroup FitMain
*/ 
class FitConfig {

public: 

   /** 
      Default constructor
   */ 
   FitConfig (unsigned int npar = 0); 

   /** 
      Destructor 
   */ 
   ~FitConfig ();    

   /**
      get the parameter settings for the i-th parameter (const method)
   */
   const ParameterSettings & ParSettings(unsigned int i) const { return fSettings[i]; }

   /**
      get the parameter settings for the i-th parameter (non-const method)
   */
   ParameterSettings & ParSettings(unsigned int i) { return fSettings[i]; }

   /**
      get the vector of parameter settings  (const method)
   */
   const std::vector<ROOT::Fit::ParameterSettings> & ParamsSettings() const { return fSettings; }

   /**
      get the vector of parameter settings  (non-const method)
   */
   std::vector<ROOT::Fit::ParameterSettings> & ParamsSettings() { return fSettings; }


   /**
      set the parameter settings from number of params and optionally a vector of values (otherwise are set to zero)
   */
   void SetParamsSettings(unsigned int npar, const double * params = 0); 

   /**
      set the parameter settings from a function
   */
   void SetParamsSettings(const ROOT::Math::IParamMultiFunction & func); 

   /**
      create a new minimizer according to chosen configuration
   */
   ROOT::Math::Minimizer * CreateMinimizer(); 


   /**
      access to the minimizer  control parameter (const method) 
   */
   const MinimizerControlParams & MinimizerOptions() const { return fMinimizerOpts; } 

   /**
      access to the minimizer  control parameter (non const method) 
   */
   MinimizerControlParams & MinimizerOptions()  { return fMinimizerOpts; } 

   
   /**
      set minimizer type 
   */
   void SetMinimizer(const std::string & type, std::string algo = "") { 
      fMinimizerType = type; 
      fMinimAlgoType = algo; 
   } 

   /**
      return type of minimizer package
   */
   const std::string & MinimizerType() const { return fMinimizerType; } 

   /**
      return type of minimizer algorithms 
   */
   const std::string & MinimizerAlgoType() const { return fMinimAlgoType; } 


   /**
      flag to check if resulting errors are be normalized according to chi2/ndf 
   */
   bool NormalizeErrors(){ return fNormErrors; } 

   /**
      set the option to normalize the error on the result  according to chi2/ndf
   */
   void SetNormErrors(bool on) { fNormErrors= on; }


   /**
      static function to control default minimizer type and algorithm
   */
   static void SetDefaultMinimizer(const std::string & type, const std::string & algo = ""); 


protected: 


private: 

   bool fNormErrors;    // flag for error normalization

   std::vector<ROOT::Fit::ParameterSettings> fSettings;  // vector with the parameter settings

   std::string fMinimizerType;  // minimizer type (MINUIT, MINUIT2, etc..)
   std::string fMinimAlgoType;  // algorithm type (MIGRAD, SIMPLEX, etc..)
   MinimizerControlParams fMinimizerOpts;   //minimizer control parameters

}; 

   } // end namespace Fit

} // end namespace ROOT


#endif /* ROOT_Fit_FitConfig */
