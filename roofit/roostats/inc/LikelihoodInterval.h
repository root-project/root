// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef RooStats_LikelihoodInterval
#define RooStats_LikelihoodInterval

#ifndef RooStats_ConfInterval
#include "RooStats/ConfInterval.h"
#endif

#ifndef ROO_ARG_SET
#include "RooArgSet.h"
#endif

#ifndef ROO_ABS_REAL
#include "RooAbsReal.h"
#endif

#include <map>

namespace RooStats {

   class LikelihoodInterval : public ConfInterval {

   public:

      // defult constructor 
      explicit LikelihoodInterval(const char* name = 0, const char* title = 0);
      /// construct the interval from a Profile Likelihood object, parameter of interest and optionally a snapshot of 
      /// POI with their best fit values 
      LikelihoodInterval(const char* name, RooAbsReal*, const RooArgSet*,  RooArgSet * = 0);
      LikelihoodInterval(const char* name, const char* title, RooAbsReal*, const RooArgSet*, RooArgSet * = 0);
      virtual ~LikelihoodInterval();
        
      virtual Bool_t IsInInterval(const RooArgSet&);

      virtual void SetConfidenceLevel(Double_t cl) {fConfidenceLevel = cl; ResetLimits(); }
      virtual Double_t ConfidenceLevel() const {return fConfidenceLevel;}
 

      // return list of parameters of interest.  User manages the return object
      virtual  RooArgSet* GetParameters() const;

      // check if parameters are correct. (dummy implementation to start)
      Bool_t CheckParameters(const RooArgSet&) const ;


      // Method to return lower limit on a given parameter 
      Double_t LowerLimit(RooRealVar& param) ;
      Double_t UpperLimit(RooRealVar& param) ;
    
      RooAbsReal* GetLikelihoodRatio() {return fLikelihoodRatio;}

      // return a pointer to a snapshot with best fit parameter of interest
      const RooArgSet * GetBestFitParameters() const { return fBestFitParams; }

   protected: 
      // reset the cached limit values
      void ResetLimits(); 

   private:

      RooArgSet   fParameters; // parameters of interest for this interval
      RooArgSet * fBestFitParams; // snapshot of the model parameters with best fit value (managed internally)
      RooAbsReal* fLikelihoodRatio; // likelihood ratio function used to make contours (managed internally)
      Double_t fConfidenceLevel; // Requested confidence level (eg. 0.95 for 95% CL)
      std::map<std::string, double> fLowerLimits; // map with cached lower limit values
      std::map<std::string, double> fUpperLimits; // map with cached upper limit values
      

      ClassDef(LikelihoodInterval,1)  // Concrete implementation of a ConfInterval based on a likelihood ratio
      
   };
}

#endif
