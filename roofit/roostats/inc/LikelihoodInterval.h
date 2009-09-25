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

namespace RooStats {

   class LikelihoodInterval : public ConfInterval {

   public:

      LikelihoodInterval();
      LikelihoodInterval(const char* name);
      LikelihoodInterval(const char* name, const char* title);
      LikelihoodInterval(const char* name, RooAbsReal*, const RooArgSet*);
      LikelihoodInterval(const char* name, const char* title, RooAbsReal*, const RooArgSet*);
      virtual ~LikelihoodInterval();
        
      virtual Bool_t IsInInterval(RooArgSet&);
      virtual void SetConfidenceLevel(Double_t cl) {fConfidenceLevel = cl;}
      virtual Double_t ConfidenceLevel() const {return fConfidenceLevel;}
 

      // do we want it to return list of parameters
      virtual RooArgSet* GetParameters() const;

      // check if parameters are correct. (dummy implementation to start)
      Bool_t CheckParameters(RooArgSet&) const ;


      // Method to return lower limit on a given parameter 
      Double_t LowerLimit(RooRealVar& param) ;
      Double_t UpperLimit(RooRealVar& param) ;
    
      RooAbsReal* GetLikelihoodRatio() {return fLikelihoodRatio;}

   private:

      const RooArgSet* fParameters; // parameters of interest for this interval
      RooAbsReal* fLikelihoodRatio; // likelihood ratio function used to make contours
      Double_t fConfidenceLevel; // Requested confidence level (eg. 0.95 for 95% CL)

      ClassDef(LikelihoodInterval,1)  // Concrete implementation of a ConfInterval based on a likelihood ratio
      
   };
}

#endif
