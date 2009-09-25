// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_ProfileLikelihoodCalculator
#define ROOSTATS_ProfileLikelihoodCalculator

#ifndef ROOSTATS_CombinedCalculator
#include "RooStats/CombinedCalculator.h"
#endif

#include "RooStats/LikelihoodInterval.h"

namespace RooStats {
   
   class LikelihoodInterval; 

   class ProfileLikelihoodCalculator : public CombinedCalculator {

   public:

      ProfileLikelihoodCalculator();

      ProfileLikelihoodCalculator(RooAbsData& data, RooAbsPdf& pdf, const RooArgSet& paramsOfInterest, 
                                  Double_t size = 0.05, const RooArgSet* nullParams = 0 );

      ProfileLikelihoodCalculator(RooAbsData& data, ModelConfig & model, Double_t size = 0.05);


      virtual ~ProfileLikelihoodCalculator();
    
      // main interface, implemented
      virtual LikelihoodInterval* GetInterval() const ; 

      // main interface, implemented
      virtual HypoTestResult* GetHypoTest() const;   
    

   protected:

      ClassDef(ProfileLikelihoodCalculator,1) // A concrete implementation of CombinedCalculator that uses the ProfileLikelihood ratio.
   };
}
#endif
