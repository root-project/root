// @(#)root/roostats:$Id$
// Author: Sven Kreiss and Kyle Cranmer    June 2010
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "RooStats/ToyMCStudy.h"

#include "RooStats/ToyMCSampler.h"


#ifndef ROO_MSG_SERVICE
#include "RooMsgService.h"
#endif

#include "RooRandom.h"



ClassImp(RooStats::ToyMCStudy)

namespace RooStats {


// _____________________________________________________________________________
Bool_t ToyMCStudy::initialize(void) {
   coutP(Generation) << "initialize" << endl;

   //coutI(InputArguments) << "SetSeed(0)" << endl;
   //RooRandom::randomGenerator()->SetSeed(0);
   coutI(InputArguments) << "Seed is: " << RooRandom::randomGenerator()->GetSeed() << endl;

   if(!fToyMCSampler) {
      coutE(InputArguments) << "Need an instance of ToyMCSampler to run." << endl;
   }else{
      coutI(InputArguments) << "Using given ToyMCSampler." << endl;
   }

   return kFALSE;
}

// _____________________________________________________________________________
Bool_t ToyMCStudy::execute(void) {
   SamplingDistribution *sd = fToyMCSampler->GetSamplingDistributionSingleWorker(fParamPointOfInterest);
   storeDetailedOutput(*sd);

   return kFALSE;
}

// _____________________________________________________________________________
Bool_t ToyMCStudy::finalize(void) {
   coutP(Generation) << "finalize" << endl;

   if(fToyMCSampler) delete fToyMCSampler;
   fToyMCSampler = NULL;

   return kFALSE;
}


Bool_t ToyMCStudy::merge(SamplingDistribution& result) {
   // returns true if there was an error
   coutP(Generation) << "merge" << endl;

   if(!detailedData()) {
      coutE(Generation) << "No detailed output present." << endl;
      return kTRUE;
   }

   RooLinkedListIter iter = detailedData()->iterator();
   TObject *o = NULL;
   while((o = iter.Next())) {
      if(!dynamic_cast<SamplingDistribution*>(o)) {
         coutW(Generation) << "Merging Results problem: not a SamplingDistribution" << endl;
         continue;
      }

      result.Add(dynamic_cast<SamplingDistribution*>(o));
   }

   return kFALSE;
}


} // end namespace RooStats
