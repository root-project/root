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



ClassImp(RooStats::ToyMCStudy);

ClassImp(RooStats::ToyMCPayload);

using namespace std;


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
   RooDataSet* sd = fToyMCSampler->GetSamplingDistributionsSingleWorker(fParamPoint);
   ToyMCPayload *sdw = new ToyMCPayload(sd);
   storeDetailedOutput(*sdw);

   return kFALSE;
}

// _____________________________________________________________________________
Bool_t ToyMCStudy::finalize(void) {
   coutP(Generation) << "finalize" << endl;

   if(fToyMCSampler) delete fToyMCSampler;
   fToyMCSampler = NULL;

   return kFALSE;
}


RooDataSet* ToyMCStudy::merge() {
   coutP(Generation) << "merge" << endl;
   RooDataSet* samplingOutput = NULL;

   if(!detailedData()) {
      coutE(Generation) << "No detailed output present." << endl;
      return NULL;
   }

   RooLinkedListIter iter = detailedData()->iterator();
   TObject *o = NULL;
   while((o = iter.Next())) {
      ToyMCPayload *oneWorker = dynamic_cast< ToyMCPayload* >(o);
      if(!oneWorker) {
         coutW(Generation) << "Merging Results problem: not correct type" << endl;
         continue;
      }
      
      if( !samplingOutput ) samplingOutput = new RooDataSet(*oneWorker->GetSamplingDistributions());
      else samplingOutput->append( *oneWorker->GetSamplingDistributions() );

      //delete oneWorker;
   }

   return samplingOutput;
}


} // end namespace RooStats
