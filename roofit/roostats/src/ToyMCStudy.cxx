// @(#)root/roostats:$Id$
// Author: Sven Kreiss and Kyle Cranmer    June 2010
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class RooStats::ToyMCStudy
     \ingroup Roostats

ToyMCStudy is an implementation of RooAbsStudy for toy Monte Carlo sampling.
This class is automatically used by ToyMCSampler when given a ProofConfig.
This is also its intended use case.
*/

#include "RooStats/ToyMCStudy.h"

#include "RooStats/ToyMCSampler.h"


#include "RooMsgService.h"

#include "RooRandom.h"
#include "TRandom2.h"
#include "TMath.h"

#include "TEnv.h"

ClassImp(RooStats::ToyMCStudy);

ClassImp(RooStats::ToyMCPayload);

using namespace std;


namespace RooStats {

////////////////////////////////////////////////////////////////////////////////

bool ToyMCStudy::initialize(void) {
   coutP(Generation) << "initialize" << endl;

   if(!fToyMCSampler) {
      coutE(InputArguments) << "Need an instance of ToyMCSampler to run." << endl;
      return false;
   }else{
      coutI(InputArguments) << "Using given ToyMCSampler." << endl;
   }


   TString  worknumber = gEnv->GetValue("ProofServ.Ordinal","undef");
   int iworker = -1;
   if (worknumber != "undef") {
      iworker = int( worknumber.Atof()*10 + 0.1);

      // generate a seed using
      std::cout << "Current global seed is " << fRandomSeed << std::endl;
      TRandom2 r(fRandomSeed );
      // get a seed using the iworker-value
      unsigned int seed = r.Integer(TMath::Limits<unsigned int>::Max() );
      for (int i = 0; i< iworker; ++i)
         seed = r.Integer(TMath::Limits<unsigned int>::Max() );

      // initialize worker using seed from ToyMCSampler
      RooRandom::randomGenerator()->SetSeed(seed);
   }

   coutI(InputArguments) << "Worker " << iworker << " seed is: " << RooRandom::randomGenerator()->GetSeed() << endl;

   return false;
}

////////////////////////////////////////////////////////////////////////////////

bool ToyMCStudy::execute(void) {

   coutP(Generation) << "ToyMCStudy::execute - run with seed " <<   RooRandom::randomGenerator()->Integer(TMath::Limits<unsigned int>::Max() ) << std::endl;
   RooDataSet* sd = fToyMCSampler->GetSamplingDistributionsSingleWorker(fParamPoint);
   ToyMCPayload *sdw = new ToyMCPayload(sd);
   storeDetailedOutput(*sdw);

   return false;
}

////////////////////////////////////////////////////////////////////////////////

bool ToyMCStudy::finalize(void) {
   coutP(Generation) << "ToyMCStudy::finalize" << endl;

   if(fToyMCSampler) delete fToyMCSampler;
   fToyMCSampler = nullptr;

   return false;
}

////////////////////////////////////////////////////////////////////////////////

RooDataSet* ToyMCStudy::merge() {

   RooDataSet* samplingOutput = nullptr;

   if(!detailedData()) {
      coutE(Generation) << "ToyMCStudy::merge No detailed output present." << endl;
      return nullptr;
   }

   int i = 0;
   for (auto * o : static_range_cast<TObject*>(*detailedData())) {
      ToyMCPayload *oneWorker = dynamic_cast< ToyMCPayload* >(o);
      if(!oneWorker) {
         coutW(Generation) << "Merging Results problem: not correct type" << endl;
         continue;
      }

      if( !samplingOutput ) samplingOutput = new RooDataSet(*oneWorker->GetSamplingDistributions());

      else samplingOutput->append( *oneWorker->GetSamplingDistributions() );

      i++;
      //delete oneWorker;
   }
   coutP(Generation) << "Merged data from nworkers # " << i << "- merged data size is " << samplingOutput->numEntries() << std::endl;


   return samplingOutput;
}


} // end namespace RooStats
