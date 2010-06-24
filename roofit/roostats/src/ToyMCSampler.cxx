// @(#)root/roostats:$Id$
// Author: Sven Kreiss    June 2010
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "RooStats/ToyMCSampler.h"

#ifndef ROO_MSG_SERVICE
#include "RooMsgService.h"
#endif

#ifndef ROO_DATA_HIST
#include "RooDataHist.h"
#endif

namespace RooStats {

Bool_t ToyMCSampler::CheckConfig(void) {
   // only checks, no guessing/determination (do this in calculators,
   // e.g. using ModelConfig::GuessObsAndNuisance(...))
   bool goodConfig = true;

   // TODO
   //if(!fTestStat) { ooccoutE(NULL,InputArguments) << "Test statistic not set." << endl; goodConfig = false; }
   if(!fObservables) { cout << "Observables not set." << endl; goodConfig = false; }
   if(!fNullPOI) { cout << "Parameter values used to evaluate for test statistic  not set." << endl; goodConfig = false; }
   if(!fPdf) { cout << "Pdf not set." << endl; goodConfig = false; }

   return goodConfig;
}

SamplingDistribution* ToyMCSampler::GetSamplingDistribution(RooArgSet& paramPointIn) {
   CheckConfig();

   std::vector<Double_t> testStatVec;
   std::vector<Double_t> testStatWeights;

   // important to cache the paramPoint b/c test statistic might 
   // modify it from event to event
   RooArgSet* paramPoint = (RooArgSet*) paramPointIn.snapshot();
   RooArgSet *allVars = fPdf->getVariables();
   RooArgSet *saveAll = (RooArgSet*) allVars->snapshot();

   RooDataSet *nuisanceParPoints = NULL;
   if (fPriorNuisance  &&  fNuisancePars) {
      if (fExpectedNuisancePar) {
#ifdef EXPECTED_NUISANCE_PAR // under development
         nuisanceParPoints = fModel->GetPriorPdf()->generateBinned(
            *fModel->GetNuisanceParameters(),
            RooFit::ExpectedData(),
            RooFit::NumEvents(fNToys*100) // TODO Good value?
         );
         if(fNToys != nuisanceParPoints->numEntries()) {
            cout << "Overwriting fNToys with " << nuisanceParPoints->numEntries() << endl;
            fNToys = nuisanceParPoints->numEntries();
         }
#endif
      }else{
         nuisanceParPoints = fPriorNuisance->generate(*fNuisancePars, fNToys);
      }
   }


   for (Int_t i = 0; i < fNToys; ++i) {

      if ( i% 500 == 0 && i > 0 ) 
         oocoutP((TObject*)0,Generation) 
            << "....... on toy number " << i << " / " << fNToys << std::endl;

      if (nuisanceParPoints) {
	 // set variables to requested parameter point
	 *allVars = *paramPoint;
	 // set nuisance parameters to randomized value
	 *allVars = *nuisanceParPoints->get(i);

	 // generate toy data for this parameter point
         RooAbsData* toydata = GenerateToyData(*allVars);

	 // evaluate test statistic, that only depends on null POI
         testStatVec.push_back(fTestStat->Evaluate(*toydata, *fNullPOI));
         testStatWeights.push_back(nuisanceParPoints->weight());
         delete toydata;
      }else{
	 // set variables to requested parameter point
	 *allVars = *paramPoint;
	 // generate toy data for this parameter point
         RooAbsData* toydata = GenerateToyData(*allVars);
	 // evaluate test statistic, that only depends on null POI
         testStatVec.push_back(fTestStat->Evaluate(*toydata, *fNullPOI));
         delete toydata;
      }
   }
   delete nuisanceParPoints;

   *allVars = *saveAll;
   delete saveAll;
   delete allVars;

#ifdef EXPECTED_NUISANCE_PAR
   if (testStatWeights.size()) {
      return new SamplingDistribution(
         fSamplingDistName.c_str(),
         "Sampling Distribution of Test Statistic (Expected NuisPar)",
         testStatVec,
         testStatWeights,
         fTestStat->GetVarName()
      );
   }
#endif
   return new SamplingDistribution(
      fSamplingDistName.c_str(),
      fSamplingDistName.c_str(),
      testStatVec,
      fTestStat->GetVarName()
   );
}


RooAbsData* ToyMCSampler::GenerateToyData(RooArgSet& /*nullPOI*/) const {
   // This method generates a toy data set for the given parameter point taking
   // global observables into account.

   RooArgSet observables(*fObservables);
   if(fGlobalObservables  &&  fGlobalObservables->getSize()) {
      observables.remove(*fGlobalObservables);

      // generate one set of global observables and assign it
      RooDataSet *one = fPdf->generate(*fGlobalObservables, 1);
      const RooArgSet *values = one->get();
      RooArgSet *allVars = fPdf->getVariables();
      *allVars = *values;
      delete allVars;
      delete values;
      delete one;
   }

   RooAbsData* data = NULL;
   if(fNEvents == 0 )  {
      if( fPdf->canBeExtended() && fPdf->expectedEvents(observables) > 0) {
         if(fGenerateBinned) data = fPdf->generateBinned(observables, RooFit::Extended());
         else                data = fPdf->generate      (observables, RooFit::Extended());
      }
      else {
         oocoutE((TObject*)0,InputArguments) 
            << "ToyMCSampler: Error : pdf is not extended and number of events per toy is zero"
            << endl;
      }
   } else {
      if(fGenerateBinned) data = fPdf->generateBinned(observables, fNEvents);
      else                data = fPdf->generate      (observables, fNEvents);
   }

   return data;
}



} // end namespace RooStats
