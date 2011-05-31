// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Sven Kreiss   23/05/10
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**
Same purpose as HybridCalculatorOriginal, but different implementation.

This is the "generic" version that works with any TestStatSampler. The
HybridCalculator derives from this class but explicitly uses the
ToyMCSampler as its TestStatSampler.
*/

#include "RooStats/HypoTestCalculatorGeneric.h"
#include "RooStats/ToyMCSampler.h"
#include "RooStats/RatioOfProfiledLikelihoodsTestStat.h"

#include "RooAddPdf.h"


ClassImp(RooStats::HypoTestCalculatorGeneric)

using namespace RooStats;


//___________________________________
HypoTestCalculatorGeneric::HypoTestCalculatorGeneric(
                                     const RooAbsData &data,
                                     const ModelConfig &altModel,
                                     const ModelConfig &nullModel,
                                     TestStatSampler *sampler
                                     ) :
   fAltModel(&altModel),
   fNullModel(&nullModel),
   fData(&data),
   fTestStatSampler(sampler),
   fDefaultSampler(0),
   fDefaultTestStat(0)
{
   // Constructor. When test stat sampler is not provided
   // uses ToyMCSampler and RatioOfProfiledLikelihoodsTestStat
   // and nToys = 1000.
   // User can : GetTestStatSampler()->SetNToys( # )
   if(!sampler){
      fDefaultTestStat
         = new RatioOfProfiledLikelihoodsTestStat(*nullModel.GetPdf(),
                                                  *altModel.GetPdf(),
                                                  altModel.GetSnapshot());

      fDefaultSampler = new ToyMCSampler(*fDefaultTestStat, 1000);
      fTestStatSampler = fDefaultSampler;
   }


}

//_____________________________________________________________
void HypoTestCalculatorGeneric::SetupSampler(const ModelConfig& model) const {
   // common setup for both models
   fNullModel->LoadSnapshot();
   fTestStatSampler->SetObservables(*fNullModel->GetObservables());
   fTestStatSampler->SetParametersForTestStat(*fNullModel->GetParametersOfInterest());

   // for this model
   model.LoadSnapshot();
   fTestStatSampler->SetSamplingDistName(model.GetName());
   fTestStatSampler->SetPdf(*model.GetPdf());
   fTestStatSampler->SetGlobalObservables(*model.GetGlobalObservables());
   fTestStatSampler->SetNuisanceParameters(*model.GetNuisanceParameters());
}

//____________________________________________________
HypoTestCalculatorGeneric::~HypoTestCalculatorGeneric()  {
   if(fDefaultSampler)    delete fDefaultSampler;
   if(fDefaultTestStat)   delete fDefaultTestStat;
}

//____________________________________________________
HypoTestResult* HypoTestCalculatorGeneric::GetHypoTest() const {

   // several possibilities:
   // no prior nuisance given and no nuisance parameters: ok
   // no prior nuisance given but nuisance parameters: error
   // prior nuisance given for some nuisance parameters:
   //   - nuisance parameters are constant, so they don't float in test statistic
   //   - nuisance parameters are floating, so they do float in test statistic

   // initial setup
   const_cast<ModelConfig*>(fNullModel)->GuessObsAndNuisance(*fData);
   const_cast<ModelConfig*>(fAltModel)->GuessObsAndNuisance(*fData);

   if(fNullModel->GetSnapshot() == NULL) {
      oocoutE((TObject*)0,Generation) << "Null model needs a snapshot. Set using modelconfig->SetSnapshot(poi)." << endl;
      return 0;
   }

   // CheckHook
   if(CheckHook() != 0) {
      oocoutE((TObject*)0,Generation) << "There was an error in CheckHook(). Stop." << endl;
      return 0;
   }

   // get a big list of all variables for convenient switching
   RooArgSet *nullParams = fNullModel->GetPdf()->getParameters(*fData);
   RooArgSet *altParams = fAltModel->GetPdf()->getParameters(*fData);
   // save all parameters so we can set them back to what they were
   RooArgSet *bothParams = fNullModel->GetPdf()->getParameters(*fData);
   bothParams->add(*altParams,false);
   RooArgSet *saveAll = (RooArgSet*) bothParams->snapshot();


   // evaluate test statistic on data
   RooArgSet nullP(*fNullModel->GetSnapshot());
   double obsTestStat = fTestStatSampler->EvaluateTestStatistic(*const_cast<RooAbsData*>(fData), nullP);
   oocoutP((TObject*)0,Generation) << "Test Statistic on data: " << obsTestStat << endl;

   // set parameters back ... in case the evaluation of the test statistic
   // modified something (e.g. a nuisance parameter that is not randomized
   // must be set here)
   *bothParams = *saveAll;

   // Generate sampling distribution for null
   SetupSampler(*fNullModel);
   RooArgSet paramPointNull(*fNullModel->GetParametersOfInterest());
   if(PreNullHook(&paramPointNull, obsTestStat) != 0) {
      oocoutE((TObject*)0,Generation) << "PreNullHook did not return 0." << endl;
   }
   SamplingDistribution* samp_null = fTestStatSampler->GetSamplingDistribution(paramPointNull);

   // set parameters back
   *bothParams = *saveAll;

   // Generate sampling distribution for alternate
   SetupSampler(*fAltModel);
   RooArgSet paramPointAlt(*fAltModel->GetParametersOfInterest());
   if(PreAltHook(&paramPointAlt, obsTestStat) != 0) {
      oocoutE((TObject*)0,Generation) << "PreAltHook did not return 0." << endl;
   }
   SamplingDistribution* samp_alt = fTestStatSampler->GetSamplingDistribution(paramPointAlt);


   // create result
   string resultname = "HypoTestCalculator_result";
   HypoTestResult* res = new HypoTestResult(resultname.c_str());
   res->SetPValueIsRightTail(fTestStatSampler->GetTestStatistic()->PValueIsRightTail());
   res->SetTestStatisticData(obsTestStat);
   res->SetAltDistribution(samp_alt);
   res->SetNullDistribution(samp_null);

   *bothParams = *saveAll;
   delete bothParams;
   delete saveAll;
   delete altParams;
   delete nullParams;

   return res;
}




