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

#include "RooStats/HybridCalculatorGeneric.h"
#include "RooStats/HybridPlot.h"

#include "RooStats/ToyMCSampler.h"
#include "RooStats/RatioOfProfiledLikelihoodsTestStat.h"

#include "RooAddPdf.h"


ClassImp(RooStats::HybridCalculatorGeneric)

using namespace RooStats;


//___________________________________
HybridCalculatorGeneric::HybridCalculatorGeneric(
                                     const RooAbsData &data,
                                     const ModelConfig &altModel,
                                     const ModelConfig &nullModel,
                                     TestStatSampler *sampler
                                     ) :
   fAltModel(&altModel),
   fNullModel(&nullModel),
   fData(&data),
   fPriorNuisanceNull(0),
   fPriorNuisanceAlt(0),
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
void HybridCalculatorGeneric::SetupSampler(const ModelConfig& model) const {
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

   if( (&model == fNullModel) && fPriorNuisanceNull){
     // Setup Priors for ad hoc Hybrid
     fTestStatSampler->SetPriorNuisance(fPriorNuisanceNull);
   } else if( (&model == fAltModel) && fPriorNuisanceAlt){
     // Setup Priors for ad hoc Hybrid
     fTestStatSampler->SetPriorNuisance(fPriorNuisanceAlt);
   } else if(model.GetNuisanceParameters()==NULL ||
             model.GetNuisanceParameters()->getSize()==0){
     oocoutI((TObject*)0,InputArguments)
       << "No nuisance parameters specified and no prior forced, reduces to simple hypothesis testing with no uncertainty" << endl;
   } else{
     // TODO principled case:
     // must create posterior from Model.PriorPdf and Model.Pdf

     // Note, we do not want to use "prior" for nuisance parameters:
     // fTestStatSampler->SetPriorNuisance(const_cast<RooAbsPdf*>(model.GetPriorPdf()));

     oocoutE((TObject*)0,InputArguments)  << "infering posterior from ModelConfig is not yet implemented" << endl;
   }
}

//____________________________________________________
HybridCalculatorGeneric::~HybridCalculatorGeneric()  {
   //  if(fPriorNuisanceNull) delete fPriorNuisanceNull;
   //  if(fPriorNuisanceAlt)  delete fPriorNuisanceAlt;
   if(fDefaultSampler)    delete fDefaultSampler;
   if(fDefaultTestStat)   delete fDefaultTestStat;
}

//____________________________________________________
HypoTestResult* HybridCalculatorGeneric::GetHypoTest() const {

   // several possibilities:
   // no prior nuisance given and no nuisance parameters: ok
   // no prior nuisance given but nuisance parameters: error
   // prior nuisance given for some nuisance parameters:
   //   - nuisance parameters are constant, so they don't float in test statistic
   //   - nuisance parameters are floating, so they do float in test statistic

   // TODO skreiss: really necessary?
   const_cast<ModelConfig*>(fNullModel)->GuessObsAndNuisance(*fData);
   const_cast<ModelConfig*>(fAltModel)->GuessObsAndNuisance(*fData);

   if( (fNullModel->GetNuisanceParameters()
        && fNullModel->GetNuisanceParameters()->getSize()>0
        && !fPriorNuisanceNull)
     || (fAltModel->GetNuisanceParameters()
         && fAltModel->GetNuisanceParameters()->getSize()>0
         && !fPriorNuisanceAlt)
       ){
     oocoutE((TObject*)0,InputArguments)  << "Must ForceNuisancePdf, inferring posterior from ModelConfig is not yet implemented" << endl;
     return 0;
   }

   if(   (!fNullModel->GetNuisanceParameters() && fPriorNuisanceNull)
      || (!fAltModel->GetNuisanceParameters()  && fPriorNuisanceAlt)
      || (fNullModel->GetNuisanceParameters()  && fNullModel->GetNuisanceParameters()->getSize()==0 && fPriorNuisanceNull)
       || (fAltModel->GetNuisanceParameters()  && fAltModel->GetNuisanceParameters()->getSize()>0   && !fPriorNuisanceAlt)
       ){
     oocoutE((TObject*)0,InputArguments)  << "Nuisance PDF specified, but the pdf doesn't know which parameters are the nuisance parameters.  Must set nuisance parameters in the ModelConfig" << endl;
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

   // Generate sampling distribution for null
   SetupSampler(*fNullModel);
   if(PreNullHook(obsTestStat) != 0) {
      oocoutE((TObject*)0,Generation) << "PreNullHook did not return 0." << endl;
   }
   RooArgSet poiNull(*fNullModel->GetParametersOfInterest());
   SamplingDistribution* samp_null = fTestStatSampler->GetSamplingDistribution(poiNull);

   // set parameters back
   *bothParams = *saveAll;

   // Generate sampling distribution for alternate
   SetupSampler(*fAltModel);
   if(PreAltHook(obsTestStat) != 0) {
      oocoutE((TObject*)0,Generation) << "PreAltHook did not return 0." << endl;
   }
   RooArgSet poiAlt(*fAltModel->GetParametersOfInterest());
   SamplingDistribution* samp_alt = fTestStatSampler->GetSamplingDistribution(poiAlt);


   string resultname = "HybridCalculator_result";
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




