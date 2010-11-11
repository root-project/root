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
Same purpose as HybridCalculatorOld, but different implementation.
*/

#include "RooStats/HybridCalculator.h"
#include "RooStats/HybridPlot.h"

#include "RooStats/ToyMCSampler.h"
#include "RooStats/RatioOfProfiledLikelihoodsTestStat.h"

#include "RooAddPdf.h"


ClassImp(RooStats::HybridCalculator)

using namespace RooStats;


//___________________________________
HybridCalculator::HybridCalculator(
				     RooAbsData &data,
				     ModelConfig &altModel,
				     ModelConfig &nullModel,
				     TestStatSampler *sampler
				     ) :
   fAltModel(altModel),
   fNullModel(nullModel),
   fData(data),
   fPriorNuisanceNull(0),
   fPriorNuisanceAlt(0),
   fTestStatSampler(sampler),
   fDefaultSampler(0),
   fDefaultTestStat(0),
   fToysInTails(0.0),
   fNullImportanceDensity(NULL),
   fNullImportanceSnapshot(NULL)
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
void HybridCalculator::SetupSampler(ModelConfig& model) const {
   // common setup for both models
   fNullModel.LoadSnapshot();
   fTestStatSampler->SetObservables(*fNullModel.GetObservables());
   fTestStatSampler->SetParametersForTestStat(*fNullModel.GetParametersOfInterest());

   // for this model
   model.LoadSnapshot();
   fTestStatSampler->SetSamplingDistName(model.GetName());
   fTestStatSampler->SetPdf(*model.GetPdf());
   fTestStatSampler->SetGlobalObservables(*model.GetGlobalObservables());
   fTestStatSampler->SetNuisanceParameters(*model.GetNuisanceParameters());

   if( (&model == &fNullModel) && fPriorNuisanceNull){
     // Setup Priors for ad hoc Hybrid
     fTestStatSampler->SetPriorNuisance(fPriorNuisanceNull);
   } else if( (&model == &fAltModel) && fPriorNuisanceAlt){
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
HybridCalculator::~HybridCalculator()  {
  //  if(fPriorNuisanceNull) delete fPriorNuisanceNull;
  //  if(fPriorNuisanceAlt)  delete fPriorNuisanceAlt;
  if(fDefaultSampler)    delete fDefaultSampler;
  if(fDefaultTestStat)   delete fDefaultTestStat;

}

void HybridCalculator::SetAdaptiveLimits(Double_t obsTestStat, Bool_t forNull) const {
   // Configures the ToyMCSampler (if used) to use adaptive sampling and
   // keep going until the requested number of toys is reached in the
   // tails.

   ToyMCSampler *ts = dynamic_cast<ToyMCSampler*>(fTestStatSampler);
   if(ts) {
      if(( forNull &&  fTestStatSampler->GetTestStatistic()->PValueIsRightTail()) ||
         (!forNull && !fTestStatSampler->GetTestStatistic()->PValueIsRightTail())
      ) {
         ts->SetToysRightTail(fToysInTails, obsTestStat);
      }else{
         ts->SetToysLeftTail(fToysInTails, obsTestStat);
      }
   }
}


//____________________________________________________
HypoTestResult* HybridCalculator::GetHypoTest() const {

  // several possibilities:
  // no prior nuisance given and no nuisance parameters: ok
  // no prior nuisance given but nuisance parameters: error
  // prior nuisance given for some nuisance parameters:
  //   - nuisance parameters are constant, so they don't float in test statistic
  //   - nuisance parameters are floating, so they do float in test statistic

   fNullModel.GuessObsAndNuisance(fData);
   fAltModel.GuessObsAndNuisance(fData);

   if( (fNullModel.GetNuisanceParameters() 
	&& fNullModel.GetNuisanceParameters()->getSize()>0 
	&& !fPriorNuisanceNull)
     || (fAltModel.GetNuisanceParameters() 
	 && fAltModel.GetNuisanceParameters()->getSize()>0 
	 && !fPriorNuisanceAlt) 
       ){
     oocoutE((TObject*)0,InputArguments)  << "Must ForceNuisancePdf, inferring posterior from ModelConfig is not yet implemented" << endl;
     return 0;
   }

   if(   (!fNullModel.GetNuisanceParameters() && fPriorNuisanceNull)
      || (!fAltModel.GetNuisanceParameters()  && fPriorNuisanceAlt)
      || (fNullModel.GetNuisanceParameters()  && fNullModel.GetNuisanceParameters()->getSize()==0 && fPriorNuisanceNull)
       || (fAltModel.GetNuisanceParameters()  && fAltModel.GetNuisanceParameters()->getSize()>0   && !fPriorNuisanceAlt) 
       ){
     oocoutE((TObject*)0,InputArguments)  << "Nuisance PDF specified, but the pdf doesn't know which parameters are the nuisance parameters.  Must set nuisance parameters in the ModelConfig" << endl;
     return 0;
   }


   // get a big list of all variables for convenient switching
   RooArgSet *nullParams = fNullModel.GetPdf()->getParameters(fData);
   RooArgSet *altParams = fAltModel.GetPdf()->getParameters(fData);
   // save all parameters so we can set them back to what they were
   RooArgSet *bothParams = fNullModel.GetPdf()->getParameters(fData);
   bothParams->add(*altParams,false);
   RooArgSet *saveAll = (RooArgSet*) bothParams->snapshot();


   // evaluate test statistic on data
   RooArgSet nullP(*fNullModel.GetSnapshot());
   double obsTestStat = fTestStatSampler->EvaluateTestStatistic(fData, nullP);


   // Generate sampling distribution for null (use importance sampling of possible,
   // importance density is determined from fAltModel).
   SamplingDistribution* samp_null = NULL;
   if(fNullImportanceDensity)
      // given null importance density
      samp_null = GenerateSamplingDistribution(&fNullModel, obsTestStat, fNullImportanceDensity, fNullImportanceSnapshot);
   else {
      samp_null = GenerateSamplingDistribution(&fNullModel, obsTestStat);
   }

   // set parameters back
   *bothParams = *saveAll;
   // Generate sampling dist for alternate (no importance sampling as otherModel is not given)
   SamplingDistribution* samp_alt = GenerateSamplingDistribution(&fAltModel, obsTestStat);


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


SamplingDistribution* HybridCalculator::GenerateSamplingDistribution(
   ModelConfig *thisModel,
   double obsTestStat,
   RooAbsPdf *impDens,
   RooArgSet *impSnapshot
) const {
   // Generates Sampling Distribution for the given model (thisModel).
   // Also handles Importance sampling (for ToyMCSampler) if
   // otherModel is given.

   oocoutP((TObject*)0,Generation) << "Generate sampling distribution for " << thisModel->GetTitle() << endl;

   SamplingDistribution *result = NULL;
   RooArgSet *params = thisModel->GetPdf()->getParameters(fData);

   // Importance Sampling setup
   ToyMCSampler *ts = dynamic_cast<ToyMCSampler*>(fTestStatSampler);
   if(ts  &&  impDens) {
      oocoutI((TObject*)0,InputArguments) << "Importance Sampling" << endl;

      ts->SetImportanceDensity(impDens);
      if(impSnapshot)
         ts->SetImportanceSnapshot(*impSnapshot);
   }else{
      // deactivate importance sampling (might be set from previous run)
      ts->SetImportanceDensity(NULL);
   }

   SetupSampler(*thisModel);

   // Adaptive sampling
   if(fToysInTails) {
      // In case of ToyMCSampler, do a more efficient adaptive
      // sampling using functions built into ToyMCSampler.
      SetAdaptiveLimits(obsTestStat, kTRUE);

      // else, simply rerun fTestStatSampler until given number of toys
      // in tails is reached.
      result = fTestStatSampler->GetSamplingDistribution(*params);
      while(
         (((1. - result->CDF(obsTestStat))*result->GetSize() < fToysInTails)
            && fTestStatSampler->GetTestStatistic()->PValueIsRightTail())  ||
         ((result->CDF(obsTestStat)*result->GetSize() < fToysInTails)
            && !fTestStatSampler->GetTestStatistic()->PValueIsRightTail())
      ) {
         oocoutP((TObject*)NULL, Generation) << "Adaptive Sampling: rerun generation." << endl;
         SamplingDistribution *additional = fTestStatSampler->GetSamplingDistribution(*params);
         result->Add(additional);
         delete additional;
      }
   }else{
      result = fTestStatSampler->GetSamplingDistribution(*params);
   }


   return result;
}



void HybridCalculator::SetMaxToys(Double_t t) {
   // Set a maximum number of toys. To be used in combination with
   // adaptive sampling.

   ToyMCSampler *ts = dynamic_cast<ToyMCSampler*>(fTestStatSampler);
   if(ts) ts->SetMaxToys(t);
   else oocoutE((TObject*)NULL, InputArguments) << "Not supported for this sampler." << endl;
}





