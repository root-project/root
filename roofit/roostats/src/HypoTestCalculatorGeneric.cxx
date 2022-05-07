// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Sven Kreiss   23/05/10
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class RooStats::HypoTestCalculatorGeneric
    \ingroup Roostats

Common base class for the Hypothesis Test Calculators.
It is not designed to use directly but via its derived classes

Same purpose as HybridCalculatorOriginal, but different implementation.

This is the "generic" version that works with any TestStatSampler. The
HybridCalculator derives from this class but explicitly uses the
ToyMCSampler as its TestStatSampler.

*/

#include "RooStats/HypoTestCalculatorGeneric.h"
#include "RooStats/ToyMCSampler.h"
#include "RooStats/RatioOfProfiledLikelihoodsTestStat.h"
#include "RooStats/HypoTestCalculator.h"
#include "RooStats/HypoTestInverterResult.h"

#include "RooAddPdf.h"

#include "RooRandom.h"


ClassImp(RooStats::HypoTestCalculatorGeneric);

using namespace RooStats;
using namespace std;

////////////////////////////////////////////////////////////////////////////////
/// Constructor. When test stat sampler is not provided
/// uses ToyMCSampler and RatioOfProfiledLikelihoodsTestStat
/// and nToys = 1000.
/// User can : GetTestStatSampler()->SetNToys( # )

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
   fDefaultTestStat(0),
   fAltToysSeed(0)
{
   if(!sampler){
      fDefaultTestStat
         = new RatioOfProfiledLikelihoodsTestStat(*nullModel.GetPdf(),
                                                  *altModel.GetPdf(),
                                                  altModel.GetSnapshot());

      fDefaultSampler = new ToyMCSampler(*fDefaultTestStat, 1000);
      fTestStatSampler = fDefaultSampler;
   }


}

////////////////////////////////////////////////////////////////////////////////
/// common setup for both models

void HypoTestCalculatorGeneric::SetupSampler(const ModelConfig& model) const {
   fNullModel->LoadSnapshot();
   fTestStatSampler->SetObservables(*fNullModel->GetObservables());
   fTestStatSampler->SetParametersForTestStat(*fNullModel->GetParametersOfInterest());

   // for this model
   model.LoadSnapshot();
   fTestStatSampler->SetSamplingDistName(model.GetName());
   fTestStatSampler->SetPdf(*model.GetPdf());
   fTestStatSampler->SetNuisanceParameters(*model.GetNuisanceParameters());
   // global observables or nuisance pdf will be set by the derived classes
   // (e.g. Frequentist or HybridCalculator)
}

////////////////////////////////////////////////////////////////////////////////

HypoTestCalculatorGeneric::~HypoTestCalculatorGeneric()  {
   if(fDefaultSampler)    delete fDefaultSampler;
   if(fDefaultTestStat)   delete fDefaultTestStat;
}

////////////////////////////////////////////////////////////////////////////////
/// several possibilities:
/// no prior nuisance given and no nuisance parameters: ok
/// no prior nuisance given but nuisance parameters: error
/// prior nuisance given for some nuisance parameters:
///   - nuisance parameters are constant, so they don't float in test statistic
///   - nuisance parameters are floating, so they do float in test statistic

HypoTestResult* HypoTestCalculatorGeneric::GetHypoTest() const {

   // initial setup
   PreHook();
   const_cast<ModelConfig*>(fNullModel)->GuessObsAndNuisance(*fData);
   const_cast<ModelConfig*>(fAltModel)->GuessObsAndNuisance(*fData);

   const RooArgSet * nullSnapshot = fNullModel->GetSnapshot();
   if(nullSnapshot == NULL) {
      oocoutE(nullptr,Generation) << "Null model needs a snapshot. Set using modelconfig->SetSnapshot(poi)." << endl;
      return 0;
   }

   // CheckHook
   if(CheckHook() != 0) {
      oocoutE(nullptr,Generation) << "There was an error in CheckHook(). Stop." << endl;
      return 0;
   }

   if (!fTestStatSampler  || !fTestStatSampler->GetTestStatistic() ) {
      oocoutE(nullptr,InputArguments) << "Test Statistic Sampler or Test Statistics not defined. Stop." << endl;
      return 0;
   }

   // get a big list of all variables for convenient switching
   RooArgSet *nullParams = fNullModel->GetPdf()->getParameters(*fData);
   RooArgSet *altParams = fAltModel->GetPdf()->getParameters(*fData);
   // save all parameters so we can set them back to what they were
   RooArgSet *bothParams = fNullModel->GetPdf()->getParameters(*fData);
   bothParams->add(*altParams,false);
   RooArgSet *saveAll = (RooArgSet*) bothParams->snapshot();

   // check whether we have a ToyMCSampler and if so, keep a pointer to it
   ToyMCSampler* toymcs = dynamic_cast<ToyMCSampler*>( fTestStatSampler );


   // evaluate test statistic on data
   RooArgSet nullP(*nullSnapshot);
   double obsTestStat;

   RooArgList* allTS = NULL;
   if( toymcs ) {
      allTS = toymcs->EvaluateAllTestStatistics(*const_cast<RooAbsData*>(fData), nullP);
      if (!allTS) return 0;
      //oocoutP(nullptr,Generation) << "All Test Statistics on data: " << endl;
      //allTS->Print("v");
      RooRealVar* firstTS = (RooRealVar*)allTS->at(0);
      obsTestStat = firstTS->getVal();
      if (allTS->getSize()<=1) {
        delete allTS;
        allTS= 0;  // don't save
      }
   }else{
      obsTestStat = fTestStatSampler->EvaluateTestStatistic(*const_cast<RooAbsData*>(fData), nullP);
   }
   oocoutP(nullptr,Generation) << "Test Statistic on data: " << obsTestStat << endl;

   // set parameters back ... in case the evaluation of the test statistic
   // modified something (e.g. a nuisance parameter that is not randomized
   // must be set here)
   bothParams->assign(*saveAll);



   // Generate sampling distribution for null
   SetupSampler(*fNullModel);
   RooArgSet paramPointNull(*fNullModel->GetParametersOfInterest());
   if(PreNullHook(&paramPointNull, obsTestStat) != 0) {
      oocoutE(nullptr,Generation) << "PreNullHook did not return 0." << endl;
   }
   SamplingDistribution* samp_null = NULL;
   RooDataSet* detOut_null = NULL;
   if(toymcs) {
      detOut_null = toymcs->GetSamplingDistributions(paramPointNull);
      if( detOut_null ) {
        samp_null = new SamplingDistribution( detOut_null->GetName(), detOut_null->GetTitle(), *detOut_null );
        if (detOut_null->get()->getSize()<=1) {
          delete detOut_null;
          detOut_null= 0;
        }
      }
   }else samp_null = fTestStatSampler->GetSamplingDistribution(paramPointNull);

   // set parameters back
   bothParams->assign(*saveAll);

   // Generate sampling distribution for alternate
   SetupSampler(*fAltModel);
   RooArgSet paramPointAlt(*fAltModel->GetParametersOfInterest());
   if(PreAltHook(&paramPointAlt, obsTestStat) != 0) {
      oocoutE(nullptr,Generation) << "PreAltHook did not return 0." << endl;
   }
   SamplingDistribution* samp_alt = NULL;
   RooDataSet* detOut_alt = NULL;
   if(toymcs) {

      // case of re-using same toys for every points
      // set a given seed
      unsigned int prevSeed = 0;
      if (fAltToysSeed > 0) {
         prevSeed = RooRandom::integer(std::numeric_limits<unsigned int>::max()-1)+1;  // want to avoid zero value
         RooRandom::randomGenerator()->SetSeed(fAltToysSeed);
      }

      detOut_alt = toymcs->GetSamplingDistributions(paramPointAlt);
      if( detOut_alt ) {
        samp_alt = new SamplingDistribution( detOut_alt->GetName(), detOut_alt->GetTitle(), *detOut_alt );
        if (detOut_alt->get()->getSize()<=1) {
          delete detOut_alt;
          detOut_alt= 0;
        }
      }

      // restore the seed
      if (prevSeed > 0) {
         RooRandom::randomGenerator()->SetSeed(prevSeed);
      }

   }else samp_alt = fTestStatSampler->GetSamplingDistribution(paramPointAlt);


   // create result
   string resultname = "HypoTestCalculator_result";
   HypoTestResult* res = new HypoTestResult(resultname.c_str());
   res->SetPValueIsRightTail(fTestStatSampler->GetTestStatistic()->PValueIsRightTail());
   res->SetTestStatisticData(obsTestStat);
   res->SetAltDistribution(samp_alt);
   res->SetNullDistribution(samp_null);
   res->SetAltDetailedOutput( detOut_alt );
   res->SetNullDetailedOutput( detOut_null );
   res->SetAllTestStatisticsData( allTS );

   const RooArgSet *aset = GetFitInfo();
   if (aset != NULL) {
      RooDataSet *dset = new RooDataSet("", "", *aset);
      dset->add(*aset);
      res->SetFitInfo( dset );
   }

   bothParams->assign(*saveAll);
   delete allTS;
   delete bothParams;
   delete saveAll;
   delete altParams;
   delete nullParams;
   delete nullSnapshot;
   PostHook();
   return res;
}

////////////////////////////////////////////////////////////////////////////////
/// to re-use same toys for alternate hypothesis

void HypoTestCalculatorGeneric::UseSameAltToys()  {
   fAltToysSeed = RooRandom::integer(std::numeric_limits<unsigned int>::max()-1)+1;
}
