// @(#)root/roostats:$Id: FrequentistCalculator.cxx 37084 2010-11-29 21:37:13Z moneta $
// Author: Kyle Cranmer, Sven Kreiss   23/05/10
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "RooStats/FrequentistCalculator.h"
#include "RooStats/ToyMCSampler.h"
#include "RooStats/RooStatsUtils.h"
#include "RooStats/DetailedOutputAggregator.h"
#include "RooMinimizer.h"
#include "RooProfileLL.h"

/** \class RooStats::FrequentistCalculator
    \ingroup Roostats

Does a frequentist hypothesis test.

Hypothesis Test Calculator using a full frequentist procedure for sampling the
test statistic distribution.
The nuisance parameters are fixed to their MLEs.
The use of ToyMCSampler as the TestStatSampler is assumed.

*/

ClassImp(RooStats::FrequentistCalculator);

using namespace RooStats;
using std::endl;

////////////////////////////////////////////////////////////////////////////////

void FrequentistCalculator::PreHook() const {
   if (fFitInfo != nullptr) {
      delete fFitInfo;
      fFitInfo = nullptr;
   }
   if (fStoreFitInfo) {
      fFitInfo = new RooArgSet();
   }
}

////////////////////////////////////////////////////////////////////////////////

void FrequentistCalculator::PostHook() const {
}

////////////////////////////////////////////////////////////////////////////////

int FrequentistCalculator::PreNullHook(RooArgSet *parameterPoint, double obsTestStat) const {

   // ****** any TestStatSampler ********

   // create profile keeping everything but nuisance parameters fixed
   std::unique_ptr<RooArgSet> allParams{fNullModel->GetPdf()->getParameters(*fData)};
   RemoveConstantParameters(&*allParams);

   // note: making nll or profile class variables can only be done in the constructor
   // as all other hooks are const (which has to be because GetHypoTest is const). However,
   // when setting it only in constructor, they would have to be changed every time SetNullModel
   // or SetAltModel is called. Simply put, converting them into class variables breaks
   // encapsulation.

   bool doProfile = true;
   RooArgSet allButNuisance(*allParams);
   if( fNullModel->GetNuisanceParameters() ) {
      allButNuisance.remove(*fNullModel->GetNuisanceParameters());
      if( fConditionalMLEsNull ) {
         oocoutI(nullptr,InputArguments) << "Using given conditional MLEs for Null." << std::endl;
         allParams->assign(*fConditionalMLEsNull);
         // LM: fConditionalMLEsNull must be nuisance parameters otherwise an error message will be printed
         allButNuisance.add( *fConditionalMLEsNull );
         if (fNullModel->GetNuisanceParameters()) {
            RooArgSet remain(*fNullModel->GetNuisanceParameters());
            remain.remove(*fConditionalMLEsNull,true,true);
            if( remain.empty() ) doProfile = false;
         }
      }
   }else{
      doProfile = false;
   }
   if (doProfile) {
      oocoutI(nullptr,InputArguments) << "Profiling conditional MLEs for Null." << std::endl;
      RooFit::MsgLevel msglevel = RooMsgService::instance().globalKillBelow();
      RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL);

      RooArgSet conditionalObs;
      if (fNullModel->GetConditionalObservables()) conditionalObs.add(*fNullModel->GetConditionalObservables());
      RooArgSet globalObs;
      if (fNullModel->GetGlobalObservables()) globalObs.add(*fNullModel->GetGlobalObservables());

      auto& config = GetGlobalRooStatsConfig();
      std::unique_ptr<RooAbsReal> nll{fNullModel->GetPdf()->createNLL(*const_cast<RooAbsData*>(fData), RooFit::CloneData(false), RooFit::Constrain(*allParams),
                                                        RooFit::GlobalObservables(globalObs),
                                                        RooFit::ConditionalObservables(conditionalObs),
                                                        RooFit::Offset(config.useLikelihoodOffset))};
      std::unique_ptr<RooAbsArg> profileOwner{nll->createProfile(allButNuisance)};
      auto profile = dynamic_cast<RooProfileLL*>(profileOwner.get());
      // set minimier options
      profile->minimizer()->setPrintLevel(ROOT::Math::MinimizerOptions::DefaultPrintLevel()-1);
      profile->getVal(); // this will do fit and set nuisance parameters to profiled values

      // Hack to extract a RooFitResult
      if (fStoreFitInfo) {
         std::unique_ptr<RooFitResult> result {profile->minimizer()->save()};
         std::unique_ptr<RooArgSet> detOutput {DetailedOutputAggregator::GetAsArgSet(result.get(), "fitNull_")};
         fFitInfo->addOwned(*detOutput);
      }

      RooMsgService::instance().setGlobalKillBelow(msglevel);

      // set in test statistics conditional and global observables
      // (needed to get correct model likelihood)
      TestStatistic * testStatistic = nullptr;
      auto testStatSampler = GetTestStatSampler();
      if (testStatSampler) testStatistic = testStatSampler->GetTestStatistic();
      if (testStatistic) {
         testStatistic->SetConditionalObservables(conditionalObs);
         testStatistic->SetGlobalObservables(globalObs);
      }

   }

   // add nuisance parameters to parameter point
   if(fNullModel->GetNuisanceParameters())
      parameterPoint->add(*fNullModel->GetNuisanceParameters());


   // ***** ToyMCSampler specific *******

   // check whether TestStatSampler is a ToyMCSampler
   ToyMCSampler *toymcs = dynamic_cast<ToyMCSampler*>(GetTestStatSampler());
   if(toymcs) {
      oocoutI(nullptr,InputArguments) << "Using a ToyMCSampler. Now configuring for Null." << std::endl;

      // variable number of toys
      if(fNToysNull >= 0) toymcs->SetNToys(fNToysNull);

      // set the global observables to be generated by the ToyMCSampler
      toymcs->SetGlobalObservables(*fNullModel->GetGlobalObservables());

      // adaptive sampling
      if(fNToysNullTail) {
         oocoutI(nullptr,InputArguments) << "Adaptive Sampling" << std::endl;
         if(GetTestStatSampler()->GetTestStatistic()->PValueIsRightTail()) {
            toymcs->SetToysRightTail(fNToysNullTail, obsTestStat);
         }else{
            toymcs->SetToysLeftTail(fNToysNullTail, obsTestStat);
         }
      }else{
         toymcs->SetToysBothTails(0, 0, obsTestStat); // disable adaptive sampling
      }

      GetNullModel()->LoadSnapshot();
   }

   return 0;
}

////////////////////////////////////////////////////////////////////////////////

int FrequentistCalculator::PreAltHook(RooArgSet *parameterPoint, double obsTestStat) const {

   // ****** any TestStatSampler ********

   // create profile keeping everything but nuisance parameters fixed
   std::unique_ptr<RooArgSet> allParams{fAltModel->GetPdf()->getParameters(*fData)};
   RemoveConstantParameters(&*allParams);

   bool doProfile = true;
   RooArgSet allButNuisance(*allParams);
   if( fAltModel->GetNuisanceParameters() ) {
      allButNuisance.remove(*fAltModel->GetNuisanceParameters());
      if( fConditionalMLEsAlt ) {
         oocoutI(nullptr,InputArguments) << "Using given conditional MLEs for Alt." << std::endl;
         allParams->assign(*fConditionalMLEsAlt);
         // LM: fConditionalMLEsAlt must be nuisance parameters otherwise an error message will be printed
         allButNuisance.add( *fConditionalMLEsAlt );
         if (fAltModel->GetNuisanceParameters()) {
            RooArgSet remain(*fAltModel->GetNuisanceParameters());
            remain.remove(*fConditionalMLEsAlt,true,true);
            if( remain.empty() ) doProfile = false;
         }
      }
   }else{
      doProfile = false;
   }
   if (doProfile) {
      oocoutI(nullptr,InputArguments) << "Profiling conditional MLEs for Alt." << std::endl;
      RooFit::MsgLevel msglevel = RooMsgService::instance().globalKillBelow();
      RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL);

      RooArgSet conditionalObs;
      if (fAltModel->GetConditionalObservables()) conditionalObs.add(*fAltModel->GetConditionalObservables());
      RooArgSet globalObs;
      if (fAltModel->GetGlobalObservables()) globalObs.add(*fAltModel->GetGlobalObservables());

      const auto& config = GetGlobalRooStatsConfig();
      std::unique_ptr<RooAbsReal> nll{fAltModel->GetPdf()->createNLL(*const_cast<RooAbsData*>(fData), RooFit::CloneData(false), RooFit::Constrain(*allParams),
                                                       RooFit::GlobalObservables(globalObs),
                                                       RooFit::ConditionalObservables(conditionalObs),
                                                       RooFit::Offset(config.useLikelihoodOffset))};

      std::unique_ptr<RooAbsReal> profileOwner{nll->createProfile(allButNuisance)};
      auto profile = dynamic_cast<RooProfileLL*>(profileOwner.get());
      // set minimizer options
      profile->minimizer()->setPrintLevel(ROOT::Math::MinimizerOptions::DefaultPrintLevel()-1); // use -1 to make more silent
      profile->getVal(); // this will do fit and set nuisance parameters to profiled values

      // Hack to extract a RooFitResult
      if (fStoreFitInfo) {
         std::unique_ptr<RooFitResult> result {profile->minimizer()->save()};
         std::unique_ptr<RooArgSet> detOutput {DetailedOutputAggregator::GetAsArgSet(result.get(), "fitAlt_")};
         fFitInfo->addOwned(*detOutput);
      }

      RooMsgService::instance().setGlobalKillBelow(msglevel);

      // set in test statistics conditional and global observables
      // (needed to get correct model likelihood)
      TestStatistic * testStatistic = nullptr;
      auto testStatSampler = GetTestStatSampler();
      if (testStatSampler) testStatistic = testStatSampler->GetTestStatistic();
      if (testStatistic) {
         testStatistic->SetConditionalObservables(conditionalObs);
         testStatistic->SetGlobalObservables(globalObs);
      }

   }

   // add nuisance parameters to parameter point
   if(fAltModel->GetNuisanceParameters())
      parameterPoint->add(*fAltModel->GetNuisanceParameters());

   // ***** ToyMCSampler specific *******

   // check whether TestStatSampler is a ToyMCSampler
   ToyMCSampler *toymcs = dynamic_cast<ToyMCSampler*>(GetTestStatSampler());
   if(toymcs) {
      oocoutI(nullptr,InputArguments) << "Using a ToyMCSampler. Now configuring for Alt." << std::endl;

      // variable number of toys
      if(fNToysAlt >= 0) toymcs->SetNToys(fNToysAlt);

      // set the global observables to be generated by the ToyMCSampler
      toymcs->SetGlobalObservables(*fAltModel->GetGlobalObservables());

      // adaptive sampling
      if(fNToysAltTail) {
         oocoutI(nullptr,InputArguments) << "Adaptive Sampling" << std::endl;
         if(GetTestStatSampler()->GetTestStatistic()->PValueIsRightTail()) {
            toymcs->SetToysLeftTail(fNToysAltTail, obsTestStat);
         }else{
            toymcs->SetToysRightTail(fNToysAltTail, obsTestStat);
         }
      }else{
         toymcs->SetToysBothTails(0, 0, obsTestStat); // disable adaptive sampling
      }

   }

   return 0;
}
