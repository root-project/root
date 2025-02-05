// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Sven Kreiss   23/05/10
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class RooStats::AsymptoticCalculator
    \ingroup Roostats

Hypothesis Test Calculator based on the asymptotic formulae for the profile
likelihood ratio.

It performs hypothesis tests using the asymptotic formula for the profile likelihood, and
uses the Asimov data set to compute expected significances or limits.

See G. Cowan, K. Cranmer, E. Gross and O. Vitells: Asymptotic formulae for
likelihood- based tests of new physics. Eur. Phys. J., C71:1–19, 2011.
It provides methods to perform hypothesis tests using the likelihood function,
and computes the \f$p\f$-values for the null and the alternate hypothesis using the asymptotic
formulae for the profile likelihood ratio described in the given paper.

The calculator provides methods to produce the Asimov dataset, *i.e.* a dataset
generated where the observed values are equal to the expected ones.
The Asimov data set is then used to compute the observed asymptotic \f$p\f$-value for
the alternate hypothesis and the asymptotic expected \f$p\f$-values.

The asymptotic formulae are valid only for one POI (parameter of interest). So
the calculator works only for one-dimensional (one POI) models.
If more than one POI exists, only the first one is used.

The calculator can generate Asimov datasets from two kinds of PDFs:
- "Counting" distributions: RooPoisson, RooGaussian, or products of RooPoissons.
- Extended, *i.e.* number of events can be read off from extended likelihood term.
*/


#include "RooStats/AsymptoticCalculator.h"
#include "RooStats/ModelConfig.h"
#include "RooStats/ProfileLikelihoodTestStat.h"
#include "RooStats/RooStatsUtils.h"

#include "RooArgSet.h"
#include "RooArgList.h"
#include "RooProdPdf.h"
#include "RooSimultaneous.h"
#include "RooDataSet.h"
#include "RooCategory.h"
#include "RooRealVar.h"
#include "RooMinimizer.h"
#include "RooFitResult.h"
#include "Math/MinimizerOptions.h"
#include "RooPoisson.h"
#include "RooUniform.h"
#include "RooGamma.h"
#include "RooGaussian.h"
#include "RooMultiVarGaussian.h"
#include "RooBifurGauss.h"
#include "RooLognormal.h"
#include "RooDataHist.h"
#include <cmath>
#include <typeinfo>

#include "Math/BrentRootFinder.h"
#include "Math/WrappedFunction.h"

#include "TStopwatch.h"

#include <ROOT/RSpan.hxx>

using namespace RooStats;
using std::string, std::unique_ptr;


namespace {

/// Control print level  (0 minimal, 1 normal, 2 debug).
int &fgPrintLevel()
{

   static int val = 1;
   return val;
}

// Forward declaration.
double EvaluateNLL(RooStats::ModelConfig const &modelConfig, RooAbsData &data, const RooArgSet *poiSet = nullptr);

} // namespace

////////////////////////////////////////////////////////////////////////////////
/// set print level (static function)
///
///  - 0 minimal,
///  - 1 normal,
///  - 2 debug

void AsymptoticCalculator::SetPrintLevel(int level) {
   fgPrintLevel() = level;
}

////////////////////////////////////////////////////////////////////////////////
/// constructor for asymptotic calculator from Data set  and ModelConfig

AsymptoticCalculator::AsymptoticCalculator(
   RooAbsData &data,
   const ModelConfig &altModel,
   const ModelConfig &nullModel, bool nominalAsimov) :
      HypoTestCalculatorGeneric(data, altModel, nullModel, nullptr),
      fOneSided(false), fOneSidedDiscovery(false), fNominalAsimov(nominalAsimov),
      fUseQTilde(-1),
      fNLLObs(0), fNLLAsimov(0),
      fAsimovData(nullptr)
{
   if (!Initialize()) return;

   int verbose = fgPrintLevel();
   // try to guess default configuration
   // (this part should be only in constructor because the null snapshot might change during HypoTestInversion
   const RooArgSet * nullSnapshot = GetNullModel()->GetSnapshot();
   assert(nullSnapshot);
   RooRealVar * muNull  = dynamic_cast<RooRealVar*>(nullSnapshot->first() );
   assert(muNull);
   if (muNull->getVal() == muNull->getMin()) {
      fOneSidedDiscovery = true;
      if (verbose > 0)
         oocoutI(nullptr,InputArguments) << "AsymptotiCalculator: Minimum of POI is " << muNull->getMin() << " corresponds to null  snapshot   - default configuration is  one-sided discovery formulae  " << std::endl;
   }

}

////////////////////////////////////////////////////////////////////////////////
/// Initialize the calculator
/// The initialization will perform a global fit of the model to the data
/// and build an Asimov data set.
/// It will then also fit the model to the Asimov data set to find the likelihood value
/// of the Asimov data set
/// nominalAsimov is an option for using Asimov data set obtained using nominal nuisance parameter values
/// By default the nuisance parameters are fitted to the data
/// NOTE: If a fit has been done before, one for speeding up could set all the initial parameters
/// to the fit value and in addition set the null snapshot to the best fit

bool AsymptoticCalculator::Initialize() const {

   int verbose = fgPrintLevel();
   if (verbose >= 0)
      oocoutP(nullptr,Eval) << "AsymptoticCalculator::Initialize...." << std::endl;


   RooAbsPdf * nullPdf = GetNullModel()->GetPdf();
   if (!nullPdf) {
      oocoutE(nullptr,InputArguments) << "AsymptoticCalculator::Initialize - ModelConfig has not a pdf defined" << std::endl;
      return false;
   }
   RooAbsData * obsData = const_cast<RooAbsData *>(GetData() );
   if (!obsData ) {
      oocoutE(nullptr,InputArguments) << "AsymptoticCalculator::Initialize - data set has not been defined" << std::endl;
      return false;
   }
   RooAbsData & data = *obsData;



   const RooArgSet * poi = GetNullModel()->GetParametersOfInterest();
   if (!poi || poi->empty()) {
      oocoutE(nullptr,InputArguments) << "AsymptoticCalculator::Initialize -  ModelConfig has not POI defined." << std::endl;
      return false;
   }
   if (poi->size() > 1) {
      oocoutW(nullptr,InputArguments) << "AsymptoticCalculator::Initialize - ModelConfig has more than one POI defined \n\t"
                                          << "The asymptotic calculator works for only one POI - consider as POI only the first parameter"
                                          << std::endl;
   }


   // This will set the poi value to the null snapshot value in the ModelConfig
   const RooArgSet * nullSnapshot = GetNullModel()->GetSnapshot();
   if(nullSnapshot == nullptr || nullSnapshot->empty()) {
      oocoutE(nullptr,InputArguments) << "AsymptoticCalculator::Initialize - Null model needs a snapshot. Set using modelconfig->SetSnapshot(poi)." << std::endl;
      return false;
   }

   // GetNullModel()->Print();
   // std::cout << "ASymptotic calc: null snapshot\n";
   // nullSnapshot->Print("v");
   // std::cout << "PDF  variables ";
   // nullPdf->getVariables()->Print("v");

   // keep snapshot for the initial parameter values (need for nominal Asimov)
   RooArgSet nominalParams;
   std::unique_ptr<RooArgSet> allParams{nullPdf->getParameters(data)};
   RemoveConstantParameters(&*allParams);
   if (fNominalAsimov) {
      allParams->snapshot(nominalParams);
   }
   fBestFitPoi.removeAll();
   fBestFitParams.removeAll();
   fAsimovGlobObs.removeAll();

   // evaluate the unconditional nll for the full model on the  observed data
   if (verbose >= 0)
      oocoutP(nullptr,Eval) << "AsymptoticCalculator::Initialize - Find  best unconditional NLL on observed data" << std::endl;
   fNLLObs = EvaluateNLL(*GetNullModel(), data);
   // fill also snapshot of best poi
   poi->snapshot(fBestFitPoi);
   RooRealVar * muBest = dynamic_cast<RooRealVar*>(fBestFitPoi.first());
   assert(muBest);
   if (verbose >= 0)
      oocoutP(nullptr,Eval) << "Best fitted POI value = " << muBest->getVal() << " +/- " << muBest->getError() << std::endl;
   // keep snapshot of all best fit parameters
   allParams->snapshot(fBestFitParams);

   // compute Asimov data set for the background (alt poi ) value
   const RooArgSet * altSnapshot = GetAlternateModel()->GetSnapshot();
   if(altSnapshot == nullptr || altSnapshot->empty()) {
      oocoutE(nullptr,InputArguments) << "Alt (Background)  model needs a snapshot. Set using modelconfig->SetSnapshot(poi)." << std::endl;
      return false;
   }

   RooArgSet poiAlt(*altSnapshot);  // this is the poi snapshot of B (i.e. for mu=0)

   oocoutP(nullptr,Eval) << "AsymptoticCalculator: Building Asimov data Set" << std::endl;

   // check that in case of binned models the n number of bins of the observables are consistent
   // with the number of bins  in the observed data
   // This number will be used for making the Asimov data set so it will be more consistent with the
   // observed data
   int prevBins = 0;
   RooRealVar * xobs = nullptr;
   if (GetNullModel()->GetObservables() && GetNullModel()->GetObservables()->size() == 1 ) {
      xobs = static_cast<RooRealVar*>((GetNullModel()->GetObservables())->first());
      if (data.IsA() == RooDataHist::Class() ) {
         if (data.numEntries() != xobs->getBins() ) {
            prevBins = xobs->getBins();
            oocoutW(nullptr,InputArguments) << "AsymptoticCalculator: number of bins in " << xobs->GetName() << " are different than data bins "
                                                << " set the same data bins " << data.numEntries() << " in range "
                                                << " [ " << xobs->getMin() << " , " << xobs->getMax() << " ]" << std::endl;
            xobs->setBins(data.numEntries());
         }
      }
   }

   if (!fNominalAsimov) {
      if (verbose >= 0)
         oocoutI(nullptr,InputArguments) << "AsymptoticCalculator: Asimov data will be generated using fitted nuisance parameter values" << std::endl;
      RooArgSet * tmp = (RooArgSet*) poiAlt.snapshot();
      fAsimovData = MakeAsimovData( data, *GetNullModel(), poiAlt, fAsimovGlobObs,tmp);
   }

   else {
      // assume use current value of nuisance as nominal ones
      if (verbose >= 0)
         oocoutI(nullptr,InputArguments) << "AsymptoticCalculator: Asimovdata set will be generated using nominal (current) nuisance parameter values" << std::endl;
      nominalParams.assign(poiAlt); // set poi to alt value but keep nuisance at the nominal one
      fAsimovData = MakeAsimovData( *GetNullModel(), nominalParams, fAsimovGlobObs);
   }

   if (!fAsimovData) {
      oocoutE(nullptr,InputArguments) << "AsymptoticCalculator: Error : Asimov data set could not be generated " << std::endl;
      return false;
   }

   // set global observables to their Asimov values
   RooArgSet globObs;
   RooArgSet globObsSnapshot;
   if (GetNullModel()->GetGlobalObservables()  ) {
      globObs.add(*GetNullModel()->GetGlobalObservables());
      assert(globObs.size() == fAsimovGlobObs.size() );
      // store previous snapshot value
      globObs.snapshot(globObsSnapshot);
      globObs.assign(fAsimovGlobObs);
   }


   // evaluate  the likelihood. Since we use on Asimov data , conditional and unconditional values should be the same
   // do conditional fit since is faster

   RooRealVar * muAlt = static_cast<RooRealVar*>(poiAlt.first());
   assert(muAlt);
   if (verbose >= 0) {
      oocoutP(nullptr, Eval)
         << "AsymptoticCalculator::Initialize Find  best conditional NLL on ASIMOV data set for given alt POI ( "
         << muAlt->GetName() << " ) = " << muAlt->getVal() << std::endl;
   }

   fNLLAsimov =  EvaluateNLL(*GetNullModel(), *fAsimovData, &poiAlt );
   // for unconditional fit
   //fNLLAsimov =  EvaluateNLL( *nullPdf, *fAsimovData);
   //poi->Print("v");

   // restore previous value
   globObs.assign(globObsSnapshot);

   // restore number of bins
   if (prevBins > 0 && xobs) xobs->setBins(prevBins);

   fIsInitialized = true;
   return true;
}

namespace {

double EvaluateNLL(RooStats::ModelConfig const& modelConfig, RooAbsData& data, const RooArgSet *poiSet)
{
    int verbose = fgPrintLevel();

    RooAbsPdf &pdf = *modelConfig.GetPdf();

    RooFit::MsgLevel msglevel = RooMsgService::instance().globalKillBelow();
    if (verbose < 2) RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL);


    std::unique_ptr<RooArgSet> allParams{pdf.getParameters(data)};
    RooStats::RemoveConstantParameters(&*allParams);
    // add constraint terms for all non-constant parameters

    // need to call constrain for RooSimultaneous until stripDisconnected problem fixed
    auto& config = GetGlobalRooStatsConfig();
    std::unique_ptr<RooAbsReal> nll{modelConfig.createNLL(data, RooFit::Constrain(*allParams), RooFit::Offset(config.useLikelihoodOffset))};

    std::unique_ptr<RooArgSet> attachedSet{nll->getVariables()};

    // if poi are specified - do a conditional fit
    RooArgSet paramsSetConstant;
    // support now only one POI
    if (poiSet && !poiSet->empty()) {
       RooRealVar * muTest = static_cast<RooRealVar*> (poiSet->first());
       RooRealVar * poiVar = dynamic_cast<RooRealVar*>(attachedSet->find( muTest->GetName() ) );
       if (poiVar && !poiVar->isConstant() ) {
          poiVar->setVal(  muTest->getVal() );
          poiVar->setConstant();
          paramsSetConstant.add(*poiVar);
       }
       if (poiSet->size() > 1)
          std::cout << "Model with more than one POI are not supported - ignore extra parameters, consider only first one" << std::endl;



       // This for more than one POI (not yet supported)
       //
       // RooLinkedListIter it = poiSet->iterator();
       // RooRealVar* tmpPar = nullptr, *tmpParA=nullptr;
       // while((tmpPar = (RooRealVar*)it.Next())){
       //    tmpParA =  ((RooRealVar*)attachedSet->find(tmpPar->GetName()));
       //    tmpParA->setVal( tmpPar->getVal() );
       //    if (!tmpParA->isConstant() ) {
       //       tmpParA->setConstant();
       //       paramsSetConstant.add(*tmpParA);
       //    }
       // }

       // check if there are non-const parameters so it is worth to do the minimization

    }

    TStopwatch tw;
    tw.Start();
    double val =  -1;

    //check if needed to skip the fit
    RooArgSet nllParams(*attachedSet);
    RooStats::RemoveConstantParameters(&nllParams);
    bool skipFit = (nllParams.empty());

    if (skipFit) {
       val = nll->getVal(); // just evaluate nll in conditional fits with model without nuisance params
    } else {

       int minimPrintLevel = verbose;

       RooMinimizer minim(*nll);
       int strategy = ROOT::Math::MinimizerOptions::DefaultStrategy();
       minim.setStrategy( strategy);
       minim.setEvalErrorWall(config.useEvalErrorWall);
       // use tolerance - but never smaller than 1 (default in RooMinimizer)
       double tol =  ROOT::Math::MinimizerOptions::DefaultTolerance();
       tol = std::max(tol,1.0); // 1.0 is the minimum value used in RooMinimizer
       minim.setEps( tol );
       //LM: RooMinimizer.setPrintLevel has +1 offset - so subtract  here -1
       minim.setPrintLevel(minimPrintLevel-1);
       int status = -1;
       minim.optimizeConst(2);
       TString minimizer = ""; // empty string to take RooMinimizer default initially
       TString algorithm = ROOT::Math::MinimizerOptions::DefaultMinimizerAlgo();

       if (verbose > 0) {
          std::cout << "AsymptoticCalculator::EvaluateNLL  ........ using " << minimizer << " / " << algorithm
                    << " with strategy  " << strategy << " and tolerance " << tol << std::endl;
       }

       for (int tries = 1, maxtries = 4; tries <= maxtries; ++tries) {
          //    status = minim.minimize(fMinimizer, ROOT::Math::MinimizerOptions::DefaultMinimizerAlgo().c_str());
          status = minim.minimize(minimizer, algorithm);
          // RooMinimizer::minimize returns -1  when the fit fails
          if (status >= 0) {
             break;
          } else {
             if (tries == 1) {
                std::cout << "    ----> Doing a re-scan first\n";
                minim.minimize(minimizer,"Scan");
             }
             if (tries == 2) {
                if (ROOT::Math::MinimizerOptions::DefaultStrategy() == 0 ) {
                   std::cout << "    ----> trying with strategy = 1\n";
                   minim.setStrategy(1);
                }
                else
                   tries++; // skip this trial if strategy is already 1
             }
             if (tries == 3) {
                std::cout << "    ----> trying with improve\n";
                minimizer = "Minuit";
                algorithm = "migradimproved";
             }
          }
       }

       std::unique_ptr<RooFitResult> result;

       // ignore errors in Hesse or in Improve and also when matrix was made pos def (status returned = 1)
       if (status >= 0) {
          result = std::unique_ptr<RooFitResult>{minim.save()};
       }
       if (result){
          if (RooStats::NLLOffsetMode() != "initial") {
             val = result->minNll();
          } else {
             bool previous = RooAbsReal::hideOffset();
             RooAbsReal::setHideOffset(true) ;
             val = nll->getVal();
             if (!previous)  RooAbsReal::setHideOffset(false) ;
          }

       }
       else {
          oocoutE(nullptr,Fitting) << "FIT FAILED !- return a NaN NLL " << std::endl;
          val =  TMath::QuietNaN();
       }

       minim.optimizeConst(false);
    }

    double muTest = 0;
    if (verbose > 0) {
       std::cout << "AsymptoticCalculator::EvaluateNLL -  value = " << val;
       if (poiSet) {
          muTest = ( static_cast<RooRealVar*>(poiSet->first()) )->getVal();
          std::cout << " for poi fixed at = " << muTest;
       }
       if (!skipFit) {
          std::cout << "\tfit time : ";
          tw.Print();
       } else {
          std::cout << std::endl;
       }
    }

    // reset the parameter free which where set as constant
    if (poiSet && !paramsSetConstant.empty()) SetAllConstant(paramsSetConstant,false);


    if (verbose < 2) RooMsgService::instance().setGlobalKillBelow(msglevel);

    return val;
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
/// It performs an hypothesis tests using the likelihood function
/// and computes the p values for the null and the alternate using the asymptotic
/// formulae for the profile likelihood ratio.
/// See G. Cowan, K. Cranmer, E. Gross and O. Vitells.
/// Asymptotic formulae for likelihood- based tests of new physics. Eur. Phys. J., C71:1–19, 2011.
/// The formulae are valid only for one POI. If more than one POI exists consider as POI only the
/// first one

HypoTestResult* AsymptoticCalculator::GetHypoTest() const {
   int verbose = fgPrintLevel();

   // re-initialized the calculator in case it is needed (pdf or data modified)
   if (!fIsInitialized) {
      if (!Initialize() ) {
         oocoutE(nullptr,InputArguments) << "AsymptoticCalculator::GetHypoTest - Error initializing Asymptotic calculator - return nullptr result " << std::endl;
         return nullptr;
      }
   }

   if (!fAsimovData) {
       oocoutE(nullptr,InputArguments) << "AsymptoticCalculator::GetHypoTest - Asimov data set has not been generated - return nullptr result " << std::endl;
       return nullptr;
   }

   assert(GetNullModel() );
   assert(GetData() );

   RooAbsPdf * nullPdf = GetNullModel()->GetPdf();
   assert(nullPdf);

   // make conditional fit on null snapshot of poi

   const RooArgSet * nullSnapshot = GetNullModel()->GetSnapshot();
   assert(nullSnapshot && !nullSnapshot->empty());

   // use as POI the nullSnapshot
   // if more than one POI exists, consider only the first one
   RooArgSet poiTest(*nullSnapshot);

   if (poiTest.size() > 1)  {
      oocoutW(nullptr,InputArguments) << "AsymptoticCalculator::GetHypoTest: snapshot has more than one POI - assume as POI first parameter " << std::endl;
   }

   std::unique_ptr<RooArgSet> allParams{nullPdf->getParameters(*GetData() )};
   allParams->assign(fBestFitParams);

   // set the one-side condition
   // (this works when we have only one params of interest
   RooRealVar * muHat =  dynamic_cast<RooRealVar*> (  fBestFitPoi.first() );
   assert(muHat && "no best fit parameter defined");
   RooRealVar * muTest = dynamic_cast<RooRealVar*> ( nullSnapshot->find(muHat->GetName() ) );
   assert(muTest && "poi snapshot is not existing");



   if (verbose> 0) {
      std::cout << std::endl;
      oocoutI(nullptr,Eval) << "AsymptoticCalculator::GetHypoTest: - perform  an hypothesis test for  POI ( " << muTest->GetName() << " ) = " << muTest->getVal() << std::endl;
      oocoutP(nullptr,Eval) << "AsymptoticCalculator::GetHypoTest -  Find  best conditional NLL on OBSERVED data set ..... " << std::endl;
   }

   // evaluate the conditional NLL on the observed data for the snapshot value
   double condNLL = EvaluateNLL(*GetNullModel(), const_cast<RooAbsData&>(*GetData()), &poiTest);

   double qmu = 2.*(condNLL - fNLLObs);



   if (verbose > 0)
      oocoutP(nullptr,Eval) << "\t OBSERVED DATA :  qmu   = " << qmu << " condNLL = " << condNLL << " uncond " << fNLLObs << std::endl;


   // this tolerance is used to avoid having negative qmu due to numerical errors
   double tol = 2.E-3 * std::max(1.,ROOT::Math::MinimizerOptions::DefaultTolerance());
   if (qmu < -tol || TMath::IsNaN(fNLLObs) ) {

      if (qmu < 0) {
         oocoutW(nullptr,Minimization) << "AsymptoticCalculator:  Found a negative value of the qmu - retry to do the unconditional fit "
                                           << std::endl;
      } else {
         oocoutW(nullptr, Minimization)
            << "AsymptoticCalculator:  unconditional fit failed before - retry to do it now " << std::endl;
      }

      double nll = EvaluateNLL(*GetNullModel(), const_cast<RooAbsData&>(*GetData()));

      if (nll < fNLLObs || (TMath::IsNaN(fNLLObs) && !TMath::IsNaN(nll) ) ) {
         oocoutW(nullptr,Minimization) << "AsymptoticCalculator:  Found a better unconditional minimum "
                                           << " old NLL = " << fNLLObs << " old muHat " << muHat->getVal() << std::endl;

         // update values
         fNLLObs = nll;
         const RooArgSet * poi = GetNullModel()->GetParametersOfInterest();
         assert(poi);
         fBestFitPoi.removeAll();
         poi->snapshot(fBestFitPoi);
         // restore also muHad since previous pointer has been deleted
         muHat =  dynamic_cast<RooRealVar*> (  fBestFitPoi.first() );
         assert(muHat);

        oocoutW(nullptr,Minimization) << "AsymptoticCalculator:  New minimum  found for                       "
                                          << "    NLL = " << fNLLObs << "    muHat  " << muHat->getVal() << std::endl;


        qmu = 2.*(condNLL - fNLLObs);

        if (verbose > 0)
           oocoutP(nullptr,Eval) << "After unconditional refit,  new qmu value is " << qmu << std::endl;

      }
   }

   if (qmu < -tol ) {
      oocoutE(nullptr,Minimization) << "AsymptoticCalculator:  qmu is still < 0  for mu = "
                                        <<  muTest->getVal() << " return a dummy result "
                                        << std::endl;
      return new HypoTestResult();
   }
   if (TMath::IsNaN(qmu) ) {
      oocoutE(nullptr,Minimization) << "AsymptoticCalculator:  failure in fitting for qmu or qmuA "
                                        <<  muTest->getVal() << " return a dummy result "
                                        << std::endl;
      return new HypoTestResult();
   }





   // compute conditional ML on Asimov data set
   // (need to const cast because it uses fitTo which is a non const method
   // RooArgSet asimovGlobObs;
   // RooAbsData * asimovData = (const_cast<AsymptoticCalculator*>(this))->MakeAsimovData( poi, asimovGlobObs);
   // set global observables to their Asimov values
   RooArgSet globObs;
   RooArgSet globObsSnapshot;
   if (GetNullModel()->GetGlobalObservables()  ) {
      globObs.add(*GetNullModel()->GetGlobalObservables());
      // store previous snapshot value
      globObs.snapshot(globObsSnapshot);
      globObs.assign(fAsimovGlobObs);
   }


   if (verbose > 0) oocoutP(nullptr,Eval) << "AsymptoticCalculator::GetHypoTest -- Find  best conditional NLL on ASIMOV data set .... " << std::endl;

   double condNLL_A = EvaluateNLL(*GetNullModel(), *fAsimovData, &poiTest);


   double qmu_A = 2.*(condNLL_A - fNLLAsimov  );

   if (verbose > 0)
      oocoutP(nullptr,Eval) << "\t ASIMOV data qmu_A = " << qmu_A << " condNLL = " << condNLL_A << " uncond " << fNLLAsimov << std::endl;

   if (qmu_A < -tol || TMath::IsNaN(fNLLAsimov) ) {

      if (qmu_A < 0) {
        oocoutW(nullptr, Minimization)
           << "AsymptoticCalculator:  Found a negative value of the qmu Asimov- retry to do the unconditional fit "
           << std::endl;
      } else {
        oocoutW(nullptr, Minimization)
           << "AsymptoticCalculator:  Fit failed for  unconditional the qmu Asimov- retry  unconditional fit "
           << std::endl;
      }

      double nll = EvaluateNLL(*GetNullModel(), *fAsimovData);

      if (nll < fNLLAsimov || (TMath::IsNaN(fNLLAsimov) && !TMath::IsNaN(nll) )) {
         oocoutW(nullptr,Minimization) << "AsymptoticCalculator:  Found a better unconditional minimum for Asimov data set"
                                           << " old NLL = " << fNLLAsimov << std::endl;

         // update values
         fNLLAsimov = nll;

         oocoutW(nullptr,Minimization) << "AsymptoticCalculator:  New minimum  found for                       "
                                           << "    NLL = " << fNLLAsimov << std::endl;
         qmu_A = 2.*(condNLL_A - fNLLAsimov);

        if (verbose > 0)
           oocoutP(nullptr,Eval) << "After unconditional Asimov refit,  new qmu_A value is " << qmu_A << std::endl;

      }
   }

   if (qmu_A < - tol) {
      oocoutE(nullptr,Minimization) << "AsymptoticCalculator:  qmu_A is still < 0  for mu = "
                                        <<  muTest->getVal() << " return a dummy result "
                                        << std::endl;
      return new HypoTestResult();
   }
   if (TMath::IsNaN(qmu) ) {
      oocoutE(nullptr,Minimization) << "AsymptoticCalculator:  failure in fitting for qmu or qmuA "
                                        <<  muTest->getVal() << " return a dummy result "
                                        << std::endl;
      return new HypoTestResult();
   }


   // restore previous value of global observables
   globObs.assign(globObsSnapshot);

   // now we compute p-values using the asymptotic formulae
   // described in the paper
   //  Cowan et al, Eur.Phys.J. C (2011) 71:1554

   // first try to guess automatically if needed to use qtilde (or ttilde in case of two sided)
   // if explicitly fUseQTilde this was not set
   // qtilde is in this case used if poi is bounded at the value of the alt hypothesis
   //  for Qtilde (need to distinguish case when qmu > qmuA = mu^2/ sigma^2)
   // (see Cowan et al, Eur.Phys.J. C(2011) 71:1554 paper equations 64 and 65
   // (remember qmu_A = mu^2/sigma^2 )
   bool useQTilde = false;
   // default case (check if poi is limited or not to a zero value)
   if (!fOneSidedDiscovery) { // qtilde is not a discovery test
      if (fUseQTilde == -1 && !fOneSidedDiscovery) {
         // alternate snapshot is value for which background is zero (for limits)
         RooRealVar * muAlt = dynamic_cast<RooRealVar*>(GetAlternateModel()->GetSnapshot()->first() );
         // null snapshot is value for which background is zero (for discovery)
         //RooRealVar * muNull = dynamic_cast<RooRealVar*>(GetNullModel()->GetSnapshot()->first() );
         assert(muAlt != nullptr );
         if (muTest->getMin() == muAlt->getVal()   ) {
            fUseQTilde = 1;
            oocoutI(nullptr,InputArguments) << "Minimum of POI is " << muTest->getMin() << " corresponds to alt  snapshot   - using qtilde asymptotic formulae  " << std::endl;
         } else {
            fUseQTilde = 0;
            oocoutI(nullptr,InputArguments) << "Minimum of POI is " << muTest->getMin() << " is different to alt snapshot " << muAlt->getVal()
                                                << " - using standard q asymptotic formulae  " << std::endl;
         }
      }
      useQTilde = fUseQTilde;
   }


   //check for one side condition (remember this is valid only for one poi)
   if (fOneSided ) {
      if ( muHat->getVal() > muTest->getVal() ) {
         oocoutI(nullptr,Eval) << "Using one-sided qmu - setting qmu to zero  muHat = " << muHat->getVal()
                                   << " muTest = " << muTest->getVal() << std::endl;
         qmu = 0;
      }
   }
   if (fOneSidedDiscovery ) {
      if ( muHat->getVal() < muTest->getVal() ) {
         oocoutI(nullptr,Eval) << "Using one-sided discovery qmu - setting qmu to zero  muHat = " << muHat->getVal()
                                   << " muTest = " << muTest->getVal() << std::endl;
         qmu = 0;
      }
   }

   // fix for negative qmu values due to numerical errors
   if (qmu < 0 && qmu > -tol) qmu = 0;
   if (qmu_A < 0 && qmu_A > -tol) qmu_A = 0;

   // asymptotic formula for pnull and from  paper Eur.Phys.J C 2011  71:1554
   // we have 4 different cases:
   //          t(mu), t_tilde(mu) for the 2-sided
   //          q(mu) and q_tilde(mu) for the one -sided test statistics

   double pnull = -1;
   double palt = -1;

   // asymptotic formula for pnull (for only one POI)
   // From fact that qmu is a chi2 with ndf=1

   double sqrtqmu = (qmu > 0) ? std::sqrt(qmu) : 0;
   double sqrtqmu_A = (qmu_A > 0) ? std::sqrt(qmu_A) : 0;


   if (fOneSided || fOneSidedDiscovery) {
      // for one-sided PL (q_mu : equations 56,57)
      if (verbose>2) {
         if (fOneSided) {
            oocoutI(nullptr,Eval) << "Using one-sided limit asymptotic formula (qmu)" << std::endl;
         } else {
            oocoutI(nullptr, Eval) << "Using one-sided discovery asymptotic formula (q0)" << std::endl;
         }
      }
      pnull = ROOT::Math::normal_cdf_c( sqrtqmu, 1.);
      palt = ROOT::Math::normal_cdf( sqrtqmu_A - sqrtqmu, 1.);
   }
   else  {
      // for 2-sided PL (t_mu : equations 35,36 in asymptotic paper)
      if (verbose > 2) oocoutI(nullptr,Eval) << "Using two-sided asymptotic  formula (tmu)" << std::endl;
      pnull = 2.*ROOT::Math::normal_cdf_c( sqrtqmu, 1.);
      palt = ROOT::Math::normal_cdf_c( sqrtqmu + sqrtqmu_A, 1.) +
         ROOT::Math::normal_cdf_c( sqrtqmu - sqrtqmu_A, 1.);

   }

   if (useQTilde ) {
      if (fOneSided) {
         // for bounded one-sided (q_mu_tilde: equations 64,65)
         if ( qmu > qmu_A && (qmu_A > 0 || qmu > tol) ) { // to avoid case 0/0
            if (verbose > 2) oocoutI(nullptr,Eval) << "Using qmu_tilde (qmu is greater than qmu_A)" << std::endl;
            pnull = ROOT::Math::normal_cdf_c( (qmu + qmu_A)/(2 * sqrtqmu_A), 1.);
            palt = ROOT::Math::normal_cdf_c( (qmu - qmu_A)/(2 * sqrtqmu_A), 1.);
         }
      }
      else {
         // for 2 sided bounded test statistic  (N.B there is no one sided discovery qtilde)
         // t_mu_tilde: equations 43,44 in asymptotic paper
         if ( qmu >  qmu_A  && (qmu_A > 0 || qmu > tol)  ) {
            if (verbose > 2) oocoutI(nullptr,Eval) << "Using tmu_tilde (qmu is greater than qmu_A)" << std::endl;
            pnull = ROOT::Math::normal_cdf_c(sqrtqmu,1.) +
                    ROOT::Math::normal_cdf_c( (qmu + qmu_A)/(2 * sqrtqmu_A), 1.);
            palt = ROOT::Math::normal_cdf_c( sqrtqmu_A + sqrtqmu, 1.) +
                   ROOT::Math::normal_cdf_c( (qmu - qmu_A)/(2 * sqrtqmu_A), 1.);
         }
      }
   }



   // create an HypoTest result but where the sampling distributions are set to zero
   string resultname = "HypoTestAsymptotic_result";
   HypoTestResult* res = new HypoTestResult(resultname.c_str(), pnull, palt);

   if (verbose > 0) {
      oocoutP(nullptr, Eval) << "poi = " << muTest->getVal() << " qmu = " << qmu << " qmu_A = " << qmu_A
                             << " sigma = " << muTest->getVal() / sqrtqmu_A << "  CLsplusb = " << pnull
                             << " CLb = " << palt << " CLs = " << res->CLs() << std::endl;
   }

   return res;

}

struct PaltFunction {
   PaltFunction( double offset, double pval, int icase) :
      fOffset(offset), fPval(pval), fCase(icase) {}
   double operator() (double x) const {
      return ROOT::Math::normal_cdf_c(x + fOffset) + ROOT::Math::normal_cdf_c(fCase*(x - fOffset)) - fPval;
   }
   double fOffset;
   double fPval;
   int fCase;
};

////////////////////////////////////////////////////////////////////////////////
/// function given the null and the alt p value - return the expected one given the N - sigma value

double AsymptoticCalculator::GetExpectedPValues(double pnull, double palt, double nsigma, bool useCls, bool oneSided ) {
   if (oneSided) {
      double sqrtqmu =  ROOT::Math::normal_quantile_c( pnull,1.);
      double sqrtqmu_A =  ROOT::Math::normal_quantile( palt,1.) + sqrtqmu;
      double clsplusb = ROOT::Math::normal_cdf_c( sqrtqmu_A - nsigma, 1.);
      if (!useCls) return clsplusb;
      double clb = ROOT::Math::normal_cdf( nsigma, 1.);
      return (clb == 0) ? -1 : clsplusb / clb;
   }

   // case of 2 sided test statistic
   // need to compute numerically
   double sqrttmu =  ROOT::Math::normal_quantile_c( 0.5*pnull,1.);
   if (sqrttmu == 0) {
      // here cannot invert the function - skip the point
      return -1;
   }
   // invert formula for palt to get sqrttmu_A
   PaltFunction f( sqrttmu, palt, -1);
   ROOT::Math::BrentRootFinder brf;
   ROOT::Math::WrappedFunction<PaltFunction> wf(f);
   brf.SetFunction( wf, 0, 20);
   bool ret = brf.Solve();
   if (!ret) {
      oocoutE(nullptr,Eval)  << "Error finding expected p-values - return -1" << std::endl;
      return -1;
   }
   double sqrttmu_A = brf.Root();

   // now invert for expected value
   PaltFunction f2( sqrttmu_A,  ROOT::Math::normal_cdf( nsigma, 1.), 1);
   ROOT::Math::WrappedFunction<PaltFunction> wf2(f2);
   brf.SetFunction(wf2,0,20);
   ret = brf.Solve();
   if (!ret) {
      oocoutE(nullptr,Eval)  << "Error finding expected p-values - return -1" << std::endl;
      return -1;
   }
   return  2*ROOT::Math::normal_cdf_c( brf.Root(),1.);
}

namespace {

////////////////////////////////////////////////////////////////////////////////
/// Fill bins by looping recursively on observables.

void FillBins(const RooAbsPdf & pdf, const RooArgList &obs, RooAbsData & data, int &index,  double &binVolume, int &ibin) {

   bool debug = (fgPrintLevel() >= 2);

   RooRealVar * v = dynamic_cast<RooRealVar*>(&(obs[index]) );
   if (!v) return;

   RooArgSet obstmp(obs);
   double expectedEvents = pdf.expectedEvents(obstmp);

   if (debug) std::cout << "looping on observable " << v->GetName() << std::endl;
   for (int i = 0; i < v->getBins(); ++i) {
      v->setBin(i);
      if (index < int(obs.size()) -1) {
         index++;  // increase index
         double prevBinVolume = binVolume;
         binVolume *= v->getBinWidth(i); // increase bin volume
         FillBins(pdf, obs, data, index,  binVolume, ibin);
         index--; // decrease index
         binVolume = prevBinVolume; // decrease also bin volume
      }
      else {

         // this is now a new bin - compute the pdf in this bin
         double totBinVolume = binVolume * v->getBinWidth(i);
         double fval = pdf.getVal(&obstmp)*totBinVolume;

         if (fval*expectedEvents <= 0)
         {
            if (fval*expectedEvents < 0) {
               oocoutW(nullptr,InputArguments)
                   << "AsymptoticCalculator::" << __func__
                   << "(): Bin " << i << " of " << v->GetName() << " has negative expected events! Please check your inputs." << std::endl;
            }
            else {
               oocoutW(nullptr,InputArguments)
                   << "AsymptoticCalculator::" << __func__
                   << "(): Bin " << i << " of " << v->GetName() << " has zero expected events - skip it" << std::endl;
            }
         }
         // have a cut off for overflows ??
         else {
            data.add(obs, fval*expectedEvents);
         }

         if (debug) {
            std::cout << "bin " << ibin << "\t";
            for (std::size_t j=0; j < obs.size(); ++j) { std::cout << "  " <<  (static_cast<RooRealVar&>( obs[j])).getVal(); }
            std::cout << " w = " << fval*expectedEvents;
            std::cout << std::endl;
         }
         ibin++;
      }
   }
   //reset bin values
   if (debug) {
      std::cout << "ending loop on .. " << v->GetName() << std::endl;
   }

   v->setBin(0);

}

bool setObsToExpected(std::span<RooAbsArg *> servers, const RooArgSet &obs, std::string const &errPrefix)
{
   RooRealVar *myobs = nullptr;
   RooAbsReal *myexp = nullptr;
   for (RooAbsArg *a : servers) {
      if (obs.contains(*a)) {
         if (myobs != nullptr) {
            oocoutF(nullptr,Generation) << errPrefix << "Has two observables ?? " << std::endl;
            return false;
         }
         myobs = dynamic_cast<RooRealVar *>(a);
         if (myobs == nullptr) {
            oocoutF(nullptr,Generation) << errPrefix << "Observable is not a RooRealVar??" << std::endl;
            return false;
         }
      } else {
         if (!a->isConstant() ) {
            if (myexp != nullptr) {
               oocoutE(nullptr,Generation) << errPrefix << "Has two non-const arguments  " << std::endl;
               return false;
            }
            myexp = dynamic_cast<RooAbsReal *>(a);
            if (myexp == nullptr) {
               oocoutF(nullptr,Generation) << errPrefix << "Expected is not a RooAbsReal??" << std::endl;
               return false;
            }
         }
      }
   }
   if (myobs == nullptr)  {
      oocoutF(nullptr,Generation) << errPrefix << "No observable?" << std::endl;
      return false;
   }
   if (myexp == nullptr) {
      oocoutF(nullptr,Generation) << errPrefix << "No observable?" << std::endl;
      return false;
   }

   myobs->setVal(myexp->getVal());

   if (fgPrintLevel() > 2) {
      std::cout << "SetObsToExpected : setting " << myobs->GetName() << " to expected value " << myexp->getVal() << " of " << myexp->GetName() << std::endl;
   }

   return true;
}

////////////////////////////////////////////////////////////////////////////////
/// set observed value to the expected one
/// works for Gaussian, Poisson or LogNormal
/// assumes mean parameter value is the argument not constant and not depending on observables
/// (if more than two arguments are not constant will use first one but print a warning !)
/// need to iterate on the components of the Poisson to get n and nu (nu can be a RooAbsReal)
/// (code from G. Petrucciani and extended by L.M.)

bool SetObsToExpected(RooAbsPdf &pdf, const RooArgSet &obs)
{
   std::string const &errPrefix = "AsymptoticCalculator::SetObsExpected( " + std::string{pdf.ClassName()} + " ) : ";
   std::vector<RooAbsArg *> servers;
   for (RooAbsArg *a : pdf.servers()) {
      servers.emplace_back(a);
   }
   return setObsToExpected(servers, obs, errPrefix);
}

bool setObsToExpectedMultiVarGauss(RooMultiVarGaussian &mvgauss, const RooArgSet &obs)
{
   // In the case of the multi-variate Gaussian, we need to iterate over the
   // dimensions and treat the servers for each dimension separately.

   std::string const &errPrefix = "AsymptoticCalculator::SetObsExpected( " + std::string{mvgauss.ClassName()} + " ) : ";
   std::vector<RooAbsArg *> servers{nullptr, nullptr};
   bool ret = true;
   for (std::size_t iDim = 0; iDim < mvgauss.xVec().size(); ++iDim) {
      servers[0] = &mvgauss.xVec()[iDim];
      servers[1] = &mvgauss.muVec()[iDim];
      ret &= setObsToExpected(servers, obs, errPrefix + " : dim " + std::to_string(iDim) + " ");
   }
   return ret;
}

////////////////////////////////////////////////////////////////////////////////
/// Inpspect a product pdf to find all the Poisson or Gaussian parts to set the observed
/// values to expected ones.

bool setObsToExpectedProdPdf(RooProdPdf &prod, const RooArgSet &obs)
{
    bool ret = true;
    for (auto *a : prod.pdfList()) {
        if (!a->dependsOn(obs)) continue;
        RooPoisson *pois = nullptr;
        RooGaussian *gauss = nullptr;
        RooMultiVarGaussian *mvgauss = nullptr;
        // should try to add also lognormal case ?
        if ((pois = dynamic_cast<RooPoisson *>(a)) != nullptr) {
            ret &= SetObsToExpected(*pois, obs);
            pois->setNoRounding(true);  //needed since expected value is not an integer
        } else if ((gauss = dynamic_cast<RooGaussian *>(a)) != nullptr) {
            ret &= SetObsToExpected(*gauss, obs);
        } else if ((mvgauss = dynamic_cast<RooMultiVarGaussian *>(a)) != nullptr) {
            ret &= setObsToExpectedMultiVarGauss(*mvgauss, obs);
        } else if (RooProdPdf *subprod = dynamic_cast<RooProdPdf *>(a)) {
            ret &= setObsToExpectedProdPdf(*subprod, obs);
        } else {
        oocoutE(nullptr, InputArguments)
           << "Illegal term in counting model: "
           << "the PDF " << a->GetName() << " depends on the observables, but is not a Poisson, Gaussian or Product"
           << std::endl;
        return false;
        }
    }

    return ret;
}

////////////////////////////////////////////////////////////////////////////////
/// Generate counting Asimov data for the case when the pdf cannot be extended.
/// This function assumes that the pdf is a RooPoisson or can be decomposed in a product of RooPoisson,
/// or is a RooGaussian. Otherwise, we cannot know how to make the Asimov data sets.

RooAbsData *GenerateCountingAsimovData(RooAbsPdf & pdf, const RooArgSet & observables,  const RooRealVar & , RooCategory * channelCat) {
    RooArgSet obs(observables);
    RooProdPdf *prod = dynamic_cast<RooProdPdf *>(&pdf);
    RooPoisson *pois = nullptr;
    RooGaussian *gauss = nullptr;
    RooMultiVarGaussian *mvgauss = nullptr;

    if (fgPrintLevel() > 1)
       std::cout << "generate counting Asimov data for pdf of type " << pdf.ClassName() << std::endl;

    bool r = false;
    if (prod != nullptr) {
        r = setObsToExpectedProdPdf(*prod, observables);
    } else if ((pois = dynamic_cast<RooPoisson *>(&pdf)) != nullptr) {
        r = SetObsToExpected(*pois, observables);
        // we need in this case to set Poisson to real values
        pois->setNoRounding(true);
    } else if ((gauss = dynamic_cast<RooGaussian *>(&pdf)) != nullptr) {
        r = SetObsToExpected(*gauss, observables);
    } else if ((mvgauss = dynamic_cast<RooMultiVarGaussian *>(&pdf)) != nullptr) {
        r = setObsToExpectedMultiVarGauss(*mvgauss, observables);
    } else {
       oocoutE(nullptr,InputArguments) << "A counting model pdf must be either a RooProdPdf or a RooPoisson or a RooGaussian" << std::endl;
    }
    if (!r) return nullptr;
    int icat = 0;
    if (channelCat) {
       icat = channelCat->getCurrentIndex();
    }

    RooDataSet *ret = new RooDataSet("CountingAsimovData" + std::to_string(icat),
                                     "CountingAsimovData" + std::to_string(icat), obs);
    ret->add(obs);
    return ret;
}

////////////////////////////////////////////////////////////////////////////////
/// Compute the asimov data set for an observable of a pdf.
/// It generates binned data following the binning of the observables.
// TODO: (possibility to change number of bins)
// TODO: implement integration over bin content

RooAbsData * GenerateAsimovDataSinglePdf(const RooAbsPdf & pdf, const RooArgSet & allobs,  const RooRealVar & weightVar, RooCategory * channelCat) {

   int printLevel = fgPrintLevel();

   // Get observables defined by the pdf associated with this state
   std::unique_ptr<RooArgSet> obs(pdf.getObservables(allobs) );


   // if pdf cannot be extended assume is then a counting experiment
   if (!pdf.canBeExtended() ) return GenerateCountingAsimovData(const_cast<RooAbsPdf&>(pdf), *obs, weightVar, channelCat);

   RooArgSet obsAndWeight(*obs);
   obsAndWeight.add(weightVar);

   std::unique_ptr<RooDataSet> asimovData;
   if (channelCat) {
      int icat = channelCat->getCurrentIndex();
      asimovData = std::make_unique<RooDataSet>("AsimovData" + std::to_string(icat),
                                  "combAsimovData" + std::to_string(icat),
                                  RooArgSet(obsAndWeight,*channelCat),RooFit::WeightVar(weightVar));
   }
   else {
      asimovData = std::make_unique<RooDataSet>("AsimovData","AsimovData",RooArgSet(obsAndWeight),RooFit::WeightVar(weightVar));
   }

    // This works only for 1D observables
    //RooRealVar* thisObs = ((RooRealVar*)obstmp->first());

    RooArgList obsList(*obs);

    // loop on observables and on the bins
    if (printLevel >= 2) {
       std::cout << "Generating Asimov data for pdf " << pdf.GetName() << std::endl;
       std::cout << "list of observables  " << std::endl;
       obsList.Print();
    }

    int obsIndex = 0;
    double binVolume = 1;
    int nbins = 0;
    FillBins(pdf, obsList, *asimovData, obsIndex, binVolume, nbins);
    if (printLevel >= 2)
       std::cout << "filled from " << pdf.GetName() << "   " << nbins << " nbins " << " volume is " << binVolume << std::endl;

    // for (int iobs = 0; iobs < obsList.size(); ++iobs) {
    //    RooRealVar * thisObs = dynamic_cast<RooRealVar*> &obsList[i];
    //    if (thisObs == 0) continue;
    //    // loop on the bin contents
    //    for(int  ibin=0; ibin<thisObs->numBins(); ++ibin){
    //       thisObs->setBin(ibin);

    //   thisNorm=pdftmp->getVal(obstmp)*thisObs->getBinWidth(jj);
    //   if (thisNorm*expectedEvents <= 0)
    //   {
    //     std::cout << "WARNING::Detected bin with zero expected events! Please check your inputs." << std::endl;
    //   }
    //   // have a cut off for overflows ??
    //   obsDataUnbinned->add(*mc->GetObservables(), thisNorm*expectedEvents);
    // }

    if (printLevel >= 1)
    {
      asimovData->Print();
    }
    if( TMath::IsNaN(asimovData->sumEntries()) ){
      std::cout << "sum entries is nan"<< std::endl;
      assert(0);
      asimovData = nullptr;
    }

    return asimovData.release();

}

} // namespace

////////////////////////////////////////////////////////////////////////////////
/// generate the asimov data for the observables (not the global ones)
/// need to deal with the case of a sim pdf

RooAbsData * AsymptoticCalculator::GenerateAsimovData(const RooAbsPdf & pdf, const RooArgSet & observables  )  {

   int printLevel = fgPrintLevel();

   RooRealVar weightVar{"binWeightAsimov", "binWeightAsimov", 1, 0, 1.e30};

   if (printLevel > 1) std::cout <<" Generate Asimov data for observables"<< std::endl;
  //RooDataSet* simData=nullptr;
   const RooSimultaneous* simPdf = dynamic_cast<const RooSimultaneous*>(&pdf);
   if (!simPdf) {
      // generate data for non sim pdf
      return GenerateAsimovDataSinglePdf( pdf, observables, weightVar, nullptr);
   }

   std::map<std::string, std::unique_ptr<RooDataSet>> asimovDataMap;

  //look at category of simpdf
  RooCategory& channelCat = const_cast<RooCategory&>(dynamic_cast<const RooCategory&>(simPdf->indexCat()));
  int nrIndices = channelCat.numTypes();
  if( nrIndices == 0 ) {
    oocoutW(nullptr,Generation) << "Simultaneous pdf does not contain any categories." << std::endl;
  }
  for (int i=0;i<nrIndices;i++){
    channelCat.setIndex(i);
    //iFrame++;
    // Get pdf associated with state from simpdf
    RooAbsPdf* pdftmp = simPdf->getPdf(channelCat.getCurrentLabel()) ;
    assert(pdftmp != nullptr);

    if (printLevel > 1)
    {
      std::cout << "on type " << channelCat.getCurrentLabel() << " " << channelCat.getCurrentIndex() << std::endl;
    }

    std::unique_ptr<RooDataSet> dataSinglePdf{static_cast<RooDataSet*>(GenerateAsimovDataSinglePdf( *pdftmp, observables, weightVar, &channelCat))};
    if (!dataSinglePdf) {
       oocoutE(nullptr,Generation) << "Error generating an Asimov data set for pdf " << pdftmp->GetName() << std::endl;
       return nullptr;
    }

    if (asimovDataMap.count(string(channelCat.getCurrentLabel())) != 0) {
      oocoutE(nullptr,Generation) << "AsymptoticCalculator::GenerateAsimovData(): The PDF for " << channelCat.getCurrentLabel()
          << " was already defined. It will be overridden. The faulty category definitions follow:" << std::endl;
      channelCat.Print("V");
    }

    if (printLevel > 1)
    {
      std::cout << "channel: " << channelCat.getCurrentLabel() << ", data: ";
      dataSinglePdf->Print();
      std::cout << std::endl;
    }

    asimovDataMap[string(channelCat.getCurrentLabel())] = std::move(dataSinglePdf);
  }

  RooArgSet obsAndWeight(observables);
  obsAndWeight.add(weightVar);


  return new RooDataSet("asimovDataFullModel","asimovDataFullModel",RooArgSet(obsAndWeight,channelCat),
                                          RooFit::Index(channelCat),RooFit::Import(asimovDataMap),RooFit::WeightVar(weightVar));
}

////////////////////////////////////////////////////////////////////////////////
/// Make the Asimov data from the ModelConfig and list of poi
/// \param realData Real data
/// \param model Model config defining the pdf and the parameters
/// \param paramValues The snapshot of POI and parameters used for finding the best nuisance parameter values (conditioned at these values)
/// \param[out] asimovGlobObs Global observables set to values satisfying the constraints
/// \param genPoiValues Optional. A different set of POI values used for generating. By default the same POI are used for generating and for finding the nuisance parameters
/// given an observed data set, a model and a snapshot of the poi.
/// \return The asimov data set. The user takes ownership.
///

RooAbsData * AsymptoticCalculator::MakeAsimovData(RooAbsData & realData, const ModelConfig & model, const  RooArgSet & paramValues, RooArgSet & asimovGlobObs, const RooArgSet * genPoiValues )  {

   int verbose = fgPrintLevel();


   RooArgSet  poi(*model.GetParametersOfInterest());
   poi.assign(paramValues);
   
   // set poi constant for conditional MLE
   // need to fit nuisance parameters at their conditional MLE value
   RooArgSet paramsSetConstant;
   for (auto *tmpPar : static_range_cast<RooRealVar *>(poi)) {
      tmpPar->setConstant();
      if (verbose>0)
         std::cout << "MakeAsimov: Setting poi " << tmpPar->GetName() << " to a constant value = " << tmpPar->getVal() << std::endl;
      paramsSetConstant.add(*tmpPar);
   }

   // find conditional value of the nuisance parameters
   bool hasFloatParams = false;
   RooArgSet  constrainParams;
   if (model.GetNuisanceParameters()) {
      constrainParams.add(*model.GetNuisanceParameters());
      RooStats::RemoveConstantParameters(&constrainParams);
      if (!constrainParams.empty()) hasFloatParams = true;

   } else {
      // Do we have free parameters anyway that need fitting?
      std::unique_ptr<RooArgSet> params(model.GetPdf()->getParameters(realData)); 
      for (auto const *rrv : dynamic_range_cast<RooRealVar *>(*params)) {
         if ( rrv != nullptr && rrv->isConstant() == false ) { hasFloatParams = true; break; }
      }
   }
   if (hasFloatParams) {
      // models need to be fitted to find best nuisance parameter values

      TStopwatch tw2; tw2.Start();
      int minimPrintLevel = ROOT::Math::MinimizerOptions::DefaultPrintLevel();
      if (verbose>0) {
         std::cout << "MakeAsimov: doing a conditional fit for finding best nuisance values " << std::endl;
         minimPrintLevel = verbose;
         if (verbose>1) {
            std::cout << "POI values:\n"; poi.Print("v");
            if (verbose > 2) {
               std::cout << "Nuis param values:\n";
               constrainParams.Print("v");
            }
         }
      }
      RooFit::MsgLevel msglevel = RooMsgService::instance().globalKillBelow();
      if (verbose < 2) RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL);

      std::string minimizerAlgo = ROOT::Math::MinimizerOptions::DefaultMinimizerAlgo();
      std::vector<RooCmdArg> args{
         RooFit::Minimizer("",minimizerAlgo.c_str()), // empty mimimizer type to select default
         RooFit::Strategy(ROOT::Math::MinimizerOptions::DefaultStrategy()),
         RooFit::PrintLevel(minimPrintLevel-1),
         RooFit::Hesse(false),
         RooFit::Constrain(constrainParams),
         RooFit::Offset(GetGlobalRooStatsConfig().useLikelihoodOffset),
         RooFit::EvalErrorWall(GetGlobalRooStatsConfig().useEvalErrorWall)
      };

      RooLinkedList argList;
      for (auto& arg : args) {
        argList.Add(&arg);
      }
      model.fitTo(realData, argList);
      if (verbose>0) { std::cout << "fit time "; tw2.Print();}
      if (verbose > 1) {
         // after the fit the nuisance parameters will have their best fit value
         if (model.GetNuisanceParameters() ) {
            std::cout << "Nuisance parameters after fit for asimov dataset: " << std::endl;
            model.GetNuisanceParameters()->Print("V");
         }
      }

      if (verbose < 2) RooMsgService::instance().setGlobalKillBelow(msglevel);

   }

   // restore the parameters which were set constant
   SetAllConstant(paramsSetConstant, false);

   std::unique_ptr<RooArgSet> allParams{model.GetPdf()->getParameters(realData)};

   RooStats::RemoveConstantParameters(allParams.get());

   // if a RooArgSet of poi is passed , different poi will be used for generating the Asimov data set
   if (genPoiValues) {
    allParams->assign(*genPoiValues);
   }

   // now do the actual generation of the AsimovData Set
   // no need to pass parameters values since we have set them before
   return MakeAsimovData(model, *allParams, asimovGlobObs);
}

////////////////////////////////////////////////////////////////////////////////
/// \param model ModelConfig that contains the model pdf and the model parameters
/// \param allParamValues The parameters of the model will be set to the values given in this set
/// \param[out] asimovGlobObs Global observables set to values satisfying the constraints
/// \return Asimov data set. The user takes ownership.
///
/// The parameter values (including the nuisance parameter) can result from a fit to data or be at the nominal values.
///

RooAbsData * AsymptoticCalculator::MakeAsimovData(const ModelConfig & model, const  RooArgSet & allParamValues, RooArgSet & asimovGlobObs)  {

   int verbose = fgPrintLevel();

   TStopwatch tw;
   tw.Start();

   // set the parameter values (do I need the poi to be constant ? )
   // the nuisance parameter values could be set at their fitted value (the MLE)
   if (!allParamValues.empty()) {
      std::unique_ptr<RooArgSet> allVars{model.GetPdf()->getVariables()};
      allVars->assign(allParamValues);
   }


   // generate the Asimov data set for the observables
   RooAbsData * asimov = GenerateAsimovData(*model.GetPdf() , *model.GetObservables() );

   if (verbose>0) {
      std::cout << "Generated Asimov data for observables "; (model.GetObservables() )->Print();
      if (verbose > 1) {
         if (asimov->numEntries() == 1 ) {
            std::cout << "--- Asimov data values \n";
            asimov->get()->Print("v");
         }
         else {
            std::cout << "--- Asimov data numEntries = " << asimov->numEntries() << " sumOfEntries = " << asimov->sumEntries() << std::endl;
         }
         std::cout << "\ttime for generating : ";  tw.Print();
      }
   }


    // Now need to have in ASIMOV the data sets also the global observables
   // Their values must be the one satisfying the constraint.
   // to do it make a nuisance pdf with all product of constraints and then
   // assign to each constraint a glob observable value = to the current fitted nuisance parameter value
   // IN general  one should solve in general the system of equations f( gobs| nuispar ) = 0 where f are the
   //  derivatives of the constraint with respect the nuisance parameter and they are evaluated at the best fit nuisance
   // parameter points
   // As simple solution assume that constrain has a direct dependence on the nuisance parameter, i.e.
   // Constraint (gobs, func( nuispar) ) and the condition is satisfied for
   // gobs = func( nuispar) where nunispar is at the MLE value


   if (model.GetGlobalObservables() && !model.GetGlobalObservables()->empty()) {

      if (verbose>1) {
         std::cout << "Generating Asimov data for global observables " << std::endl;
      }

      RooArgSet gobs(*model.GetGlobalObservables());

      // snapshot data global observables
      RooArgSet snapGlobalObsData;
      SetAllConstant(gobs, true);
      gobs.snapshot(snapGlobalObsData);


      RooArgSet nuis;
      if (model.GetNuisanceParameters()) nuis.add(*model.GetNuisanceParameters());
      if (nuis.empty()) {
            oocoutW(nullptr,Generation) << "AsymptoticCalculator::MakeAsimovData: model does not have nuisance parameters but has global observables"
                                            << " set global observables to model values " << std::endl;
            asimovGlobObs.assign(gobs);
            return asimov;
      }

      // part 1: create the nuisance pdf
      std::unique_ptr<RooAbsPdf> nuispdf(RooStats::MakeNuisancePdf(model,"TempNuisPdf") );
      if (nuispdf == nullptr) {
            oocoutF(nullptr, Generation) << "AsymptoticCalculator::MakeAsimovData: model has nuisance parameters and "
                                            "global obs but no nuisance pdf "
                                         << std::endl;
      }
      // unfold the nuisance pdf if it is a prod pdf
      RooArgList pdfList;
      RooProdPdf *prod = dynamic_cast<RooProdPdf *>(nuispdf.get());
      if (prod ) {
         pdfList.add(prod->pdfList());
      } else {
         // nothing to unfold - just use the pdf
         pdfList.add(*nuispdf);
      }

      for (auto *cterm : static_range_cast<RooAbsPdf *>(pdfList)) {
         assert(dynamic_cast<RooAbsPdf *>(static_cast<RooAbsArg *>(cterm)) &&
                "AsimovUtils: a factor of the nuisance pdf is not a Pdf!");

         if (!cterm->dependsOn(nuis)) continue; // dummy constraints
         // skip also the case of uniform components
         if (typeid(*cterm) == typeid(RooUniform)) continue;

         std::unique_ptr<RooArgSet> cpars(cterm->getParameters(&gobs));
         std::unique_ptr<RooArgSet> cgobs(cterm->getObservables(&gobs));
         if (cgobs->size() > 1) {
            oocoutE(nullptr,Generation) << "AsymptoticCalculator::MakeAsimovData: constraint term  " <<  cterm->GetName()
                                            << " has multiple global observables -cannot generate - skip it" << std::endl;
            continue;
         }
         else if (cgobs->empty()) {
            oocoutW(nullptr, Generation)
               << "AsymptoticCalculator::MakeAsimovData: constraint term  " << cterm->GetName()
                                            << " has no global observables - skip it" << std::endl;
            continue;
         }
         // the variable representing the global observable
         RooRealVar &rrv = dynamic_cast<RooRealVar &>(*cgobs->first());

         // remove the constant parameters in cpars
         RooStats::RemoveConstantParameters(cpars.get());
         if (cpars->size() != 1) {
            oocoutE(nullptr, Generation)
               << "AsymptoticCalculator::MakeAsimovData:constraint term "
                                            << cterm->GetName() << " has multiple floating params - cannot generate - skip it " << std::endl;
            continue;
         }

         bool foundServer = false;
         // note : this will work only for this type of constraints
         // expressed as RooPoisson, RooGaussian, RooLognormal, RooGamma
         TClass * cClass = cterm->IsA();
         if (verbose > 2) std::cout << "Constraint " << cterm->GetName() << " of type " << cClass->GetName() << std::endl;
         if ( cClass != RooGaussian::Class() && cClass != RooPoisson::Class() &&
              cClass != RooGamma::Class() && cClass != RooLognormal::Class() &&
              cClass != RooBifurGauss::Class()  ) {
            TString className =  (cClass) ?  cClass->GetName() : "undefined";
            oocoutW(nullptr, Generation)
               << "AsymptoticCalculator::MakeAsimovData:constraint term "
                                            << cterm->GetName() << " of type " << className
                                            << " is a non-supported type - result might be not correct " << std::endl;
         }

         // in case of a Poisson constraint make sure the rounding is not set
         if (cClass == RooPoisson::Class() ) {
            RooPoisson * pois = static_cast<RooPoisson*>(cterm);
            assert(dynamic_cast<RooPoisson *>(cterm));
            pois->setNoRounding(true);
         }

         // look at server of the constraint term and check if the global observable is part of the server
         RooAbsArg * arg = cterm->findServer(rrv);
         if (!arg) {
            // special case is for the Gamma where one might define the global observable n and you have a Gamma(b, n+1, ...._
            // in this case n+1 is the server and we don;t have a direct dependency, but we want to set n to the b value
            // so in case of the Gamma ignore this test
            if ( cClass != RooGamma::Class() ) {
               oocoutE(nullptr, Generation)
                  << "AsymptoticCalculator::MakeAsimovData:constraint term "
                                               << cterm->GetName() << " has no direct dependence on global observable- cannot generate it " << std::endl;
               continue;
            }
         }

         // loop on the server of the constraint term
         // need to treat the Gamma as a special case
         // the mode of the Gamma is (k-1)*theta where theta is the inverse of the rate parameter.
         // we assume that the global observable is defined as ngobs = k-1 and the theta parameter has the name theta otherwise we use other procedure which might be wrong
         RooAbsReal * thetaGamma = nullptr;
         if ( cClass == RooGamma::Class() ) {
            for (RooAbsArg *a2 : cterm->servers()) {
               if (TString(a2->GetName()).Contains("theta") ) {
                  thetaGamma = dynamic_cast<RooAbsReal*>(a2);
                  break;
               }
            }
            if (thetaGamma == nullptr) {
               oocoutI(nullptr, Generation)
                  << "AsymptoticCalculator::MakeAsimovData:constraint term "
                                               << cterm->GetName() << " is a Gamma distribution and no server named theta is found. Assume that the Gamma scale is  1 " << std::endl;
            }
            else if (verbose>2) {
                  std::cout << "Gamma constraint has a scale " << thetaGamma->GetName() << "  = " << thetaGamma->getVal() << std::endl;
            }
         }
         for (RooAbsArg *a2 : cterm->servers()) {
            RooAbsReal * rrv2 = dynamic_cast<RooAbsReal *>(a2);
            if (verbose > 2) std::cout << "Loop on constraint server term  " << a2->GetName() << std::endl;
            if (rrv2 && rrv2->dependsOn(nuis) ) {


               // found server depending on nuisance
               if (foundServer) {
                  oocoutE(nullptr,Generation) << "AsymptoticCalculator::MakeAsimovData:constraint term "
                                            << cterm->GetName() << " constraint term has more server depending on nuisance- cannot generate it " <<
                     std::endl;
                  foundServer = false;
                  break;
               }
               if (thetaGamma && thetaGamma->getVal() > 0) {
                  rrv.setVal( rrv2->getVal() / thetaGamma->getVal() );
               } else {
                  rrv.setVal(rrv2->getVal());
               }
               foundServer = true;

               if (verbose > 2) {
                  std::cout << "setting global observable " << rrv.GetName() << " to value " << rrv.getVal()
                            << " which comes from " << rrv2->GetName() << std::endl;
               }
            }
         }

         if (!foundServer) {
            oocoutE(nullptr,Generation) << "AsymptoticCalculator::MakeAsimovData - can't find nuisance for constraint term - global observables will not be set to Asimov value " << cterm->GetName() << std::endl;
            std::cerr << "Parameters: " << std::endl;
            cpars->Print("V");
            std::cerr << "Observables: " << std::endl;
            cgobs->Print("V");
         }
      }

      // make a snapshot of global observables
      // needed this ?? (LM)

      asimovGlobObs.removeAll();
      SetAllConstant(gobs, true);
      gobs.snapshot(asimovGlobObs);

      // revert global observables to the data value
      gobs.assign(snapGlobalObsData);

      if (verbose>0) {
         std::cout << "Generated Asimov data for global observables ";
         if (verbose == 1) gobs.Print();
      }

      if (verbose > 1) {
         std::cout << "\nGlobal observables for data: " << std::endl;
         gobs.Print("V");
         std::cout << "\nGlobal observables for asimov: " << std::endl;
         asimovGlobObs.Print("V");
      }


   }

   return asimov;

}
