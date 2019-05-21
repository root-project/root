// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
// Additional Contributions: Giovanni Petrucciani
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class RooStats::ProfileLikelihoodTestStat
    \ingroup Roostats

ProfileLikelihoodTestStat is an implementation of the TestStatistic interface
that calculates the profile likelihood ratio at a particular parameter point
given a dataset. It does not constitute a statistical test, for that one may
either use:

  - the ProfileLikelihoodCalculator that relies on asymptotic properties of the
    Profile Likelihood Ratio
  - the NeymanConstruction class with this class as a test statistic
  - the HybridCalculator class with this class as a test statistic


*/

#include "RooStats/ProfileLikelihoodTestStat.h"
#include "RooFitResult.h"
#include "RooPullVar.h"
#include "RooStats/DetailedOutputAggregator.h"

#include "RooProfileLL.h"
#include "RooNLLVar.h"
#include "RooMsgService.h"
#include "RooMinimizer.h"
#include "RooArgSet.h"
#include "RooDataSet.h"
#include "TStopwatch.h"

#include "RooStats/RooStatsUtils.h"

using namespace std;

Bool_t RooStats::ProfileLikelihoodTestStat::fgAlwaysReuseNll = kTRUE ;

void RooStats::ProfileLikelihoodTestStat::SetAlwaysReuseNLL(Bool_t flag) { fgAlwaysReuseNll = flag ; }

////////////////////////////////////////////////////////////////////////////////
/// internal function to evaluate test statistics
/// can do depending on type:
/// -  type  = 0 standard evaluation,
/// -  type = 1 find only unconditional NLL minimum,
/// -  type = 2 conditional MLL

Double_t RooStats::ProfileLikelihoodTestStat::EvaluateProfileLikelihood(int type, RooAbsData& data, RooArgSet& paramsOfInterest) {

       if( fDetailedOutputEnabled && fDetailedOutput ) {
          delete fDetailedOutput;
          fDetailedOutput = 0;
       }
       if( fDetailedOutputEnabled && !fDetailedOutput ) {
          fDetailedOutput = new RooArgSet();
       }

       //data.Print("V");

       TStopwatch tsw;
       tsw.Start();

       double initial_mu_value  = 0;
       RooRealVar* firstPOI = dynamic_cast<RooRealVar*>( paramsOfInterest.first());
       if (firstPOI) initial_mu_value = firstPOI->getVal();
       //paramsOfInterest.getRealValue(firstPOI->GetName());
       if (fPrintLevel > 1) {
            cout << "POIs: " << endl;
            paramsOfInterest.Print("v");
       }

       RooFit::MsgLevel msglevel = RooMsgService::instance().globalKillBelow();
       if (fPrintLevel < 3) RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL);

       // simple
       Bool_t reuse=(fReuseNll || fgAlwaysReuseNll) ;

       Bool_t created(kFALSE) ;
       if (!reuse || fNll==0) {
          RooArgSet* allParams = fPdf->getParameters(data);
          RooStats::RemoveConstantParameters(allParams);

          // need to call constrain for RooSimultaneous until stripDisconnected problem fixed
          fNll = fPdf->createNLL(data, RooFit::CloneData(kFALSE),RooFit::Constrain(*allParams),
                                 RooFit::GlobalObservables(fGlobalObs), RooFit::ConditionalObservables(fConditionalObs), RooFit::Offset(fLOffset));

          if (fPrintLevel > 0 && fLOffset) cout << "ProfileLikelihoodTestStat::Evaluate - Use Offset in creating NLL " << endl ;

          created = kTRUE ;
          delete allParams;
          if (fPrintLevel > 1) cout << "creating NLL " << fNll << " with data = " << &data << endl ;
       }
       if (reuse && !created) {
         if (fPrintLevel > 1) cout << "reusing NLL " << fNll << " new data = " << &data << endl ;
         fNll->setData(data,kFALSE) ;
       }
       // print data in case of number counting (simple data sets)
       if (fPrintLevel > 1 && data.numEntries() == 1) {
          std::cout << "Data set used is:  ";
          RooStats::PrintListContent(*data.get(0), std::cout);
       }


       // make sure we set the variables attached to this nll
       RooArgSet* attachedSet = fNll->getVariables();

       *attachedSet = paramsOfInterest;
       RooArgSet* origAttachedSet = (RooArgSet*) attachedSet->snapshot();

       ///////////////////////////////////////////////////////////////////////
       // New profiling based on RooMinimizer (allows for Minuit2)
       // based on major speed increases seen by CMS for complex problems


       // other order
       // get the numerator
       RooArgSet* snap =  (RooArgSet*)paramsOfInterest.snapshot();

       tsw.Stop();
       double createTime = tsw.CpuTime();
       tsw.Start();

       // get the denominator
       double uncondML = 0;
       double fit_favored_mu = 0;
       int statusD = 0;
       RooArgSet * detOutput = 0;
       if (type != 2) {
          // minimize and count eval errors
          fNll->clearEvalErrorLog();
          if (fPrintLevel>1) std::cout << "Do unconditional fit" << std::endl;
     RooFitResult* result = GetMinNLL();
          if (result) {
             uncondML = result->minNll();
             statusD = result->status();

             // get best fit value for one-sided interval
             if (firstPOI) fit_favored_mu = attachedSet->getRealValue(firstPOI->GetName()) ;

             // save this snapshot
             if( fDetailedOutputEnabled ) {
                detOutput = DetailedOutputAggregator::GetAsArgSet(result, "fitUncond_", fDetailedOutputWithErrorsAndPulls);
                fDetailedOutput->addOwned(*detOutput);
                delete detOutput;
             }
             delete result;
          }
          else {
             return TMath::SignalingNaN();   // this should not really happen
          }
       }
       tsw.Stop();
       double fitTime1  = tsw.CpuTime();

       //double ret = 0;
       int statusN = 0;
       tsw.Start();

       double condML = 0;

       bool doConditionalFit = (type != 1);

       // skip the conditional ML (the numerator) only when fit value is smaller than test value
       if (!fSigned && type==0 &&
           ((fLimitType==oneSided          && fit_favored_mu >= initial_mu_value) ||
            (fLimitType==oneSidedDiscovery && fit_favored_mu <= initial_mu_value))) {
          doConditionalFit = false;
          condML = uncondML;
       }

       if (doConditionalFit) {

          if (fPrintLevel>1) std::cout << "Do conditional fit " << std::endl;


          //       cout <<" reestablish snapshot"<<endl;
          *attachedSet = *snap;


          // set the POI to constant
          RooLinkedListIter it = paramsOfInterest.iterator();
          RooRealVar* tmpPar = NULL, *tmpParA=NULL;
          while((tmpPar = (RooRealVar*)it.Next())){
             tmpParA =  dynamic_cast<RooRealVar*>( attachedSet->find(tmpPar->GetName()));
             if (tmpParA) tmpParA->setConstant();
          }


          // check if there are non-const parameters so it is worth to do the minimization
          RooArgSet allParams(*attachedSet);
          RooStats::RemoveConstantParameters(&allParams);

          // in case no nuisance parameters are present
          // no need to minimize just evaluate the nll
          if (allParams.getSize() == 0 ) {
             // be sure to evaluate with offsets
             if (fLOffset) RooAbsReal::setHideOffset(false);
             condML = fNll->getVal();
             if (fLOffset) RooAbsReal::setHideOffset(true);
          }
          else {
            fNll->clearEvalErrorLog();
            RooFitResult* result = GetMinNLL();
            if (result) {
               condML = result->minNll();
               statusN = result->status();
               if( fDetailedOutputEnabled ) {
                  detOutput = DetailedOutputAggregator::GetAsArgSet(result, "fitCond_", fDetailedOutputWithErrorsAndPulls);
                  fDetailedOutput->addOwned(*detOutput);
                  delete detOutput;
               }
               delete result;
            }
            else {
               return TMath::SignalingNaN();   // this should not really happen
            }
          }

       }

       tsw.Stop();
       double fitTime2 = tsw.CpuTime();

       double pll = 0;
       if (type != 0)  {
          // for conditional only or unconditional fits
          // need to compute nll value without the offset
          if (fLOffset) {
             RooAbsReal::setHideOffset(kFALSE) ;
             pll = fNll->getVal();
          }
          else {
             if (type == 1)
                pll = uncondML;
             else if (type == 2)
                pll = condML;
          }
       }
       else {  // type == 0
          // for standard profile likelihood evaluations
         pll = condML-uncondML;

         if (fSigned) {
            if (pll<0.0) {
               if (fPrintLevel > 0) std::cout << "pll is negative - setting it to zero " << std::endl;
               pll = 0.0;   // bad fit
            }
           if (fLimitType==oneSidedDiscovery ? (fit_favored_mu < initial_mu_value)
                                             : (fit_favored_mu > initial_mu_value))
             pll = -pll;
         }
       }

       if (fPrintLevel > 0) {
          std::cout << "EvaluateProfileLikelihood - ";
          if (type <= 1)
             std::cout << "mu hat = " << fit_favored_mu  <<  ", uncond ML = " << uncondML;
          if (type != 1)
             std::cout << ", cond ML = " << condML;
          if (type == 0)
             std::cout << " pll = " << pll;
          std::cout << " time (create/fit1/2) " << createTime << " , " << fitTime1 << " , " << fitTime2
                    << std::endl;
       }


       // need to restore the values ?
       *attachedSet = *origAttachedSet;

       delete attachedSet;
       delete origAttachedSet;
       delete snap;

       if (!reuse) {
    delete fNll;
    fNll = 0;
       }

       RooMsgService::instance().setGlobalKillBelow(msglevel);

       if(statusN!=0 || statusD!=0) {
         return -1; // indicate failed fit (WVE is not used anywhere yet)
       }

       return pll;

     }

////////////////////////////////////////////////////////////////////////////////
/// find minimum of NLL using RooMinimizer

RooFitResult* RooStats::ProfileLikelihoodTestStat::GetMinNLL() {

   RooMinimizer minim(*fNll);
   minim.setStrategy(fStrategy);
   //LM: RooMinimizer.setPrintLevel has +1 offset - so subtract  here -1 + an extra -1
   int level = (fPrintLevel == 0) ? -1 : fPrintLevel -2;
   minim.setPrintLevel(level);
   minim.setEps(fTolerance);
   // this causes a memory leak
   minim.optimizeConst(2);
   TString minimizer = fMinimizer;
   TString algorithm = ROOT::Math::MinimizerOptions::DefaultMinimizerAlgo();
   if (algorithm == "Migrad") algorithm = "Minimize"; // prefer to use Minimize instead of Migrad
   int status;
   for (int tries = 1, maxtries = 4; tries <= maxtries; ++tries) {
      status = minim.minimize(minimizer,algorithm);
      if (status%1000 == 0) {  // ignore erros from Improve
         break;
      } else if (tries < maxtries) {
         cout << "    ----> Doing a re-scan first" << endl;
         minim.minimize(minimizer,"Scan");
         if (tries == 2) {
            if (fStrategy == 0 ) {
               cout << "    ----> trying with strategy = 1" << endl;;
               minim.setStrategy(1);
            }
            else
               tries++; // skip this trial if strategy is already 1
         }
         if (tries == 3) {
            cout << "    ----> trying with improve" << endl;;
            minimizer = "Minuit";
            algorithm = "migradimproved";
         }
      }
   }

   //how to get cov quality faster?
   return minim.save();
   //minim.optimizeConst(false);
}
