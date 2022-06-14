// @(#)root/roostats:$Id$
// Author: Kyle Cranmer    June 2010
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_MaxLikelihoodEstimateTestStat
#define ROOSTATS_MaxLikelihoodEstimateTestStat




#include "Rtypes.h"

#include "RooFitResult.h"
#include "RooStats/TestStatistic.h"
#include "RooAbsPdf.h"
#include "RooRealVar.h"
#include "RooMinimizer.h"
#include "Math/MinimizerOptions.h"
#include "RooStats/RooStatsUtils.h"



namespace RooStats {

/** \class MaxLikelihoodEstimateTestStat
    \ingroup Roostats
MaxLikelihoodEstimateTestStat: TestStatistic that returns maximum likelihood
estimate of a specified parameter.
*/

class MaxLikelihoodEstimateTestStat: public TestStatistic {

   public:

   //__________________________________
   MaxLikelihoodEstimateTestStat() :
   fPdf(nullptr),fParameter(nullptr), fUpperLimit(true)
   {
     /// constructor
     ///      fPdf = pdf;
     ///      fParameter = parameter;

   fMinimizer=::ROOT::Math::MinimizerOptions::DefaultMinimizerType().c_str();
   fStrategy=::ROOT::Math::MinimizerOptions::DefaultStrategy();
   fPrintLevel=::ROOT::Math::MinimizerOptions::DefaultPrintLevel();

   }
   //__________________________________
   MaxLikelihoodEstimateTestStat(RooAbsPdf& pdf, RooRealVar& parameter) :
   fPdf(&pdf),fParameter(&parameter), fUpperLimit(true)
   {
      // constructor
      //      fPdf = pdf;
      //      fParameter = parameter;
   fMinimizer=::ROOT::Math::MinimizerOptions::DefaultMinimizerType().c_str();
   fStrategy=::ROOT::Math::MinimizerOptions::DefaultStrategy();
   fPrintLevel=::ROOT::Math::MinimizerOptions::DefaultPrintLevel();

   }

  //______________________________
  double Evaluate(RooAbsData& data, RooArgSet& /*nullPOI*/) override {


    RooFit::MsgLevel msglevel = RooMsgService::instance().globalKillBelow();
    RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL);

    /*
    // this is more straight forward, but produces a lot of messages
    RooFitResult* res = fPdf.fitTo(data, RooFit::CloneData(false),RooFit::Minos(0),RooFit::Hesse(false), RooFit::Save(1),RooFit::PrintLevel(-1),RooFit::PrintEvalErrors(0));
    RooRealVar* mle = (RooRealVar*) res->floatParsFinal().find(fParameter.GetName());
    double ret = mle->getVal();
    delete res;
    return ret;
    */

    RooArgSet* allParams = fPdf->getParameters(data);
    RooStats::RemoveConstantParameters(allParams);

    // need to call constrain for RooSimultaneous until stripDisconnected problem fixed
    RooAbsReal* nll = fPdf->createNLL(data, RooFit::CloneData(false),RooFit::Constrain(*allParams),RooFit::ConditionalObservables(fConditionalObs));

    //RooAbsReal* nll = fPdf->createNLL(data, RooFit::CloneData(false));

    // RooAbsReal* profile = nll->createProfile(RooArgSet());
    // profile->getVal();
    // RooArgSet* vars = profile->getVariables();
    // RooMsgService::instance().setGlobalKillBelow(msglevel);
    // double ret = vars->getRealValue(fParameter->GetName());
    // delete vars;
    // delete nll;
    // delete profile;
    // return ret;


     RooMinimizer minim(*nll);
     minim.setStrategy(fStrategy);
     //LM: RooMinimizer.setPrintLevel has +1 offset - so subtract  here -1
     minim.setPrintLevel(fPrintLevel-1);
     int status = -1;
     //   minim.optimizeConst(true);
     for (int tries = 0, maxtries = 4; tries <= maxtries; ++tries) {
     //    status = minim.minimize(fMinimizer, ROOT::Math::MinimizerOptions::DefaultMinimizerAlgo().c_str());
        status = minim.minimize(fMinimizer, "Minimize");
        if (status == 0) {
           break;
        } else {
           if (tries > 1) {
         printf("    ----> Doing a re-scan first\n");
         minim.minimize(fMinimizer,"Scan");
       }
           if (tries > 2) {
         printf("    ----> trying with strategy = 1\n");
              minim.setStrategy(1);
           }
        }
     }
     //std::cout << "BEST FIT values " << std::endl;
     //allParams->Print("V");

     RooMsgService::instance().setGlobalKillBelow(msglevel);
     delete nll;

     if (status != 0) return -1;
     return fParameter->getVal();


  }

  const TString GetVarName() const override {
    TString varName = Form("Maximum Likelihood Estimate of %s",fParameter->GetName());
    return varName;
  }


  virtual void PValueIsRightTail(bool isright) {  fUpperLimit = isright; }
  bool PValueIsRightTail(void) const override { return fUpperLimit; }

   // set the conditional observables which will be used when creating the NLL
   // so the pdf's will not be normalized on the conditional observables when computing the NLL
   void SetConditionalObservables(const RooArgSet& set) override {fConditionalObs.removeAll(); fConditionalObs.add(set);}


   private:
      RooAbsPdf *fPdf;
      RooRealVar *fParameter;
      RooArgSet fConditionalObs;
      bool fUpperLimit;
      TString fMinimizer;
      Int_t fStrategy;
      Int_t fPrintLevel;



   protected:
   ClassDefOverride(MaxLikelihoodEstimateTestStat,2)
};

}


#endif
