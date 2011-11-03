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

//_________________________________________________
/*
BEGIN_HTML
<p>
MaxLikelihoodEstimateTestStat: TestStatistic that returns maximum likelihood estimate of a specified parameter.
</p>
END_HTML
*/
//

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#ifndef ROO_NLL_VAR
#include "RooNLLVar.h"
#endif

#include "RooFitResult.h"
#include "RooStats/TestStatistic.h"
#include "RooAbsPdf.h"
#include "RooRealVar.h"
#include "RooMinimizer.h"
#include "Math/MinimizerOptions.h"


namespace RooStats {

class MaxLikelihoodEstimateTestStat: public TestStatistic {

   public:

   //__________________________________
   MaxLikelihoodEstimateTestStat() :
   fPdf(NULL),fParameter(NULL), fUpperLimit(true)
   {
     // constructor
     //      fPdf = pdf;
     //      fParameter = parameter;

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
  virtual Double_t Evaluate(RooAbsData& data, RooArgSet& /*nullPOI*/) {
      
    
    RooFit::MsgLevel msglevel = RooMsgService::instance().globalKillBelow();
    RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL);

    /*
    // this is more straight forward, but produces a lot of messages
    RooFitResult* res = fPdf.fitTo(data, RooFit::CloneData(kFALSE),RooFit::Minos(0),RooFit::Hesse(false), RooFit::Save(1),RooFit::PrintLevel(-1),RooFit::PrintEvalErrors(0));
    RooRealVar* mle = (RooRealVar*) res->floatParsFinal().find(fParameter.GetName());
    double ret = mle->getVal();
    delete res;
    return ret;
    */

    RooArgSet* allParams = fPdf->getParameters(data);
    RooStats::RemoveConstantParameters(allParams);

    // need to call constrain for RooSimultaneous until stripDisconnected problem fixed
    RooAbsReal* nll = (RooNLLVar*) fPdf->createNLL(data, RooFit::CloneData(kFALSE),RooFit::Constrain(*allParams));

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
     //LM: RooMinimizer.setPrintLevel has +1 offset - so subtruct  here -1
     minim.setPrintLevel(fPrintLevel-1);
     int status = -1;
     //	minim.optimizeConst(true);
     for (int tries = 0, maxtries = 4; tries <= maxtries; ++tries) {
	  //	 status = minim.minimize(fMinimizer, ROOT::Math::MinimizerOptions::DefaultMinimizerAlgo().c_str());
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
     std::cout << "BEST FIT values " << std::endl;
     allParams->Print("V");

     RooMsgService::instance().setGlobalKillBelow(msglevel);
     delete nll;

     if (status != 0) return -1; 
     return fParameter->getVal();


  }
  
  virtual const TString GetVarName() const { 
    TString varName = Form("Maximum Likelihood Estimate of %s",fParameter->GetName());
    return varName;
  }

      
  virtual void PValueIsRightTail(bool isright) {  fUpperLimit = isright; }
  virtual bool PValueIsRightTail(void) const { return fUpperLimit; }


   private:
      RooAbsPdf *fPdf;
      RooRealVar *fParameter;
      bool fUpperLimit;
      TString fMinimizer;
      Int_t fStrategy;
      Int_t fPrintLevel;



   protected:
   ClassDef(MaxLikelihoodEstimateTestStat,1)
};

}


#endif
