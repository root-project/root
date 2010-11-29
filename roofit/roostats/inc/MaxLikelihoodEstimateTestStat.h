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
   }
   //__________________________________
   MaxLikelihoodEstimateTestStat(RooAbsPdf& pdf, RooRealVar& parameter) :
   fPdf(&pdf),fParameter(&parameter), fUpperLimit(true)
   {
      // constructor
      //      fPdf = pdf;
      //      fParameter = parameter;
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

    RooAbsReal* nll = fPdf->createNLL(data, RooFit::CloneData(false));
    RooAbsReal* profile = nll->createProfile(RooArgSet());
    profile->getVal();
    RooArgSet* vars = profile->getVariables();
    RooMsgService::instance().setGlobalKillBelow(msglevel);
    double ret = vars->getRealValue(fParameter->GetName());
    delete vars;
    delete nll;
    delete profile;
    return ret;

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

   protected:
   ClassDef(MaxLikelihoodEstimateTestStat,1)
};

}


#endif
