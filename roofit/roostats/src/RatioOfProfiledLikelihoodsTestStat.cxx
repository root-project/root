// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "RooStats/RatioOfProfiledLikelihoodsTestStat.h"

#include "RooArgSet.h"
#include "RooAbsData.h"
#include "TMath.h"
#include "RooMsgService.h"
#include "RooGlobalFunc.h"


Bool_t RooStats::RatioOfProfiledLikelihoodsTestStat::fgAlwaysReuseNll = kTRUE ;

void RooStats::RatioOfProfiledLikelihoodsTestStat::SetAlwaysReuseNLL(Bool_t flag) { fgAlwaysReuseNll = flag ; }

    //__________________________________________
Double_t RooStats::RatioOfProfiledLikelihoodsTestStat::ProfiledLikelihood(RooAbsData& data, RooArgSet& poi, RooAbsPdf& pdf) {
   // returns -logL(poi, conditonal MLE of nuisance params)
   // subtract off the global MLE or not depending on the option
   // It is the numerator or the denominator of the ratio (depending on the pdf)
   //L.M. : not sure why this method is needed now

   int type = (fSubtractMLE) ? 0 : 2; 
       
   // null
   if ( &pdf == fNullProfile.GetPdf() )
      return fNullProfile.EvaluateProfileLikelihood(type, data, poi);
   else if (&pdf == fAltProfile.GetPdf() )
      return fAltProfile.EvaluateProfileLikelihood(type, data, poi);
   
   oocoutE((TObject*)NULL,InputArguments) << "RatioOfProfiledLikelihoods::ProfileLikelihood - invalid pdf used for computing the profiled likelihood - return NaN" 
                         << std::endl;

   return TMath::QuietNaN(); 
      
}
    
//__________________________________________
Double_t  RooStats::RatioOfProfiledLikelihoodsTestStat::Evaluate(RooAbsData& data, RooArgSet& nullParamsOfInterest) {
   // evaluate the ratio of profile likelihood
   
   
   int type = (fSubtractMLE) ? 0 : 2; 
       
   // null
   double nullNLL = fNullProfile.EvaluateProfileLikelihood(type, data, nullParamsOfInterest);
   const RooArgSet *nullset = fNullProfile.GetDetailedOutput();
   
   // alt 
   double altNLL = fAltProfile.EvaluateProfileLikelihood(type, data, *fAltPOI);
   const RooArgSet *altset = fAltProfile.GetDetailedOutput();
   
   if (fDetailedOutput != NULL) {
      delete fDetailedOutput;
      fDetailedOutput = NULL;
   }
   if (fDetailedOutputEnabled) {
      fDetailedOutput = new RooArgSet();
      RooRealVar* var(0);
      for(TIterator *it = nullset->createIterator();(var = dynamic_cast<RooRealVar*>(it->Next()));) {
         RooRealVar* cloneVar = new RooRealVar(TString::Format("nullprof_%s", var->GetName()),
                                               TString::Format("%s for null", var->GetTitle()), var->getVal());
         fDetailedOutput->addOwned(*cloneVar);
      }
      for(TIterator *it = altset->createIterator();(var = dynamic_cast<RooRealVar*>(it->Next()));) {
         RooRealVar* cloneVar = new RooRealVar(TString::Format("altprof_%s", var->GetName()),
                                               TString::Format("%s for null", var->GetTitle()), var->getVal());
         fDetailedOutput->addOwned(*cloneVar);
      }
   }
   
/*
// set variables back to where they were
nullParamsOfInterest = *saveNullPOI;
*allVars = *saveAll;
delete saveAll;
delete allVars;
*/

   return nullNLL -altNLL;
}
