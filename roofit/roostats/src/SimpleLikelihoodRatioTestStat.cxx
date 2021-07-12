// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class RooStats::SimpleLikelihoodRatioTestStat
    \ingroup Roostats

TestStatistic class that returns -log(L[null] / L[alt]) where
L is the likelihood.
It is often called as the LEP Test statistic.


*/

#include "RooStats/SimpleLikelihoodRatioTestStat.h"
#include "RooStats/RooStatsUtils.h"

Bool_t RooStats::SimpleLikelihoodRatioTestStat::fgAlwaysReuseNll = kTRUE ;

////////////////////////////////////////////////////////////////////////////////

void RooStats::SimpleLikelihoodRatioTestStat::SetAlwaysReuseNLL(Bool_t flag) { fgAlwaysReuseNll = flag ; }

Double_t RooStats::SimpleLikelihoodRatioTestStat::Evaluate(RooAbsData& data, RooArgSet& nullPOI) {

   if (fFirstEval && ParamsAreEqual()) {
      oocoutW(fNullParameters,InputArguments)
         << "Same RooArgSet used for null and alternate, so you must explicitly SetNullParameters and SetAlternateParameters or the likelihood ratio will always be 1."
         << std::endl;
   }

   // strip pdfs of constraints (which cancel out in the ratio) to avoid unnecessary computations and computational errors
   if (fFirstEval) {
      fNullPdf = RooStats::MakeUnconstrainedPdf(*fNullPdf, *fNullPdf->getObservables(data));
      fAltPdf  = RooStats::MakeUnconstrainedPdf(*fAltPdf , *fAltPdf->getObservables(data) );
   }

   fFirstEval = false;

   RooFit::MsgLevel msglevel = RooMsgService::instance().globalKillBelow();
   RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL);

   Bool_t reuse = (fReuseNll || fgAlwaysReuseNll) ;

   Bool_t created = kFALSE ;
   if (!fNllNull) {
      RooArgSet* allParams = fNullPdf->getParameters(data);
      fNllNull = fNullPdf->createNLL(data, RooFit::CloneData(kFALSE),RooFit::Constrain(*allParams),RooFit::GlobalObservables(fGlobalObs),RooFit::ConditionalObservables(fConditionalObs));
      delete allParams;
      created = kTRUE ;
   }
   if (reuse && !created) {
      fNllNull->setData(data, kFALSE) ;
   }

   // make sure we set the variables attached to this nll
   RooArgSet* attachedSet = fNllNull->getVariables();
   attachedSet->assign(*fNullParameters);
   attachedSet->assign(nullPOI);
   double nullNLL = fNllNull->getVal();

   //std::cout << std::endl << "SLRTS: null params:" << std::endl;
   //attachedSet->Print("v");


   if (!reuse) {
      delete fNllNull ; fNllNull = NULL ;
   }
   delete attachedSet;

   created = kFALSE ;
   if (!fNllAlt) {
      RooArgSet* allParams = fAltPdf->getParameters(data);
      fNllAlt = fAltPdf->createNLL(data, RooFit::CloneData(kFALSE),RooFit::Constrain(*allParams),RooFit::GlobalObservables(fGlobalObs),RooFit::ConditionalObservables(fConditionalObs));
      delete allParams;
      created = kTRUE ;
   }
   if (reuse && !created) {
      fNllAlt->setData(data, kFALSE) ;
   }
   // make sure we set the variables attached to this nll
   attachedSet = fNllAlt->getVariables();
   attachedSet->assign(*fAltParameters);
   double altNLL = fNllAlt->getVal();

   //std::cout << std::endl << "SLRTS: alt params:" << std::endl;
   //attachedSet->Print("v");


//   std::cout << std::endl << "SLRTS null NLL: " << nullNLL << "    alt NLL: " << altNLL << std::endl << std::endl;


   if (!reuse) {
      delete fNllAlt ; fNllAlt = NULL ;
   }
   delete attachedSet;



   // save this snapshot
   if( fDetailedOutputEnabled ) {
      if( !fDetailedOutput ) {
         fDetailedOutput = new RooArgSet( *(new RooRealVar("nullNLL","null NLL",0)), "detailedOut_SLRTS" );
         fDetailedOutput->add( *(new RooRealVar("altNLL","alternate NLL",0)) );
      }
      fDetailedOutput->setRealValue( "nullNLL", nullNLL );
      fDetailedOutput->setRealValue( "altNLL", altNLL );

//             std::cout << std::endl << "STORING THIS AS DETAILED OUTPUT:" << std::endl;
//             fDetailedOutput->Print("v");
//             std::cout << std::endl;
   }


   RooMsgService::instance().setGlobalKillBelow(msglevel);
   return nullNLL - altNLL;
}
