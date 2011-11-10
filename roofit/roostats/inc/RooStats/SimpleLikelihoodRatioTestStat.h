// @(#)root/roostats:$Id$
// Author: Kyle Cranmer and Sven Kreiss    June 2010
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_SimpleLikelihoodRatioTestStat
#define ROOSTATS_SimpleLikelihoodRatioTestStat

//_________________________________________________
/*
 BEGIN_HTML
 <p>
 SimpleLikelihoodRatioTestStat: TestStatistic that returns -log(L[null] / L[alt]) where
 L is the likelihood.
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

#include "RooStats/TestStatistic.h"
#include "RooWorkspace.h"

namespace RooStats {

class SimpleLikelihoodRatioTestStat : public TestStatistic {

   public:

      //__________________________________
      SimpleLikelihoodRatioTestStat() :
         fNullPdf(NULL), fAltPdf(NULL)
      {
         // Constructor for proof. Do not use.
         fFirstEval = true;
         fNullParameters = NULL;
         fAltParameters = NULL;
	 fReuseNll=kFALSE ;
	 fNllNull=NULL ;
	 fNllAlt=NULL ;
      }

      //__________________________________
      SimpleLikelihoodRatioTestStat(
         RooAbsPdf& nullPdf,
         RooAbsPdf& altPdf
      ) :
         fFirstEval(true)
      {
         // Takes null and alternate parameters from PDF. Can be overridden.

         fNullPdf = &nullPdf;
         fAltPdf = &altPdf;

         RooArgSet * allNullVars = fNullPdf->getVariables();
         fNullParameters = (RooArgSet*) allNullVars->snapshot();
         delete allNullVars; 

         RooArgSet * allAltVars = fAltPdf->getVariables();
         fAltParameters = (RooArgSet*) allAltVars->snapshot();
         delete allAltVars;

	 fReuseNll=kFALSE ;
	 fNllNull=NULL ;
	 fNllAlt=NULL ;
      }
      //__________________________________
      SimpleLikelihoodRatioTestStat(
         RooAbsPdf& nullPdf,
         RooAbsPdf& altPdf,
         const RooArgSet& nullParameters,
         const RooArgSet& altParameters
      ) :
         fFirstEval(true)
      {
         // Takes null and alternate parameters from values in nullParameters
         // and altParameters. Can be overridden.
         fNullPdf = &nullPdf;
         fAltPdf = &altPdf;

         fNullParameters = (RooArgSet*) nullParameters.snapshot();
         fAltParameters = (RooArgSet*) altParameters.snapshot();
	 fReuseNll=kFALSE ;
	 fNllNull=NULL ;
	 fNllAlt=NULL ;
      }

      //______________________________
      virtual ~SimpleLikelihoodRatioTestStat() {
         if (fNullParameters) delete fNullParameters;
         if (fAltParameters) delete fAltParameters;
	 if (fNllNull) delete fNllNull ;
	 if (fNllAlt) delete fNllAlt ;
      }

     static void setAlwaysReuseNLL(Bool_t flag) { fAlwaysReuseNll = flag ; }
     void setReuseNLL(Bool_t flag) { fReuseNll = flag ; }

      //_________________________________________
      void SetNullParameters(const RooArgSet& nullParameters) {
         if (fNullParameters) delete fNullParameters;
         fFirstEval = true;
         //      if(fNullParameters) delete fNullParameters;
         fNullParameters = (RooArgSet*) nullParameters.snapshot();
      }

      //_________________________________________
      void SetAltParameters(const RooArgSet& altParameters) {
         if (fAltParameters) delete fAltParameters;
         fFirstEval = true;
         //      if(fAltParameters) delete fAltParameters;
         fAltParameters = (RooArgSet*) altParameters.snapshot();
      }

      //______________________________
      bool ParamsAreEqual() {
         // this should be possible with RooAbsCollection
         if (!fNullParameters->equals(*fAltParameters)) return false;

         RooAbsReal* null;
         RooAbsReal* alt;

         TIterator* nullIt = fNullParameters->createIterator();
         TIterator* altIt = fAltParameters->createIterator();
         bool ret = true;
         while ((null = (RooAbsReal*) nullIt->Next()) && (alt = (RooAbsReal*) altIt->Next())) {
            if (null->getVal() != alt->getVal()) ret = false;
         }
         delete nullIt;
         delete altIt;
         return ret;
      }

      //______________________________
      virtual Double_t Evaluate(RooAbsData& data, RooArgSet& nullPOI) {

         if (fFirstEval && ParamsAreEqual()) {
            oocoutW(fNullParameters,InputArguments)
               << "Same RooArgSet used for null and alternate, so you must explicitly SetNullParameters and SetAlternateParameters or the likelihood ratio will always be 1."
               << endl;
         }
         fFirstEval = false;

         RooFit::MsgLevel msglevel = RooMsgService::instance().globalKillBelow();
         RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL);

	 Bool_t reuse = (fReuseNll || fAlwaysReuseNll) ;

	 Bool_t created = kFALSE ;
	 if (!fNllNull) {
	   fNllNull = (RooNLLVar*) fNullPdf->createNLL(data, RooFit::CloneData(kFALSE));
	   created = kTRUE ;
	 }
	 if (reuse && !created) {
	   fNllNull->setData(data, kFALSE) ;
	 }

         // make sure we set the variables attached to this nll
         RooArgSet* attachedSet = fNllNull->getVariables();
         *attachedSet = *fNullParameters;
         *attachedSet = nullPOI;
         double nullNLL = fNllNull->getVal();

         if (!reuse) {
            delete fNllNull ; fNllNull = NULL ;
         }
         delete attachedSet;

	 created = kFALSE ;
	 if (!fNllAlt) {
	   fNllAlt = (RooNLLVar*) fAltPdf->createNLL(data, RooFit::CloneData(kFALSE));
	   created = kTRUE ;
	 }
	 if (reuse && !created) {
	   fNllAlt->setData(data, kFALSE) ;
	 }
         // make sure we set the variables attached to this nll
         attachedSet = fNllAlt->getVariables();
         *attachedSet = *fAltParameters;
         double altNLL = fNllAlt->getVal();

         if (!reuse) { 
            delete fNllAlt ; fNllAlt = NULL ;
         }
         delete attachedSet;

         RooMsgService::instance().setGlobalKillBelow(msglevel);
         return nullNLL - altNLL;
      }

      virtual const TString GetVarName() const {
         return "log(L(#mu_{1}) / L(#mu_{0}))";
      }

   private:

      RooAbsPdf* fNullPdf;
      RooAbsPdf* fAltPdf;
      RooArgSet* fNullParameters;
      RooArgSet* fAltParameters;
      bool fFirstEval;

      RooNLLVar* fNllNull ;
      RooNLLVar* fNllAlt ;
      static Bool_t fAlwaysReuseNll ;
      Bool_t fReuseNll ;


   protected:
   ClassDef(SimpleLikelihoodRatioTestStat,1)
};

}

#endif
