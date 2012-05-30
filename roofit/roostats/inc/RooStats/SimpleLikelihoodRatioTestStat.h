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

#ifndef ROO_REAL_VAR
#include "RooRealVar.h"
#endif

#ifndef ROOSTATS_TestStatistic
#include "RooStats/TestStatistic.h"
#endif


namespace RooStats {

class SimpleLikelihoodRatioTestStat : public TestStatistic {

   public:

      //__________________________________
      SimpleLikelihoodRatioTestStat() :
         fNullPdf(NULL), fAltPdf(NULL)
      {
         // Constructor for proof. Do not use.
         fFirstEval = true;
	     fDetailedOutputEnabled = false;
        fDetailedOutput = NULL;
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

	     fDetailedOutputEnabled = false;
        fDetailedOutput = NULL;

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

	     fDetailedOutputEnabled = false;
        fDetailedOutput = NULL;

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
	 if (fDetailedOutput) delete fDetailedOutput;
      }

     static void SetAlwaysReuseNLL(Bool_t flag) { fAlwaysReuseNll = flag ; }
     void SetReuseNLL(Bool_t flag) { fReuseNll = flag ; }

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
               << std::endl;
         }
         fFirstEval = false;

         RooFit::MsgLevel msglevel = RooMsgService::instance().globalKillBelow();
         RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL);

	 Bool_t reuse = (fReuseNll || fAlwaysReuseNll) ;

	 Bool_t created = kFALSE ;
	 if (!fNllNull) {
      RooArgSet* allParams = fNullPdf->getParameters(data);
	   fNllNull = (RooNLLVar*) fNullPdf->createNLL(data, RooFit::CloneData(kFALSE),RooFit::Constrain(*allParams));
	   delete allParams;
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
         
         //cout << std::endl << "SLRTS: null params:" << std::endl;
         //attachedSet->Print("v");
         

         if (!reuse) {
            delete fNllNull ; fNllNull = NULL ;
         }
         delete attachedSet;

	 created = kFALSE ;
	 if (!fNllAlt) {
      RooArgSet* allParams = fAltPdf->getParameters(data);
	   fNllAlt = (RooNLLVar*) fAltPdf->createNLL(data, RooFit::CloneData(kFALSE),RooFit::Constrain(*allParams));
	   delete allParams;
	   created = kTRUE ;
	 }
	 if (reuse && !created) {
	   fNllAlt->setData(data, kFALSE) ;
	 }
         // make sure we set the variables attached to this nll
         attachedSet = fNllAlt->getVariables();
         *attachedSet = *fAltParameters;
         double altNLL = fNllAlt->getVal();

         //cout << std::endl << "SLRTS: alt params:" << std::endl;
         //attachedSet->Print("v");


         //cout << std::endl << "SLRTS null NLL: " << nullNLL << "    alt NLL: " << altNLL << std::endl << std::endl;


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

//             cout << std::endl << "STORING THIS AS DETAILED OUTPUT:" << std::endl;
//             fDetailedOutput->Print("v");
//             cout << std::endl;
         }


         RooMsgService::instance().setGlobalKillBelow(msglevel);
         return nullNLL - altNLL;
      }

      virtual void EnableDetailedOutput( bool e=true ) { fDetailedOutputEnabled = e; fDetailedOutput = NULL; }
      virtual const RooArgSet* GetDetailedOutput(void) const { return fDetailedOutput; }

      virtual const TString GetVarName() const {
         return "log(L(#mu_{1}) / L(#mu_{0}))";
      }

   private:

      RooAbsPdf* fNullPdf;
      RooAbsPdf* fAltPdf;
      RooArgSet* fNullParameters;
      RooArgSet* fAltParameters;
      bool fFirstEval;
      
      bool fDetailedOutputEnabled;
      RooArgSet* fDetailedOutput; //!

      RooNLLVar* fNllNull ;  //! transient copy of the null NLL
      RooNLLVar* fNllAlt ; //!  transient copy of the alt NLL
      static Bool_t fAlwaysReuseNll ;
      Bool_t fReuseNll ;


   protected:
   ClassDef(SimpleLikelihoodRatioTestStat,2)
};

}

#endif
