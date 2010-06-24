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

namespace RooStats {

class SimpleLikelihoodRatioTestStat: public TestStatistic {

   public:

    //__________________________________
  SimpleLikelihoodRatioTestStat(RooAbsPdf& nullPdf, RooAbsPdf& altPdf) :
    fNullPdf(nullPdf), fAltPdf(altPdf)
    {
      // b/c null and alternate parameters are not set, it will take
      // values from PDF. Can override
      fFirstEval=true;
      fNullParameters = (RooArgSet*)fNullPdf.getVariables()->snapshot();      
      fAltParameters = (RooArgSet*)fAltPdf.getVariables()->snapshot();          
    }


    //__________________________________
  SimpleLikelihoodRatioTestStat(RooAbsPdf& nullPdf, RooAbsPdf& altPdf, const RooArgSet& nullParameters, const RooArgSet& altParameters) :
    fNullPdf(nullPdf), fAltPdf(altPdf)
      {
	// constructor, explicitly set parameters.
	// note to developers, constructor with two const RooArgSet*
	// causes problems if the usage is
	// null.GetSnapshot(), alt.GetSnapshot()
	// because both are pointers to the same set, and you will
	// get duplicate values.

	fFirstEval=true;
	fNullParameters = (RooArgSet*)nullParameters.snapshot();
	fAltParameters = (RooArgSet*)altParameters.snapshot();

      }

    //_________________________________________
    void SetNullParameters(const RooArgSet& nullParameters){
      delete fNullParameters;
      fFirstEval=true;
      //      if(fNullParameters) delete fNullParameters;
      fNullParameters = (RooArgSet*)nullParameters.snapshot();
    }

    //_________________________________________
    void SetAltParameters(const RooArgSet& altParameters){
      delete fAltParameters;
      fFirstEval=true;
      //      if(fAltParameters) delete fAltParameters;
      fAltParameters = (RooArgSet*)altParameters.snapshot();
    }
    //______________________________
    virtual ~SimpleLikelihoodRatioTestStat() {
      delete fNullParameters;
      delete fAltParameters;
    }

    //______________________________
    bool ParamsAreEqual(){
      // this should be possible with RooAbsCollection
      if(!fNullParameters->equals(*fAltParameters))
	return false;

      RooAbsReal* null;
      RooAbsReal* alt;

      TIterator* nullIt = fNullParameters->createIterator();
      TIterator* altIt = fAltParameters->createIterator();
      bool ret =true;
      while((null = (RooAbsReal*) nullIt->Next()) && 
	    (alt = (RooAbsReal*) altIt->Next()) ){
	if(null->getVal() != alt->getVal() ) ret = false;
      }
      delete nullIt;
      delete altIt;
      return ret;    
    }

    //______________________________
    virtual Double_t Evaluate(RooAbsData& data, RooArgSet& /*nullPOI*/) {
            
      if( fFirstEval && ParamsAreEqual()){
	  oocoutW(fNullParameters,InputArguments) << "Same RooArgSet used for null and alternate, so you must explicitly SetNullParameters and SetAlternateParameters or the likelihood ratio will always be 1." << endl; 
	}
      fFirstEval=false;

         RooFit::MsgLevel msglevel = RooMsgService::instance().globalKillBelow();
         RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL);


         RooAbsReal *nll;
         nll = fNullPdf.createNLL(data, RooFit::CloneData(kFALSE));
	 // make sure we set the variables attached to this nll
	 RooArgSet* attachedSet = nll->getVariables();
	 *attachedSet = *fNullParameters;
         double nullNLL = nll->getVal();
         delete nll;
	 delete attachedSet;

         nll = fAltPdf.createNLL(data, RooFit::CloneData(kFALSE));
	 // make sure we set the variables attached to this nll
	 attachedSet = nll->getVariables();
	 *attachedSet = *fAltParameters;
         double altNLL = nll->getVal();
         delete nll;
	 delete attachedSet;


         RooMsgService::instance().setGlobalKillBelow(msglevel);
         return nullNLL - altNLL;
      }

      virtual const TString GetVarName() const { return "log(L(#mu_{1}) / L(#mu_{0}))"; }



   private:
      RooAbsPdf& fNullPdf;
      RooAbsPdf& fAltPdf;
      RooArgSet* fNullParameters;
      RooArgSet* fAltParameters;
      bool fFirstEval;

   protected:
   ClassDef(SimpleLikelihoodRatioTestStat,1)
};

}


#endif
