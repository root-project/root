// @(#)root/roostats:$Id$
// Authors: Kyle Cranmer, Sven Kreiss    June 2010
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_RatioOfProfiledLikelihoodsTestStat
#define ROOSTATS_RatioOfProfiledLikelihoodsTestStat

//_________________________________________________
/*
BEGIN_HTML
<p>
TestStatistic that returns the ratio of profiled likelihoods.
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

#ifndef ROOSTATS_TestStatistic
#include "RooStats/TestStatistic.h"
#endif

#ifndef ROOSTATS_ProfileLikelihoodTestStat
#include "RooStats/ProfileLikelihoodTestStat.h"
#endif

namespace RooStats {

class RatioOfProfiledLikelihoodsTestStat: public TestStatistic {

  public:

   RatioOfProfiledLikelihoodsTestStat() :
      fNullProfile(),
      fAltProfile(),
      fAltPOI(NULL),
      fSubtractMLE(true),
      fDetailedOutputEnabled(false),
      fDetailedOutput(NULL)
   {
      // Proof constructor. Don't use.
   }

  RatioOfProfiledLikelihoodsTestStat(RooAbsPdf& nullPdf, RooAbsPdf& altPdf, 
				     const RooArgSet* altPOI=0) :
    fNullProfile(nullPdf), 
    fAltProfile(altPdf), 
    fSubtractMLE(true),
    fDetailedOutputEnabled(false),
    fDetailedOutput(NULL)
      {
	/*
         Calculates the ratio of profiled likelihoods. 

	 By default the calculation is:

	    Lambda(mu_alt , conditional MLE for alt nuisance) 
	log --------------------------------------------
   	    Lambda(mu_null , conditional MLE for null nuisance)

	where Lambda is the profile likeihood ratio, so the 
	MLE for the null and alternate are subtracted off.

	If SetSubtractMLE(false) then it calculates:

	    L(mu_alt , conditional MLE for alt nuisance) 
	log --------------------------------------------
	    L(mu_null , conditional MLE for null nuisance)


	The values of the parameters of interest for the alternative 
	hypothesis are taken at the time of the construction.
	If empty, it treats all free parameters as nuisance parameters.

	The value of the parameters of interest for the null hypotheses 
	are given at each call of Evaluate(data,nullPOI).
	*/
	if(altPOI)
	  fAltPOI = (RooArgSet*) altPOI->snapshot();
	else
	  fAltPOI = new RooArgSet(); // empty set

      }

    //__________________________________________
    ~RatioOfProfiledLikelihoodsTestStat(void) {
      if(fAltPOI) delete fAltPOI;
      if(fDetailedOutput) delete fDetailedOutput;
    }
    
    //__________________________________________
    Double_t ProfiledLikelihood(RooAbsData& data, RooArgSet& poi, RooAbsPdf& pdf) {
      // returns -logL(poi, conditonal MLE of nuisance params)
      // it does not subtract off the global MLE
      // because  nuisance parameters of null and alternate may not
      // be the same.
      RooAbsReal* nll = pdf.createNLL(data, RooFit::CloneData(kFALSE));      
      RooAbsReal* profile = nll->createProfile(poi);
      // make sure we set the variables attached to this nll
      RooArgSet* attachedSet = nll->getVariables();
      *attachedSet = poi;
      // now evaluate profile to set nuisance to conditional MLE values
      double nllVal =  profile->getVal();
      // but we may want the nll value without subtracting off the MLE      
      if(!fSubtractMLE) nllVal = nll->getVal();

      delete attachedSet;
      delete profile;
      delete nll;
      
      return nllVal;
    }
    
    //__________________________________________
    virtual Double_t Evaluate(RooAbsData& data, RooArgSet& nullParamsOfInterest) {
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
    
    virtual void EnableDetailedOutput( bool e=true ) { 
	    fDetailedOutputEnabled = e; 
	    fNullProfile.EnableDetailedOutput(fDetailedOutputEnabled);
	    fAltProfile.EnableDetailedOutput(fDetailedOutputEnabled);
    }

    static void SetAlwaysReuseNLL(Bool_t flag) { fgAlwaysReuseNll = flag ; }

   void SetReuseNLL(Bool_t flag) { 
      fNullProfile.SetReuseNLL(flag);  
      fAltProfile.SetReuseNLL(flag);  
   }

   void SetMinimizer(const char* minimizer){ 
      fNullProfile.SetMinimizer(minimizer);  
      fAltProfile.SetMinimizer(minimizer);  
   }
   void SetStrategy(Int_t strategy){
      fNullProfile.SetStrategy(strategy);  
      fAltProfile.SetStrategy(strategy);  
   }
   void SetTolerance(Double_t tol){
      fNullProfile.SetTolerance(tol);  
      fAltProfile.SetTolerance(tol);  
   }
   void SetPrintLevel(Int_t printLevel){
      fNullProfile.SetPrintLevel(printLevel);  
      fAltProfile.SetPrintLevel(printLevel);  
   }
   
     virtual const RooArgSet* GetDetailedOutput(void) const {
	     // Returns detailed output. The value returned by this function is updated after each call to Evaluate().
	     // The returned RooArgSet contains the following for the alternative and null hypotheses:
	     // <ul>
	     // <li> the minimum nll, fitstatus and convergence quality for each fit </li> 
	     // <li> for each fit and for each non-constant parameter, the value, error and pull of the parameter are stored </li>
	     // </ul>
	     return fDetailedOutput;
     }
    

   virtual const TString GetVarName() const { return "log(L(#mu_{1},#hat{#nu}_{1}) / L(#mu_{0},#hat{#nu}_{0}))"; }
    
    //    const bool PValueIsRightTail(void) { return false; } // overwrites default
    
    void SetSubtractMLE(bool subtract){fSubtractMLE = subtract;}
    
  private:
    ProfileLikelihoodTestStat fNullProfile;
    ProfileLikelihoodTestStat fAltProfile;

    RooArgSet* fAltPOI;
    Bool_t fSubtractMLE;
   static Bool_t fgAlwaysReuseNll ;

    bool fDetailedOutputEnabled;
    RooArgSet* fDetailedOutput;

    
  protected:
    ClassDef(RatioOfProfiledLikelihoodsTestStat,3)
};

}


#endif
