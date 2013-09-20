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

#ifndef ROOSTATS_ProfileLikelihoodTestStat
#define ROOSTATS_ProfileLikelihoodTestStat

//_________________________________________________
/*
BEGIN_HTML
<p>
ProfileLikelihoodTestStat is an implementation of the TestStatistic interface that calculates the profile
likelihood ratio at a particular parameter point given a dataset.  It does not constitute a statistical test, for that one may either use:
<ul>
 <li> the ProfileLikelihoodCalculator that relies on asymptotic properties of the Profile Likelihood Ratio</li>
 <li> the Neyman Construction classes with this class as a test statistic</li>
 <li> the Hybrid Calculator class with this class as a test statistic</li>
</ul>

</p>
END_HTML
*/
//

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#ifndef ROOSTATS_TestStatistic
#include "RooStats/TestStatistic.h"
#endif


#ifndef ROO_REAL_VAR
#include "RooRealVar.h"
#endif

#ifndef ROO_NLL_VAR
#include "RooNLLVar.h"
#endif

#ifndef ROOTT_Math_MinimizerOptions
#include "Math/MinimizerOptions.h"
#endif


namespace RooStats {

  class ProfileLikelihoodTestStat : public TestStatistic{

     enum LimitType {twoSided, oneSided, oneSidedDiscovery};

   public:
     ProfileLikelihoodTestStat() {
        // Proof constructor. Do not use.
        fPdf = 0;
        fNll = 0;
        fCachedBestFitParams = 0;
        fLastData = 0;
	fLimitType = twoSided;
	fSigned = false;
        fDetailedOutputWithErrorsAndPulls = false;
        fDetailedOutputEnabled = false;
        fDetailedOutput = NULL;
	fLOffset = kFALSE ;
      
        fVarName = "Profile Likelihood Ratio";
        fReuseNll = false;
	fMinimizer=::ROOT::Math::MinimizerOptions::DefaultMinimizerType().c_str();
	fStrategy=::ROOT::Math::MinimizerOptions::DefaultStrategy();
        fTolerance=TMath::Max(1.,::ROOT::Math::MinimizerOptions::DefaultTolerance());
	fPrintLevel=::ROOT::Math::MinimizerOptions::DefaultPrintLevel();

     }
     ProfileLikelihoodTestStat(RooAbsPdf& pdf) {
       fPdf = &pdf;
       fNll = 0;
       fCachedBestFitParams = 0;
       fLastData = 0;
       fLimitType = twoSided;
       fSigned = false;
       fDetailedOutputWithErrorsAndPulls = false;
       fDetailedOutputEnabled = false;
       fDetailedOutput = NULL;
       fLOffset = kFALSE ;
      
       fVarName = "Profile Likelihood Ratio";
       fReuseNll = false;
       fMinimizer=::ROOT::Math::MinimizerOptions::DefaultMinimizerType().c_str();
       fStrategy=::ROOT::Math::MinimizerOptions::DefaultStrategy();
       // avoid default tolerance to be too small (1. is default in RooMinimizer)
       fTolerance=TMath::Max(1.,::ROOT::Math::MinimizerOptions::DefaultTolerance());
       fPrintLevel=::ROOT::Math::MinimizerOptions::DefaultPrintLevel();
     }
     virtual ~ProfileLikelihoodTestStat() {
       if(fNll) delete fNll;
       if(fCachedBestFitParams) delete fCachedBestFitParams;
       if(fDetailedOutput) delete fDetailedOutput;
     }

     //LM use default copy constructor and assignment copying the pointers. Is this what we want ?

     void SetOneSided(Bool_t flag=true) {fLimitType = (flag ? oneSided : twoSided);}
     void SetOneSidedDiscovery(Bool_t flag=true) {fLimitType = (flag ? oneSidedDiscovery : twoSided);}
     void SetSigned(Bool_t flag=true) {fSigned = flag;}  // +/- t_mu instead of t_mu>0 with one-sided settings
     //void SetOneSidedDiscovery(Bool_t flag=true) {fOneSidedDiscovery = flag;}

     bool IsTwoSided() const { return fLimitType == twoSided; }
     bool IsOneSidedDiscovery() const { return fLimitType == oneSidedDiscovery; }

     static void SetAlwaysReuseNLL(Bool_t flag);

     void SetReuseNLL(Bool_t flag) { fReuseNll = flag ; }
     void SetLOffset(Bool_t flag=kTRUE) { fLOffset = flag ; }

     void SetMinimizer(const char* minimizer){ fMinimizer=minimizer;}
     void SetStrategy(Int_t strategy){fStrategy=strategy;}
     void SetTolerance(double tol){fTolerance=tol;}
     void SetPrintLevel(Int_t printlevel){fPrintLevel=printlevel;}
    
     // Main interface to evaluate the test statistic on a dataset
     virtual Double_t Evaluate(RooAbsData& data, RooArgSet& paramsOfInterest) {
        return EvaluateProfileLikelihood(0, data, paramsOfInterest);
     }

     // evaluate  the profile likelihood ratio (type = 0) or the minimum of likelihood (type=1) or the conditional LL (type = 2) 
     virtual Double_t EvaluateProfileLikelihood(int type, RooAbsData &data, RooArgSet & paramsOfInterest);
     
     virtual void EnableDetailedOutput( bool e=true, bool withErrorsAndPulls=false ) {
        fDetailedOutputEnabled = e;
        fDetailedOutputWithErrorsAndPulls = withErrorsAndPulls;
        delete fDetailedOutput;
        fDetailedOutput = NULL;
     }
     virtual const RooArgSet* GetDetailedOutput(void) const {
	     // Returns detailed output. The value returned by this function is updated after each call to Evaluate().
	     // The returned RooArgSet contains the following:
	     // <ul>
	     // <li> the minimum nll, fitstatus and convergence quality for each fit </li> 
	     // <li> for each fit and for each non-constant parameter, the value, error and pull of the parameter are stored </li>
	     // </ul>
	     return fDetailedOutput;
     }
         
     // set the conditional observables which will be used when creating the NLL
     // so the pdf's will not be normalized on the conditional observables when computing the NLL 
     virtual void SetConditionalObservables(const RooArgSet& set) {fConditionalObs.removeAll(); fConditionalObs.add(set);}

     virtual void SetVarName(const char* name) { fVarName = name; }
     virtual const TString GetVarName() const {return fVarName;}

     virtual RooAbsPdf * GetPdf() const { return fPdf; }

      
      //      const bool PValueIsRightTail(void) { return false; } // overwrites default

  private:

     RooFitResult* GetMinNLL();

   private:

      RooAbsPdf* fPdf;
      RooAbsReal* fNll; //!
      const RooArgSet* fCachedBestFitParams;
      RooAbsData* fLastData;
      //      Double_t fLastMLE;
      LimitType fLimitType;
      Bool_t fSigned;
      
      // this will store a snapshot of the unconditional nuisance
      // parameter fit.
      bool fDetailedOutputEnabled;
      bool fDetailedOutputWithErrorsAndPulls;
      RooArgSet* fDetailedOutput; //!
      RooArgSet fConditionalObs;    // conditional observables 
      
      TString fVarName;

      static Bool_t fgAlwaysReuseNll ;
      Bool_t fReuseNll ;
      TString fMinimizer;
      Int_t fStrategy;
      Double_t fTolerance; 
      Int_t fPrintLevel;
      Bool_t fLOffset ;

   protected:

      ClassDef(ProfileLikelihoodTestStat,9)   // implements the profile likelihood ratio as a test statistic to be used with several tools
   };
}


#endif
