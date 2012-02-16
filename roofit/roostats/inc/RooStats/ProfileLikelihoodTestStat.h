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

#include <vector>

#include "RooStats/RooStatsUtils.h"

//#include "RooStats/DistributionCreator.h"
#include "RooStats/SamplingDistribution.h"
#include "RooStats/TestStatistic.h"

#include "RooStats/RooStatsUtils.h"

#include "RooRealVar.h"
#include "RooProfileLL.h"
#include "RooNLLVar.h"
#include "RooMsgService.h"

#include "RooMinuit.h"
#include "RooMinimizer.h"
#include "Math/MinimizerOptions.h"
#include "TStopwatch.h"

namespace RooStats {

  class ProfileLikelihoodTestStat : public TestStatistic{

     enum LimitType {twoSided, oneSided, oneSidedDiscovery};

   public:
     ProfileLikelihoodTestStat() {
        // Proof constructor. Do not use.
        fPdf = 0;
        fProfile = 0;
        fNll = 0;
        fCachedBestFitParams = 0;
        fLastData = 0;
	fLimitType = twoSided;
	fSigned = false;
        fReuseNll = false;
	fMinimizer=::ROOT::Math::MinimizerOptions::DefaultMinimizerType().c_str();
	fStrategy=::ROOT::Math::MinimizerOptions::DefaultStrategy();
        fTolerance=TMath::Max(1.,::ROOT::Math::MinimizerOptions::DefaultTolerance());
	fPrintLevel=::ROOT::Math::MinimizerOptions::DefaultPrintLevel();

     }
     ProfileLikelihoodTestStat(RooAbsPdf& pdf) {
       fPdf = &pdf;
       fProfile = 0;
       fNll = 0;
       fCachedBestFitParams = 0;
       fLastData = 0;
       fLimitType = twoSided;
       fSigned = false;
       fReuseNll = false;
       fMinimizer=::ROOT::Math::MinimizerOptions::DefaultMinimizerType().c_str();
       fStrategy=::ROOT::Math::MinimizerOptions::DefaultStrategy();
       // avoid default tolerance to be too small (1. is default in RooMinimizer)
       fTolerance=TMath::Max(1.,::ROOT::Math::MinimizerOptions::DefaultTolerance());
       fPrintLevel=::ROOT::Math::MinimizerOptions::DefaultPrintLevel();
     }
     virtual ~ProfileLikelihoodTestStat() {
       //       delete fRand;
       //       delete fTestStatistic;
       if(fProfile) delete fProfile;
       if(fNll) delete fNll;
       if(fCachedBestFitParams) delete fCachedBestFitParams;
     }
     void SetOneSided(Bool_t flag=true) {fLimitType = (flag ? oneSided : twoSided);}
     void SetOneSidedDiscovery(Bool_t flag=true) {fLimitType = (flag ? oneSidedDiscovery : twoSided);}
     void SetSigned(Bool_t flag=true) {fSigned = flag;}  // +/- t_mu instead of t_mu>0 with one-sided settings

     static void SetAlwaysReuseNLL(Bool_t flag) { fgAlwaysReuseNll = flag ; }
     void SetReuseNLL(Bool_t flag) { fReuseNll = flag ; }

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
    
      virtual const TString GetVarName() const {return "Profile Likelihood Ratio";}
      
      //      const bool PValueIsRightTail(void) { return false; } // overwrites default

  private:

     double GetMinNLL(int& status);

   private:
      RooProfileLL* fProfile; //!
      RooAbsPdf* fPdf;
      RooNLLVar* fNll; //!
      const RooArgSet* fCachedBestFitParams;
      RooAbsData* fLastData;
      //      Double_t fLastMLE;
      LimitType fLimitType;
      Bool_t fSigned;

      static Bool_t fgAlwaysReuseNll ;
      Bool_t fReuseNll ;
      TString fMinimizer;
      Int_t fStrategy;
      Double_t fTolerance; 
      Int_t fPrintLevel;

   protected:
      ClassDef(ProfileLikelihoodTestStat,7)   // implements the profile likelihood ratio as a test statistic to be used with several tools
   };
}


#endif
