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


#include "Rtypes.h"

#include "RooStats/TestStatistic.h"


#include "RooRealVar.h"

#include "Math/MinimizerOptions.h"

#include "RooStats/RooStatsUtils.h"


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
   fLOffset = RooStats::IsNLLOffset() ;

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
       fLOffset = RooStats::IsNLLOffset() ;

       fVarName = "Profile Likelihood Ratio";
       fReuseNll = false;
       fMinimizer=::ROOT::Math::MinimizerOptions::DefaultMinimizerType().c_str();
       fStrategy=::ROOT::Math::MinimizerOptions::DefaultStrategy();
       // avoid default tolerance to be too small (1. is default in RooMinimizer)
       fTolerance=TMath::Max(1.,::ROOT::Math::MinimizerOptions::DefaultTolerance());
       fPrintLevel=::ROOT::Math::MinimizerOptions::DefaultPrintLevel();
     }

     ~ProfileLikelihoodTestStat() override {
       if(fNll) delete fNll;
       if(fCachedBestFitParams) delete fCachedBestFitParams;
       if(fDetailedOutput) delete fDetailedOutput;
     }

     void SetOneSided(bool flag=true) {fLimitType = (flag ? oneSided : twoSided);}
     void SetOneSidedDiscovery(bool flag=true) {fLimitType = (flag ? oneSidedDiscovery : twoSided);}
     void SetSigned(bool flag=true) {fSigned = flag;}  // +/- t_mu instead of t_mu>0 with one-sided settings

     bool IsTwoSided() const { return fLimitType == twoSided; }
     bool IsOneSidedDiscovery() const { return fLimitType == oneSidedDiscovery; }

     static void SetAlwaysReuseNLL(bool flag);

     void SetReuseNLL(bool flag) { fReuseNll = flag ; }
     void SetLOffset(bool flag=true) { fLOffset = flag ; }

     void SetMinimizer(const char* minimizer){ fMinimizer=minimizer;}
     void SetStrategy(Int_t strategy){fStrategy=strategy;}
     void SetTolerance(double tol){fTolerance=tol;}
     void SetPrintLevel(Int_t printlevel){fPrintLevel=printlevel;}

     /// Main interface to evaluate the test statistic on a dataset
     double Evaluate(RooAbsData& data, RooArgSet& paramsOfInterest) override {
        return EvaluateProfileLikelihood(0, data, paramsOfInterest);
     }

     /// evaluate  the profile likelihood ratio (type = 0) or the minimum of likelihood (type=1) or the conditional LL (type = 2)
     virtual double EvaluateProfileLikelihood(int type, RooAbsData &data, RooArgSet & paramsOfInterest);

     virtual void EnableDetailedOutput( bool e=true, bool withErrorsAndPulls=false ) {
        fDetailedOutputEnabled = e;
        fDetailedOutputWithErrorsAndPulls = withErrorsAndPulls;
        delete fDetailedOutput;
        fDetailedOutput = NULL;
     }
     /// Returns detailed output. The value returned by this function is updated after each call to Evaluate().
     /// The returned RooArgSet contains the following:
     ///
     ///  - the minimum nll, fitstatus and convergence quality for each fit </li>
     ///  - for each fit and for each non-constant parameter, the value, error and pull of the parameter are stored </li>
     ///
     const RooArgSet* GetDetailedOutput(void) const override {
      return fDetailedOutput;
     }

     /// set the conditional observables which will be used when creating the NLL
     /// so the pdf's will not be normalized on the conditional observables when computing the NLL
     void SetConditionalObservables(const RooArgSet& set) override {fConditionalObs.removeAll(); fConditionalObs.add(set);}

     /// set the global observables which will be used when creating the NLL
     /// so the constraint pdf's will be normalized correctly on the global observables when computing the NLL
     void SetGlobalObservables(const RooArgSet& set) override {fGlobalObs.removeAll(); fGlobalObs.add(set);}

     virtual void SetVarName(const char* name) { fVarName = name; }
     const TString GetVarName() const override {return fVarName;}

     virtual RooAbsPdf * GetPdf() const { return fPdf; }

  private:

     RooFitResult* GetMinNLL();

   private:

      RooAbsPdf* fPdf;
      RooAbsReal* fNll; //!
      const RooArgSet* fCachedBestFitParams;
      RooAbsData* fLastData;
      //      double fLastMLE;
      LimitType fLimitType;
      bool fSigned;

      /// this will store a snapshot of the unconditional nuisance
      /// parameter fit.
      bool fDetailedOutputEnabled;
      bool fDetailedOutputWithErrorsAndPulls;
      RooArgSet* fDetailedOutput; ///<!
      RooArgSet fConditionalObs;  ///< conditional observables
      RooArgSet fGlobalObs;       ///< global observables

      TString fVarName;

      static bool fgAlwaysReuseNll ;
      bool fReuseNll ;
      TString fMinimizer;
      Int_t fStrategy;
      double fTolerance;
      Int_t fPrintLevel;
      bool fLOffset ;

   protected:

      ClassDefOverride(ProfileLikelihoodTestStat,10)   // implements the profile likelihood ratio as a test statistic to be used with several tools
   };
}


#endif
