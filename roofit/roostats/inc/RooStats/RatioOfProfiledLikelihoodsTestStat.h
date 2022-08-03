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


#include "Rtypes.h"

#include "RooStats/TestStatistic.h"

#include "RooStats/ProfileLikelihoodTestStat.h"


namespace RooStats {

   class RatioOfProfiledLikelihoodsTestStat: public TestStatistic {

   public:

      RatioOfProfiledLikelihoodsTestStat() :
         fNullProfile(),
         fAltProfile(),
         fAltPOI(nullptr),
         fSubtractMLE(true),
         fDetailedOutputEnabled(false),
         fDetailedOutput(nullptr)
      {
         // Proof constructor. Don't use.
      }

      RatioOfProfiledLikelihoodsTestStat(RooAbsPdf& nullPdf, RooAbsPdf& altPdf,
                                         const RooArgSet* altPOI=nullptr) :
         fNullProfile(nullPdf),
         fAltProfile(altPdf),
         fSubtractMLE(true),
         fDetailedOutputEnabled(false),
         fDetailedOutput(nullptr)
      {
         //  Calculates the ratio of profiled likelihoods.

         if(altPOI)
            fAltPOI = (RooArgSet*) altPOI->snapshot();
         else
            fAltPOI = new RooArgSet(); // empty set

      }

      //__________________________________________
      ~RatioOfProfiledLikelihoodsTestStat(void) override {
         if(fAltPOI) delete fAltPOI;
         if(fDetailedOutput) delete fDetailedOutput;
      }


      /// returns -logL(poi, conditional MLE of nuisance params)
      /// it does not subtract off the global MLE
      /// because  nuisance parameters of null and alternate may not
      /// be the same.
      double ProfiledLikelihood(RooAbsData& data, RooArgSet& poi, RooAbsPdf& pdf);

      /// evaluate the ratio of profile likelihood
      double Evaluate(RooAbsData& data, RooArgSet& nullParamsOfInterest) override;

      virtual void EnableDetailedOutput( bool e=true ) {
         fDetailedOutputEnabled = e;
         fNullProfile.EnableDetailedOutput(fDetailedOutputEnabled);
         fAltProfile.EnableDetailedOutput(fDetailedOutputEnabled);
      }

      static void SetAlwaysReuseNLL(bool flag);

      void SetReuseNLL(bool flag) {
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
      void SetTolerance(double tol){
         fNullProfile.SetTolerance(tol);
         fAltProfile.SetTolerance(tol);
      }
      void SetPrintLevel(Int_t printLevel){
         fNullProfile.SetPrintLevel(printLevel);
         fAltProfile.SetPrintLevel(printLevel);
      }

      /// set the conditional observables which will be used when creating the NLL
      /// so the pdf's will not be normalized on the conditional observables when computing the NLL
      void SetConditionalObservables(const RooArgSet& set) override {
         fNullProfile.SetConditionalObservables(set);
         fAltProfile.SetConditionalObservables(set);
      }

      /// set the global observables which will be used when creating the NLL
      /// so the constraint pdf's will be normalized correctly on the global observables when computing the NLL
      void SetGlobalObservables(const RooArgSet& set) override {
         fNullProfile.SetGlobalObservables(set);
         fAltProfile.SetGlobalObservables(set);
      }

      /// Returns detailed output. The value returned by this function is updated after each call to Evaluate().
      /// The returned RooArgSet contains the following for the alternative and null hypotheses:
      ///  - the minimum nll, fitstatus and convergence quality for each fit
      ///  - for each fit and for each non-constant parameter, the value, error and pull of the parameter are stored
      const RooArgSet* GetDetailedOutput(void) const override {
         return fDetailedOutput;
      }




      const TString GetVarName() const override { return "log(L(#mu_{1},#hat{#nu}_{1}) / L(#mu_{0},#hat{#nu}_{0}))"; }

      //    const bool PValueIsRightTail(void) { return false; } // overwrites default

      void SetSubtractMLE(bool subtract){fSubtractMLE = subtract;}

   private:

      ProfileLikelihoodTestStat fNullProfile;
      ProfileLikelihoodTestStat fAltProfile;

      RooArgSet* fAltPOI;
      bool fSubtractMLE;
      static bool fgAlwaysReuseNll ;

      bool fDetailedOutputEnabled;
      RooArgSet* fDetailedOutput;


   protected:
      ClassDefOverride(RatioOfProfiledLikelihoodsTestStat,3)  // implements the ratio of profiled likelihood as test statistic
   };

}


#endif
