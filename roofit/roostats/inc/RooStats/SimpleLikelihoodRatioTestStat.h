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

#include "Rtypes.h"

#include "RooAbsPdf.h"
#include "RooRealVar.h"

#include "RooStats/TestStatistic.h"

namespace RooStats {

   class SimpleLikelihoodRatioTestStat : public TestStatistic {

   public:

      /// Constructor for proof. Do not use.
      SimpleLikelihoodRatioTestStat() :
         fNullPdf(nullptr), fAltPdf(nullptr)
      {
         fFirstEval = true;
         fDetailedOutputEnabled = false;
         fDetailedOutput = nullptr;
         fNullParameters = nullptr;
         fAltParameters = nullptr;
         fReuseNll=false ;
      }

      /// Takes null and alternate parameters from PDF. Can be overridden.
      SimpleLikelihoodRatioTestStat(
         RooAbsPdf& nullPdf,
         RooAbsPdf& altPdf
      ) :
         fFirstEval(true)
      {
         fNullPdf = &nullPdf;
         fAltPdf = &altPdf;

         std::unique_ptr<RooArgSet> allNullVars{fNullPdf->getVariables()};
         fNullParameters = (RooArgSet*) allNullVars->snapshot();

         std::unique_ptr<RooArgSet> allAltVars{fAltPdf->getVariables()};
         fAltParameters = (RooArgSet*) allAltVars->snapshot();

         fDetailedOutputEnabled = false;
         fDetailedOutput = nullptr;

         fReuseNll=false ;
      }

      /// Takes null and alternate parameters from values in nullParameters
      /// and altParameters. Can be overridden.
      SimpleLikelihoodRatioTestStat(
         RooAbsPdf& nullPdf,
         RooAbsPdf& altPdf,
         const RooArgSet& nullParameters,
         const RooArgSet& altParameters
      ) :
         fFirstEval(true)
      {
         fNullPdf = &nullPdf;
         fAltPdf = &altPdf;

         fNullParameters = (RooArgSet*) nullParameters.snapshot();
         fAltParameters = (RooArgSet*) altParameters.snapshot();

         fDetailedOutputEnabled = false;
         fDetailedOutput = nullptr;

         fReuseNll=false ;
      }

      ~SimpleLikelihoodRatioTestStat() override {
         if (fNullParameters) delete fNullParameters;
         if (fAltParameters) delete fAltParameters;
         if (fDetailedOutput) delete fDetailedOutput;
      }

      static void SetAlwaysReuseNLL(bool flag);

      void SetReuseNLL(bool flag) { fReuseNll = flag ; }

      void SetNullParameters(const RooArgSet& nullParameters) {
         if (fNullParameters) delete fNullParameters;
         fFirstEval = true;
         fNullParameters = (RooArgSet*) nullParameters.snapshot();
      }

      void SetAltParameters(const RooArgSet& altParameters) {
         if (fAltParameters) delete fAltParameters;
         fFirstEval = true;
         fAltParameters = (RooArgSet*) altParameters.snapshot();
      }

      /// this should be possible with RooAbsCollection
      bool ParamsAreEqual() {
         if (!fNullParameters->equals(*fAltParameters)) return false;

         bool ret = true;

         for (auto nullIt = fNullParameters->begin(), altIt = fAltParameters->begin();
              nullIt != fNullParameters->end() && altIt != fAltParameters->end(); ++nullIt, ++altIt) {
            RooAbsReal *null = static_cast<RooAbsReal *>(*nullIt);
            RooAbsReal *alt = static_cast<RooAbsReal *>(*altIt);
            if (null->getVal() != alt->getVal())
               ret = false;
         }

         return ret;
      }


      /// set the conditional observables which will be used when creating the NLL
      /// so the pdf's will not be normalized on the conditional observables when computing the NLL
      void SetConditionalObservables(const RooArgSet& set) override {fConditionalObs.removeAll(); fConditionalObs.add(set);}

      /// set the global observables which will be used when creating the NLL
      /// so the constraint pdf's will be normalized correctly on the global observables when computing the NLL
      void SetGlobalObservables(const RooArgSet& set) override {fGlobalObs.removeAll(); fGlobalObs.add(set);}

      double Evaluate(RooAbsData& data, RooArgSet& nullPOI) override;

      virtual void EnableDetailedOutput( bool e=true ) { fDetailedOutputEnabled = e; fDetailedOutput = nullptr; }
      const RooArgSet* GetDetailedOutput(void) const override { return fDetailedOutput; }

      const TString GetVarName() const override {
         return "log(L(#mu_{1}) / L(#mu_{0}))";
      }

   private:

      RooAbsPdf* fNullPdf;
      RooAbsPdf* fAltPdf;
      RooArgSet* fNullParameters;
      RooArgSet* fAltParameters;
      RooArgSet fConditionalObs;
      RooArgSet fGlobalObs;
      bool fFirstEval;

      bool fDetailedOutputEnabled;
      RooArgSet* fDetailedOutput; ///<!

      std::unique_ptr<RooAbsReal> fNllNull; ///<! transient copy of the null NLL
      std::unique_ptr<RooAbsReal> fNllAlt;  ///<!  transient copy of the alt NLL
      static bool fgAlwaysReuseNll ;
      bool fReuseNll ;


   protected:
   ClassDefOverride(SimpleLikelihoodRatioTestStat,4)
};

}

#endif
