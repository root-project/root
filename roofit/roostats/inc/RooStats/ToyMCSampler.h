// @(#)root/roostats:$Id$
// Author: Sven Kreiss and Kyle Cranmer    June 2010
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
// Additions and modifications by Mario Pelliccioni
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_ToyMCSampler
#define ROOSTATS_ToyMCSampler

#include "RooStats/TestStatSampler.h"
#include "RooStats/SamplingDistribution.h"
#include "RooStats/TestStatistic.h"
#include "RooStats/ModelConfig.h"
#include "RooStats/ProofConfig.h"

#include "RooWorkspace.h"
#include "RooMsgService.h"
#include "RooAbsPdf.h"
#include "RooRealVar.h"
#include "RooDataSet.h"

#include <vector>
#include <list>
#include <string>
#include <sstream>
#include <memory>

namespace RooStats {

  class DetailedOutputAggregator;

class NuisanceParametersSampler {

   public:
      NuisanceParametersSampler(RooAbsPdf *prior=NULL, const RooArgSet *parameters=NULL, Int_t nToys=1000, bool asimov=false) :
         fPrior(prior),
         fParams(parameters),
         fNToys(nToys),
         fExpected(asimov),
         fIndex(0)
      {
         if(prior) Refresh();
      }
      virtual ~NuisanceParametersSampler() = default;

      void NextPoint(RooArgSet& nuisPoint, Double_t& weight);

   protected:
      void Refresh();

   private:
      RooAbsPdf *fPrior;           // prior for nuisance parameters
      const RooArgSet *fParams;    // nuisance parameters
      Int_t fNToys;
      bool fExpected;

      std::unique_ptr<RooAbsData> fPoints;         // generated nuisance parameter points
      Int_t fIndex;                // current index in fPoints array
};

class ToyMCSampler: public TestStatSampler {

   public:

      ToyMCSampler();
      ToyMCSampler(TestStatistic &ts, Int_t ntoys);
      ~ToyMCSampler() override;

      static void SetAlwaysUseMultiGen(bool flag);

      void SetUseMultiGen(bool flag) { fUseMultiGen = flag ; }

      /// main interface
      SamplingDistribution* GetSamplingDistribution(RooArgSet& paramPoint) override;
      virtual RooDataSet* GetSamplingDistributions(RooArgSet& paramPoint);
      virtual RooDataSet* GetSamplingDistributionsSingleWorker(RooArgSet& paramPoint);

      virtual SamplingDistribution* AppendSamplingDistribution(
         RooArgSet& allParameters,
         SamplingDistribution* last,
         Int_t additionalMC
      );


      /// The pdf can be NULL in which case the density from SetPdf()
      /// is used. The snapshot and TestStatistic is also optional.
      virtual void AddTestStatistic(TestStatistic* t = NULL) {
         if( t == NULL ) {
            oocoutI(nullptr,InputArguments) << "No test statistic given. Doing nothing." << std::endl;
            return;
         }

         fTestStatistics.push_back( t );
      }

      /// generates toy data
      ///   without weight
      virtual RooAbsData* GenerateToyData(RooArgSet& paramPoint, RooAbsPdf& pdf) const {
         if(fExpectedNuisancePar) oocoutE(nullptr,InputArguments) << "ToyMCSampler: using expected nuisance parameters but ignoring weight. Use GetSamplingDistribution(paramPoint, weight) instead." << std::endl;
         double weight;
         return GenerateToyData(paramPoint, weight, pdf);
      }
      virtual RooAbsData* GenerateToyData(RooArgSet& paramPoint) const { return GenerateToyData(paramPoint,*fPdf); }
      /// generates toy data
      ///   with weight
      virtual RooAbsData* GenerateToyData(RooArgSet& paramPoint, double& weight, RooAbsPdf& pdf) const;
      virtual RooAbsData* GenerateToyData(RooArgSet& paramPoint, double& weight) const { return GenerateToyData(paramPoint,weight,*fPdf); }

      /// generate global observables
      virtual void GenerateGlobalObservables(RooAbsPdf& pdf) const;


      /// Main interface to evaluate the test statistic on a dataset
      virtual Double_t EvaluateTestStatistic(RooAbsData& data, RooArgSet& nullPOI, int i ) {
         return fTestStatistics[i]->Evaluate(data, nullPOI);
      }
      Double_t EvaluateTestStatistic(RooAbsData& data, RooArgSet& nullPOI) override { return EvaluateTestStatistic( data,nullPOI, 0 ); }
      virtual RooArgList* EvaluateAllTestStatistics(RooAbsData& data, const RooArgSet& poi);


      virtual TestStatistic* GetTestStatistic(unsigned int i) const {
         if( fTestStatistics.size() <= i ) return NULL;
         return fTestStatistics[i];
      }
      TestStatistic* GetTestStatistic(void) const override { return GetTestStatistic(0); }

      Double_t ConfidenceLevel() const override { return 1. - fSize; }
      void Initialize(
         RooAbsArg& /*testStatistic*/,
         RooArgSet& /*paramsOfInterest*/,
         RooArgSet& /*nuisanceParameters*/
      ) override {}

      virtual Int_t GetNToys(void) { return fNToys; }
      virtual void SetNToys(const Int_t ntoy) { fNToys = ntoy; }
      /// Forces the generation of exactly `n` events even for extended PDFs. Set to 0 to
      /// use the Poisson-distributed events from the extended PDF.
      virtual void SetNEventsPerToy(const Int_t nevents) {
         fNEvents = nevents;
      }


      /// Set the Pdf, add to the the workspace if not already there
      void SetParametersForTestStat(const RooArgSet& nullpoi) override {
         fParametersForTestStat.reset( nullpoi.snapshot() );
      }

      void SetPdf(RooAbsPdf& pdf) override { fPdf = &pdf; ClearCache(); }

      /// How to randomize the prior. Set to NULL to deactivate randomization.
      void SetPriorNuisance(RooAbsPdf* pdf) override {
         fPriorNuisance = pdf;
         if (fNuisanceParametersSampler) {
            delete fNuisanceParametersSampler;
            fNuisanceParametersSampler = NULL;
         }
      }
      /// specify the nuisance parameters (eg. the rest of the parameters)
      void SetNuisanceParameters(const RooArgSet& np) override { fNuisancePars = &np; }
      /// specify the observables in the dataset (needed to evaluate the test statistic)
      void SetObservables(const RooArgSet& o) override { fObservables = &o; }
      /// specify the conditional observables
      void SetGlobalObservables(const RooArgSet& o) override { fGlobalObservables = &o; }


      /// set the size of the test (rate of Type I error) ( Eg. 0.05 for a 95% Confidence Interval)
      void SetTestSize(Double_t size) override { fSize = size; }
      /// set the confidence level for the interval (eg. 0.95 for a 95% Confidence Interval)
      void SetConfidenceLevel(Double_t cl) override { fSize = 1. - cl; }

      /// Set the TestStatistic (want the argument to be a function of the data & parameter points
      virtual void SetTestStatistic(TestStatistic *testStatistic, unsigned int i) {
         if( fTestStatistics.size() < i ) {
            oocoutE(nullptr,InputArguments) << "Cannot set test statistic for this index." << std::endl;
            return;
         }
         if( fTestStatistics.size() == i)
            fTestStatistics.push_back(testStatistic);
         else
            fTestStatistics[i] = testStatistic;
      }
      void SetTestStatistic(TestStatistic *t) override { return SetTestStatistic(t,0); }

      virtual void SetExpectedNuisancePar(bool i = true) { fExpectedNuisancePar = i; }
      virtual void SetAsimovNuisancePar(bool i = true) { fExpectedNuisancePar = i; }

      /// Checks for sufficient information to do a GetSamplingDistribution(...).
      bool CheckConfig(void);

      /// control to use bin data generation (=> see RooFit::AllBinned() option)
      void SetGenerateBinned(bool binned = true) { fGenerateBinned = binned; }
      /// name of the tag for individual components to be generated binned (=> see RooFit::GenBinned() option)
      void SetGenerateBinnedTag( const char* binnedTag = "" ) { fGenerateBinnedTag = binnedTag; }
      /// set auto binned generation (=> see RooFit::AutoBinned() option)
      void SetGenerateAutoBinned( bool autoBinned = true ) { fGenerateAutoBinned = autoBinned; }

      /// Set the name of the sampling distribution used for plotting
      void SetSamplingDistName(const char* name) override { if(name) fSamplingDistName = name; }
      std::string GetSamplingDistName(void) { return fSamplingDistName; }

      /// This option forces a maximum number of total toys.
      void SetMaxToys(Double_t t) { fMaxToys = t; }

      void SetToysLeftTail(Double_t toys, Double_t threshold) {
         fToysInTails = toys;
         fAdaptiveLowLimit = threshold;
         fAdaptiveHighLimit = RooNumber::infinity();
      }
      void SetToysRightTail(Double_t toys, Double_t threshold) {
         fToysInTails = toys;
         fAdaptiveHighLimit = threshold;
         fAdaptiveLowLimit = -RooNumber::infinity();
      }
      void SetToysBothTails(Double_t toys, Double_t low_threshold, Double_t high_threshold) {
         fToysInTails = toys;
         fAdaptiveHighLimit = high_threshold;
         fAdaptiveLowLimit = low_threshold;
      }

      /// calling with argument or NULL deactivates proof
      void SetProofConfig(ProofConfig *pc = NULL) { fProofConfig = pc; }

      void SetProtoData(const RooDataSet* d) { fProtoData = d; }

   protected:

      const RooArgList* EvaluateAllTestStatistics(RooAbsData& data, const RooArgSet& poi, DetailedOutputAggregator& detOutAgg);

      /// helper for GenerateToyData
      RooAbsData* Generate(RooAbsPdf &pdf, RooArgSet &observables, const RooDataSet *protoData=NULL, int forceEvents=0) const;

      /// helper method for clearing  the cache
      virtual void ClearCache();


      /// densities, snapshots, and test statistics to reweight to
      RooAbsPdf *fPdf; ///< model (can be alt or null)
      std::unique_ptr<const RooArgSet> fParametersForTestStat;
      std::vector<TestStatistic*> fTestStatistics;

      std::string fSamplingDistName; ///< name of the model
      RooAbsPdf *fPriorNuisance;     ///< prior pdf for nuisance parameters
      const RooArgSet *fNuisancePars;
      const RooArgSet *fObservables;
      const RooArgSet *fGlobalObservables;
      Int_t fNToys;   ///< number of toys to generate
      Int_t fNEvents; ///< number of events per toy (may be ignored depending on settings)
      Double_t fSize;
      bool fExpectedNuisancePar; ///< whether to use expectation values for nuisance parameters (ie Asimov data set)
      bool fGenerateBinned;
      TString fGenerateBinnedTag;
      bool fGenerateAutoBinned;

      /// minimum no of toys in tails for adaptive sampling
      /// (taking weights into account, therefore double)
      /// Default: 0.0 which means no adaptive sampling
      Double_t fToysInTails;
      /// maximum no of toys
      /// (taking weights into account, therefore double)
      Double_t fMaxToys;
      /// tails
      Double_t fAdaptiveLowLimit;
      Double_t fAdaptiveHighLimit;

      const RooDataSet *fProtoData; ///< in dev

      ProofConfig *fProofConfig;   ///<!

      mutable NuisanceParametersSampler *fNuisanceParametersSampler; ///<!

      // objects below cache information and are mutable and non-persistent
      mutable std::unique_ptr<RooArgSet> _allVars; ///<!
      mutable std::vector<RooAbsPdf*> _pdfList;    ///<! We don't own those objects
      mutable std::vector<std::unique_ptr<RooArgSet>> _obsList;         ///<!
      mutable std::vector<std::unique_ptr<RooAbsPdf::GenSpec>> _gsList; ///<!
      mutable std::unique_ptr<RooAbsPdf::GenSpec> _gs1; ///<! GenSpec #1
      mutable std::unique_ptr<RooAbsPdf::GenSpec> _gs2; ///<! GenSpec #2
      mutable std::unique_ptr<RooAbsPdf::GenSpec> _gs3; ///<! GenSpec #3
      mutable std::unique_ptr<RooAbsPdf::GenSpec> _gs4; ///<! GenSpec #4

      static bool fgAlwaysUseMultiGen ;  ///< Use PrepareMultiGen always
      bool fUseMultiGen ;                ///< Use PrepareMultiGen?

   protected:
   ClassDefOverride(ToyMCSampler, 4) // A simple implementation of the TestStatSampler interface
};
}


#endif
