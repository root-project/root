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

//_________________________________________________
/*
BEGIN_HTML
<p>
ToyMCSampler is an implementation of the TestStatSampler interface.
It generates Toy Monte Carlo for a given parameter point and evaluates a
TestStatistic.
</p>

<p>
For parallel runs, ToyMCSampler can be given an instance of ProofConfig
and then run in parallel using proof or proof-lite. Internally, it uses
ToyMCStudy with the RooStudyManager.
</p>
END_HTML
*/
//

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#include <vector>
#include <sstream>

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

namespace RooStats {

// only used inside ToyMCSampler, ie "private" in the cxx file
class NuisanceParametersSampler;

class ToyMCSampler: public TestStatSampler {

   public:
      ToyMCSampler() :
         fTestStat(NULL), fSamplingDistName("temp"), fNToys(1)
      {
         // Proof constructor. Do not use.

         fPdf = NULL;
         fPriorNuisance = NULL;
         fNullPOI = NULL;
         fNuisancePars = NULL;
         fObservables = NULL;
         fGlobalObservables = NULL;

         fSize = 0.05;
         fNEvents = 0;
         fGenerateBinned = kFALSE;
         fExpectedNuisancePar = kFALSE;

         fToysInTails = 0.0;
         fMaxToys = RooNumber::infinity();
         fAdaptiveLowLimit = -RooNumber::infinity();
         fAdaptiveHighLimit = RooNumber::infinity();

         fImportanceDensity = NULL;
         fImportanceSnapshot = NULL;
         fProtoData = NULL;

         fProofConfig = NULL;
         fNuisanceParametersSampler = NULL;

	_allVars = NULL ;
	_gs1 = NULL ;
	_gs2 = NULL ;
	_gs3 = NULL ;
	_gs4 = NULL ;

	fUseMultiGen = kFALSE ;
      }
      ToyMCSampler(TestStatistic &ts, Int_t ntoys) :
         fTestStat(&ts), fSamplingDistName(ts.GetVarName()), fNToys(ntoys)
      {
         fPdf = NULL;
         fPriorNuisance = NULL;
         fNullPOI = NULL;
         fNuisancePars = NULL;
         fObservables = NULL;
         fGlobalObservables = NULL;

         fSize = 0.05;
         fNEvents = 0;
         fGenerateBinned = kFALSE;
         fExpectedNuisancePar = kFALSE;

         fToysInTails = 0.0;
         fMaxToys = RooNumber::infinity();
         fAdaptiveLowLimit = -RooNumber::infinity();
         fAdaptiveHighLimit = RooNumber::infinity();

         fImportanceDensity = NULL;
         fImportanceSnapshot = NULL;
         fProtoData = NULL;

         fProofConfig = NULL;
         fNuisanceParametersSampler = NULL;

	_allVars = NULL ;
	_gs1 = NULL ;
	_gs2 = NULL ;
	_gs3 = NULL ;
	_gs4 = NULL ;

	fUseMultiGen = kFALSE ;
      }


      virtual ~ToyMCSampler();

      static void setAlwaysUseMultiGen(Bool_t flag) { fAlwaysUseMultiGen = flag ; }
      void setUseMultiGen(Bool_t flag) { fUseMultiGen = flag ; }

      // main interface
      virtual SamplingDistribution* GetSamplingDistribution(RooArgSet& paramPoint);

      virtual SamplingDistribution* GetSamplingDistributionSingleWorker(RooArgSet& paramPoint);


      // generates toy data
      //   without weight
      virtual RooAbsData* GenerateToyData(RooArgSet& paramPoint) const {
         if(fExpectedNuisancePar) oocoutE((TObject*)NULL,InputArguments) << "ToyMCSampler: using expected nuisance parameters but ignoring weight. Use GetSamplingDistribution(paramPoint, weight) instead." << endl;
         double weight;
         return GenerateToyData(paramPoint, weight);
      }
      //   importance sampling without weight does not make sense
      //   so the equivalent function to the one above is omitted
      //
      //   with weight
      virtual RooAbsData* GenerateToyData(RooArgSet& paramPoint, double& weight) const;
      virtual RooAbsData* GenerateToyDataImportanceSampling(RooArgSet& paramPoint, double& weight) const;

      // generate global observables
      virtual void GenerateGlobalObservables(void) const;



      // Extended interface to append to sampling distribution more samples
      virtual SamplingDistribution* AppendSamplingDistribution(RooArgSet& allParameters, 
							       SamplingDistribution* last, 
							       Int_t additionalMC) {

	Int_t tmp = fNToys;
	fNToys = additionalMC;
	SamplingDistribution* newSamples = GetSamplingDistribution(allParameters);
	fNToys = tmp;
	
	if(last){
	  last->Add(newSamples);
	  delete newSamples;
	  return last;
	}

	return newSamples;
      }


      // Main interface to evaluate the test statistic on a dataset
      virtual Double_t EvaluateTestStatistic(RooAbsData& data, RooArgSet& nullPOI) {
         return fTestStat->Evaluate(data, nullPOI);
      }

      virtual TestStatistic* GetTestStatistic() const { return fTestStat; }
      virtual Double_t ConfidenceLevel() const { return 1. - fSize; }
      virtual void Initialize(
         RooAbsArg& /*testStatistic*/,
         RooArgSet& /*paramsOfInterest*/,
         RooArgSet& /*nuisanceParameters*/
      ) {}

      virtual Int_t GetNToys(void) { return fNToys; }
      virtual void SetNToys(const Int_t ntoy) { fNToys = ntoy; }
      virtual void SetNEventsPerToy(const Int_t nevents) {
         // Forces n events even for extended PDFs. Set NEvents=0 to
         // use the Poisson distributed events from the extended PDF.
         fNEvents = nevents;
      }


      // specify the values of parameters used when evaluating test statistic
      virtual void SetParametersForTestStat(const RooArgSet& nullpoi) { fNullPOI = (RooArgSet*)nullpoi.snapshot(); }
      // Set the Pdf, add to the the workspace if not already there
      virtual void SetPdf(RooAbsPdf& pdf) { fPdf = &pdf; }
      // How to randomize the prior. Set to NULL to deactivate randomization.
      virtual void SetPriorNuisance(RooAbsPdf* pdf) { fPriorNuisance = pdf; }
      // specify the nuisance parameters (eg. the rest of the parameters)
      virtual void SetNuisanceParameters(const RooArgSet& np) { fNuisancePars = &np; }
      // specify the observables in the dataset (needed to evaluate the test statistic)
      virtual void SetObservables(const RooArgSet& o) { fObservables = &o; }
      // specify the conditional observables
      virtual void SetGlobalObservables(const RooArgSet& o) { fGlobalObservables = &o; }


      // set the size of the test (rate of Type I error) ( Eg. 0.05 for a 95% Confidence Interval)
      virtual void SetTestSize(Double_t size) { fSize = size; }
      // set the confidence level for the interval (eg. 0.95 for a 95% Confidence Interval)
      virtual void SetConfidenceLevel(Double_t cl) { fSize = 1. - cl; }

      // Set the TestStatistic (want the argument to be a function of the data & parameter points
      virtual void SetTestStatistic(TestStatistic *testStatistic) { fTestStat = testStatistic; }

      virtual void SetExpectedNuisancePar(Bool_t i = kTRUE) { fExpectedNuisancePar = i; }
      virtual void SetAsimovNuisancePar(Bool_t i = kTRUE) { fExpectedNuisancePar = i; }

      // Checks for sufficient information to do a GetSamplingDistribution(...).
      Bool_t CheckConfig(void);

      // control to use bin data generation
      void SetGenerateBinned(bool binned = true) { fGenerateBinned = binned; }

      // Set the name of the sampling distribution used for plotting
      void SetSamplingDistName(const char* name) { if(name) fSamplingDistName = name; }
      string GetSamplingDistName(void) { return fSamplingDistName; }

      // This option forces a maximum number of total toys.
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

      // for importance sampling, specifies the pdf to sample from
      void SetImportanceDensity(RooAbsPdf *p) {
         if(p) oocoutW((TObject*)NULL,InputArguments) << "ToyMCSampler Importance Sampling: This is in beta." << endl;
         fImportanceDensity = p;
      }
      // for importance sampling, a snapshot of the parameters used in importance density
      void SetImportanceSnapshot(const RooArgSet &s) { fImportanceSnapshot = &s; }

      // calling with argument or NULL deactivates proof
      void SetProofConfig(ProofConfig *pc = NULL) { fProofConfig = pc; }

      void SetProtoData(const RooDataSet* d) { fProtoData = d; }

   protected:

      // helper for GenerateToyData
      RooAbsData* Generate(RooAbsPdf &pdf, RooArgSet &observables, const RooDataSet *protoData=NULL, int forceEvents=0) const;



      TestStatistic *fTestStat; // test statistic that is being sampled
      RooAbsPdf *fPdf; // model
      string fSamplingDistName; // name of the model
      RooAbsPdf *fPriorNuisance; // prior pdf for nuisance parameters
      RooArgSet *fNullPOI; // parameters of interest
      const RooArgSet *fNuisancePars;
      const RooArgSet *fObservables;
      const RooArgSet *fGlobalObservables;
      Int_t fNToys; // number of toys to generate
      Int_t fNEvents; // number of events per toy (may be ignored depending on settings)
      Double_t fSize;
      Bool_t fExpectedNuisancePar; // whether to use expectation values for nuisance parameters (ie Asimov data set)
      Bool_t fGenerateBinned;

      // minimum no of toys in tails for adaptive sampling
      // (taking weights into account, therefore double)
      // Default: 0.0 which means no adaptive sampling
      Double_t fToysInTails;
      // maximum no of toys
      // (taking weights into account, therefore double)
      Double_t fMaxToys;
      // tails
      Double_t fAdaptiveLowLimit;
      Double_t fAdaptiveHighLimit;

      RooAbsPdf *fImportanceDensity; // in dev
      const RooArgSet *fImportanceSnapshot; // in dev

      const RooDataSet *fProtoData; // in dev

      ProofConfig *fProofConfig;   //!

      mutable NuisanceParametersSampler *fNuisanceParametersSampler; //!

      // objects below cache information and are mutable and non-persistent
      mutable RooArgSet* _allVars ; //! 
      mutable list<RooAbsPdf*> _pdfList ; //!
      mutable list<RooArgSet*> _obsList ; //!
      mutable list<RooAbsPdf::GenSpec*> _gsList ; //!      
      mutable RooAbsPdf::GenSpec* _gs1 ; //! GenSpec #1 
      mutable RooAbsPdf::GenSpec* _gs2 ; //! GenSpec #2
      mutable RooAbsPdf::GenSpec* _gs3 ; //! GenSpec #3
      mutable RooAbsPdf::GenSpec* _gs4 ; //! GenSpec #4
      
      static Bool_t fAlwaysUseMultiGen ;  // Use PrepareMultiGen always
      Bool_t fUseMultiGen ; // Use PrepareMultiGen?

   protected:
   ClassDef(ToyMCSampler,2) // A simple implementation of the TestStatSampler interface
};
}


#endif
