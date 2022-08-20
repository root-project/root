// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_DebuggingSampler
#define ROOSTATS_DebuggingSampler

/** \class DebuggingSampler
    \ingroup Roostats

DebuggingSampler is a simple implementation of the DistributionCreator interface used for debugging.
The sampling distribution is uniformly random between [0,1] and is INDEPENDENT of the data.  So it is not useful
for true statistical tests, but it is useful for debugging.

*/

#include "Rtypes.h"

#include <vector>

#include "RooStats/TestStatSampler.h"
#include "RooStats/SamplingDistribution.h"

#include "RooRealVar.h"
#include "TRandom.h"

namespace RooStats {

 class DebuggingSampler: public TestStatSampler {

   public:
     DebuggingSampler() {
       fTestStatistic = new RooRealVar("UniformTestStatistic","UniformTestStatistic",0,0,1);
       fRand = new TRandom();
     }
     ~DebuggingSampler() override {
       delete fRand;
       delete fTestStatistic;
     }

     /// Main interface to get a ConfInterval, pure virtual
     SamplingDistribution* GetSamplingDistribution(RooArgSet& paramsOfInterest) override  {
       (void)paramsOfInterest; // avoid warning
       // normally this method would be complex, but here it is simple for debugging
       std::vector<double> testStatVec;
       for(Int_t i=0; i<1000; ++i){
    testStatVec.push_back( fRand->Uniform() );
       }
       return new SamplingDistribution("UniformSamplingDist", "for debugging", testStatVec );
     }

     /// Main interface to evaluate the test statistic on a dataset
     double EvaluateTestStatistic(RooAbsData& /*data*/, RooArgSet& /*paramsOfInterest*/) override  {
       //       data = data; // avoid warning
       //       paramsOfInterest = paramsOfInterest; // avoid warning
       return fRand->Uniform();
     }

      /// Get the TestStatistic
      TestStatistic* GetTestStatistic()  const override {
         std::cout << "GetTestStatistic() IS NOT IMPLEMENTED FOR THIS SAMPLER. Returning nullptr." << std::endl;
         return nullptr; /*fTestStatistic;*/
      }

      /// Get the Confidence level for the test
      double ConfidenceLevel()  const override {return 1.-fSize;}

      /// Common Initialization
      void Initialize(RooAbsArg& /* testStatistic */, RooArgSet& /* paramsOfInterest */, RooArgSet& /* nuisanceParameters */ ) override {
      }

      /// Set the Pdf, add to the workspace if not already there
      void SetPdf(RooAbsPdf&) override {}

      /// specify the parameters of interest in the interval
      virtual void SetParameters(RooArgSet&) {}
      /// specify the nuisance parameters (eg. the rest of the parameters)
      void SetNuisanceParameters(const RooArgSet&) override {}
      /// specify the values of parameters used when evaluating test statistic
      void SetParametersForTestStat(const RooArgSet& ) override {}
      /// specify the conditional observables
      void SetGlobalObservables(const RooArgSet& ) override {}


      /// set the size of the test (rate of Type I error) ( Eg. 0.05 for a 95% Confidence Interval)
      void SetTestSize(double size) override {fSize = size;}
      /// set the confidence level for the interval (eg. 0.95 for a 95% Confidence Interval)
      void SetConfidenceLevel(double cl) override {fSize = 1.-cl;}

      /// Set the TestStatistic (want the argument to be a function of the data & parameter points
      void SetTestStatistic(TestStatistic* /*testStatistic*/) override {
         std::cout << "SetTestStatistic(...) IS NOT IMPLEMENTED FOR THIS SAMPLER" << std::endl;
      }

   private:
      double fSize;
      RooRealVar* fTestStatistic;
      TRandom* fRand;

   protected:
      ClassDefOverride(DebuggingSampler,1)   // A simple implementation of the DistributionCreator interface
   };
}


#endif
