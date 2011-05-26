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

//_________________________________________________
/*
BEGIN_HTML
<p>
DebuggingSampler is a simple implementation of the DistributionCreator interface used for debugging.
The sampling distribution is uniformly random between [0,1] and is INDEPENDENT of the data.  So it is not useful
for true statistical tests, but it is useful for debugging.
</p>
END_HTML
*/
//

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

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
     virtual ~DebuggingSampler() {
       delete fRand;
       delete fTestStatistic;
     }
    
      // Main interface to get a ConfInterval, pure virtual
     virtual SamplingDistribution* GetSamplingDistribution(RooArgSet& paramsOfInterest)  {
       paramsOfInterest = paramsOfInterest; // avoid warning
       // normally this method would be complex, but here it is simple for debugging
       std::vector<Double_t> testStatVec;
       for(Int_t i=0; i<1000; ++i){
	 testStatVec.push_back( fRand->Uniform() );
       }
       return new SamplingDistribution("UniformSamplingDist", "for debugging", testStatVec );
     } 

      // Main interface to evaluate the test statistic on a dataset
     virtual Double_t EvaluateTestStatistic(RooAbsData& /*data*/, RooArgSet& /*paramsOfInterest*/)  {
       //       data = data; // avoid warning
       //       paramsOfInterest = paramsOfInterest; // avoid warning
       return fRand->Uniform();
     }

      // Get the TestStatistic
      virtual TestStatistic* GetTestStatistic()  const {
         // TODO change to Roo... notifications
         cout << "GetTestStatistic() IS NOT IMPLEMENTED FOR THIS SAMPLER. Returning NULL." << endl;
         return NULL; /*fTestStatistic;*/
      }
    
      // Get the Confidence level for the test
      virtual Double_t ConfidenceLevel()  const {return 1.-fSize;}  

      // Common Initialization
      virtual void Initialize(RooAbsArg& /* testStatistic */, RooArgSet& /* paramsOfInterest */, RooArgSet& /* nuisanceParameters */ ) {
      }

      // Set the Pdf, add to the the workspace if not already there
      virtual void SetPdf(RooAbsPdf&) {}

      // specify the parameters of interest in the interval
      virtual void SetParameters(RooArgSet&) {}
      // specify the nuisance parameters (eg. the rest of the parameters)
      virtual void SetNuisanceParameters(const RooArgSet&) {}
      // specify the values of parameters used when evaluating test statistic
      virtual void SetParametersForTestStat(const RooArgSet& ) {}
      // specify the conditional observables
      virtual void SetGlobalObservables(const RooArgSet& ) {}


      // set the size of the test (rate of Type I error) ( Eg. 0.05 for a 95% Confidence Interval)
      virtual void SetTestSize(Double_t size) {fSize = size;}
      // set the confidence level for the interval (eg. 0.95 for a 95% Confidence Interval)
      virtual void SetConfidenceLevel(Double_t cl) {fSize = 1.-cl;}

      // Set the TestStatistic (want the argument to be a function of the data & parameter points
      virtual void SetTestStatistic(TestStatistic* /*testStatistic*/) {
         // TODO change to Roo... notifications
         cout << "SetTestStatistic(...) IS NOT IMPLEMENTED FOR THIS SAMPLER" << endl;
      }
      
   private:
      Double_t fSize;
      RooRealVar* fTestStatistic;
      TRandom* fRand;

   protected:
      ClassDef(DebuggingSampler,1)   // A simple implementation of the DistributionCreator interface
   };
}


#endif
